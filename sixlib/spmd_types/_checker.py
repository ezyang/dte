"""
SPMD type checking logic and tensor type tracking.

This module provides:
- Functions to track SPMD types on tensors
- Type inference/checking logic for operations
- TorchFunctionMode for automatic type propagation
"""

from __future__ import annotations

from enum import auto, Enum
from typing import Callable

import torch
import torch.overrides
from sixlib.spmd_types._collectives import (
    all_gather,
    all_reduce,
    all_to_all,
    redistribute,
    reduce_scatter,
)
from sixlib.spmd_types._local import convert, reinterpret
from sixlib.spmd_types.types import (
    DeviceMeshAxis,
    format_axis,
    I,
    Invariant,
    LocalSpmdType,
    P,
    Partial,
    PerMeshAxisLocalSpmdType,
    PerMeshAxisSpmdType,
    R,
    Replicate,
    Shard,
    SpmdTypeError,
    V,
    Varying,
)

# =============================================================================
# Fix Suggestion Engine
# =============================================================================

# Each entry: (from_type_class, to_type_instance, operation_str, consequence_str)
# When a type error occurs, we try replacing one operand's type and re-running
# inference. If it succeeds, we report the fix.
_FIX_CANDIDATES: list[tuple[type, PerMeshAxisLocalSpmdType, str, str]] = [
    (
        Invariant,
        R,
        "reinterpret(tensor, {axis_arg}, src=I, dst=R)",
        "no-op forward, all-reduce in backward",
    ),
    (
        Invariant,
        V,
        "reinterpret(tensor, {axis_arg}, src=I, dst=V)",
        "no-op forward, all-reduce in backward",
    ),
    (
        Invariant,
        P,
        "convert(tensor, {axis_arg}, src=I, dst=P)",
        "zeros non-rank-0 in forward, no-op backward",
    ),
    (
        Partial,
        R,
        "all_reduce(tensor, {axis_arg}, src=P, dst=R)",
        "all-reduce in forward, all-reduce in backward",
    ),
    (
        Partial,
        I,
        "all_reduce(tensor, {axis_arg}, src=P, dst=I)",
        "all-reduce in forward, no-op backward",
    ),
    (
        Replicate,
        P,
        "convert(tensor, {axis_arg}, src=R, dst=P)",
        "zeros non-rank-0 in forward, zeros non-rank-0 in backward",
    ),
]
# NB: We don't suggest R->I because compute typically happens on R, not I.

# NB: We don't suggest
# (Partial, V, "reduce_scatter(tensor, axis, src=P, dst=V)", ...)
# because this requires reasoning about the desired size of the output; if the
# original code is the correct size, the rewrite is incorrect.

# NB: We don't suggest
# (Varying, P, "reinterpret(tensor, axis, src=V, dst=P)", ...)
# because the operation requiring P should have just been fed V directly.


def _suggest_fixes(
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisLocalSpmdType],
    infer_fn: Callable[
        [DeviceMeshAxis, list[PerMeshAxisLocalSpmdType]], PerMeshAxisLocalSpmdType
    ],
) -> list[tuple[str, str, type]]:
    """Try each candidate fix and return suggestions for ones that work.

    For each candidate (from_type, to_type, ...):
    1. Check if from_type exists in axis_types
    2. Replace one occurrence with to_type
    3. Call infer_fn on the modified list
    4. If no exception, this is a valid fix — include it

    The ``infer_fn`` must be a *raw* inference function that raises plain
    ``SpmdTypeError`` without calling ``_format_error_with_suggestions``.
    This makes recursion structurally impossible.

    Args:
        axis: The mesh axis where the type error occurred.
        axis_types: The list of per-axis SPMD types that caused the error.
        infer_fn: A raw inference function that takes (axis, axis_types) and
            returns the inferred output type, or raises ``SpmdTypeError``.

    Returns:
        List of (operation_str, consequence_str, from_type_class) tuples.
    """
    # Compute the axis argument text once
    if isinstance(axis, str):
        axis_arg = repr(axis)  # e.g. "'tp'"
    else:
        axis_arg = "pg"

    suggestions: list[tuple[str, str, type]] = []
    for from_type_class, to_type, operation_template, consequence in _FIX_CANDIDATES:
        # Find the first operand matching from_type_class
        idx = None
        for i, t in enumerate(axis_types):
            if isinstance(t, from_type_class):
                idx = i
                break
        if idx is None:
            continue
        # Try replacing that operand
        modified = list(axis_types)
        modified[idx] = to_type
        try:
            fix_output = infer_fn(axis, modified)
        except SpmdTypeError:
            continue
        # Filter: does this fix preserve the natural output of the other operands?
        remaining = [t for i, t in enumerate(axis_types) if i != idx]
        if remaining:
            try:
                natural_output = infer_fn(axis, remaining)
                if fix_output != natural_output:
                    continue  # Fix changes the output type — skip
            except (SpmdTypeError, ValueError):
                pass  # Can't determine natural output — keep the suggestion
        operation = operation_template.format(axis_arg=axis_arg)
        suggestions.append((operation, consequence, from_type_class))
    return suggestions


def _format_error_with_suggestions(
    base_msg: str,
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisLocalSpmdType],
    infer_fn: Callable[
        [DeviceMeshAxis, list[PerMeshAxisLocalSpmdType]], PerMeshAxisLocalSpmdType
    ],
) -> str:
    """Format an error message, appending fix suggestions if any exist.

    Args:
        base_msg: The base error message to display.
        axis: The mesh axis where the type error occurred.
        axis_types: The list of per-axis SPMD types that caused the error.
        infer_fn: A raw inference function used to discover valid fixes.
    """
    suggestions = _suggest_fixes(axis, axis_types, infer_fn)
    if not suggestions:
        return base_msg
    lines = [base_msg, "Are you missing a collective? e.g.,"]
    for operation, consequence, from_type_class in suggestions:
        lines.append(
            f"  {operation} on the {from_type_class.__name__} operand ({consequence})"
        )
    return "\n".join(lines)


# =============================================================================
# SPMD Type Tracking on Tensors
# =============================================================================

# Attribute name for storing SPMD types on tensors
_LOCAL_TYPE_ATTR = "_local_type"


def _validate_and_canonicalize(types: LocalSpmdType) -> LocalSpmdType:
    """Validate and canonicalize a LocalSpmdType.

    Validates that all values are valid local SPMD types (R, I, V, or P).
    Shard types are not valid local SPMD types — they are used as arguments to
    collective operations but must not be stored on tensors.

    Canonicalizes by removing V entries: omitted mesh axes default to Varying,
    so explicit V entries are redundant.  Stripping them ensures that
    ``{"tp": R, "dp": V}`` and ``{"tp": R}`` compare as equal.

    Args:
        types: A LocalSpmdType dict mapping mesh axes to per-axis SPMD types.

    Raises:
        TypeError: If any value is not a PerMeshAxisLocalSpmdType (R, I, V, or P).
    """
    for axis, typ in types.items():
        if not isinstance(typ, PerMeshAxisLocalSpmdType):
            if isinstance(typ, Shard):
                raise TypeError(
                    f"Shard type {typ!r} on axis {format_axis(axis)} cannot be stored "
                    f"as a local SPMD type. Shard is only valid as src/dst in "
                    f"collective operations. Use V instead for local type tracking."
                )
            raise TypeError(
                f"Expected PerMeshAxisLocalSpmdType (R, I, V, or P) on axis "
                f"{format_axis(axis)}, got {typ!r}"
            )
    return {axis: typ for axis, typ in types.items() if typ is not V}


def has_local_type(tensor: torch.Tensor) -> bool:
    """Return True if the tensor has SPMD type annotations.

    Args:
        tensor: The tensor to check for SPMD type annotations.
    """
    return hasattr(tensor, _LOCAL_TYPE_ATTR)


def get_local_type(tensor: torch.Tensor) -> LocalSpmdType:
    """Get the SPMD types stored on a tensor.

    Args:
        tensor: The tensor to retrieve SPMD type annotations from.

    Raises:
        AttributeError: If the tensor has no SPMD type annotations.
            Use ``has_local_type`` to check first.
    """
    result = getattr(tensor, _LOCAL_TYPE_ATTR, None)
    if result is None:
        raise AttributeError(
            "Tensor has no SPMD type annotations. "
            "Use has_local_type() to check, or assert_local_type() to annotate."
        )
    return result


def _set_local_type(tensor: torch.Tensor, types: LocalSpmdType) -> torch.Tensor:
    """Set SPMD types on a tensor (internal). Returns the tensor for chaining.

    Args:
        tensor: The tensor to annotate with SPMD types.
        types: A LocalSpmdType dict mapping mesh axes to per-axis SPMD types.
    """
    setattr(tensor, _LOCAL_TYPE_ATTR, _validate_and_canonicalize(types))
    return tensor


def assert_local_type(tensor: torch.Tensor, types: LocalSpmdType) -> torch.Tensor:
    """
    Assert or set SPMD types on a tensor.

    If the tensor has no SPMD types, sets them.
    If the tensor already has SPMD types, checks they equal the provided types.

    If a mesh axis is omitted from types, it is assumed that the tensor varies
    over that mesh axis.

    Returns the tensor for chaining.

    Args:
        tensor: The tensor to assert or set SPMD types on.
        types: A LocalSpmdType dict mapping mesh axes to per-axis SPMD types
            (must be R, I, V, or P — not Shard).

    Raises:
        TypeError: If types contain invalid type objects (must be R/I/V/P)
        AssertionError: If existing types don't match provided types
    """
    canonical = _validate_and_canonicalize(types)
    if not has_local_type(tensor):
        return _set_local_type(tensor, canonical)

    # existing is already canonical (_set_local_type canonicalizes on store)
    existing = get_local_type(tensor)
    if existing != canonical:
        raise AssertionError(
            f"SPMD type mismatch: tensor has {existing}, expected {canonical}"
        )
    return tensor


def get_axis_local_type(
    tensor: torch.Tensor, axis: DeviceMeshAxis
) -> PerMeshAxisLocalSpmdType:
    """Get the SPMD type for a specific mesh axis.

    Returns V (Varying) if the axis is not explicitly stored, since omitted
    axes default to Varying.

    Raises:
        ValueError: If the tensor has no SPMD type annotations.

    Args:
        tensor: The tensor to query.
        axis: The mesh axis to look up (string name or ProcessGroup).
    """
    if not has_local_type(tensor):
        raise ValueError(
            "get_axis_local_type: tensor has no SPMD type annotations. "
            "Use has_local_type() to check first, or assert_local_type() to annotate."
        )
    return get_local_type(tensor).get(axis, V)


# =============================================================================
# Type Inference Logic
# =============================================================================


class OpLinearity(Enum):
    """Classifies how a torch op interacts with Partial (P) types.

    - NONLINEAR: P cannot propagate (safe default for unclassified ops).
    - LINEAR: Linear map on direct sum; all-P -> P.
      Examples: addition, subtraction, concat, clone.
    - MULTILINEAR: Linear in each factor separately; P in one factor with R
      in others -> P, but P in multiple factors is forbidden.
      Examples: multiplication, matmul, einsum.
    """

    NONLINEAR = auto()
    LINEAR = auto()
    MULTILINEAR = auto()


def _infer_local_type_for_axis_raw(
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisLocalSpmdType],
    out_partial: bool = False,
    linearity: OpLinearity = OpLinearity.NONLINEAR,
) -> PerMeshAxisLocalSpmdType:
    """Raw inference logic — raises plain ``SpmdTypeError`` without suggestions.

    The public wrapper ``infer_local_type_for_axis`` catches these errors and
    enriches them with fix suggestions.

    Args:
        axis: The mesh axis name (used for error messages).
        axis_types: List of input SPMD types for this axis.
        out_partial: If True, reinterpret the inferred result as Partial.
        linearity: How the op interacts with Partial types.
    """
    if not axis_types:
        if out_partial:
            return P
        raise ValueError(f"No types provided for axis {format_axis(axis)}")

    # Check type compatibility and infer output type
    type_set = set(axis_types)

    if len(type_set) == 1:
        inferred_type = axis_types[0]
        if inferred_type is P:
            if linearity is OpLinearity.NONLINEAR:
                raise SpmdTypeError(
                    f"Partial type on axis {format_axis(axis)} cannot propagate through "
                    f"non-linear ops. Use all_reduce or reduce_scatter first. "
                    f"Found types: {axis_types}"
                )
            elif linearity is OpLinearity.MULTILINEAR:
                p_count = sum(1 for t in axis_types if t is P)
                if p_count > 1:
                    raise SpmdTypeError(
                        f"Partial in multiple factors of multilinear op on axis "
                        f"{format_axis(axis)} is forbidden. "
                        f"Found types: {axis_types}"
                    )
                # Single P → P (unary multilinear is fine)
            # LINEAR: all-P → P, pass through
    elif type_set == {R, V}:
        # Mixed replicate/varying -> varying
        inferred_type = V
    elif I in type_set and len(type_set) > 1:
        raise SpmdTypeError(
            f"Invariant type on axis {format_axis(axis)} cannot mix with other types. "
            f"Found types: {axis_types}"
        )
    elif P in type_set:
        # P mixed with non-P types
        if linearity is OpLinearity.MULTILINEAR:
            p_count = sum(1 for t in axis_types if t is P)
            non_p = type_set - {P}
            if p_count > 1:
                raise SpmdTypeError(
                    f"Partial in multiple factors of multilinear op on axis "
                    f"{format_axis(axis)} is forbidden. "
                    f"Found types: {axis_types}"
                )
            if non_p == {R}:
                inferred_type = P  # P in one factor, R in others → P
            else:
                raise SpmdTypeError(
                    f"Partial type on axis {format_axis(axis)} can only multiply with "
                    f"Replicate. Found types: {axis_types}"
                )
        else:
            # NONLINEAR and LINEAR both reject P mixed with non-P
            raise SpmdTypeError(
                f"Partial type on axis {format_axis(axis)} can only combine with partial. "
                f"Found types: {axis_types}"
            )
    else:
        raise SpmdTypeError(
            f"Incompatible types on axis {format_axis(axis)}: {axis_types}"
        )

    # Apply out_partial: reinterpret as P
    if out_partial:
        if inferred_type is V or inferred_type is P:
            return P
        elif inferred_type is R:
            raise SpmdTypeError(
                f"out_partial_axes includes axis {format_axis(axis)} but inferred type "
                f"is R (Replicate). A replicated result cannot be partial — this likely "
                f"indicates an unsharded contraction dimension. "
                f"Input types: {axis_types}"
            )
        else:
            raise SpmdTypeError(
                f"Cannot mark axis {format_axis(axis)} as partial with type {inferred_type}. "
                f"out_partial_axes is only valid for V or P types."
            )

    return inferred_type


def infer_local_type_for_axis(
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisLocalSpmdType],
    out_partial: bool = False,
    linearity: OpLinearity = OpLinearity.NONLINEAR,
) -> PerMeshAxisLocalSpmdType:
    """
    Infer the output SPMD type for a single mesh axis given input types.

    Args:
        axis: The mesh axis name (for error messages)
        axis_types: List of input types for this axis
        out_partial: If True, reinterpret the result as Partial
        linearity: How the op interacts with Partial types

    Returns:
        The inferred output type

    Raises:
        SpmdTypeError: If the input types are incompatible
    """
    try:
        return _infer_local_type_for_axis_raw(axis, axis_types, out_partial, linearity)
    except SpmdTypeError as e:
        raise SpmdTypeError(
            _format_error_with_suggestions(
                str(e),
                axis,
                axis_types,
                lambda a, t: _infer_local_type_for_axis_raw(
                    a, t, out_partial, linearity
                ),
            )
        ) from None


def infer_output_types(
    input_types_list: list[LocalSpmdType],
    out_partial_axes: set[DeviceMeshAxis] | None = None,
    linearity: OpLinearity = OpLinearity.NONLINEAR,
) -> LocalSpmdType:
    """
    Infer output SPMD types from a list of input types.

    This implements the typing rules for operations like einsum:
    - If all operands are R -> output is R
    - If all operands are I -> output is I
    - If all operands are V -> output is V
    - If all operands are P -> output is P (linear ops only)
    - Mixed R/V -> output is V
    - I cannot mix with other types
    - P cannot mix with non-P types

    Args:
        input_types_list: List of LocalSpmdType dicts, one per operand
        out_partial_axes: Optional set of mesh axis names to mark as partial
        linearity: How the op interacts with Partial types

    Returns:
        LocalSpmdType dict for the output
    """
    if out_partial_axes is None:
        out_partial_axes = set()

    # Collect all mesh axes mentioned
    all_axes: set[DeviceMeshAxis] = set()
    for types in input_types_list:
        all_axes.update(types.keys())
    all_axes.update(out_partial_axes)

    # Infer output type for each axis
    output_types: LocalSpmdType = {}
    for axis in all_axes:
        axis_types = []
        for types in input_types_list:
            axis_types.append(types.get(axis, V))

        output_types[axis] = infer_local_type_for_axis(
            axis, axis_types, out_partial=axis in out_partial_axes, linearity=linearity
        )

    return output_types


# =============================================================================
# SPMD Function Registry
# =============================================================================

# Default src/dst for each SPMD collective/local op.  When a kwarg is omitted
# by the caller the Python default from the function signature applies; we
# record those defaults here so that __torch_function__ can recover them
# (handle_torch_function does not forward defaults).
# A value of ``None`` means the parameter is required (no default).

_OP_LINEARITY: dict[Callable, OpLinearity] = {
    # Linear (direct sum): all tensor args participate linearly together
    torch.add: OpLinearity.LINEAR,
    torch.sub: OpLinearity.LINEAR,
    torch.subtract: OpLinearity.LINEAR,
    torch.Tensor.clone: OpLinearity.LINEAR,
    torch.cat: OpLinearity.LINEAR,
    torch.stack: OpLinearity.LINEAR,
    # Multilinear (product): linear in each tensor arg separately
    torch.mul: OpLinearity.MULTILINEAR,
    torch.multiply: OpLinearity.MULTILINEAR,
    torch.matmul: OpLinearity.MULTILINEAR,
    torch.einsum: OpLinearity.MULTILINEAR,
}


def _iter_tensor_args(args: tuple, kwargs: dict):
    """Yield all tensor arguments from args and kwargs.

    Flattens one level of list/tuple in positional args (for ops like
    torch.cat/stack).

    Args:
        args: Positional arguments to the torch operation.
        kwargs: Keyword arguments to the torch operation.
    """
    for arg in args:
        if isinstance(arg, torch.Tensor):
            yield arg
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, torch.Tensor):
                    yield item
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            yield v


def _check_all_typed(args: tuple, kwargs: dict) -> None:
    """Raise ``SpmdTypeError`` if typed and untyped tensors are mixed.

    Called once at the top of the regular-op path in strict mode so that
    ``_collect_tensor_types`` itself stays simple and unconditional.

    Uses ``has_local_type`` rather than truthiness of the types dict because
    ``_validate_and_canonicalize`` strips V entries — a tensor annotated as all-V stores
    ``{}`` but should still count as typed.

    Args:
        args: Positional arguments to the torch operation.
        kwargs: Keyword arguments to the torch operation.
    """
    has_typed = False
    has_untyped = False
    for t in _iter_tensor_args(args, kwargs):
        if has_local_type(t):
            has_typed = True
        else:
            has_untyped = True
        if has_typed and has_untyped:
            raise SpmdTypeError(
                "Strict mode: operation mixes tensors with SPMD type annotations "
                "and tensors without. All tensor operands must be annotated."
            )


def _collect_tensor_types(args: tuple, kwargs: dict) -> list[LocalSpmdType]:
    """Collect SPMD types from all typed tensor arguments.

    Walks args (flattening one level of list/tuple for ops like torch.cat/stack)
    and kwargs values.  Skips tensors without annotations; since omitted axes
    default to V, unannotated tensors contribute nothing to inference.

    Args:
        args: Positional arguments to the torch operation.
        kwargs: Keyword arguments to the torch operation.
    """
    result: list[LocalSpmdType] = []
    for t in _iter_tensor_args(args, kwargs):
        if has_local_type(t):
            result.append(get_local_type(t))
    return result


def _set_result_types(result: object, output_types: LocalSpmdType) -> None:
    """Set SPMD types on the result tensor(s).

    Args:
        result: The operation result — a single tensor, or a list/tuple of tensors.
        output_types: The LocalSpmdType dict to set on each result tensor.
    """
    if isinstance(result, torch.Tensor):
        _set_local_type(result, output_types)
    elif isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, torch.Tensor):
                _set_local_type(item, output_types)


# Every function in this registry must accept (x: Tensor, axis, *, src=..., dst=...)
# as its leading arguments, since __torch_function__ recovers src/dst from
# args[0], args[1], and kwargs.
_SPMD_FUNCTION_DEFAULTS: dict[Callable, dict[str, PerMeshAxisSpmdType | None]] = {
    all_reduce: {"src": P, "dst": None},
    all_gather: {"src": V, "dst": None},
    reduce_scatter: {"src": P, "dst": V},
    all_to_all: {"src": V, "dst": V},
    redistribute: {"src": None, "dst": None},
    reinterpret: {"src": None, "dst": None},
    convert: {"src": None, "dst": None},
}


# =============================================================================
# TorchFunctionMode for SPMD Type Tracking
# =============================================================================


class SpmdTypeMode(torch.overrides.TorchFunctionMode):
    """
    TorchFunctionMode for tracking SPMD types on tensors.

    When active, this mode intercepts torch operations and propagates
    SPMD types from inputs to outputs according to the typing rules.

    For SPMD collectives and local ops (all_reduce, all_gather, etc.), the
    mode validates that the input tensor's type on the relevant mesh axis
    matches the declared ``src`` and sets the output type to ``dst``.  Types
    on all other mesh axes are copied through unchanged.

    Type checking runs *after* the function executes so that runtime errors
    (shape mismatches, invalid arguments) surface before type errors.

    Args:
        strict: If True, raises ``SpmdTypeError`` when a regular torch op
            mixes typed and untyped tensor operands.  Once all tensors are
            annotated, keep this on to prevent unannotated tensors from
            silently slipping through.
    """

    def __init__(self, strict: bool = False):
        super().__init__()
        self.strict = strict

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # Run the function first (catches runtime errors before type errors)
        result = func(*args, **kwargs)

        # Strict mode: all tensor operands must be annotated (applies to both
        # SPMD collectives and regular torch ops).
        if self.strict:
            _check_all_typed(args, kwargs)

        if func in _SPMD_FUNCTION_DEFAULTS:
            # Special spmd collective/reinterpret/collect
            x = args[0]
            if has_local_type(x):
                defaults = _SPMD_FUNCTION_DEFAULTS[func]
                axis = args[1]
                src = kwargs.get("src", defaults["src"])
                dst = kwargs.get("dst", defaults["dst"])

                # Decay Shard to Varying for local SPMD type checking.
                # S(i) is a global SPMD refinement; locally it behaves as V.
                if isinstance(src, Shard):
                    src = V
                if isinstance(dst, Shard):
                    dst = V

                # Validate input type on the axis matches src
                input_type = get_axis_local_type(x, axis)
                if src is not None:
                    if input_type != src:
                        raise SpmdTypeError(
                            f"{func.__name__}: expected input type {src} on axis "
                            f"{format_axis(axis)}, got {input_type}"
                        )

                # Build output types: copy all axes from input, override this axis
                output_types = get_local_type(x).copy()
                if dst is not None:
                    output_types[axis] = dst
                _set_local_type(result, output_types)
        else:
            # Regular torch op: propagate types according to non-comms rules
            input_types_list = _collect_tensor_types(args, kwargs)
            if input_types_list:
                linearity = _OP_LINEARITY.get(func, OpLinearity.NONLINEAR)
                output_types = infer_output_types(input_types_list, linearity=linearity)
                _set_result_types(result, output_types)

        return result
