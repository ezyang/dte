"""
SPMD type checking logic and tensor type tracking.

This module provides:
- Functions to track SPMD types on tensors
- Type inference/checking logic for operations
- TorchFunctionMode for automatic type propagation
"""

from typing import Callable

import torch
import torch.distributed as dist
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
        "reinterpret(tensor, axis, src=I, dst=R)",
        "no-op forward, all-reduce in backward",
    ),
    (
        Invariant,
        V,
        "reinterpret(tensor, axis, src=I, dst=V)",
        "no-op forward, all-reduce in backward",
    ),
    (
        Invariant,
        P,
        "convert(tensor, axis, src=I, dst=P)",
        "zeros non-rank-0 in forward, no-op backward",
    ),
    (
        Partial,
        R,
        "all_reduce(tensor, axis, src=P, dst=R)",
        "all-reduce in forward, all-reduce in backward",
    ),
    (
        Partial,
        I,
        "all_reduce(tensor, axis, src=P, dst=I)",
        "all-reduce in forward, no-op backward",
    ),
    (
        Replicate,
        P,
        "convert(tensor, axis, src=R, dst=P)",
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
    axis_types: list[PerMeshAxisSpmdType],
    infer_fn: Callable[
        [DeviceMeshAxis, list[PerMeshAxisSpmdType]], PerMeshAxisSpmdType
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

    Returns list of (operation_str, consequence_str, from_type_class) tuples.
    """
    suggestions: list[tuple[str, str, type]] = []
    for from_type_class, to_type, operation, consequence in _FIX_CANDIDATES:
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
        suggestions.append((operation, consequence, from_type_class))
    return suggestions


def _format_error_with_suggestions(
    base_msg: str,
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisSpmdType],
    infer_fn: Callable[
        [DeviceMeshAxis, list[PerMeshAxisSpmdType]], PerMeshAxisSpmdType
    ],
) -> str:
    """Format an error message, appending fix suggestions if any exist."""
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
_SPMD_TYPES_ATTR = "_spmd_types"


def _canonicalize(types: LocalSpmdType) -> LocalSpmdType:
    """Canonicalize a LocalSpmdType by removing V entries.

    Omitted mesh axes default to Varying, so explicit V entries are redundant.
    Stripping them ensures that ``{"tp": R, "dp": V}`` and ``{"tp": R}``
    compare as equal.
    """
    return {axis: typ for axis, typ in types.items() if typ is not V}


def get_spmd_types(tensor: torch.Tensor) -> LocalSpmdType:
    """Get the SPMD types stored on a tensor, or empty dict if none."""
    return getattr(tensor, _SPMD_TYPES_ATTR, {})


def set_spmd_types(tensor: torch.Tensor, types: LocalSpmdType) -> torch.Tensor:
    """Set SPMD types on a tensor. Returns the tensor for chaining."""
    for axis_name, typ in types.items():
        if not isinstance(typ, (PerMeshAxisLocalSpmdType, Shard)):
            raise TypeError(
                f"Invalid type '{typ}' for axis '{axis_name}'. "
                f"Must be a PerMeshAxisSpmdType (R, I, V, P, or S(i))"
            )
    setattr(tensor, _SPMD_TYPES_ATTR, _canonicalize(types))
    return tensor


def assert_type(tensor: torch.Tensor, types: LocalSpmdType) -> torch.Tensor:
    """
    Assert or set SPMD types on a tensor.

    If the tensor has no SPMD types, sets them.
    If the tensor already has SPMD types, checks they equal the provided types.

    Both the existing and provided types are canonicalized before comparison
    (explicit V entries are stripped, since omitted axes default to V).

    Returns the tensor for chaining.

    Raises:
        TypeError: If types contain invalid type objects (not R/I/V/P/S)
        AssertionError: If existing types don't match provided types
    """
    canonical = _canonicalize(types)
    existing = get_spmd_types(tensor)
    if not existing:
        return set_spmd_types(tensor, canonical)

    # existing is already canonical (set_spmd_types canonicalizes on store)
    if existing != canonical:
        raise AssertionError(
            f"SPMD type mismatch: tensor has {existing}, expected {canonical}"
        )
    return tensor


def get_spmd_type(tensor: torch.Tensor, axis: DeviceMeshAxis) -> PerMeshAxisSpmdType:
    """Get the SPMD type for a specific mesh axis.

    Returns V (Varying) for axes not explicitly stored, since omitted axes
    default to Varying.
    """
    return get_spmd_types(tensor).get(axis, V)


def with_spmd_type(
    tensor: torch.Tensor, axis: DeviceMeshAxis, typ: PerMeshAxisSpmdType
) -> torch.Tensor:
    """Return tensor with an updated SPMD type for the given axis."""
    if not isinstance(typ, (PerMeshAxisLocalSpmdType, Shard)):
        raise TypeError(
            f"Invalid type '{typ}'. Must be a PerMeshAxisSpmdType (R, I, V, P, or S(i))"
        )
    new_types = get_spmd_types(tensor).copy()
    new_types[axis] = typ
    return set_spmd_types(tensor, new_types)


# =============================================================================
# Type Inference Logic
# =============================================================================


def _infer_output_type_for_axis_raw(
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisSpmdType],
    out_partial: bool = False,
) -> PerMeshAxisSpmdType:
    """Raw inference logic — raises plain ``SpmdTypeError`` without suggestions.

    The public wrapper ``infer_output_type_for_axis`` catches these errors and
    enriches them with fix suggestions.
    """
    if not axis_types:
        if out_partial:
            return P
        raise ValueError(f"No types provided for axis {format_axis(axis)}")

    # Check type compatibility and infer output type
    type_classes = set(type(t) for t in axis_types)

    if len(type_classes) == 1:
        # All same type
        inferred_type = axis_types[0]
    elif type_classes == {Replicate, Varying}:
        # Mixed replicate/varying -> varying
        inferred_type = V
    elif Invariant in type_classes and len(type_classes) > 1:
        raise SpmdTypeError(
            f"Invariant type on axis {format_axis(axis)} cannot mix with other types. "
            f"Found types: {axis_types}"
        )
    elif Partial in type_classes:
        # Partial can only appear alone (for linear ops) or with another partial
        # TODO: If this op is linear, P should propagate (linear_op(P) -> P).
        # Need a mechanism to declare/detect linear ops.
        non_partial = type_classes - {Partial}
        if non_partial:
            raise SpmdTypeError(
                f"Partial type on axis {format_axis(axis)} can only combine with partial. "
                f"Found types: {axis_types}"
            )
        inferred_type = P
    else:
        raise SpmdTypeError(
            f"Incompatible types on axis {format_axis(axis)}: {axis_types}"
        )

    # Apply out_partial: reinterpret as P
    if out_partial:
        if inferred_type is V or isinstance(inferred_type, Shard):
            return P
        elif inferred_type is R:
            # R with out_partial becomes P (via reinterpret(R,P))
            return P
        elif inferred_type is P:
            # Already partial
            return P
        else:
            raise SpmdTypeError(
                f"Cannot mark axis {format_axis(axis)} as partial with type {inferred_type}. "
                f"out_partial_axes is only valid for V, S(i), R, or P types."
            )

    return inferred_type


def infer_output_type_for_axis(
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisSpmdType],
    out_partial: bool = False,
) -> PerMeshAxisSpmdType:
    """
    Infer the output SPMD type for a single mesh axis given input types.

    Args:
        axis: The mesh axis name (for error messages)
        axis_types: List of input types for this axis
        out_partial: If True, reinterpret the result as Partial

    Returns:
        The inferred output type

    Raises:
        SpmdTypeError: If the input types are incompatible
    """
    try:
        return _infer_output_type_for_axis_raw(axis, axis_types, out_partial)
    except SpmdTypeError as e:
        raise SpmdTypeError(
            _format_error_with_suggestions(
                str(e),
                axis,
                axis_types,
                lambda a, t: _infer_output_type_for_axis_raw(a, t, out_partial),
            )
        ) from None


def infer_output_types(
    input_types_list: list[LocalSpmdType],
    out_partial_axes: set[DeviceMeshAxis] | None = None,
) -> LocalSpmdType:
    """
    Infer output SPMD types from a list of input types.

    This implements the typing rules for operations like einsum:
    - If all operands are R -> output is R
    - If all operands are I -> output is I
    - If all operands are V -> output is V
    - If all operands are P -> output is P
    - Mixed R/V -> output is V
    - I cannot mix with other types
    - P cannot mix with non-P types

    Args:
        input_types_list: List of LocalSpmdType dicts, one per operand
        out_partial_axes: Optional set of mesh axis names to mark as partial

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
            typ = types.get(axis)
            if typ is not None:
                axis_types.append(typ)

        if not axis_types:
            # Axis only in out_partial_axes
            if axis in out_partial_axes:
                output_types[axis] = P
            continue

        output_types[axis] = infer_output_type_for_axis(
            axis, axis_types, out_partial=axis in out_partial_axes
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

_MUL_OPS: set[Callable] = {torch.mul, torch.multiply}


def _infer_mul_output_type_for_axis_raw(
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisSpmdType],
) -> PerMeshAxisSpmdType:
    """Raw mul inference logic — raises plain ``SpmdTypeError`` without suggestions.

    Mul is linear in each argument, so P * R -> P is valid, but P * P is not.
    When no Partial is involved, delegates to the standard raw inference rule.
    """
    type_classes = set(type(t) for t in axis_types)
    if Partial not in type_classes:
        return _infer_output_type_for_axis_raw(axis, axis_types)

    # Partial is present
    non_partial = type_classes - {Partial}
    if not non_partial:
        # All P: P * P is forbidden
        # TODO: If this op is linear, P should propagate (linear_op(P) -> P).
        # Need a mechanism to declare/detect linear ops.
        raise SpmdTypeError(
            f"Partial * Partial on axis {format_axis(axis)} is forbidden (not linear). "
            f"Found types: {axis_types}"
        )
    if non_partial == {Replicate}:
        # P * R -> P (scaling partial by replicate)
        return P
    # TODO: If this op is linear, P should propagate (linear_op(P) -> P).
    # Need a mechanism to declare/detect linear ops.
    raise SpmdTypeError(
        f"Partial type on axis {format_axis(axis)} can only multiply with Replicate. "
        f"Found types: {axis_types}"
    )


def _infer_mul_output_type_for_axis(
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisSpmdType],
) -> PerMeshAxisSpmdType:
    """Infer output type for a mul op on a single mesh axis.

    Mul is linear in each argument, so P * R -> P is valid, but P * P is not.
    When no Partial is involved, delegates to the standard inference rule.
    """
    try:
        return _infer_mul_output_type_for_axis_raw(axis, axis_types)
    except SpmdTypeError as e:
        raise SpmdTypeError(
            _format_error_with_suggestions(
                str(e),
                axis,
                axis_types,
                _infer_mul_output_type_for_axis_raw,
            )
        ) from None


def _infer_mul_output_types(
    input_types_list: list[LocalSpmdType],
) -> LocalSpmdType:
    """Infer output types for a mul op across all mesh axes."""
    all_axes: set[DeviceMeshAxis] = set()
    for types in input_types_list:
        all_axes.update(types.keys())

    output_types: LocalSpmdType = {}
    for axis in all_axes:
        axis_types = []
        for types in input_types_list:
            typ = types.get(axis)
            if typ is not None:
                axis_types.append(typ)
        if axis_types:
            output_types[axis] = _infer_mul_output_type_for_axis(axis, axis_types)
    return output_types


def _collect_tensor_types(args: tuple, kwargs: dict) -> list[LocalSpmdType]:
    """Collect SPMD types from all tensor arguments.

    Walks args (flattening one level of list/tuple for ops like torch.cat/stack)
    and kwargs values. Untyped tensors (empty dict) are included only if they
    have explicit types set; since canonicalized V is stripped, untyped tensors
    contribute nothing.
    """
    result: list[LocalSpmdType] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            types = get_spmd_types(arg)
            if types:
                result.append(types)
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, torch.Tensor):
                    types = get_spmd_types(item)
                    if types:
                        result.append(types)
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            types = get_spmd_types(v)
            if types:
                result.append(types)
    return result


def _set_result_types(result: object, output_types: LocalSpmdType) -> None:
    """Set SPMD types on the result tensor(s)."""
    if isinstance(result, torch.Tensor):
        set_spmd_types(result, output_types)
    elif isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, torch.Tensor):
                set_spmd_types(item, output_types)


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
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # Run the function first (catches runtime errors before type errors)
        result = func(*args, **kwargs)

        if func in _SPMD_FUNCTION_DEFAULTS:
            defaults = _SPMD_FUNCTION_DEFAULTS[func]
            x = args[0]
            axis = args[1]
            src = kwargs.get("src", defaults["src"])
            dst = kwargs.get("dst", defaults["dst"])

            # Validate input type on the axis matches src
            input_type = get_spmd_type(x, axis)
            if src is not None:
                # Use == so Shard(0) == Shard(0) works; singletons also pass.
                if input_type != src:
                    raise SpmdTypeError(
                        f"{func.__name__}: expected input type {src} on axis "
                        f"{format_axis(axis)}, got {input_type}"
                    )

            # Build output types: copy all axes from input, override this axis
            output_types = get_spmd_types(x).copy()
            if dst is not None:
                output_types[axis] = dst
            set_spmd_types(result, output_types)
        else:
            # Regular torch op: propagate types according to non-comms rules
            input_types_list = _collect_tensor_types(args, kwargs)
            if input_types_list:
                if func in _MUL_OPS:
                    output_types = _infer_mul_output_types(input_types_list)
                else:
                    output_types = infer_output_types(input_types_list)
                _set_result_types(result, output_types)

        return result
