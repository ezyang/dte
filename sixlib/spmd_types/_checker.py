"""
SPMD type checking logic and tensor type tracking.

This module provides:
- Functions to track SPMD types on tensors
- Type inference/checking logic for operations
- TorchFunctionMode for automatic type propagation
"""

import torch
import torch.distributed as dist
from sixlib.spmd_types.types import (
    DeviceMeshAxis,
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
    V,
    Varying,
)

# =============================================================================
# SPMD Type Tracking on Tensors
# =============================================================================

# Attribute name for storing SPMD types on tensors
_SPMD_TYPES_ATTR = "_spmd_types"


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
    setattr(tensor, _SPMD_TYPES_ATTR, types)
    return tensor


def assert_spmd_types(tensor: torch.Tensor, types: LocalSpmdType) -> torch.Tensor:
    """
    Assert or set SPMD types on a tensor.

    If the tensor has no SPMD types, sets them.
    If the tensor already has SPMD types, checks they equal the provided types.

    Returns the tensor for chaining.

    Raises:
        TypeError: If types contain invalid values
        AssertionError: If existing types don't match provided types
    """
    existing = get_spmd_types(tensor)
    if not existing:
        return set_spmd_types(tensor, types)

    if existing != types:
        raise AssertionError(
            f"SPMD type mismatch: tensor has {existing}, expected {types}"
        )
    return tensor


def get_spmd_type(
    tensor: torch.Tensor, axis: DeviceMeshAxis
) -> PerMeshAxisSpmdType | None:
    """Get the SPMD type for a specific mesh axis, or None if not tracked."""
    return get_spmd_types(tensor).get(axis)


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
        TypeError: If the input types are incompatible
    """
    if not axis_types:
        if out_partial:
            return P
        raise ValueError(f"No types provided for axis '{axis}'")

    # Check type compatibility and infer output type
    type_classes = set(type(t) for t in axis_types)

    if len(type_classes) == 1:
        # All same type
        inferred_type = axis_types[0]
    elif type_classes == {Replicate, Varying}:
        # Mixed replicate/varying -> varying
        inferred_type = V
    elif Invariant in type_classes and len(type_classes) > 1:
        raise TypeError(
            f"Invariant type on axis '{axis}' cannot mix with other types. "
            f"Found types: {axis_types}"
        )
    elif Partial in type_classes:
        # Partial can only appear alone (for linear ops) or with another partial
        non_partial = type_classes - {Partial}
        if non_partial:
            raise TypeError(
                f"Partial type on axis '{axis}' can only combine with partial. "
                f"Found types: {axis_types}"
            )
        inferred_type = P
    else:
        raise TypeError(f"Incompatible types on axis '{axis}': {axis_types}")

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
            raise TypeError(
                f"Cannot mark axis '{axis}' as partial with type {inferred_type}. "
                f"out_partial_axes is only valid for V, S(i), R, or P types."
            )

    return inferred_type


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
# TorchFunctionMode for SPMD Type Tracking
# =============================================================================


class SpmdTypeMode(torch.overrides.TorchFunctionMode):
    """
    TorchFunctionMode for tracking SPMD types on tensors.

    When active, this mode intercepts torch operations and propagates
    SPMD types from inputs to outputs according to the typing rules.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # For now, just call the function and propagate types for known ops
        result = func(*args, **kwargs)

        # TODO: Add type propagation rules for various ops
        # For now, we rely on explicit einsum calls to handle type checking

        return result
