"""
SPMD type definitions for distributed tensor expressions.

This module provides:
- Per-mesh-axis local SPMD types (R, I, V, P, S)
- PartitionSpec for global SPMD
- Type aliases
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

# =============================================================================
# Per-Mesh-Axis Local SPMD Type Hierarchy
# =============================================================================


class PerMeshAxisLocalSpmdType:
    """
    Base class for local SPMD types on a single mesh axis.

    Describes how a tensor is distributed across ranks on one axis of the
    device mesh, as well as how the gradients are distributed. The four
    concrete types are: Replicate (R), Invariant (I), Varying (V), Partial (P).
    """

    def backward_type(self) -> PerMeshAxisLocalSpmdType:
        """Return the type that gradients have in the backward pass."""
        raise NotImplementedError


class Replicate(PerMeshAxisLocalSpmdType):
    """
    Replicate type - data is replicated across ranks.

    The gradient of replicate is partial.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "R"

    def backward_type(self) -> PerMeshAxisLocalSpmdType:
        return P


class Invariant(PerMeshAxisLocalSpmdType):
    """
    Invariant type - data is replicated across ranks, gradient is also invariant.

    Unlike replicate, the gradient is expected to already be synchronized.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "I"

    def backward_type(self) -> PerMeshAxisLocalSpmdType:
        return I


class Varying(PerMeshAxisLocalSpmdType):
    """
    Varying type - data differs across ranks.

    The gradient of varying is also varying.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "V"

    def backward_type(self) -> PerMeshAxisLocalSpmdType:
        return V


class Partial(PerMeshAxisLocalSpmdType):
    """
    Partial type - pending sum across ranks.

    The gradient of partial is replicate.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "P"

    def backward_type(self) -> PerMeshAxisLocalSpmdType:
        return R


@dataclass(frozen=True)
class Shard:
    """
    A special version of Varying that specifies that the tensor is sharded on
    a particular dimension.

    This is not a true type (notice it doesn't inherit from PerMeshAxisLocalSpmdType)
    but it is accepted at any src/dst argument and, from a typing perspective,
    is equivalent to Varying.  However, it does change the semantics of operations
    by changing you from stack/unbind semantics to concat/split semantics,
    where the concatenation occurs on the dimension specified by Shard.
    Intuitively, if you have a tensor that is sharded on tensor dim i, and you
    do an all-gather, you typically want to concatenate the result on dim i.

    In global SPMD types, a per-mesh-axis Shard can also be used to manipulate
    the PartitionSpec in a mesh-oriented way, although the PartitionSpec is still
    the canonical way of representing this typing information.
    """

    dim: int

    def __repr__(self):
        return f"S({self.dim})"

    def backward_type(self) -> Shard:
        return self


# Single character singletons for ease of pattern matching
R = Replicate()
I = Invariant()
V = Varying()
P = Partial()
S = Shard  # S(i) creates a Shard with dim=i

# Type aliases
PerMeshAxisSpmdType = PerMeshAxisLocalSpmdType | Shard

# Axis identifier: either a mesh axis name (string) or a ProcessGroup directly
DeviceMeshAxis: TypeAlias = "str | ProcessGroup"

# LocalSpmdType maps axis identifiers to per-axis SPMD types
LocalSpmdType: TypeAlias = "dict[DeviceMeshAxis, PerMeshAxisSpmdType]"

# =============================================================================
# PartitionSpec for Global SPMD
# =============================================================================


class PartitionSpec(tuple):
    """
    A partition spec describes how tensor dimensions map to mesh axes.

    Each element corresponds to a tensor dimension and specifies zero, one, or
    multiple mesh axis names that shard that dimension. For example:
        - PartitionSpec('tp', None) means dim 0 is sharded on 'tp', dim 1 is replicated
        - PartitionSpec(('dp', 'tp'), None) means dim 0 is sharded on both 'dp' and 'tp'
        - PartitionSpec() means fully replicated
    """

    def __new__(cls, *args: str | tuple[str, ...] | None):
        return super().__new__(cls, args)

    def __repr__(self):
        pr = repr(tuple(self))[1:-1]
        if not self:
            return "PartitionSpec()"
        return f"PartitionSpec({pr})"



GlobalSpmdType: TypeAlias = "tuple[LocalSpmdType, PartitionSpec]"


class SpmdTypeError(RuntimeError):
    """Error raised for SPMD type mismatches.

    Inherits from RuntimeError (not TypeError) so that it is not swallowed by
    Python's binary-operator dispatch machinery when raised inside
    ``__torch_function__``.  Python interprets a TypeError from an operator
    dunder as "this type doesn't support the operation" and silently falls
    through to reflected operations, masking the real error message.
    """

    pass


def _canonicalize_shard(typ: PerMeshAxisSpmdType, ndim: int) -> PerMeshAxisSpmdType:
    """Resolve negative dims in Shard types. Returns typ unchanged if not Shard.

    Args:
        typ: The per-mesh-axis SPMD type, possibly a Shard with a negative dim.
        ndim: The number of dimensions of the tensor, used to resolve negative dims.
    """
    if isinstance(typ, Shard) and typ.dim < 0:
        return Shard(typ.dim % ndim)
    return typ


def format_axis(axis: DeviceMeshAxis) -> str:
    """Format a mesh axis for display in error messages.

    For string axes, returns the repr (e.g., ``'tp'``).
    For ProcessGroup axes, uses ``group_desc`` when available to produce a
    bare human-readable name (e.g., ``TP``) instead of the default opaque
    object repr.

    Args:
        axis: The mesh axis identifier, either a string name or a ProcessGroup.
    """
    if isinstance(axis, str):
        return repr(axis)
    # ProcessGroup - use group_desc for a readable name
    desc = getattr(axis, "group_desc", None)
    if desc and desc != "undefined":
        return desc
    return repr(axis)
