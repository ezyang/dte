"""
SPMD type definitions for distributed tensor expressions.

This module provides:
- Per-mesh-axis local SPMD types (R, I, V, P, S)
- PartitionSpec for global SPMD
- Type aliases
"""

from dataclasses import dataclass
from typing import Union

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

    def backward_type(self) -> "PerMeshAxisLocalSpmdType":
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

    def backward_type(self) -> "PerMeshAxisLocalSpmdType":
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

    def backward_type(self) -> "PerMeshAxisLocalSpmdType":
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

    def backward_type(self) -> "PerMeshAxisLocalSpmdType":
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

    def backward_type(self) -> "PerMeshAxisLocalSpmdType":
        return R


@dataclass(frozen=True)
class Shard:
    """
    Shard type - varying on a specific tensor dimension.

    Like Varying but rank-preserving; operates on a specific tensor dimension
    rather than adding/removing dimensions. Used for concat semantics in
    collectives (vs stack semantics for V).

    The gradient of Shard(i) is also Shard(i).
    """

    dim: int

    def __repr__(self):
        return f"S({self.dim})"

    def backward_type(self) -> "Shard":
        return self


# Single character singletons for ease of pattern matching
R = Replicate()
I = Invariant()
V = Varying()
P = Partial()
S = Shard  # S(i) creates a Shard with dim=i

# Type aliases
PerMeshAxisSpmdType = Union[PerMeshAxisLocalSpmdType, Shard]

# Axis identifier: either a mesh axis name (string) or a ProcessGroup directly
DeviceMeshAxis = str | ProcessGroup

# LocalSpmdType maps axis identifiers to per-axis SPMD types
LocalSpmdType = dict[DeviceMeshAxis, PerMeshAxisSpmdType]

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

    When printed with a tensor shape, we use notation like f32[8@tp,16] to show
    that dim 0 (size 8) is sharded on 'tp' while dim 1 (size 16) is replicated.
    """

    def __new__(cls, *args: str | tuple[str, ...] | None):
        return super().__new__(cls, args)

    def __repr__(self):
        if not self:
            return "PartitionSpec()"
        parts = []
        for s in self:
            if s is None:
                parts.append("None")
            elif isinstance(s, tuple):
                parts.append(f"({', '.join(repr(x) for x in s)})")
            else:
                parts.append(repr(s))
        return f"PartitionSpec({', '.join(parts)})"

    def get_mesh_axes(self) -> set[str]:
        """Return all mesh axis names mentioned in this spec."""
        axes = set()
        for s in self:
            if s is None:
                continue
            elif isinstance(s, tuple):
                axes.update(s)
            else:
                axes.add(s)
        return axes


GlobalSpmdType = tuple[LocalSpmdType, PartitionSpec]


class SpmdTypeError(RuntimeError):
    """Error raised for SPMD type mismatches.

    Inherits from RuntimeError (not TypeError) so that it is not swallowed by
    Python's binary-operator dispatch machinery when raised inside
    ``__torch_function__``.  Python interprets a TypeError from an operator
    dunder as "this type doesn't support the operation" and silently falls
    through to reflected operations, masking the real error message.
    """

    pass


def format_axis(axis: DeviceMeshAxis) -> str:
    """Format a mesh axis for display in error messages.

    For string axes, returns the repr (e.g., ``'tp'``).
    For ProcessGroup axes, uses ``group_desc`` when available to produce a
    bare human-readable name (e.g., ``TP``) instead of the default opaque
    object repr.
    """
    if isinstance(axis, str):
        return repr(axis)
    # ProcessGroup - use group_desc for a readable name
    desc = getattr(axis, "group_desc", None)
    if desc and desc != "undefined":
        return desc
    return repr(axis)
