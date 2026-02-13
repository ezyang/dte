"""Global device mesh management for SPMD operations."""

from __future__ import annotations

from torch.distributed import ProcessGroup

# Global device mesh storage
_global_mesh = None


def set_mesh(mesh):
    """
    Set the global device mesh for distributed operations.

    Args:
        mesh: A DeviceMesh object that maps axis names to process groups.
              Must have a `get_group(axis_name)` method.
    """
    global _global_mesh
    _global_mesh = mesh


def get_mesh():
    """
    Get the current global device mesh.

    Returns:
        The global DeviceMesh, or None if not set.
    """
    return _global_mesh


def _get_mesh_axis_group(axis: str | ProcessGroup) -> ProcessGroup:
    """Get the process group for a mesh axis.

    If axis is already a ProcessGroup, return it directly.
    If axis is a string, look it up from the global mesh.

    Args:
        axis: Either a ProcessGroup (returned as-is) or a string axis name
            to look up from the global mesh.
    """
    if isinstance(axis, ProcessGroup):
        return axis
    # It's a string axis name
    if _global_mesh is None:
        raise RuntimeError(
            "No global mesh set. Call set_mesh() with a DeviceMesh before using "
            "distributed operations with axis names, or pass a ProcessGroup directly."
        )
    return _global_mesh.get_group(axis)
