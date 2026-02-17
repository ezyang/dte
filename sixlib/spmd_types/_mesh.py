"""Global device mesh management for SPMD operations."""

from __future__ import annotations

import threading

from torch.distributed import ProcessGroup

# Thread-local device mesh storage
_tls = threading.local()


def set_mesh(mesh):
    """
    Set the device mesh for the current thread.

    Args:
        mesh: A DeviceMesh object that maps axis names to process groups.
              Must have a `get_group(axis_name)` method.
    """
    _tls.mesh = mesh


def get_mesh():
    """
    Get the current thread's device mesh.

    Returns:
        The thread-local DeviceMesh, or None if not set.
    """
    return getattr(_tls, "mesh", None)


def _get_mesh_axis_group(axis: str | ProcessGroup) -> ProcessGroup:
    """Get the process group for a mesh axis.

    If axis is already a ProcessGroup, return it directly.
    If axis is a string, look it up from the current thread's mesh.

    Args:
        axis: Either a ProcessGroup (returned as-is) or a string axis name
            to look up from the current thread's mesh.
    """
    if isinstance(axis, ProcessGroup):
        return axis
    # It's a string axis name
    mesh = get_mesh()
    if mesh is None:
        raise RuntimeError(
            "No global mesh set. Call set_mesh() with a DeviceMesh before using "
            "distributed operations with axis names, or pass a ProcessGroup directly."
        )
    return mesh.get_group(axis)
