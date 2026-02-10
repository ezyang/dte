"""Patchable reference to torch.distributed.

Callers should ``import sixlib.spmd_types._dist as _dist`` and access
``_dist.dist`` to use the current dist implementation. Do NOT use
``from ... import dist`` as that captures a snapshot.
"""

import torch.distributed as _torch_dist

dist = _torch_dist


def set_dist(module) -> None:
    """Replace the dist implementation (e.g., with comms_wrapper)."""
    global dist
    if module is None:
        module = _torch_dist
    dist = module
