"""
Shared test utilities for SPMD type system tests.

Provides FakeMesh and LocalTensorTestCase for simulating distributed operations
in a single process using PyTorch's LocalTensorMode.
"""

import unittest

import torch
import torch.distributed as dist
from sixlib.spmd_types import set_mesh
from sixlib.spmd_types._checker import (
    assert_type,
    SpmdTypeMode,
)
from sixlib.spmd_types.types import I, P, R, Shard, V
from torch.distributed._local_tensor import LocalTensorMode
from torch.testing._internal.distributed.fake_pg import FakeStore


class FakeMesh:
    """
    A fake DeviceMesh for testing that returns the default process group.

    In real distributed code, DeviceMesh maps axis names to process groups.
    For testing with LocalTensorMode, we just return the default group.
    """

    def __init__(self, world_size: int):
        self.world_size = world_size

    def get_group(self, axis_name: str):
        """Return the default process group for any axis name.

        Args:
            axis_name: The mesh axis name (ignored -- always returns the default group).
        """
        return dist.distributed_c10d._get_default_group()


class LocalTensorTestCase(unittest.TestCase):
    """
    Base test class that sets up LocalTensorMode and fake process group.
    """

    WORLD_SIZE = 3

    @classmethod
    def setUpClass(cls):
        """Initialize fake distributed environment."""
        if not dist.is_initialized():
            store = FakeStore()
            dist.init_process_group(
                backend="fake", rank=0, world_size=cls.WORLD_SIZE, store=store
            )
        # Set up the global mesh
        set_mesh(FakeMesh(cls.WORLD_SIZE))

    @classmethod
    def tearDownClass(cls):
        """Clean up distributed environment."""
        set_mesh(None)
        if dist.is_initialized():
            dist.destroy_process_group()

    def setUp(self):
        """Enter LocalTensorMode and SpmdTypeMode for each test."""
        self.mode = LocalTensorMode(self.WORLD_SIZE)
        self.mode.__enter__()
        self.spmd_mode = SpmdTypeMode()
        self.spmd_mode.__enter__()

    def tearDown(self):
        """Exit SpmdTypeMode and LocalTensorMode after each test."""
        self.spmd_mode.__exit__(None, None, None)
        self.mode.__exit__(None, None, None)

    def _generate_inputs(self, shape, axis, typ):
        """Generate input tensor with the given SPMD type on the given axis.

        Data generation:
          R, I -> same data on all ranks (replicated)
          V, P, S(i) -> different data per rank

        Args:
            shape: The tensor shape to generate.
            axis: The mesh axis name to annotate the tensor on.
            typ: The SPMD type to assign (R, I, V, P, or S(i)).
        """
        if typ is R or typ is I:
            base = torch.randn(shape)
            result = self.mode.rank_map(lambda r: base.clone())
        else:
            result = self.mode.rank_map(lambda r: torch.randn(shape) + r)
        local_typ = V if isinstance(typ, Shard) else typ
        assert_type(result, {axis: local_typ})
        return result

    def _assert_all_ranks_equal(self, lt, msg=""):
        """Assert that a LocalTensor has the same value on all ranks.

        Args:
            lt: A LocalTensor whose per-rank values should all be equal.
            msg: Optional message prefix for assertion failures.
        """
        tensors = lt._local_tensors
        ranks = list(tensors.keys())
        for i in range(1, len(ranks)):
            torch.testing.assert_close(
                tensors[ranks[0]],
                tensors[ranks[i]],
                msg=f"{msg}: rank 0 vs rank {ranks[i]}",
            )

    def _assert_ranks_different(self, lt, msg=""):
        """Assert that a LocalTensor has different values on different ranks.

        Args:
            lt: A LocalTensor whose per-rank values should not all be equal.
            msg: Optional message prefix for assertion failures.
        """
        tensors = lt._local_tensors
        ranks = list(tensors.keys())
        all_same = all(
            torch.allclose(tensors[ranks[0]], tensors[ranks[i]])
            for i in range(1, len(ranks))
        )
        self.assertFalse(all_same, f"{msg}: expected different values per rank")

    def _gradient_type(self, typ):
        """Return the gradient (dual) type for a given SPMD type."""
        base = V if isinstance(typ, Shard) else typ
        return {R: P, I: I, V: V, P: R}[base]

    def _make_random_typed(self, like_lt, typ, axis):
        """Create a random LocalTensor with correct SPMD type annotation.

        For R/I types, the same random data is replicated on all ranks.
        For V/P/S types, each rank gets independent random data.

        Args:
            like_lt: A LocalTensor whose per-rank shapes to match.
            typ: The SPMD type to assign (R, I, V, P, or S(i)).
            axis: The mesh axis name to annotate on.
        """
        base = V if isinstance(typ, Shard) else typ
        if base is R or base is I:
            t = torch.randn_like(like_lt._local_tensors[0])
            result = self.mode.rank_map(lambda r: t.clone())
        else:
            result = self.mode.rank_map(
                lambda r: torch.randn_like(like_lt._local_tensors[r])
            )
        local_typ = V if isinstance(typ, Shard) else typ
        assert_type(result, {axis: local_typ})
        return result

    def _dual_inner(self, grad, perturbation, typ):
        """Compute dual inner product <grad, perturbation> for type T.

        The inner product pairs a gradient (dual type) with a perturbation
        (primal type), accounting for the multi-rank structure of SPMD types.

        Args:
            grad: Gradient tensor (has the gradient/dual type of T).
            perturbation: Perturbation tensor (has the forward/primal type T).
            typ: The SPMD type defining the space.
        """
        base = V if isinstance(typ, Shard) else typ
        W = self.WORLD_SIZE
        ranks = range(W)
        if base is R:
            # grad is P (varying), perturbation is R (replicated)
            grad_sum = sum(grad._local_tensors[r] for r in ranks)
            return (grad_sum * perturbation._local_tensors[0]).sum().item()
        elif base is I:
            return (
                (grad._local_tensors[0] * perturbation._local_tensors[0]).sum().item()
            )
        elif base is V:
            return sum(
                (grad._local_tensors[r] * perturbation._local_tensors[r]).sum().item()
                for r in ranks
            )
        elif base is P:
            # grad is R (replicated), perturbation is P (varying)
            pert_sum = sum(perturbation._local_tensors[r] for r in ranks)
            return (grad._local_tensors[0] * pert_sum).sum().item()

    def spmd_gradcheck(self, fn, x, axis, src_type, dst_type):
        """Verify backward correctness via the adjoint identity.

        For a linear SPMD operation A with backward A*, verifies:
            <A*(g), dx>_{src} = <g, A(dx)>_{dst}
        using dual inner products that respect SPMD type semantics.

        Since all SPMD operations are linear and their backwards don't depend
        on the input value, this is an exact check (no finite differences).

        Args:
            fn: The SPMD operation to test (a callable taking one tensor).
            x: Input tensor with the correct SPMD type.
            axis: The mesh axis name.
            src_type: The source SPMD type of fn.
            dst_type: The destination SPMD type of fn.
        """
        x.requires_grad_(True)
        y = fn(x)

        # g has the gradient type of dst (the type that grad_output should have)
        grad_type = self._gradient_type(dst_type)
        g = self._make_random_typed(y, grad_type, axis)

        y.backward(g)
        grad_x = x.grad  # has gradient type of src

        # dx has the forward type of src
        dx = self._make_random_typed(x, src_type, axis)

        with torch.no_grad():
            Adx = fn(dx)

        lhs = self._dual_inner(grad_x, dx, src_type)
        rhs = self._dual_inner(g, Adx, dst_type)

        torch.testing.assert_close(
            torch.tensor(lhs),
            torch.tensor(rhs),
            msg=f"Adjoint identity failed: <A*g, dx>={lhs}, <g, Adx>={rhs}",
        )
