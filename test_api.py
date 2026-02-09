"""
Tests for dte/_api.py local SPMD type system.

Uses PyTorch's LocalTensorMode to simulate distributed operations in a single
process without requiring an actual distributed backend.
"""

import unittest
import torch
from torch.distributed._local_tensor import LocalTensor, LocalTensorMode
from torch.testing._internal.distributed.fake_pg import FakeStore
import torch.distributed as dist

from dte._api import (
    LTensor,
    einsum,
    set_mesh,
    get_mesh,
    all_reduce,
    all_gather,
    reduce_scatter,
    all_to_all,
    reinterpret,
    convert,
    redistribute,
    R, I, V, P, S,
    Replicate, Invariant, Varying, Partial, Shard,
    PartitionSpec,
)


class FakeMesh:
    """
    A fake DeviceMesh for testing that returns the default process group.

    In real distributed code, DeviceMesh maps axis names to process groups.
    For testing with LocalTensorMode, we just return the default group.
    """

    def __init__(self, world_size: int):
        self.world_size = world_size

    def get_group(self, axis_name: str):
        """Return the default process group for any axis name."""
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
                backend="fake",
                rank=0,
                world_size=cls.WORLD_SIZE,
                store=store
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
        """Enter LocalTensorMode for each test."""
        self.mode = LocalTensorMode(self.WORLD_SIZE)
        self.mode.__enter__()

    def tearDown(self):
        """Exit LocalTensorMode after each test."""
        self.mode.__exit__(None, None, None)

    def _generate_inputs(self, shape, src):
        """
        Generate input tensors based on source type.

        For replicate/invariant: same tensor on all ranks
        For varying/partial: different tensor per rank
        """
        if src in ('replicate', 'invariant'):
            # Same tensor on all ranks
            base = torch.randn(shape)
            return self.mode.rank_map(lambda r: base.clone())
        else:  # varying or partial
            # Different tensor per rank - use rank as seed for reproducibility
            return self.mode.rank_map(lambda r: torch.randn(shape) + r)

    def _assert_all_ranks_equal(self, lt, msg=""):
        """Assert that a LocalTensor has the same value on all ranks."""
        tensors = lt._local_tensors
        ranks = list(tensors.keys())
        for i in range(1, len(ranks)):
            torch.testing.assert_close(
                tensors[ranks[0]],
                tensors[ranks[i]],
                msg=f"{msg}: rank 0 vs rank {ranks[i]}"
            )

    def _assert_ranks_different(self, lt, msg=""):
        """Assert that a LocalTensor has different values on different ranks."""
        tensors = lt._local_tensors
        ranks = list(tensors.keys())
        all_same = all(
            torch.allclose(tensors[ranks[0]], tensors[ranks[i]])
            for i in range(1, len(ranks))
        )
        self.assertFalse(all_same, f"{msg}: expected different values per rank")

    def _check_backward_not_none(self, autograd_fn_class, forward_args, backward_grad_out):
        """
        Verify that an autograd Function's backward returns valid gradients (not None).

        This directly tests the backward method of autograd Functions to catch bugs
        like returning None instead of the actual gradient tensor.

        Args:
            autograd_fn_class: The torch.autograd.Function class to test
            forward_args: Tuple of args to pass to forward (will also set up ctx)
            backward_grad_out: The gradient output to pass to backward

        Returns:
            The result of the backward call
        """
        # Create a context object
        class FakeCtx:
            pass
        ctx = FakeCtx()

        # Run forward to populate ctx
        autograd_fn_class.forward(ctx, *forward_args)

        # Run backward
        result = autograd_fn_class.backward(ctx, backward_grad_out)

        # Check that the gradient for the main input (first element) is not None
        if isinstance(result, tuple):
            grad_input = result[0]
        else:
            grad_input = result

        self.assertIsNotNone(grad_input, f"{autograd_fn_class.__name__}.backward returned None")

        # Also verify it's a proper tensor/LocalTensor with the right shape
        if isinstance(backward_grad_out, LocalTensor):
            self.assertIsInstance(grad_input, LocalTensor)
        else:
            self.assertIsInstance(grad_input, torch.Tensor)

        return result


class TestLTensorCreation(unittest.TestCase):
    """Test LTensor creation and validation (no LocalTensorMode needed)."""

    def test_ltensor_creation_valid(self):
        """Test creating LTensor with valid types."""
        data = torch.randn(4, 4)
        types = {'tp': R, 'dp': V}
        lt = LTensor(data, types)
        self.assertEqual(lt.get_type('tp'), R)
        self.assertEqual(lt.get_type('dp'), V)
        self.assertIsNone(lt.get_type('nonexistent'))

    def test_ltensor_creation_all_types(self):
        """Test creating LTensor with all valid types."""
        data = torch.randn(4)
        for typ in [R, I, V, P]:
            lt = LTensor(data, {'axis': typ})
            self.assertEqual(lt.get_type('axis'), typ)

    def test_ltensor_creation_with_shard(self):
        """Test creating LTensor with Shard type."""
        data = torch.randn(4, 8)
        lt = LTensor(data, {'tp': S(1)})
        self.assertEqual(lt.get_type('tp'), S(1))
        self.assertIsInstance(lt.get_type('tp'), Shard)

    def test_ltensor_creation_invalid_type(self):
        """Test that invalid types raise TypeError."""
        data = torch.randn(4)
        with self.assertRaises(TypeError) as ctx:
            LTensor(data, {'tp': 'invalid'})
        self.assertIn("Invalid type", str(ctx.exception))

    def test_ltensor_with_type(self):
        """Test LTensor.with_type method."""
        data = torch.randn(4)
        lt = LTensor(data, {'tp': R})
        lt2 = lt.with_type('tp', V)
        # Original unchanged
        self.assertEqual(lt.get_type('tp'), R)
        # New one updated
        self.assertEqual(lt2.get_type('tp'), V)

    def test_ltensor_with_type_shard(self):
        """Test LTensor.with_type with Shard type."""
        data = torch.randn(4, 8)
        lt = LTensor(data, {'tp': R})
        lt2 = lt.with_type('tp', S(0))
        self.assertEqual(lt2.get_type('tp'), S(0))

    def test_ltensor_with_type_invalid(self):
        """Test LTensor.with_type rejects invalid types."""
        data = torch.randn(4)
        lt = LTensor(data, {'tp': R})
        with self.assertRaises(TypeError):
            lt.with_type('tp', 'bad_type')


class TestEinsumTypePropagation(unittest.TestCase):
    """Test einsum type propagation rules (no LocalTensorMode needed)."""

    def _make_ltensor(self, shape, types):
        """Helper to create LTensor with given types."""
        return LTensor(torch.randn(*shape), types)

    def test_einsum_all_replicate(self):
        """All replicate inputs -> replicate output."""
        a = self._make_ltensor((2, 3), {'tp': R})
        b = self._make_ltensor((3, 4), {'tp': R})
        result = einsum('ij,jk->ik', a, b)
        self.assertEqual(result.get_type('tp'), R)

    def test_einsum_all_invariant(self):
        """All invariant inputs -> invariant output."""
        a = self._make_ltensor((2, 3), {'tp': I})
        b = self._make_ltensor((3, 4), {'tp': I})
        result = einsum('ij,jk->ik', a, b)
        self.assertEqual(result.get_type('tp'), I)

    def test_einsum_all_varying(self):
        """All varying inputs -> varying output."""
        a = self._make_ltensor((2, 3), {'tp': V})
        b = self._make_ltensor((3, 4), {'tp': V})
        result = einsum('ij,jk->ik', a, b)
        self.assertEqual(result.get_type('tp'), V)

    def test_einsum_all_partial(self):
        """All partial inputs -> partial output (for linear ops)."""
        a = self._make_ltensor((2, 3), {'tp': P})
        b = self._make_ltensor((3, 4), {'tp': P})
        result = einsum('ij,jk->ik', a, b)
        self.assertEqual(result.get_type('tp'), P)

    def test_einsum_mixed_replicate_varying(self):
        """Mixed replicate/varying -> varying output."""
        a = self._make_ltensor((2, 3), {'tp': R})
        b = self._make_ltensor((3, 4), {'tp': V})
        result = einsum('ij,jk->ik', a, b)
        self.assertEqual(result.get_type('tp'), V)

    def test_einsum_invariant_mixing_error(self):
        """Invariant cannot mix with other types."""
        a = self._make_ltensor((2, 3), {'tp': I})
        b = self._make_ltensor((3, 4), {'tp': R})
        with self.assertRaises(TypeError) as ctx:
            einsum('ij,jk->ik', a, b)
        self.assertIn("Invariant type", str(ctx.exception))
        self.assertIn("cannot mix", str(ctx.exception))

    def test_einsum_invariant_varying_error(self):
        """Invariant cannot mix with varying."""
        a = self._make_ltensor((2, 3), {'tp': I})
        b = self._make_ltensor((3, 4), {'tp': V})
        with self.assertRaises(TypeError):
            einsum('ij,jk->ik', a, b)

    def test_einsum_partial_replicate_error(self):
        """Partial cannot mix with replicate."""
        a = self._make_ltensor((2, 3), {'tp': P})
        b = self._make_ltensor((3, 4), {'tp': R})
        with self.assertRaises(TypeError) as ctx:
            einsum('ij,jk->ik', a, b)
        self.assertIn("Partial type", str(ctx.exception))

    def test_einsum_multi_axis(self):
        """Test type propagation across multiple mesh axes."""
        a = self._make_ltensor((2, 3), {'tp': R, 'dp': V})
        b = self._make_ltensor((3, 4), {'tp': V, 'dp': R})
        result = einsum('ij,jk->ik', a, b)
        # tp: replicate + varying -> varying
        self.assertEqual(result.get_type('tp'), V)
        # dp: varying + replicate -> varying
        self.assertEqual(result.get_type('dp'), V)

    def test_einsum_out_partial_axes(self):
        """Test out_partial_axes for marking output as partial."""
        a = self._make_ltensor((2, 3), {'tp': V})
        b = self._make_ltensor((3, 4), {'tp': V})
        result = einsum('ij,jk->ik', a, b, out_partial_axes={'tp'})
        self.assertEqual(result.get_type('tp'), P)


class TestAllReduce(LocalTensorTestCase):
    """Test all_reduce operation: P -> R | I."""

    def test_all_reduce_p_to_r_forward(self):
        """all_reduce(R): P -> R, forward sums across ranks."""
        # Create "partial" input - different values per rank that need summing
        x = self._generate_inputs((4,), 'partial')

        # Get expected sum before all_reduce modifies x in-place
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        # Run all_reduce
        result = all_reduce(x, 'tp', src=P, dst=R)

        # Check all ranks have the same summed value
        self._assert_all_ranks_equal(result, "all_reduce result should be same on all ranks")
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)

    def test_all_reduce_p_to_i_forward(self):
        """all_reduce(I): P -> I, forward sums across ranks."""
        x = self._generate_inputs((4,), 'partial')
        # Get expected sum before all_reduce modifies x in-place
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        result = all_reduce(x, 'tp', src=P, dst=I)

        self._assert_all_ranks_equal(result, "all_reduce result should be same on all ranks")
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)

    def test_all_reduce_invalid_src(self):
        """all_reduce only accepts partial src."""
        x = self._generate_inputs((4,), 'replicate')
        with self.assertRaises(ValueError) as ctx:
            all_reduce(x, 'tp', src=R, dst=R)
        self.assertIn("must be P", str(ctx.exception))

    def test_all_reduce_invalid_dst(self):
        """all_reduce only accepts replicate or invariant dst."""
        x = self._generate_inputs((4,), 'partial')
        with self.assertRaises(ValueError) as ctx:
            all_reduce(x, 'tp', src=P, dst=V)
        self.assertIn("must be R or I", str(ctx.exception))


class TestAllGather(LocalTensorTestCase):
    """Test all_gather operation: V -> R | I."""

    def test_all_gather_v_to_r_forward(self):
        """all_gather(R): V -> R, gathers shards from all ranks."""
        # Create varying input - different per rank
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)]))

        result = all_gather(x, 'tp', src=V, dst=R, gather_dim=0)

        # Result should be [0, 1, 2] on all ranks (concatenation)
        expected = torch.tensor([0.0, 1.0, 2.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)

    def test_all_gather_v_to_i_forward(self):
        """all_gather(I): V -> I, gathers shards from all ranks."""
        x = self.mode.rank_map(lambda r: torch.tensor([float(r) * 2]))

        result = all_gather(x, 'tp', src=V, dst=I, gather_dim=0)

        expected = torch.tensor([0.0, 2.0, 4.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)

    def test_all_gather_shard_to_r_forward(self):
        """all_gather(R): S(i) -> R, gathers shards from all ranks."""
        # Create sharded input on dim 0
        x = self.mode.rank_map(lambda r: torch.tensor([float(r), float(r) + 0.5]))

        result = all_gather(x, 'tp', src=S(0), dst=R, gather_dim=0)

        # Result should be concatenated shards
        expected = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)

    def test_all_gather_invalid_src(self):
        """all_gather only accepts V or S(i) src."""
        x = self._generate_inputs((4,), 'replicate')
        with self.assertRaises(ValueError) as ctx:
            all_gather(x, 'tp', src=R, dst=R)
        self.assertIn("must be V or S(i)", str(ctx.exception))

    def test_all_gather_invalid_dst(self):
        """all_gather only accepts replicate or invariant dst."""
        x = self._generate_inputs((4,), 'varying')
        with self.assertRaises(ValueError) as ctx:
            all_gather(x, 'tp', src=V, dst=P)
        self.assertIn("must be R or I", str(ctx.exception))


class TestReduceScatter(LocalTensorTestCase):
    """Test reduce_scatter operation: P -> V."""

    def test_reduce_scatter_forward(self):
        """reduce_scatter: P -> V, reduces and scatters."""
        # Create input with world_size chunks per rank
        # Each rank has [A_r, B_r, C_r] where total length is world_size * chunk_size
        chunk_size = 2
        x = self.mode.rank_map(lambda r: torch.arange(self.WORLD_SIZE * chunk_size, dtype=torch.float) + r)

        result = reduce_scatter(x, 'tp', src=P, dst=V, scatter_dim=0)

        # Each rank r gets the sum of chunk r from all ranks
        # Rank 0 gets sum of chunks [0:2] from all ranks
        # Rank 1 gets sum of chunks [2:4] from all ranks
        # Rank 2 gets sum of chunks [4:6] from all ranks
        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(chunk_size)
            for src_rank in range(self.WORLD_SIZE):
                chunk_start = r * chunk_size
                chunk_end = (r + 1) * chunk_size
                expected += x._local_tensors[src_rank][chunk_start:chunk_end]
            torch.testing.assert_close(
                result._local_tensors[r],
                expected,
                msg=f"rank {r}"
            )

    def test_reduce_scatter_to_shard(self):
        """reduce_scatter: P -> S(i), reduces and scatters."""
        chunk_size = 2
        x = self.mode.rank_map(lambda r: torch.arange(self.WORLD_SIZE * chunk_size, dtype=torch.float) + r)

        result = reduce_scatter(x, 'tp', src=P, dst=S(0), scatter_dim=0)

        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(chunk_size)
            for src_rank in range(self.WORLD_SIZE):
                chunk_start = r * chunk_size
                chunk_end = (r + 1) * chunk_size
                expected += x._local_tensors[src_rank][chunk_start:chunk_end]
            torch.testing.assert_close(
                result._local_tensors[r],
                expected,
                msg=f"rank {r}"
            )

    def test_reduce_scatter_invalid_src(self):
        """reduce_scatter only accepts P src."""
        x = self._generate_inputs((6,), 'varying')
        with self.assertRaises(ValueError) as ctx:
            reduce_scatter(x, 'tp', src=V, dst=V)
        self.assertIn("must be P", str(ctx.exception))

    def test_reduce_scatter_invalid_dst(self):
        """reduce_scatter only accepts V or S(i) dst."""
        x = self._generate_inputs((6,), 'partial')
        with self.assertRaises(ValueError) as ctx:
            reduce_scatter(x, 'tp', src=P, dst=R)
        self.assertIn("must be V or S(i)", str(ctx.exception))


class TestAllToAll(LocalTensorTestCase):
    """Test all_to_all operation: V -> V."""

    def test_all_to_all_forward(self):
        """all_to_all: V -> V, transposes mesh and tensor dims."""
        # Create input: rank r has [r*3, r*3+1, r*3+2]
        # After all_to_all, rank r should get [r, r+3, r+6]
        x = self.mode.rank_map(lambda r: torch.tensor([float(r * 3 + i) for i in range(self.WORLD_SIZE)]))

        result = all_to_all(x, 'tp', src=V, dst=V, split_dim=0, concat_dim=0)

        # Check result
        for r in range(self.WORLD_SIZE):
            expected = torch.tensor([float(r + i * 3) for i in range(self.WORLD_SIZE)])
            torch.testing.assert_close(result._local_tensors[r], expected, msg=f"rank {r}")

    def test_all_to_all_shard_to_shard(self):
        """all_to_all: S(i) -> S(j), resharding between dimensions."""
        # Each rank has a 2D tensor
        x = self.mode.rank_map(lambda r: torch.arange(6, dtype=torch.float).reshape(3, 2) + r * 10)

        result = all_to_all(x, 'tp', src=S(0), dst=S(1), split_dim=0, concat_dim=1)

        # Check shapes are correct
        for r in range(self.WORLD_SIZE):
            self.assertEqual(result._local_tensors[r].shape[0], 1)
            self.assertEqual(result._local_tensors[r].shape[1], 6)

    def test_all_to_all_invalid_src(self):
        """all_to_all only accepts V or S(i) src."""
        x = self._generate_inputs((6,), 'replicate')
        with self.assertRaises(ValueError) as ctx:
            all_to_all(x, 'tp', src=R, dst=V)
        self.assertIn("must be V or S(i)", str(ctx.exception))

    def test_all_to_all_invalid_dst(self):
        """all_to_all only accepts V or S(i) dst."""
        x = self._generate_inputs((6,), 'varying')
        with self.assertRaises(ValueError) as ctx:
            all_to_all(x, 'tp', src=V, dst=R)
        self.assertIn("must be V or S(i)", str(ctx.exception))


class TestReinterpret(LocalTensorTestCase):
    """Test reinterpret operations (no-op forwards, possibly comms in backwards)."""

    def test_reinterpret_r_to_v_forward(self):
        """reinterpret(R,V): R -> V, no-op forward."""
        x = self._generate_inputs((4,), 'replicate')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, 'tp', src=R, dst=V)

        # Forward is no-op, values unchanged
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_reinterpret_r_to_i_forward(self):
        """reinterpret(R,I): R -> I, no-op forward."""
        x = self._generate_inputs((4,), 'replicate')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, 'tp', src=R, dst=I)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_reinterpret_r_to_p_forward(self):
        """reinterpret(R,P): R -> P, no-op forward."""
        x = self._generate_inputs((4,), 'replicate')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, 'tp', src=R, dst=P)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_reinterpret_i_to_r_forward(self):
        """reinterpret(I,R): I -> R, no-op forward."""
        x = self._generate_inputs((4,), 'invariant')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, 'tp', src=I, dst=R)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_reinterpret_v_to_p_forward(self):
        """reinterpret(V,P): V -> P, no-op forward."""
        x = self._generate_inputs((4,), 'varying')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, 'tp', src=V, dst=P)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_reinterpret_same_type_noop(self):
        """reinterpret with same src and dst is identity."""
        x = self._generate_inputs((4,), 'replicate')
        result = reinterpret(x, 'tp', src=R, dst=R)
        # Should return same tensor
        self.assertIs(result, x)

    def test_reinterpret_shard_rejected(self):
        """reinterpret rejects Shard types."""
        x = self._generate_inputs((4,), 'replicate')
        with self.assertRaises(ValueError) as ctx:
            reinterpret(x, 'tp', src=R, dst=S(0))
        self.assertIn("does not support S(i)", str(ctx.exception))

    def test_reinterpret_unsupported_transition(self):
        """reinterpret rejects unsupported transitions."""
        x = self._generate_inputs((4,), 'partial')
        with self.assertRaises(ValueError) as ctx:
            reinterpret(x, 'tp', src=P, dst=R)
        self.assertIn("not supported", str(ctx.exception))


class TestConvert(LocalTensorTestCase):
    """Test convert operations (semantics-preserving type coercion)."""

    def test_convert_r_to_v_forward(self):
        """convert(R,V): R -> V, slices to local portion."""
        # Create replicated input [0, 1, 2, 3, 4, 5] on all ranks
        base = torch.arange(6, dtype=torch.float)
        x = self.mode.rank_map(lambda r: base.clone())

        result = convert(x, 'tp', src=R, dst=V, dim=0)

        # Each rank gets its chunk: rank 0 gets [0,1], rank 1 gets [2,3], rank 2 gets [4,5]
        for r in range(self.WORLD_SIZE):
            expected = base[r * 2:(r + 1) * 2]
            torch.testing.assert_close(result._local_tensors[r], expected, msg=f"rank {r}")

    def test_convert_r_to_shard_forward(self):
        """convert(R,S(i)): R -> S(i), slices to local portion."""
        base = torch.arange(6, dtype=torch.float)
        x = self.mode.rank_map(lambda r: base.clone())

        result = convert(x, 'tp', src=R, dst=S(0), dim=0)

        for r in range(self.WORLD_SIZE):
            expected = base[r * 2:(r + 1) * 2]
            torch.testing.assert_close(result._local_tensors[r], expected, msg=f"rank {r}")

    def test_convert_i_to_v_forward(self):
        """convert(I,V): I -> V, slices to local portion."""
        base = torch.arange(6, dtype=torch.float)
        x = self.mode.rank_map(lambda r: base.clone())

        result = convert(x, 'tp', src=I, dst=V, dim=0)

        for r in range(self.WORLD_SIZE):
            expected = base[r * 2:(r + 1) * 2]
            torch.testing.assert_close(result._local_tensors[r], expected, msg=f"rank {r}")

    def test_convert_i_to_shard_forward(self):
        """convert(I,S(i)): I -> S(i), slices to local portion."""
        base = torch.arange(6, dtype=torch.float)
        x = self.mode.rank_map(lambda r: base.clone())

        result = convert(x, 'tp', src=I, dst=S(0), dim=0)

        for r in range(self.WORLD_SIZE):
            expected = base[r * 2:(r + 1) * 2]
            torch.testing.assert_close(result._local_tensors[r], expected, msg=f"rank {r}")

    def test_convert_r_to_p_forward(self):
        """convert(R,P): R -> P, zeros out non-rank-0."""
        base = torch.tensor([1.0, 2.0, 3.0])
        x = self.mode.rank_map(lambda r: base.clone())

        result = convert(x, 'tp', src=R, dst=P)

        # Rank 0 keeps values, others are zeroed
        torch.testing.assert_close(result._local_tensors[0], base)
        for r in range(1, self.WORLD_SIZE):
            torch.testing.assert_close(
                result._local_tensors[r],
                torch.zeros_like(base),
                msg=f"rank {r} should be zeros"
            )

    def test_convert_i_to_p_forward(self):
        """convert(I,P): I -> P, zeros out non-rank-0."""
        base = torch.tensor([1.0, 2.0, 3.0])
        x = self.mode.rank_map(lambda r: base.clone())

        result = convert(x, 'tp', src=I, dst=P)

        torch.testing.assert_close(result._local_tensors[0], base)
        for r in range(1, self.WORLD_SIZE):
            torch.testing.assert_close(
                result._local_tensors[r],
                torch.zeros_like(base),
                msg=f"rank {r} should be zeros"
            )

    def test_convert_v_to_p_forward(self):
        """convert(V,P): V -> P, places data in disjoint positions."""
        # Each rank has [r]
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)]))

        result = convert(x, 'tp', src=V, dst=P, dim=0)

        # Each rank has a tensor of size world_size with its value at its position
        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(self.WORLD_SIZE)
            expected[r] = float(r)
            torch.testing.assert_close(result._local_tensors[r], expected, msg=f"rank {r}")

    def test_convert_shard_to_p_forward(self):
        """convert(S(i),P): S(i) -> P, places data in disjoint positions."""
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)]))

        result = convert(x, 'tp', src=S(0), dst=P, dim=0)

        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(self.WORLD_SIZE)
            expected[r] = float(r)
            torch.testing.assert_close(result._local_tensors[r], expected, msg=f"rank {r}")

    def test_convert_same_type_noop(self):
        """convert with same src and dst is identity."""
        x = self._generate_inputs((4,), 'replicate')
        result = convert(x, 'tp', src=R, dst=R)
        self.assertIs(result, x)

    def test_convert_r_to_i_same_as_reinterpret(self):
        """convert(R,I) should work like reinterpret(R,I)."""
        x = self._generate_inputs((4,), 'replicate')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = convert(x, 'tp', src=R, dst=I)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_convert_i_to_r_same_as_reinterpret(self):
        """convert(I,R) should work like reinterpret(I,R)."""
        x = self._generate_inputs((4,), 'invariant')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = convert(x, 'tp', src=I, dst=R)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_convert_from_partial_error(self):
        """convert cannot convert from partial."""
        x = self._generate_inputs((4,), 'partial')
        with self.assertRaises(ValueError) as ctx:
            convert(x, 'tp', src=P, dst=R)
        self.assertIn("not supported", str(ctx.exception))


class TestMeshSetup(unittest.TestCase):
    """Test mesh setup functions."""

    def test_set_and_get_mesh(self):
        """Test set_mesh and get_mesh."""
        original = get_mesh()

        fake_mesh = object()
        set_mesh(fake_mesh)
        self.assertIs(get_mesh(), fake_mesh)

        # Restore
        set_mesh(original)


class TestReinterpretCompositions(LocalTensorTestCase):
    """Test compositional reinterpret operations: I->V and I->P."""

    def test_reinterpret_i_to_v_forward(self):
        """reinterpret(I,V): I -> R -> V composition, no-op forward."""
        x = self._generate_inputs((4,), 'invariant')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, 'tp', src=I, dst=V)

        # Forward is no-op (composition of two no-ops)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_reinterpret_i_to_p_forward(self):
        """reinterpret(I,P): I -> R -> P composition, no-op forward."""
        x = self._generate_inputs((4,), 'invariant')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, 'tp', src=I, dst=P)

        # Forward is no-op (composition of two no-ops)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])


class TestAllGatherMultiDim(LocalTensorTestCase):
    """Test all_gather with different gather dimensions."""

    def test_all_gather_dim_1(self):
        """all_gather with gather_dim=1."""
        # Each rank has shape (2, 1)
        x = self.mode.rank_map(lambda r: torch.tensor([[float(r)], [float(r + 10)]]))

        result = all_gather(x, 'tp', src=V, dst=R, gather_dim=1)

        # Result should have shape (2, 3) on all ranks
        expected = torch.tensor([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)

    def test_all_gather_2d_tensors(self):
        """all_gather with 2D varying tensors."""
        # Each rank has a 2x2 matrix
        x = self.mode.rank_map(lambda r: torch.full((2, 2), float(r)))

        result = all_gather(x, 'tp', src=V, dst=R, gather_dim=0)

        # Result should be (6, 2) - concatenation along dim 0
        expected = torch.tensor([
            [0.0, 0.0], [0.0, 0.0],
            [1.0, 1.0], [1.0, 1.0],
            [2.0, 2.0], [2.0, 2.0],
        ])
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)


class TestAllToAllMultiDim(LocalTensorTestCase):
    """Test all_to_all with different split/concat dimensions."""

    def test_all_to_all_2d_same_dims(self):
        """all_to_all with 2D tensors, split and concat on same dim."""
        # Each rank has shape (3, 2) - split on dim 0
        x = self.mode.rank_map(lambda r: torch.arange(6, dtype=torch.float).reshape(3, 2) + r * 10)

        result = all_to_all(x, 'tp', split_dim=0, concat_dim=0)

        # After all_to_all:
        # Rank 0 gets [0th row from rank 0, 0th row from rank 1, 0th row from rank 2]
        # etc.
        for r in range(self.WORLD_SIZE):
            result_tensor = result._local_tensors[r]
            self.assertEqual(result_tensor.shape, (3, 2))


class TestLTensorNoTypes(unittest.TestCase):
    """Test LTensor with no type annotations."""

    def test_ltensor_empty_types(self):
        """LTensor with empty types dict."""
        data = torch.randn(4)
        lt = LTensor(data, {})
        self.assertEqual(lt.types, {})
        self.assertIsNone(lt.get_type('any_axis'))

    def test_ltensor_none_types(self):
        """LTensor with None types (default)."""
        data = torch.randn(4)
        lt = LTensor(data)
        self.assertEqual(lt.types, {})

    def test_einsum_with_no_types(self):
        """einsum with operands that have no type annotations."""
        a = LTensor(torch.randn(2, 3), {})
        b = LTensor(torch.randn(3, 4), {})
        result = einsum('ij,jk->ik', a, b)
        self.assertEqual(result.types, {})

    def test_einsum_partial_types(self):
        """einsum where only some operands have type annotations."""
        a = LTensor(torch.randn(2, 3), {'tp': R})
        b = LTensor(torch.randn(3, 4), {})  # No types
        result = einsum('ij,jk->ik', a, b)
        # The type is inherited from the one operand that has it
        self.assertEqual(result.get_type('tp'), R)


class TestEinsumSingleOperand(unittest.TestCase):
    """Test einsum with single operand (unary operations)."""

    def _make_ltensor(self, shape, types):
        return LTensor(torch.randn(*shape), types)

    def test_einsum_trace(self):
        """einsum trace operation: ii->"""
        a = self._make_ltensor((3, 3), {'tp': R})
        result = einsum('ii->', a)
        self.assertEqual(result.get_type('tp'), R)
        self.assertEqual(result.data.shape, ())

    def test_einsum_transpose(self):
        """einsum transpose operation: ij->ji"""
        a = self._make_ltensor((2, 3), {'tp': V})
        result = einsum('ij->ji', a)
        self.assertEqual(result.get_type('tp'), V)
        self.assertEqual(result.data.shape, (3, 2))

    def test_einsum_sum_reduction(self):
        """einsum sum reduction: ij->"""
        a = self._make_ltensor((2, 3), {'tp': P})
        result = einsum('ij->', a)
        self.assertEqual(result.get_type('tp'), P)


class TestTypeSingletons(unittest.TestCase):
    """Test that type objects are singletons."""

    def test_replicate_is_singleton(self):
        """R should be a singleton."""
        self.assertIs(Replicate(), R)
        self.assertIs(Replicate(), Replicate())

    def test_invariant_is_singleton(self):
        """I should be a singleton."""
        self.assertIs(Invariant(), I)
        self.assertIs(Invariant(), Invariant())

    def test_varying_is_singleton(self):
        """V should be a singleton."""
        self.assertIs(Varying(), V)
        self.assertIs(Varying(), Varying())

    def test_partial_is_singleton(self):
        """P should be a singleton."""
        self.assertIs(Partial(), P)
        self.assertIs(Partial(), Partial())

    def test_shard_equality(self):
        """Shard objects with same dim should be equal."""
        self.assertEqual(S(0), S(0))
        self.assertEqual(S(1), S(1))
        self.assertNotEqual(S(0), S(1))
        self.assertNotEqual(S(0), V)

    def test_type_repr(self):
        """Test string representation of types."""
        self.assertEqual(repr(R), "R")
        self.assertEqual(repr(I), "I")
        self.assertEqual(repr(V), "V")
        self.assertEqual(repr(P), "P")
        self.assertEqual(repr(S(0)), "S(0)")
        self.assertEqual(repr(S(1)), "S(1)")


class TestBackwardType(unittest.TestCase):
    """Test backward_type method for all types."""

    def test_replicate_backward_type(self):
        """Replicate backward type is Partial."""
        self.assertIs(R.backward_type(), P)

    def test_invariant_backward_type(self):
        """Invariant backward type is Invariant."""
        self.assertIs(I.backward_type(), I)

    def test_varying_backward_type(self):
        """Varying backward type is Varying."""
        self.assertIs(V.backward_type(), V)

    def test_partial_backward_type(self):
        """Partial backward type is Replicate."""
        self.assertIs(P.backward_type(), R)

    def test_shard_backward_type(self):
        """Shard backward type is the same Shard."""
        s = S(0)
        self.assertEqual(s.backward_type(), s)
        s2 = S(1)
        self.assertEqual(s2.backward_type(), s2)


class TestPartitionSpec(unittest.TestCase):
    """Test PartitionSpec for Global SPMD."""

    def test_empty_partition_spec(self):
        """Empty partition spec is fully replicated."""
        ps = PartitionSpec()
        self.assertEqual(len(ps), 0)
        self.assertTrue(ps.is_replicated())
        self.assertEqual(ps.get_mesh_axes(), set())

    def test_single_axis_partition_spec(self):
        """Partition spec with single mesh axis."""
        ps = PartitionSpec('tp', None)
        self.assertEqual(len(ps), 2)
        self.assertFalse(ps.is_replicated())
        self.assertEqual(ps.get_mesh_axes(), {'tp'})
        self.assertEqual(ps[0], 'tp')
        self.assertIsNone(ps[1])

    def test_multi_axis_partition_spec(self):
        """Partition spec with multiple mesh axes on same dim."""
        ps = PartitionSpec(('dp', 'tp'), None)
        self.assertEqual(len(ps), 2)
        self.assertEqual(ps.get_mesh_axes(), {'dp', 'tp'})
        self.assertEqual(ps[0], ('dp', 'tp'))

    def test_partition_spec_iteration(self):
        """Partition spec should be iterable."""
        ps = PartitionSpec('tp', 'dp', None)
        axes = list(ps)
        self.assertEqual(axes, ['tp', 'dp', None])

    def test_partition_spec_repr(self):
        """Test string representation of PartitionSpec."""
        ps = PartitionSpec()
        self.assertEqual(repr(ps), "PartitionSpec()")

        ps = PartitionSpec('tp', None)
        self.assertEqual(repr(ps), "PartitionSpec('tp', None)")

        ps = PartitionSpec(('dp', 'tp'), 'ep')
        self.assertEqual(repr(ps), "PartitionSpec(('dp', 'tp'), 'ep')")

    def test_all_none_is_replicated(self):
        """Partition spec with all None is replicated."""
        ps = PartitionSpec(None, None, None)
        self.assertTrue(ps.is_replicated())


class TestRedistribute(LocalTensorTestCase):
    """Test redistribute operation (semantics-preserving with comms)."""

    def test_redistribute_v_to_r(self):
        """redistribute(V,R) uses all_gather."""
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)]))

        result = redistribute(x, 'tp', src=V, dst=R, dim=0)

        expected = torch.tensor([0.0, 1.0, 2.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)

    def test_redistribute_v_to_i(self):
        """redistribute(V,I) uses all_gather."""
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)]))

        result = redistribute(x, 'tp', src=V, dst=I, dim=0)

        expected = torch.tensor([0.0, 1.0, 2.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)

    def test_redistribute_p_to_r(self):
        """redistribute(P,R) uses all_reduce."""
        x = self._generate_inputs((4,), 'partial')
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        result = redistribute(x, 'tp', src=P, dst=R)

        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)

    def test_redistribute_p_to_v(self):
        """redistribute(P,V) uses reduce_scatter."""
        chunk_size = 2
        x = self.mode.rank_map(lambda r: torch.arange(self.WORLD_SIZE * chunk_size, dtype=torch.float) + r)

        result = redistribute(x, 'tp', src=P, dst=V, dim=0)

        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(chunk_size)
            for src_rank in range(self.WORLD_SIZE):
                chunk_start = r * chunk_size
                chunk_end = (r + 1) * chunk_size
                expected += x._local_tensors[src_rank][chunk_start:chunk_end]
            torch.testing.assert_close(result._local_tensors[r], expected, msg=f"rank {r}")

    def test_redistribute_r_to_v_uses_convert(self):
        """redistribute(R,V) delegates to convert."""
        base = torch.arange(6, dtype=torch.float)
        x = self.mode.rank_map(lambda r: base.clone())

        result = redistribute(x, 'tp', src=R, dst=V, dim=0)

        for r in range(self.WORLD_SIZE):
            expected = base[r * 2:(r + 1) * 2]
            torch.testing.assert_close(result._local_tensors[r], expected, msg=f"rank {r}")

    def test_redistribute_r_to_p_uses_convert(self):
        """redistribute(R,P) delegates to convert."""
        base = torch.tensor([1.0, 2.0, 3.0])
        x = self.mode.rank_map(lambda r: base.clone())

        result = redistribute(x, 'tp', src=R, dst=P)

        torch.testing.assert_close(result._local_tensors[0], base)
        for r in range(1, self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], torch.zeros_like(base))

    def test_redistribute_same_type_noop(self):
        """redistribute with same src and dst is identity."""
        x = self._generate_inputs((4,), 'replicate')
        result = redistribute(x, 'tp', src=R, dst=R)
        self.assertIs(result, x)

    def test_redistribute_shard_to_shard_uses_all_to_all(self):
        """redistribute(S(i),S(j)) with different dims uses all_to_all."""
        x = self.mode.rank_map(lambda r: torch.arange(6, dtype=torch.float).reshape(3, 2) + r * 10)

        result = redistribute(x, 'tp', src=S(0), dst=S(1), dim=0)

        # Check shapes are correct
        for r in range(self.WORLD_SIZE):
            self.assertEqual(result._local_tensors[r].shape[0], 1)
            self.assertEqual(result._local_tensors[r].shape[1], 6)


class TestAutogradFunctionBackward(LocalTensorTestCase):
    """
    Test that autograd Function backward methods return valid gradients.

    These tests directly invoke the backward methods of the autograd Functions
    in _api.py to verify they return proper tensors (not None). This catches
    bugs like using dist.all_reduce (which returns None) instead of
    funcol.all_reduce(...).wait() (which returns the tensor).
    """

    def test_all_reduce_to_replicate_backward(self):
        """_AllReduceToReplicate.backward should return valid gradient."""
        from dte._api import _AllReduceToReplicate
        x = self._generate_inputs((4,), 'partial')
        grad_out = self._generate_inputs((4,), 'partial')
        self._check_backward_not_none(_AllReduceToReplicate, (x, 'tp'), grad_out)

    def test_all_reduce_to_invariant_backward(self):
        """_AllReduceToInvariant.backward should return valid gradient."""
        from dte._api import _AllReduceToInvariant
        x = self._generate_inputs((4,), 'partial')
        grad_out = self._generate_inputs((4,), 'replicate')
        self._check_backward_not_none(_AllReduceToInvariant, (x, 'tp'), grad_out)

    def test_invariant_to_replicate_backward(self):
        """_InvariantToReplicate.backward should return valid gradient.

        This is the operation that had the bug (returning None from dist.all_reduce).
        """
        from dte._api import _InvariantToReplicate
        x = self._generate_inputs((4,), 'invariant')
        grad_out = self._generate_inputs((4,), 'partial')
        self._check_backward_not_none(_InvariantToReplicate, (x, 'tp'), grad_out)

    def test_replicate_to_invariant_backward(self):
        """_ReplicateToInvariant.backward should return valid gradient."""
        from dte._api import _ReplicateToInvariant
        x = self._generate_inputs((4,), 'replicate')
        grad_out = self._generate_inputs((4,), 'invariant')
        self._check_backward_not_none(_ReplicateToInvariant, (x, 'tp'), grad_out)

    def test_replicate_to_varying_backward(self):
        """_ReplicateToVarying.backward should return valid gradient."""
        from dte._api import _ReplicateToVarying
        x = self._generate_inputs((4,), 'replicate')
        grad_out = self._generate_inputs((4,), 'varying')
        self._check_backward_not_none(_ReplicateToVarying, (x, 'tp'), grad_out)

    def test_varying_to_partial_backward(self):
        """_VaryingToPartial.backward should return valid gradient."""
        from dte._api import _VaryingToPartial
        x = self._generate_inputs((4,), 'varying')
        grad_out = self._generate_inputs((4,), 'replicate')
        self._check_backward_not_none(_VaryingToPartial, (x, 'tp'), grad_out)

    def test_replicate_to_partial_backward(self):
        """_ReplicateToPartial.backward should return valid gradient."""
        from dte._api import _ReplicateToPartial
        x = self._generate_inputs((4,), 'replicate')
        grad_out = self._generate_inputs((4,), 'replicate')
        self._check_backward_not_none(_ReplicateToPartial, (x, 'tp'), grad_out)

    def test_convert_replicate_to_varying_backward(self):
        """_ConvertReplicateToVarying.backward should return valid gradient."""
        from dte._api import _ConvertReplicateToVarying
        x = self._generate_inputs((6,), 'replicate')
        # Output is smaller (chunked)
        grad_out = self._generate_inputs((2,), 'varying')
        self._check_backward_not_none(_ConvertReplicateToVarying, (x, 'tp', 0), grad_out)

    def test_convert_invariant_to_varying_backward(self):
        """_ConvertInvariantToVarying.backward should return valid gradient."""
        from dte._api import _ConvertInvariantToVarying
        x = self._generate_inputs((6,), 'invariant')
        grad_out = self._generate_inputs((2,), 'varying')
        self._check_backward_not_none(_ConvertInvariantToVarying, (x, 'tp', 0), grad_out)

    def test_convert_replicate_to_partial_backward(self):
        """_ConvertReplicateToPartial.backward should return valid gradient."""
        from dte._api import _ConvertReplicateToPartial
        x = self._generate_inputs((4,), 'replicate')
        grad_out = self._generate_inputs((4,), 'replicate')
        self._check_backward_not_none(_ConvertReplicateToPartial, (x, 'tp'), grad_out)

    def test_convert_invariant_to_partial_backward(self):
        """_ConvertInvariantToPartial.backward should return valid gradient."""
        from dte._api import _ConvertInvariantToPartial
        x = self._generate_inputs((4,), 'invariant')
        grad_out = self._generate_inputs((4,), 'invariant')
        self._check_backward_not_none(_ConvertInvariantToPartial, (x, 'tp'), grad_out)

    def test_convert_varying_to_partial_backward(self):
        """_ConvertVaryingToPartial.backward should return valid gradient."""
        from dte._api import _ConvertVaryingToPartial
        x = self._generate_inputs((2,), 'varying')
        # Output is larger (padded)
        grad_out = self._generate_inputs((6,), 'replicate')
        self._check_backward_not_none(_ConvertVaryingToPartial, (x, 'tp', 0), grad_out)


if __name__ == '__main__':
    unittest.main()
