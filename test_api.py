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
    ALLOWED_PCAST_STATES,
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


class TestLTensorCreation(unittest.TestCase):
    """Test LTensor creation and validation (no LocalTensorMode needed)."""

    def test_ltensor_creation_valid(self):
        """Test creating LTensor with valid types."""
        data = torch.randn(4, 4)
        types = {'tp': 'replicate', 'dp': 'varying'}
        lt = LTensor(data, types)
        self.assertEqual(lt.get_type('tp'), 'replicate')
        self.assertEqual(lt.get_type('dp'), 'varying')
        self.assertIsNone(lt.get_type('nonexistent'))

    def test_ltensor_creation_all_types(self):
        """Test creating LTensor with all valid types."""
        data = torch.randn(4)
        for typ in ALLOWED_PCAST_STATES:
            lt = LTensor(data, {'axis': typ})
            self.assertEqual(lt.get_type('axis'), typ)

    def test_ltensor_creation_invalid_type(self):
        """Test that invalid types raise ValueError."""
        data = torch.randn(4)
        with self.assertRaises(ValueError) as ctx:
            LTensor(data, {'tp': 'invalid'})
        self.assertIn("Invalid type 'invalid'", str(ctx.exception))

    def test_ltensor_with_type(self):
        """Test LTensor.with_type method."""
        data = torch.randn(4)
        lt = LTensor(data, {'tp': 'replicate'})
        lt2 = lt.with_type('tp', 'varying')
        # Original unchanged
        self.assertEqual(lt.get_type('tp'), 'replicate')
        # New one updated
        self.assertEqual(lt2.get_type('tp'), 'varying')

    def test_ltensor_with_type_invalid(self):
        """Test LTensor.with_type rejects invalid types."""
        data = torch.randn(4)
        lt = LTensor(data, {'tp': 'replicate'})
        with self.assertRaises(ValueError):
            lt.with_type('tp', 'bad_type')


class TestEinsumTypePropagation(unittest.TestCase):
    """Test einsum type propagation rules (no LocalTensorMode needed)."""

    def _make_ltensor(self, shape, types):
        """Helper to create LTensor with given types."""
        return LTensor(torch.randn(*shape), types)

    def test_einsum_all_replicate(self):
        """All replicate inputs -> replicate output."""
        a = self._make_ltensor((2, 3), {'tp': 'replicate'})
        b = self._make_ltensor((3, 4), {'tp': 'replicate'})
        result = einsum('ij,jk->ik', a, b)
        self.assertEqual(result.get_type('tp'), 'replicate')

    def test_einsum_all_invariant(self):
        """All invariant inputs -> invariant output."""
        a = self._make_ltensor((2, 3), {'tp': 'invariant'})
        b = self._make_ltensor((3, 4), {'tp': 'invariant'})
        result = einsum('ij,jk->ik', a, b)
        self.assertEqual(result.get_type('tp'), 'invariant')

    def test_einsum_all_varying(self):
        """All varying inputs -> varying output."""
        a = self._make_ltensor((2, 3), {'tp': 'varying'})
        b = self._make_ltensor((3, 4), {'tp': 'varying'})
        result = einsum('ij,jk->ik', a, b)
        self.assertEqual(result.get_type('tp'), 'varying')

    def test_einsum_all_partial(self):
        """All partial inputs -> partial output (for linear ops)."""
        a = self._make_ltensor((2, 3), {'tp': 'partial'})
        b = self._make_ltensor((3, 4), {'tp': 'partial'})
        result = einsum('ij,jk->ik', a, b)
        self.assertEqual(result.get_type('tp'), 'partial')

    def test_einsum_mixed_replicate_varying(self):
        """Mixed replicate/varying -> varying output."""
        a = self._make_ltensor((2, 3), {'tp': 'replicate'})
        b = self._make_ltensor((3, 4), {'tp': 'varying'})
        result = einsum('ij,jk->ik', a, b)
        self.assertEqual(result.get_type('tp'), 'varying')

    def test_einsum_invariant_mixing_error(self):
        """Invariant cannot mix with other types."""
        a = self._make_ltensor((2, 3), {'tp': 'invariant'})
        b = self._make_ltensor((3, 4), {'tp': 'replicate'})
        with self.assertRaises(TypeError) as ctx:
            einsum('ij,jk->ik', a, b)
        self.assertIn("Invariant type", str(ctx.exception))
        self.assertIn("cannot mix", str(ctx.exception))

    def test_einsum_invariant_varying_error(self):
        """Invariant cannot mix with varying."""
        a = self._make_ltensor((2, 3), {'tp': 'invariant'})
        b = self._make_ltensor((3, 4), {'tp': 'varying'})
        with self.assertRaises(TypeError):
            einsum('ij,jk->ik', a, b)

    def test_einsum_partial_replicate_error(self):
        """Partial cannot mix with replicate."""
        a = self._make_ltensor((2, 3), {'tp': 'partial'})
        b = self._make_ltensor((3, 4), {'tp': 'replicate'})
        with self.assertRaises(TypeError) as ctx:
            einsum('ij,jk->ik', a, b)
        self.assertIn("Partial type", str(ctx.exception))

    def test_einsum_multi_axis(self):
        """Test type propagation across multiple mesh axes."""
        a = self._make_ltensor((2, 3), {'tp': 'replicate', 'dp': 'varying'})
        b = self._make_ltensor((3, 4), {'tp': 'varying', 'dp': 'replicate'})
        result = einsum('ij,jk->ik', a, b)
        # tp: replicate + varying -> varying
        self.assertEqual(result.get_type('tp'), 'varying')
        # dp: varying + replicate -> varying
        self.assertEqual(result.get_type('dp'), 'varying')


class TestAllReduce(LocalTensorTestCase):
    """Test all_reduce operation: P -> R | I."""

    def test_all_reduce_p_to_r_forward(self):
        """all_reduce(R): P -> R, forward sums across ranks."""
        # Create "partial" input - different values per rank that need summing
        x = self._generate_inputs((4,), 'partial')

        # Get expected sum before all_reduce modifies x in-place
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        # Run all_reduce
        result = all_reduce(x, 'tp', src='partial', tgt='replicate')

        # Check all ranks have the same summed value
        self._assert_all_ranks_equal(result, "all_reduce result should be same on all ranks")
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)

    def test_all_reduce_p_to_i_forward(self):
        """all_reduce(I): P -> I, forward sums across ranks."""
        x = self._generate_inputs((4,), 'partial')
        # Get expected sum before all_reduce modifies x in-place
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        result = all_reduce(x, 'tp', src='partial', tgt='invariant')

        self._assert_all_ranks_equal(result, "all_reduce result should be same on all ranks")
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)

    def test_all_reduce_invalid_src(self):
        """all_reduce only accepts partial src."""
        x = self._generate_inputs((4,), 'replicate')
        with self.assertRaises(ValueError) as ctx:
            all_reduce(x, 'tp', src='replicate', tgt='replicate')
        self.assertIn("must be 'partial'", str(ctx.exception))

    def test_all_reduce_invalid_tgt(self):
        """all_reduce only accepts replicate or invariant tgt."""
        x = self._generate_inputs((4,), 'partial')
        with self.assertRaises(ValueError) as ctx:
            all_reduce(x, 'tp', src='partial', tgt='varying')
        self.assertIn("must be 'replicate' or 'invariant'", str(ctx.exception))


class TestAllGather(LocalTensorTestCase):
    """Test all_gather operation: V -> R | I."""

    def test_all_gather_v_to_r_forward(self):
        """all_gather(R): V -> R, gathers shards from all ranks."""
        # Create varying input - different per rank
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)]))

        result = all_gather(x, 'tp', tgt='replicate', gather_dim=0)

        # Result should be [0, 1, 2] on all ranks (concatenation)
        expected = torch.tensor([0.0, 1.0, 2.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)

    def test_all_gather_v_to_i_forward(self):
        """all_gather(I): V -> I, gathers shards from all ranks."""
        x = self.mode.rank_map(lambda r: torch.tensor([float(r) * 2]))

        result = all_gather(x, 'tp', tgt='invariant', gather_dim=0)

        expected = torch.tensor([0.0, 2.0, 4.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)

    def test_all_gather_invalid_tgt(self):
        """all_gather only accepts replicate or invariant tgt."""
        x = self._generate_inputs((4,), 'varying')
        with self.assertRaises(ValueError) as ctx:
            all_gather(x, 'tp', tgt='partial')
        self.assertIn("must be 'replicate' or 'invariant'", str(ctx.exception))


class TestReduceScatter(LocalTensorTestCase):
    """Test reduce_scatter operation: P -> V."""

    def test_reduce_scatter_forward(self):
        """reduce_scatter: P -> V, reduces and scatters."""
        # Create input with world_size chunks per rank
        # Each rank has [A_r, B_r, C_r] where total length is world_size * chunk_size
        chunk_size = 2
        x = self.mode.rank_map(lambda r: torch.arange(self.WORLD_SIZE * chunk_size, dtype=torch.float) + r)

        result = reduce_scatter(x, 'tp', scatter_dim=0)

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


class TestAllToAll(LocalTensorTestCase):
    """Test all_to_all operation: V -> V."""

    def test_all_to_all_forward(self):
        """all_to_all: V -> V, transposes mesh and tensor dims."""
        # Create input: rank r has [r*3, r*3+1, r*3+2]
        # After all_to_all, rank r should get [r, r+3, r+6]
        x = self.mode.rank_map(lambda r: torch.tensor([float(r * 3 + i) for i in range(self.WORLD_SIZE)]))

        result = all_to_all(x, 'tp', split_dim=0, concat_dim=0)

        # Check result
        for r in range(self.WORLD_SIZE):
            expected = torch.tensor([float(r + i * 3) for i in range(self.WORLD_SIZE)])
            torch.testing.assert_close(result._local_tensors[r], expected, msg=f"rank {r}")


class TestReinterpret(LocalTensorTestCase):
    """Test reinterpret operations (no-op forwards, possibly comms in backwards)."""

    def test_reinterpret_r_to_v_forward(self):
        """reinterpret(R,V): R -> V, no-op forward."""
        x = self._generate_inputs((4,), 'replicate')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, 'tp', src='replicate', tgt='varying')

        # Forward is no-op, values unchanged
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_reinterpret_r_to_i_forward(self):
        """reinterpret(R,I): R -> I, no-op forward."""
        x = self._generate_inputs((4,), 'replicate')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, 'tp', src='replicate', tgt='invariant')

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_reinterpret_r_to_p_forward(self):
        """reinterpret(R,P): R -> P, no-op forward."""
        x = self._generate_inputs((4,), 'replicate')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, 'tp', src='replicate', tgt='partial')

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_reinterpret_i_to_r_forward(self):
        """reinterpret(I,R): I -> R, no-op forward."""
        x = self._generate_inputs((4,), 'invariant')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, 'tp', src='invariant', tgt='replicate')

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_reinterpret_v_to_p_forward(self):
        """reinterpret(V,P): V -> P, no-op forward."""
        x = self._generate_inputs((4,), 'varying')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, 'tp', src='varying', tgt='partial')

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_reinterpret_same_type_noop(self):
        """reinterpret with same src and tgt is identity."""
        x = self._generate_inputs((4,), 'replicate')
        result = reinterpret(x, 'tp', src='replicate', tgt='replicate')
        # Should return same tensor
        self.assertIs(result, x)

    def test_reinterpret_invalid_src(self):
        """reinterpret rejects invalid src type."""
        x = self._generate_inputs((4,), 'replicate')
        with self.assertRaises(ValueError) as ctx:
            reinterpret(x, 'tp', src='bad', tgt='varying')
        self.assertIn("Invalid src state", str(ctx.exception))

    def test_reinterpret_invalid_tgt(self):
        """reinterpret rejects invalid tgt type."""
        x = self._generate_inputs((4,), 'replicate')
        with self.assertRaises(ValueError) as ctx:
            reinterpret(x, 'tp', src='replicate', tgt='bad')
        self.assertIn("Invalid tgt state", str(ctx.exception))

    def test_reinterpret_unsupported_transition(self):
        """reinterpret rejects unsupported transitions."""
        x = self._generate_inputs((4,), 'partial')
        with self.assertRaises(ValueError) as ctx:
            reinterpret(x, 'tp', src='partial', tgt='replicate')
        self.assertIn("not supported", str(ctx.exception))


class TestConvert(LocalTensorTestCase):
    """Test convert operations (semantics-preserving type coercion)."""

    def test_convert_r_to_v_forward(self):
        """convert(R,V): R -> V, slices to local portion."""
        # Create replicated input [0, 1, 2, 3, 4, 5] on all ranks
        base = torch.arange(6, dtype=torch.float)
        x = self.mode.rank_map(lambda r: base.clone())

        result = convert(x, 'tp', src='replicate', tgt='varying', dim=0)

        # Each rank gets its chunk: rank 0 gets [0,1], rank 1 gets [2,3], rank 2 gets [4,5]
        for r in range(self.WORLD_SIZE):
            expected = base[r * 2:(r + 1) * 2]
            torch.testing.assert_close(result._local_tensors[r], expected, msg=f"rank {r}")

    def test_convert_i_to_v_forward(self):
        """convert(I,V): I -> V, slices to local portion."""
        base = torch.arange(6, dtype=torch.float)
        x = self.mode.rank_map(lambda r: base.clone())

        result = convert(x, 'tp', src='invariant', tgt='varying', dim=0)

        for r in range(self.WORLD_SIZE):
            expected = base[r * 2:(r + 1) * 2]
            torch.testing.assert_close(result._local_tensors[r], expected, msg=f"rank {r}")

    def test_convert_r_to_p_forward(self):
        """convert(R,P): R -> P, zeros out non-rank-0."""
        base = torch.tensor([1.0, 2.0, 3.0])
        x = self.mode.rank_map(lambda r: base.clone())

        result = convert(x, 'tp', src='replicate', tgt='partial')

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

        result = convert(x, 'tp', src='invariant', tgt='partial')

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

        result = convert(x, 'tp', src='varying', tgt='partial', dim=0)

        # Each rank has a tensor of size world_size with its value at its position
        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(self.WORLD_SIZE)
            expected[r] = float(r)
            torch.testing.assert_close(result._local_tensors[r], expected, msg=f"rank {r}")

    def test_convert_same_type_noop(self):
        """convert with same src and tgt is identity."""
        x = self._generate_inputs((4,), 'replicate')
        result = convert(x, 'tp', src='replicate', tgt='replicate')
        self.assertIs(result, x)

    def test_convert_r_to_i_same_as_reinterpret(self):
        """convert(R,I) should work like reinterpret(R,I)."""
        x = self._generate_inputs((4,), 'replicate')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = convert(x, 'tp', src='replicate', tgt='invariant')

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_convert_i_to_r_same_as_reinterpret(self):
        """convert(I,R) should work like reinterpret(I,R)."""
        x = self._generate_inputs((4,), 'invariant')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = convert(x, 'tp', src='invariant', tgt='replicate')

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_convert_invalid_src(self):
        """convert rejects invalid src type."""
        x = self._generate_inputs((4,), 'replicate')
        with self.assertRaises(ValueError) as ctx:
            convert(x, 'tp', src='bad', tgt='varying')
        self.assertIn("Invalid src state", str(ctx.exception))

    def test_convert_invalid_tgt(self):
        """convert rejects invalid tgt type."""
        x = self._generate_inputs((4,), 'replicate')
        with self.assertRaises(ValueError) as ctx:
            convert(x, 'tp', src='replicate', tgt='bad')
        self.assertIn("Invalid tgt state", str(ctx.exception))

    def test_convert_from_partial_error(self):
        """convert cannot convert from partial."""
        x = self._generate_inputs((4,), 'partial')
        with self.assertRaises(ValueError) as ctx:
            convert(x, 'tp', src='partial', tgt='replicate')
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

        result = reinterpret(x, 'tp', src='invariant', tgt='varying')

        # Forward is no-op (composition of two no-ops)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])

    def test_reinterpret_i_to_p_forward(self):
        """reinterpret(I,P): I -> R -> P composition, no-op forward."""
        x = self._generate_inputs((4,), 'invariant')
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, 'tp', src='invariant', tgt='partial')

        # Forward is no-op (composition of two no-ops)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])


class TestAllGatherMultiDim(LocalTensorTestCase):
    """Test all_gather with different gather dimensions."""

    def test_all_gather_dim_1(self):
        """all_gather with gather_dim=1."""
        # Each rank has shape (2, 1)
        x = self.mode.rank_map(lambda r: torch.tensor([[float(r)], [float(r + 10)]]))

        result = all_gather(x, 'tp', tgt='replicate', gather_dim=1)

        # Result should have shape (2, 3) on all ranks
        expected = torch.tensor([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)

    def test_all_gather_2d_tensors(self):
        """all_gather with 2D varying tensors."""
        # Each rank has a 2x2 matrix
        x = self.mode.rank_map(lambda r: torch.full((2, 2), float(r)))

        result = all_gather(x, 'tp', tgt='replicate', gather_dim=0)

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
        a = LTensor(torch.randn(2, 3), {'tp': 'replicate'})
        b = LTensor(torch.randn(3, 4), {})  # No types
        result = einsum('ij,jk->ik', a, b)
        # The type is inherited from the one operand that has it
        self.assertEqual(result.get_type('tp'), 'replicate')


class TestEinsumSingleOperand(unittest.TestCase):
    """Test einsum with single operand (unary operations)."""

    def _make_ltensor(self, shape, types):
        return LTensor(torch.randn(*shape), types)

    def test_einsum_trace(self):
        """einsum trace operation: ii->"""
        a = self._make_ltensor((3, 3), {'tp': 'replicate'})
        result = einsum('ii->', a)
        self.assertEqual(result.get_type('tp'), 'replicate')
        self.assertEqual(result.data.shape, ())

    def test_einsum_transpose(self):
        """einsum transpose operation: ij->ji"""
        a = self._make_ltensor((2, 3), {'tp': 'varying'})
        result = einsum('ij->ji', a)
        self.assertEqual(result.get_type('tp'), 'varying')
        self.assertEqual(result.data.shape, (3, 2))

    def test_einsum_sum_reduction(self):
        """einsum sum reduction: ij->"""
        a = self._make_ltensor((2, 3), {'tp': 'partial'})
        result = einsum('ij->', a)
        self.assertEqual(result.get_type('tp'), 'partial')


class TestPcast(unittest.TestCase):
    """Test that pcast is an alias for reinterpret."""

    def test_pcast_is_reinterpret(self):
        """pcast should be the same function as reinterpret."""
        from dte._api import pcast
        self.assertIs(pcast, reinterpret)


if __name__ == '__main__':
    unittest.main()
