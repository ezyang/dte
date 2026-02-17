"""
Tests for collective operations: all_reduce, all_gather, reduce_scatter, all_to_all.

Covers: _collectives.py
"""

import unittest

import torch
from sixlib.spmd_types import (
    all_gather,
    all_reduce,
    all_to_all,
    I,
    P,
    R,
    reduce_scatter,
    S,
    V,
)
from sixlib.spmd_types._checker import (
    assert_type,
    get_axis_local_type,
)
from sixlib.spmd_types._test_utils import LocalTensorTestCase
from sixlib.spmd_types.types import SpmdTypeError


class TestAllReduce(LocalTensorTestCase):
    """Test all_reduce operation: P -> R | I."""

    def test_all_reduce_p_to_r(self):
        """all_reduce(R): P -> R, sums across ranks. Tests forward and backward."""
        # Create "partial" input - different values per rank that need summing
        x = self._generate_inputs((4,), "tp", P)

        # Get expected sum before all_reduce modifies x in-place
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        # Run all_reduce
        result = all_reduce(x, "tp", src=P, dst=R)

        # Check all ranks have the same summed value
        self._assert_all_ranks_equal(
            result, "all_reduce result should be same on all ranks"
        )
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_all_reduce_p_to_i(self):
        """all_reduce(I): P -> I, sums across ranks. Tests forward and backward."""
        x = self._generate_inputs((4,), "tp", P)
        # Get expected sum before all_reduce modifies x in-place
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        result = all_reduce(x, "tp", src=P, dst=I)

        self._assert_all_ranks_equal(
            result, "all_reduce result should be same on all ranks"
        )
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)
        self.assertIs(get_axis_local_type(result, "tp"), I)

    def test_all_reduce_invalid_src(self):
        """all_reduce only accepts partial src."""
        x = self._generate_inputs((4,), "tp", R)
        with self.assertRaises(ValueError) as ctx:
            all_reduce(x, "tp", src=R, dst=R)
        self.assertIn("must be P", str(ctx.exception))

    def test_all_reduce_invalid_dst(self):
        """all_reduce only accepts replicate or invariant dst."""
        x = self._generate_inputs((4,), "tp", P)
        with self.assertRaises(ValueError) as ctx:
            all_reduce(x, "tp", src=P, dst=V)
        self.assertIn("must be R or I", str(ctx.exception))

    def test_all_reduce_p_to_r_inplace(self):
        """all_reduce(R, inplace=True): P -> R in-place. Result shares storage with input."""
        x = self._generate_inputs((4,), "tp", P)
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        # Save data pointers before
        input_ptrs = {r: x._local_tensors[r].data_ptr() for r in range(self.WORLD_SIZE)}

        result = all_reduce(x, "tp", src=P, dst=R, inplace=True)

        # Check correctness
        self._assert_all_ranks_equal(
            result, "all_reduce inplace result should be same on all ranks"
        )
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)
        self.assertIs(get_axis_local_type(result, "tp"), R)

        # Check in-place: returned tensor shares storage with input
        for r in range(self.WORLD_SIZE):
            self.assertEqual(
                result._local_tensors[r].data_ptr(),
                input_ptrs[r],
                f"rank {r}: inplace result should share storage with input",
            )

    def test_all_reduce_p_to_i_inplace(self):
        """all_reduce(I, inplace=True): P -> I in-place. Result shares storage with input."""
        x = self._generate_inputs((4,), "tp", P)
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        # Save data pointers before
        input_ptrs = {r: x._local_tensors[r].data_ptr() for r in range(self.WORLD_SIZE)}

        result = all_reduce(x, "tp", src=P, dst=I, inplace=True)

        # Check correctness
        self._assert_all_ranks_equal(
            result, "all_reduce inplace result should be same on all ranks"
        )
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)
        self.assertIs(get_axis_local_type(result, "tp"), I)

        # Check in-place: returned tensor shares storage with input
        for r in range(self.WORLD_SIZE):
            self.assertEqual(
                result._local_tensors[r].data_ptr(),
                input_ptrs[r],
                f"rank {r}: inplace result should share storage with input",
            )

    def test_all_reduce_preserves_other_axes(self):
        """all_reduce preserves SPMD types on other mesh axes."""
        x = self.mode.rank_map(lambda r: torch.randn(4) + r)
        assert_type(x, {"tp": P, "dp": R})
        result = all_reduce(x, "tp", src=P, dst=R)
        self.assertIs(get_axis_local_type(result, "tp"), R)
        self.assertIs(get_axis_local_type(result, "dp"), R)

    def test_all_reduce_type_mismatch(self):
        """all_reduce raises SpmdTypeError when input type doesn't match src."""
        x = self._generate_inputs((4,), "tp", R)
        with self.assertRaises(SpmdTypeError):
            all_reduce(x, "tp", src=P, dst=R)


class TestAllGather(LocalTensorTestCase):
    """Test all_gather operation: V -> R | I."""

    def test_all_gather_v_to_r(self):
        """all_gather(R): V -> R, gathers shards from all ranks."""
        # Create varying input - different per rank
        x = self.mode.rank_map(lambda r: torch.tensor(float(r)))
        assert_type(x, {"tp": V})

        result = all_gather(x, "tp", src=V, dst=R)

        # Result should be [0, 1, 2] on all ranks (stack)
        expected = torch.tensor([0.0, 1.0, 2.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_all_gather_v_to_i(self):
        """all_gather(I): V -> I, gathers shards from all ranks."""
        x = self.mode.rank_map(lambda r: torch.tensor(float(r) * 2))
        assert_type(x, {"tp": V})

        result = all_gather(x, "tp", src=V, dst=I)

        expected = torch.tensor([0.0, 2.0, 4.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, "tp"), I)

    def test_all_gather_shard_to_r(self):
        """all_gather(R): S(i) -> R, gathers shards from all ranks."""
        # Create sharded input on dim 0
        x = self.mode.rank_map(lambda r: torch.tensor([float(r), float(r) + 0.5]))
        assert_type(x, {"tp": V})

        result = all_gather(x, "tp", src=S(0), dst=R)

        # Result should be concatenated shards
        expected = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_all_gather_invalid_src(self):
        """all_gather only accepts V or S(i) src."""
        x = self._generate_inputs((4,), "tp", R)
        with self.assertRaises(ValueError) as ctx:
            all_gather(x, "tp", src=R, dst=R)
        self.assertIn("must be V or S(i)", str(ctx.exception))

    def test_all_gather_invalid_dst(self):
        """all_gather only accepts replicate or invariant dst."""
        x = self._generate_inputs((4,), "tp", V)
        with self.assertRaises(ValueError) as ctx:
            all_gather(x, "tp", src=V, dst=P)
        self.assertIn("must be R or I", str(ctx.exception))


class TestReduceScatter(LocalTensorTestCase):
    """Test reduce_scatter operation: P -> V."""

    def test_reduce_scatter(self):
        """reduce_scatter: P -> V, reduces and scatters."""
        # Create input with world_size chunks per rank
        # Each rank has [A_r, B_r, C_r] where total length is world_size * chunk_size
        chunk_size = 2
        x = self.mode.rank_map(
            lambda r: torch.arange(
                self.WORLD_SIZE * chunk_size, dtype=torch.float
            ).reshape(self.WORLD_SIZE, chunk_size)
            + r
        )
        assert_type(x, {"tp": P})

        result = reduce_scatter(x, "tp", src=P, dst=V, scatter_dim=0)

        # Each rank r gets the sum of chunk r from all ranks
        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(chunk_size)
            for src_rank in range(self.WORLD_SIZE):
                expected += x._local_tensors[src_rank][r]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_reduce_scatter_to_shard(self):
        """reduce_scatter: P -> S(i), reduces and scatters."""
        chunk_size = 2
        x = self.mode.rank_map(
            lambda r: torch.arange(self.WORLD_SIZE * chunk_size, dtype=torch.float) + r
        )
        assert_type(x, {"tp": P})

        result = reduce_scatter(x, "tp", src=P, dst=S(0))

        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(chunk_size)
            for src_rank in range(self.WORLD_SIZE):
                chunk_start = r * chunk_size
                chunk_end = (r + 1) * chunk_size
                expected += x._local_tensors[src_rank][chunk_start:chunk_end]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_reduce_scatter_invalid_src(self):
        """reduce_scatter only accepts P (or V, which is auto-reinterpreted) src."""
        x = self._generate_inputs((6,), "tp", R)
        with self.assertRaises(ValueError) as ctx:
            reduce_scatter(x, "tp", src=R, dst=V)
        self.assertIn("must be P", str(ctx.exception))

    def test_reduce_scatter_invalid_dst(self):
        """reduce_scatter only accepts V or S(i) dst."""
        x = self._generate_inputs((6,), "tp", P)
        with self.assertRaises(ValueError) as ctx:
            reduce_scatter(x, "tp", src=P, dst=R)
        self.assertIn("must be V or S(i)", str(ctx.exception))


class TestAllToAll(LocalTensorTestCase):
    """Test all_to_all operation: V -> V."""

    def test_all_to_all(self):
        """all_to_all: V -> V, transposes mesh and tensor dims."""
        # Create input: rank r has [r*3, r*3+1, r*3+2]
        # After all_to_all, rank r should get [r, r+3, r+6]
        x = self.mode.rank_map(
            lambda r: torch.tensor([float(r * 3 + i) for i in range(self.WORLD_SIZE)])
        )
        assert_type(x, {"tp": V})

        result = all_to_all(x, "tp", src=V, dst=V, split_dim=0, concat_dim=0)

        # Check result
        for r in range(self.WORLD_SIZE):
            expected = torch.tensor([float(r + i * 3) for i in range(self.WORLD_SIZE)])
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_all_to_all_shard_to_shard(self):
        """all_to_all: S(i) -> S(j), resharding between dimensions."""
        # Each rank has a 2D tensor
        x = self.mode.rank_map(
            lambda r: torch.arange(6, dtype=torch.float).reshape(3, 2) + r * 10
        )
        assert_type(x, {"tp": V})

        result = all_to_all(x, "tp", src=S(0), dst=S(1))

        # Check shapes are correct
        for r in range(self.WORLD_SIZE):
            self.assertEqual(result._local_tensors[r].shape[0], 1)
            self.assertEqual(result._local_tensors[r].shape[1], 6)
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_all_to_all_invalid_src(self):
        """all_to_all only accepts V or S(i) src."""
        x = self._generate_inputs((6,), "tp", R)
        with self.assertRaises(ValueError) as ctx:
            all_to_all(x, "tp", src=R, dst=V)
        self.assertIn("must be V or S(i)", str(ctx.exception))

    def test_all_to_all_invalid_dst(self):
        """all_to_all only accepts V or S(i) dst."""
        x = self._generate_inputs((6,), "tp", V)
        with self.assertRaises(ValueError) as ctx:
            all_to_all(x, "tp", src=V, dst=R)
        self.assertIn("must be V or S(i)", str(ctx.exception))


class TestAllGatherMultiDim(LocalTensorTestCase):
    """Test all_gather with different gather dimensions."""

    def test_all_gather_dim_1(self):
        """all_gather with S(1) concatenates on dim 1."""
        # Each rank has shape (2, 1)
        x = self.mode.rank_map(lambda r: torch.tensor([[float(r)], [float(r + 10)]]))
        assert_type(x, {"tp": V})

        result = all_gather(x, "tp", src=S(1), dst=R)

        # Result should have shape (2, 3) on all ranks (cat along dim 1)
        expected = torch.tensor([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_all_gather_2d_tensors(self):
        """all_gather with 2D varying tensors."""
        # Each rank has a 2x2 matrix
        x = self.mode.rank_map(lambda r: torch.full((2, 2), float(r)))
        assert_type(x, {"tp": V})

        result = all_gather(x, "tp", src=V, dst=R)

        # Result should be (3, 2, 2) - stack along dim 0
        expected = torch.stack(
            [torch.full((2, 2), float(r)) for r in range(self.WORLD_SIZE)], dim=0
        )
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, "tp"), R)


class TestAllToAllMultiDim(LocalTensorTestCase):
    """Test all_to_all with different split/concat dimensions."""

    def test_all_to_all_2d_same_dims(self):
        """all_to_all with 2D tensors, split and concat on same dim."""
        # Each rank has shape (3, 2) - split on dim 0
        x = self.mode.rank_map(
            lambda r: torch.arange(6, dtype=torch.float).reshape(3, 2) + r * 10
        )
        assert_type(x, {"tp": V})

        result = all_to_all(x, "tp", split_dim=0, concat_dim=0)

        # After all_to_all:
        # Rank 0 gets [0th row from rank 0, 0th row from rank 1, 0th row from rank 2]
        # etc.
        for r in range(self.WORLD_SIZE):
            result_tensor = result._local_tensors[r]
            self.assertEqual(result_tensor.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
