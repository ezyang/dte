"""
Tests for cross-module SPMD operations: redistribute, negative dim sharding.

Covers: __init__.py (public API), integration across modules.
"""

import unittest

import torch
from sixlib.spmd_types import (
    all_gather,
    convert,
    I,
    P,
    R,
    redistribute,
    reduce_scatter,
    S,
    V,
)
from sixlib.spmd_types._checker import (
    assert_local_type,
    get_axis_local_type,
)
from sixlib.spmd_types._test_utils import LocalTensorTestCase


class TestRedistribute(LocalTensorTestCase):
    """Test redistribute operation (semantics-preserving with comms)."""

    def test_redistribute_v_to_r(self):
        """redistribute(V,R) uses all_gather."""
        x = self.mode.rank_map(lambda r: torch.tensor(float(r)))
        assert_local_type(x, {"tp": V})

        result = redistribute(x, "tp", src=V, dst=R, dim=0)

        expected = torch.tensor([0.0, 1.0, 2.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_redistribute_v_to_i(self):
        """redistribute(V,I) uses all_gather."""
        x = self.mode.rank_map(lambda r: torch.tensor(float(r)))
        assert_local_type(x, {"tp": V})

        result = redistribute(x, "tp", src=V, dst=I, dim=0)

        expected = torch.tensor([0.0, 1.0, 2.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, "tp"), I)

    def test_redistribute_p_to_r(self):
        """redistribute(P,R) uses all_reduce."""
        x = self._generate_inputs((4,), "tp", P)
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        result = redistribute(x, "tp", src=P, dst=R)

        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_redistribute_p_to_s(self):
        """redistribute(P,S(0)) uses reduce_scatter."""
        chunk_size = 2
        x = self.mode.rank_map(
            lambda r: torch.arange(self.WORLD_SIZE * chunk_size, dtype=torch.float) + r
        )
        assert_local_type(x, {"tp": P})

        result = redistribute(x, "tp", src=P, dst=S(0))

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

    def test_redistribute_r_to_v_uses_convert(self):
        """redistribute(R,V) delegates to convert."""
        base = torch.arange(6, dtype=torch.float).reshape(self.WORLD_SIZE, 2)
        x = self.mode.rank_map(lambda r: base.clone())
        assert_local_type(x, {"tp": R})

        result = redistribute(x, "tp", src=R, dst=V, dim=0)

        for r in range(self.WORLD_SIZE):
            expected = base[r]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_redistribute_r_to_p_uses_convert(self):
        """redistribute(R,P) delegates to convert."""
        base = torch.tensor([1.0, 2.0, 3.0])
        x = self.mode.rank_map(lambda r: base.clone())
        assert_local_type(x, {"tp": R})

        result = redistribute(x, "tp", src=R, dst=P)

        torch.testing.assert_close(result._local_tensors[0], base)
        for r in range(1, self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], torch.zeros_like(base))
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_redistribute_same_type_noop(self):
        """redistribute with same src and dst is identity."""
        x = self._generate_inputs((4,), "tp", R)
        result = redistribute(x, "tp", src=R, dst=R)
        self.assertIs(result, x)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_redistribute_shard_to_shard_uses_all_to_all(self):
        """redistribute(S(i),S(j)) with different dims uses all_to_all."""
        x = self.mode.rank_map(
            lambda r: torch.arange(6, dtype=torch.float).reshape(3, 2) + r * 10
        )
        assert_local_type(x, {"tp": V})

        result = redistribute(x, "tp", src=S(0), dst=S(1), dim=0)

        # Check shapes are correct
        for r in range(self.WORLD_SIZE):
            self.assertEqual(result._local_tensors[r].shape[0], 1)
            self.assertEqual(result._local_tensors[r].shape[1], 6)
        self.assertIs(get_axis_local_type(result, "tp"), V)


class TestShardNegativeDim(LocalTensorTestCase):
    """Test that Shard(-1) (negative dimension indexing) works correctly.

    S(-1) should work like S(ndim-1), similar to how PyTorch handles negative
    dim arguments in operations like torch.cat, torch.chunk, etc.
    """

    def test_reduce_scatter_shard_neg1(self):
        """reduce_scatter with dst=S(-1) should scatter along the last dim.

        Bug: _ReduceScatter.forward always divides output_shape[0] regardless of
        scatter_dim, so dst=S(-1) on a 2D tensor produces shape (2, 9) instead
        of (6, 3).
        """
        # Each rank has shape (6, 9), world_size=3.
        # dst=S(-1) means scatter on last dim (dim 1): each rank gets shape (6, 3).
        x = self.mode.rank_map(
            lambda r: torch.arange(54, dtype=torch.float).reshape(6, 9) + r
        )
        assert_local_type(x, {"tp": P})

        result = reduce_scatter(x, "tp", src=P, dst=S(-1))

        for r in range(self.WORLD_SIZE):
            self.assertEqual(
                result._local_tensors[r].shape,
                (6, 3),
                f"rank {r}: expected shape (6, 3) but got "
                f"{tuple(result._local_tensors[r].shape)}",
            )

    def test_all_gather_shard_neg1(self):
        """all_gather with src=S(-1) should gather along the last dim.

        Forward uses torch.cat(dim=-1) which handles negative dims, so this
        should work. Included to confirm forward is fine (backward calls
        reduce_scatter with S(-1) which is broken).
        """
        # Each rank has shape (2, 1), world_size=3.
        # src=S(-1) = S(1) on 2D, result should be (2, 3).
        x = self.mode.rank_map(lambda r: torch.tensor([[float(r)], [float(r + 10)]]))
        assert_local_type(x, {"tp": V})

        result = all_gather(x, "tp", src=S(-1), dst=R)

        expected = torch.tensor([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_redistribute_p_to_shard_neg1(self):
        """redistribute(P, S(-1)) delegates to reduce_scatter, which is broken for S(-1)."""
        x = self.mode.rank_map(
            lambda r: torch.arange(54, dtype=torch.float).reshape(6, 9) + r
        )
        assert_local_type(x, {"tp": P})

        result = redistribute(x, "tp", src=P, dst=S(-1))

        for r in range(self.WORLD_SIZE):
            self.assertEqual(
                result._local_tensors[r].shape,
                (6, 3),
                f"rank {r}: expected shape (6, 3) but got "
                f"{tuple(result._local_tensors[r].shape)}",
            )

    def test_convert_r_to_shard_neg1(self):
        """convert(R, S(-1)) should chunk along the last dim."""
        # Each rank has shape (2, 3), world_size=3.
        # S(-1) = S(1) on 2D, each rank gets a (2, 1) chunk.
        base = torch.arange(6, dtype=torch.float).reshape(2, 3)
        x = self.mode.rank_map(lambda r: base.clone())
        assert_local_type(x, {"tp": R})

        result = convert(x, "tp", src=R, dst=S(-1))

        for r in range(self.WORLD_SIZE):
            expected = base[:, r : r + 1]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )


if __name__ == "__main__":
    unittest.main()
