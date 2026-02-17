"""
Tests for local (no-comms) operations: reinterpret, convert.

Covers: _local.py
"""

import unittest

import torch
from sixlib.spmd_types import (
    convert,
    I,
    P,
    R,
    reinterpret,
    S,
    V,
)
from sixlib.spmd_types._checker import (
    assert_type,
    get_axis_local_type,
)
from sixlib.spmd_types._test_utils import LocalTensorTestCase


class TestReinterpret(LocalTensorTestCase):
    """Test reinterpret operations (no-op forwards, possibly comms in backwards)."""

    def test_reinterpret_r_to_v(self):
        """reinterpret(R,V): R -> V, no-op forward. Tests forward and backward."""
        x = self._generate_inputs((4,), "tp", R)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, "tp", src=R, dst=V, expert_mode=True)

        # Forward is no-op, values unchanged
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_reinterpret_r_to_i(self):
        """reinterpret(R,I): R -> I, no-op forward. Tests forward and backward."""
        x = self._generate_inputs((4,), "tp", R)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, "tp", src=R, dst=I, expert_mode=True)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, "tp"), I)

    def test_reinterpret_r_to_p(self):
        """reinterpret(R,P): R -> P, no-op forward. Tests forward and backward."""
        x = self._generate_inputs((4,), "tp", R)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, "tp", src=R, dst=P, expert_mode=True)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_reinterpret_i_to_r(self):
        """reinterpret(I,R): I -> R, no-op forward. Tests forward and backward."""
        x = self._generate_inputs((4,), "tp", I)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, "tp", src=I, dst=R)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_reinterpret_v_to_p(self):
        """reinterpret(V,P): V -> P, no-op forward. Tests forward and backward."""
        x = self._generate_inputs((4,), "tp", V)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, "tp", src=V, dst=P)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_reinterpret_same_type_noop(self):
        """reinterpret with same src and dst is identity."""
        x = self._generate_inputs((4,), "tp", R)
        result = reinterpret(x, "tp", src=R, dst=R)
        # Should return same tensor
        self.assertIs(result, x)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_reinterpret_shard_rejected(self):
        """reinterpret rejects Shard types."""
        x = self._generate_inputs((4,), "tp", R)
        with self.assertRaises(ValueError) as ctx:
            reinterpret(x, "tp", src=R, dst=S(0))
        self.assertIn("does not support S(i)", str(ctx.exception))

    def test_reinterpret_unsupported_transition(self):
        """reinterpret rejects unsupported transitions."""
        x = self._generate_inputs((4,), "tp", P)
        with self.assertRaises(ValueError) as ctx:
            reinterpret(x, "tp", src=P, dst=R)
        self.assertIn("not supported", str(ctx.exception))

    def test_reinterpret_expert_mode_required(self):
        """reinterpret(R,V), reinterpret(R,I), reinterpret(R,P) require expert_mode."""
        x = self._generate_inputs((4,), "tp", R)
        for dst in [V, I, P]:
            with self.assertRaises(ValueError) as ctx:
                reinterpret(x, "tp", src=R, dst=dst)
            self.assertIn("expert_mode", str(ctx.exception))


class TestConvert(LocalTensorTestCase):
    """Test convert operations (semantics-preserving type coercion)."""

    def test_convert_r_to_v(self):
        """convert(R,V): R -> V, slices to local portion. Tests forward and backward."""
        # Create replicated input shaped for stack semantics
        base = torch.arange(6, dtype=torch.float).reshape(self.WORLD_SIZE, 2)
        x = self.mode.rank_map(lambda r: base.clone())
        assert_type(x, {"tp": R})

        result = convert(x, "tp", src=R, dst=V)

        # Each rank gets its chunk: rank 0 gets [0,1], rank 1 gets [2,3], rank 2 gets [4,5]
        for r in range(self.WORLD_SIZE):
            expected = base[r]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_convert_r_to_shard(self):
        """convert(R,S(i)): R -> S(i), slices to local portion."""
        base = torch.arange(6, dtype=torch.float)
        x = self.mode.rank_map(lambda r: base.clone())
        assert_type(x, {"tp": R})

        result = convert(x, "tp", src=R, dst=S(0))

        for r in range(self.WORLD_SIZE):
            expected = base[r * 2 : (r + 1) * 2]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_convert_i_to_v(self):
        """convert(I,V): I -> V, slices to local portion. Tests forward and backward."""
        base = torch.arange(6, dtype=torch.float).reshape(self.WORLD_SIZE, 2)
        x = self.mode.rank_map(lambda r: base.clone())
        assert_type(x, {"tp": I})

        result = convert(x, "tp", src=I, dst=V)

        for r in range(self.WORLD_SIZE):
            expected = base[r]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_convert_i_to_shard(self):
        """convert(I,S(i)): I -> S(i), slices to local portion."""
        base = torch.arange(6, dtype=torch.float)
        x = self.mode.rank_map(lambda r: base.clone())
        assert_type(x, {"tp": I})

        result = convert(x, "tp", src=I, dst=S(0))

        for r in range(self.WORLD_SIZE):
            expected = base[r * 2 : (r + 1) * 2]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_convert_r_to_p(self):
        """convert(R,P): R -> P, zeros out non-rank-0. Tests forward and backward."""
        base = torch.tensor([1.0, 2.0, 3.0])
        x = self.mode.rank_map(lambda r: base.clone())
        assert_type(x, {"tp": R})

        result = convert(x, "tp", src=R, dst=P)

        # Rank 0 keeps values, others are zeroed
        torch.testing.assert_close(result._local_tensors[0], base)
        for r in range(1, self.WORLD_SIZE):
            torch.testing.assert_close(
                result._local_tensors[r],
                torch.zeros_like(base),
                msg=f"rank {r} should be zeros",
            )
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_convert_i_to_p(self):
        """convert(I,P): I -> P, zeros out non-rank-0. Tests forward and backward."""
        base = torch.tensor([1.0, 2.0, 3.0])
        x = self.mode.rank_map(lambda r: base.clone())
        assert_type(x, {"tp": I})

        result = convert(x, "tp", src=I, dst=P, expert_mode=True)

        torch.testing.assert_close(result._local_tensors[0], base)
        for r in range(1, self.WORLD_SIZE):
            torch.testing.assert_close(
                result._local_tensors[r],
                torch.zeros_like(base),
                msg=f"rank {r} should be zeros",
            )
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_convert_v_to_p(self):
        """convert(V,P): V -> P, places data in disjoint positions. Tests forward and backward."""
        # Each rank has [r]
        x = self.mode.rank_map(lambda r: torch.tensor(float(r)))
        assert_type(x, {"tp": V})

        result = convert(x, "tp", src=V, dst=P, expert_mode=True)

        # Each rank has a tensor of size world_size with its value at its position
        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(self.WORLD_SIZE)
            expected[r] = float(r)
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_convert_shard_to_p(self):
        """convert(S(i),P): S(i) -> P, places data in disjoint positions."""
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)]))
        assert_type(x, {"tp": V})

        result = convert(x, "tp", src=S(0), dst=P, expert_mode=True)

        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(self.WORLD_SIZE)
            expected[r] = float(r)
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_convert_same_type_noop(self):
        """convert with same src and dst is identity."""
        x = self._generate_inputs((4,), "tp", R)
        result = convert(x, "tp", src=R, dst=R)
        self.assertIs(result, x)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_convert_r_to_i_same_as_reinterpret(self):
        """convert(R,I) should work like reinterpret(R,I)."""
        x = self._generate_inputs((4,), "tp", R)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = convert(x, "tp", src=R, dst=I)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, "tp"), I)

    def test_convert_i_to_r_same_as_reinterpret(self):
        """convert(I,R) should work like reinterpret(I,R)."""
        x = self._generate_inputs((4,), "tp", I)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = convert(x, "tp", src=I, dst=R)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_convert_from_partial_error(self):
        """convert cannot convert from partial."""
        x = self._generate_inputs((4,), "tp", P)
        with self.assertRaises(ValueError) as ctx:
            convert(x, "tp", src=P, dst=R)
        self.assertIn("not supported", str(ctx.exception))

    def test_convert_expert_mode_required(self):
        """convert(I,P) and convert(V,P) require expert_mode."""
        x_i = self._generate_inputs((4,), "tp", I)
        with self.assertRaises(ValueError) as ctx:
            convert(x_i, "tp", src=I, dst=P)
        self.assertIn("expert_mode", str(ctx.exception))

        x_v = self.mode.rank_map(lambda r: torch.tensor(float(r)))
        assert_type(x_v, {"tp": V})
        with self.assertRaises(ValueError) as ctx:
            convert(x_v, "tp", src=V, dst=P)
        self.assertIn("expert_mode", str(ctx.exception))

        # S(i) -> P also requires expert_mode (S normalizes to V)
        x_s = self.mode.rank_map(lambda r: torch.tensor([float(r)]))
        assert_type(x_s, {"tp": V})
        with self.assertRaises(ValueError) as ctx:
            convert(x_s, "tp", src=S(0), dst=P)
        self.assertIn("expert_mode", str(ctx.exception))


class TestReinterpretCompositions(LocalTensorTestCase):
    """Test compositional reinterpret operations: I->V and I->P."""

    def test_reinterpret_i_to_v(self):
        """reinterpret(I,V): I -> R -> V composition, no-op forward."""
        x = self._generate_inputs((4,), "tp", I)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, "tp", src=I, dst=V)

        # Forward is no-op (composition of two no-ops)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_reinterpret_i_to_p(self):
        """reinterpret(I,P): I -> R -> P composition, no-op forward."""
        x = self._generate_inputs((4,), "tp", I)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        result = reinterpret(x, "tp", src=I, dst=P)

        # Forward is no-op (composition of two no-ops)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, "tp"), P)


if __name__ == "__main__":
    unittest.main()
