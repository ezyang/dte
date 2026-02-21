"""
Tests for global SPMD shard propagation: propagating S(i) through torch ops.

In global SPMD, the per-axis types are R, I, P, and S(i) (no V). S(i)
propagation reuses DTensor's sharding propagation to determine how shard
dimensions flow through ops, while local SPMD inference handles R/I/P.

Decision tree per mesh axis:
- Has I in any input? -> local SPMD inference only (DTensor can't distinguish R/I)
- Has S and P? -> reject (error)
- Has S? -> DTensor prop (rejects invalid combos like pointwise S+R)
- No S? -> local SPMD only

When DTensor says output is Partial (contracted sharded dim), the output type
becomes P. The user must all_reduce before using the result in nonlinear ops.
"""

import torch
from sixlib.spmd_types import (
    I,
    P,
    R,
    S,
    V,
    all_reduce,
    local_map,
    redistribute,
)
from sixlib.spmd_types._checker import (
    assert_type,
    get_local_type,
    has_local_type,
    SpmdTypeMode,
)
from sixlib.spmd_types._test_utils import LocalTensorTestCase
from sixlib.spmd_types.types import PerMeshAxisSpmdType, Shard, SpmdTypeError
from torch.distributed._local_tensor import LocalTensorMode
from torch.testing._internal.common_utils import run_tests


def _get_axis_type(tensor: torch.Tensor, axis: str) -> PerMeshAxisSpmdType:
    """Get the full SPMD type for a specific axis, including S(i).

    Unlike get_axis_local_type (which decays S to V), this returns the stored
    type which may include S(i) in global SPMD.
    """
    return get_local_type(tensor).get(axis, V)


class GlobalSpmdTestCase(LocalTensorTestCase):
    """Base test class for global SPMD tests.

    Like LocalTensorTestCase, but uses SpmdTypeMode(global_mode=True) to enable
    S(i) storage and DTensor-based shard propagation.
    """

    def setUp(self):
        self.mode = LocalTensorMode(self.WORLD_SIZE)
        self.mode.__enter__()
        self.spmd_mode = SpmdTypeMode(global_mode=True)
        self.spmd_mode.__enter__()

    def _make_input(self, shape, axis, typ):
        if typ is R or typ is I:
            base = torch.randn(shape)
            result = self.mode.rank_map(lambda r: base.clone())
        else:
            result = self.mode.rank_map(lambda r: torch.randn(shape) + r)
        assert_type(result, {axis: typ})
        return result

    def _make_multi_axis_input(self, shape, types):
        has_shard_or_varying = any(
            isinstance(t, type(S(0))) or t is V or t is P for t in types.values()
        )
        if not has_shard_or_varying:
            base = torch.randn(shape)
            result = self.mode.rank_map(lambda r: base.clone())
        else:
            result = self.mode.rank_map(lambda r: torch.randn(shape) + r)
        assert_type(result, types)
        return result


# =============================================================================
# S(i) storage
# =============================================================================


class TestShardTypeStorage(GlobalSpmdTestCase):
    """Test that S(i) can be stored on and read from tensors."""

    def test_assert_type_stores_shard(self):
        """assert_type({'tp': S(0)}) should store S(0), not decay to V."""
        x = self.mode.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {"tp": S(0)})
        self.assertEqual(_get_axis_type(x, "tp"), S(0))

    def test_assert_type_stores_shard_with_other_axes(self):
        x = self.mode.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {"dp": S(0), "tp": R})
        self.assertEqual(_get_axis_type(x, "dp"), S(0))
        self.assertEqual(_get_axis_type(x, "tp"), R)

    def test_assert_type_shard_check_matches(self):
        """Re-asserting the same S(i) type should succeed."""
        x = self.mode.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {"tp": S(0)})
        assert_type(x, {"tp": S(0)})

    def test_assert_type_shard_check_mismatch_raises(self):
        x = self.mode.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {"tp": S(0)})
        with self.assertRaises(AssertionError):
            assert_type(x, {"tp": S(1)})

    def test_assert_type_shard_vs_local_mismatch_raises(self):
        """S(0) stored, then asserting R should fail."""
        x = self.mode.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {"tp": S(0)})
        with self.assertRaises(AssertionError):
            assert_type(x, {"tp": R})


# =============================================================================
# S(i) propagation through pointwise ops
# =============================================================================


class TestGlobalSpmdPointwise(GlobalSpmdTestCase):
    """Test S(i) propagation through pointwise/elementwise ops."""

    def test_add_s0_s0(self):
        """S(0) + S(0) -> S(0)"""
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", S(0))
        result = x + y
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_add_s0_r_rejected(self):
        """S(0) + R rejected: user must redistribute R first."""
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", R)
        with self.assertRaises(SpmdTypeError):
            x + y

        # Fix: redistribute R -> S(0) first
        y_s = redistribute(y, "tp", src=R, dst=S(0))
        result = x + y_s
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_add_r_s0_rejected(self):
        """R + S(0) rejected: order doesn't matter."""
        x = self._make_input((4, 3), "tp", R)
        y = self._make_input((4, 3), "tp", S(0))
        with self.assertRaises(SpmdTypeError):
            x + y

    def test_mul_s0_r_rejected(self):
        """S(0) * R rejected: user must redistribute R first."""
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", R)
        with self.assertRaises(SpmdTypeError):
            x * y

    def test_mul_s0_s0(self):
        """S(0) * S(0) -> S(0): same shard on both inputs."""
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", S(0))
        result = x * y
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_neg_s0(self):
        """-S(0) -> S(0)"""
        x = self._make_input((4, 3), "tp", S(0))
        result = -x
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_clone_s0(self):
        x = self._make_input((4, 3), "tp", S(0))
        result = x.clone()
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_sub_s0_s0(self):
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", S(0))
        result = x - y
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_div_s0_r_rejected(self):
        """S(0) / R rejected: user must redistribute R first."""
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", R)
        with self.assertRaises(SpmdTypeError):
            x / y

    def test_abs_s0(self):
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(torch.abs(x), "tp"), S(0))

    def test_exp_s0(self):
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(torch.exp(x), "tp"), S(0))

    def test_tanh_s0(self):
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(torch.tanh(x), "tp"), S(0))

    def test_sigmoid_s0(self):
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(torch.sigmoid(x), "tp"), S(0))

    def test_relu_s0(self):
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(torch.relu(x), "tp"), S(0))

    def test_sqrt_s0(self):
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(torch.sqrt(torch.abs(x)), "tp"), S(0))

    def test_rsqrt_s0(self):
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(
            _get_axis_type(torch.rsqrt(torch.abs(x) + 1), "tp"), S(0)
        )

    def test_reciprocal_s0(self):
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(
            _get_axis_type(torch.reciprocal(torch.abs(x) + 1), "tp"), S(0)
        )

    def test_add_scalar(self):
        """S(0) + scalar -> S(0)"""
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(x + 1, "tp"), S(0))

    def test_mul_scalar(self):
        """S(0) * scalar -> S(0)"""
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(x * 2.0, "tp"), S(0))

    def test_sub_scalar(self):
        """S(0) - scalar -> S(0)"""
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(x - 0.5, "tp"), S(0))

    def test_div_scalar(self):
        """S(0) / scalar -> S(0)"""
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(x / 3.0, "tp"), S(0))


# =============================================================================
# S(i) propagation through unary shape-changing ops
# =============================================================================


class TestGlobalSpmdUnaryOps(GlobalSpmdTestCase):
    """Test S(i) propagation through transpose and reduction ops."""

    def test_transpose_s0(self):
        """t(S(0)) -> S(1): transpose swaps shard dim."""
        x = self._make_input((6, 4), "tp", S(0))
        result = torch.t(x)
        self.assertEqual(_get_axis_type(result, "tp"), S(1))

    def test_transpose_s1(self):
        """t(S(1)) -> S(0): transpose swaps shard dim."""
        x = self._make_input((6, 4), "tp", S(1))
        result = x.t()
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_sum_s0_gives_partial(self):
        """sum(S(0)) -> P: reducing a sharded tensor produces partial."""
        x = self._make_input((6, 4), "tp", S(0))
        result = torch.sum(x)
        self.assertEqual(_get_axis_type(result, "tp"), P)

    def test_transpose_then_mm(self):
        """Transpose weight then matmul: t(S(0)) -> S(1), mm(R, S(1)) -> S(1)."""
        x = self._make_input((4, 6), "tp", R)
        w = self._make_input((3, 6), "tp", S(0))
        w_t = torch.t(w)  # S(0) -> S(1), shape (6, 3)
        self.assertEqual(_get_axis_type(w_t, "tp"), S(1))
        result = torch.mm(x, w_t)
        self.assertEqual(_get_axis_type(result, "tp"), S(1))


# =============================================================================
# S(i) propagation through matmul ops
# =============================================================================


class TestGlobalSpmdMatmul(GlobalSpmdTestCase):
    """Test S(i) propagation through matmul ops."""

    def test_mm_s0_r(self):
        """[M,K]@[K,N]: M sharded -> output M sharded. S(0)@R -> S(0)."""
        x = self._make_input((4, 3), "tp", S(0))
        w = self._make_input((3, 5), "tp", R)
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_mm_r_s1(self):
        """[M,K]@[K,N]: N sharded -> output N sharded. R@S(1) -> S(1)."""
        x = self._make_input((4, 3), "tp", R)
        w = self._make_input((3, 5), "tp", S(1))
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, "tp"), S(1))

    def test_mm_r_r(self):
        """R@R -> R: no sharding, baseline case."""
        x = self._make_input((4, 3), "tp", R)
        w = self._make_input((3, 5), "tp", R)
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, "tp"), R)

    def test_matmul_s0_r(self):
        """torch.matmul with S(0)@R -> S(0)."""
        x = self._make_input((4, 3), "tp", S(0))
        w = self._make_input((3, 5), "tp", R)
        result = torch.matmul(x, w)
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_matmul_operator_s0_r(self):
        """x @ w with S(0)@R -> S(0) via __matmul__."""
        x = self._make_input((4, 3), "tp", S(0))
        w = self._make_input((3, 5), "tp", R)
        result = x @ w
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_bmm_s1_r(self):
        """bmm(S(1), R) -> S(1): batch dim replicated, M sharded."""
        x = self._make_input((2, 6, 3), "tp", S(1))
        w = self._make_input((2, 3, 4), "tp", R)
        result = torch.bmm(x, w)
        self.assertEqual(_get_axis_type(result, "tp"), S(1))

    def test_bmm_r_s2(self):
        """bmm(R, S(2)) -> S(2): N dim sharded."""
        x = self._make_input((2, 4, 3), "tp", R)
        w = self._make_input((2, 3, 6), "tp", S(2))
        result = torch.bmm(x, w)
        self.assertEqual(_get_axis_type(result, "tp"), S(2))


# =============================================================================
# Contracted sharded dim -> Partial output
# =============================================================================


class TestGlobalSpmdPartialOutput(GlobalSpmdTestCase):
    """Test that contracting a sharded dim produces Partial output.

    When both inputs shard the contracting dimension K (S(1) on a[M,K],
    S(0) on b[K,N]), DTensor says the output is Partial. The user must
    all_reduce before using the result in nonlinear ops.
    """

    def test_mm_contracted_dim_gives_partial(self):
        """mm(S(1), S(0)) -> P: contracting dim K sharded on both inputs.

        x: [M, K] with S(1) means K (dim 1) sharded.
        w: [K, N] with S(0) means K (dim 0) sharded.
        Local mm: [M, K/n] @ [K/n, N] = [M, N] partial.
        """
        x = self._make_input((4, 3), "tp", S(1))
        w = self._make_input((3, 5), "tp", S(0))
        y = torch.mm(x, w)
        self.assertEqual(_get_axis_type(y, "tp"), P)

    def test_nonlinear_on_partial_rejected(self):
        """y^2 on Partial -> error: nonlinear op on P is rejected.

        The user must insert an all_reduce before nonlinear ops.
        """
        x = self._make_input((4, 3), "tp", S(1))
        w = self._make_input((3, 5), "tp", S(0))
        y = torch.mm(x, w)
        self.assertEqual(_get_axis_type(y, "tp"), P)

        with self.assertRaises(SpmdTypeError):
            y**2

    def test_allreduce_then_nonlinear(self):
        """all_reduce P->R fixes the partial, then nonlinear ops work."""
        x = self._make_input((4, 3), "tp", S(1))
        w = self._make_input((3, 5), "tp", S(0))
        y = torch.mm(x, w)
        self.assertEqual(_get_axis_type(y, "tp"), P)

        # all_reduce: P -> R
        y_r = all_reduce(y, "tp", src=P, dst=R)
        self.assertEqual(_get_axis_type(y_r, "tp"), R)

        # Now nonlinear ops work
        z = y_r**2
        self.assertEqual(_get_axis_type(z, "tp"), R)

    def test_partial_add_partial(self):
        """P + P -> P: addition is linear, two partial sums can add."""
        x1 = self._make_input((4, 3), "tp", S(1))
        w1 = self._make_input((3, 5), "tp", S(0))
        y1 = torch.mm(x1, w1)

        x2 = self._make_input((4, 3), "tp", S(1))
        w2 = self._make_input((3, 5), "tp", S(0))
        y2 = torch.mm(x2, w2)

        result = y1 + y2
        self.assertEqual(_get_axis_type(result, "tp"), P)


# =============================================================================
# S + P rejection, S + I rejection, incompatible sharding
# =============================================================================


class TestGlobalSpmdRejection(GlobalSpmdTestCase):
    """Test cases that should be rejected in global SPMD."""

    def test_add_s_p_rejected(self):
        """S + P on the same axis must be rejected."""
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", P)
        with self.assertRaises(SpmdTypeError):
            x + y

    def test_mul_s_p_rejected(self):
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", P)
        with self.assertRaises(SpmdTypeError):
            x * y

    def test_mm_s_p_rejected(self):
        x = self._make_input((4, 3), "tp", S(0))
        w = self._make_input((3, 5), "tp", P)
        with self.assertRaises(SpmdTypeError):
            torch.mm(x, w)

    def test_add_s0_s1_rejected(self):
        """S(0) + S(1) on same axis: different shard dims, needs redistribution."""
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", S(1))
        with self.assertRaises(SpmdTypeError):
            x + y

    def test_add_s_i_rejected(self):
        """S + I: I cannot mix with other types."""
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", I)
        with self.assertRaises(SpmdTypeError):
            x + y

    def test_mm_weight_k_sharded_rejected(self):
        """mm(R, S(0)): weight's K dim (dim 0) sharded -> no feasible strategy.

        x: [M, K] with R (full).
        w: [K, N] with S(0) means K sharded -> each rank has [K/n, N].
        Local mm [M, K] @ [K/n, N] is a shape mismatch. Needs redistribution.
        """
        x = self._make_input((4, 3), "tp", R)
        w = self._make_input((3, 5), "tp", S(0))
        with self.assertRaises(SpmdTypeError):
            torch.mm(x, w)

    def test_add_s_p_has_fix_suggestion(self):
        """S+P rejection goes through local SPMD and gets fix suggestions."""
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", P)
        with self.assertRaises(SpmdTypeError) as ctx:
            x + y
        self.assertIn("all_reduce", str(ctx.exception))

    def test_add_s_i_has_fix_suggestion(self):
        """S+I rejection goes through local SPMD and gets fix suggestions."""
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", I)
        with self.assertRaises(SpmdTypeError) as ctx:
            x + y
        self.assertIn("reinterpret", str(ctx.exception))


# =============================================================================
# I fallback to local SPMD inference
# =============================================================================


class TestGlobalSpmdInvariant(GlobalSpmdTestCase):
    """Test that I falls back to local SPMD inference (DTensor can't distinguish R/I)."""

    def test_add_i_i(self):
        x = self._make_input((4, 3), "tp", I)
        y = self._make_input((4, 3), "tp", I)
        result = x + y
        self.assertEqual(_get_axis_type(result, "tp"), I)

    def test_mm_i_i(self):
        """I@I -> I: local SPMD says I, DTensor would say R."""
        x = self._make_input((4, 3), "tp", I)
        w = self._make_input((3, 5), "tp", I)
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, "tp"), I)

    def test_mul_i_i(self):
        x = self._make_input((4, 3), "tp", I)
        y = self._make_input((4, 3), "tp", I)
        result = x * y
        self.assertEqual(_get_axis_type(result, "tp"), I)

    def test_add_i_r_rejected(self):
        """I + R: I cannot mix with other types. Must reinterpret first."""
        x = self._make_input((4, 3), "tp", I)
        y = self._make_input((4, 3), "tp", R)
        with self.assertRaises(SpmdTypeError):
            x + y

    def test_mul_i_r_rejected(self):
        x = self._make_input((4, 3), "tp", I)
        y = self._make_input((4, 3), "tp", R)
        with self.assertRaises(SpmdTypeError):
            x * y

    def test_add_i_p_rejected(self):
        x = self._make_input((4, 3), "tp", I)
        y = self._make_input((4, 3), "tp", P)
        with self.assertRaises(SpmdTypeError):
            x + y

    def test_mm_i_r_rejected(self):
        """mm(I, R): I cannot mix with R."""
        x = self._make_input((4, 3), "tp", I)
        w = self._make_input((3, 5), "tp", R)
        with self.assertRaises(SpmdTypeError):
            torch.mm(x, w)


# =============================================================================
# R/P combinations without shard axes
# =============================================================================


class TestGlobalSpmdNoShardAxes(GlobalSpmdTestCase):
    """Test R and P type interactions in global SPMD (no S involved)."""

    def test_add_r_r(self):
        x = self._make_input((4, 3), "tp", R)
        y = self._make_input((4, 3), "tp", R)
        result = x + y
        self.assertEqual(_get_axis_type(result, "tp"), R)

    def test_mul_r_r(self):
        x = self._make_input((4, 3), "tp", R)
        y = self._make_input((4, 3), "tp", R)
        result = x * y
        self.assertEqual(_get_axis_type(result, "tp"), R)

    def test_mm_r_r(self):
        x = self._make_input((4, 3), "tp", R)
        w = self._make_input((3, 5), "tp", R)
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, "tp"), R)

    def test_add_p_p(self):
        """P + P -> P: linear op on two partial values."""
        x = self._make_input((4, 3), "tp", P)
        y = self._make_input((4, 3), "tp", P)
        result = x + y
        self.assertEqual(_get_axis_type(result, "tp"), P)

    def test_mul_p_r(self):
        """P * R -> P: multilinear, partial in one factor."""
        x = self._make_input((4, 3), "tp", P)
        y = self._make_input((4, 3), "tp", R)
        result = x * y
        self.assertEqual(_get_axis_type(result, "tp"), P)

    def test_nonlinear_p_rejected(self):
        """relu(P) -> error: nonlinear op on P."""
        x = self._make_input((4, 3), "tp", P)
        with self.assertRaises(SpmdTypeError):
            torch.relu(x)


# =============================================================================
# Ops accepting list of tensors and multi-output ops
# =============================================================================


class TestGlobalSpmdListAndMultiOutput(GlobalSpmdTestCase):
    """Test ops that accept tensor lists (cat, stack) and multi-output ops (sort)."""

    # --- cat ---

    def test_cat_s1_on_dim0(self):
        """cat([S(1), S(1)], dim=0): shard preserved on non-cat dim."""
        a = self._make_input((4, 6), "tp", S(1))
        b = self._make_input((4, 6), "tp", S(1))
        result = torch.cat([a, b], dim=0)
        self.assertEqual(_get_axis_type(result, "tp"), S(1))

    def test_cat_r_r(self):
        a = self._make_input((4, 3), "tp", R)
        b = self._make_input((4, 3), "tp", R)
        result = torch.cat([a, b], dim=0)
        self.assertEqual(_get_axis_type(result, "tp"), R)

    def test_cat_p_p(self):
        """cat is LINEAR: P + P -> P."""
        a = self._make_input((4, 3), "tp", P)
        b = self._make_input((4, 3), "tp", P)
        result = torch.cat([a, b], dim=0)
        self.assertEqual(_get_axis_type(result, "tp"), P)

    def test_cat_i_i(self):
        a = self._make_input((4, 3), "tp", I)
        b = self._make_input((4, 3), "tp", I)
        result = torch.cat([a, b], dim=0)
        self.assertEqual(_get_axis_type(result, "tp"), I)

    # --- stack ---

    def test_stack_s0_on_dim0(self):
        """stack([S(0), S(0)], dim=0): adds dim 0, original shard dim shifts to 1."""
        a = self._make_input((4, 6), "tp", S(0))
        b = self._make_input((4, 6), "tp", S(0))
        result = torch.stack([a, b], dim=0)
        self.assertEqual(_get_axis_type(result, "tp"), S(1))

    def test_stack_r_r(self):
        a = self._make_input((4, 3), "tp", R)
        b = self._make_input((4, 3), "tp", R)
        result = torch.stack([a, b], dim=0)
        self.assertEqual(_get_axis_type(result, "tp"), R)

    def test_stack_p_p(self):
        """stack is LINEAR: P + P -> P."""
        a = self._make_input((4, 3), "tp", P)
        b = self._make_input((4, 3), "tp", P)
        result = torch.stack([a, b], dim=0)
        self.assertEqual(_get_axis_type(result, "tp"), P)

    # --- sort (multi-output) ---

    def test_sort_r(self):
        """sort(R): both values and indices get R."""
        x = self._make_input((4, 3), "tp", R)
        values, indices = torch.sort(x)
        self.assertEqual(_get_axis_type(values, "tp"), R)
        self.assertEqual(_get_axis_type(indices, "tp"), R)

    def test_sort_s0(self):
        """sort(S(0)): sorting along last dim preserves S(0)."""
        x = self._make_input((4, 6), "tp", S(0))
        values, indices = torch.sort(x)
        self.assertEqual(_get_axis_type(values, "tp"), S(0))
        self.assertEqual(_get_axis_type(indices, "tp"), S(0))

    def test_sort_p_rejected(self):
        """sort is nonlinear: P cannot propagate."""
        x = self._make_input((4, 3), "tp", P)
        with self.assertRaises(SpmdTypeError):
            torch.sort(x)


# =============================================================================
# Multiple independent mesh axes
# =============================================================================


class TestGlobalSpmdMultiAxis(GlobalSpmdTestCase):
    """Test S(i) propagation with multiple independent mesh axes."""

    def test_mm_shard_one_axis_replicate_other(self):
        """S(0)@dp, R@tp propagates independently per axis."""
        x = self._make_multi_axis_input((4, 3), {"dp": S(0), "tp": R})
        w = self._make_multi_axis_input((3, 5), {"dp": R, "tp": R})
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, "dp"), S(0))
        self.assertEqual(_get_axis_type(result, "tp"), R)

    def test_mm_shard_both_axes(self):
        """S(0)@dp, S(0)@tp: both axes shard dim 0, propagate independently."""
        x = self._make_multi_axis_input((4, 3), {"dp": S(0), "tp": S(0)})
        w = self._make_multi_axis_input((3, 5), {"dp": R, "tp": R})
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, "dp"), S(0))
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_mm_different_shard_dims_on_different_axes(self):
        """S(0)@dp on input, S(1)@tp on weight: independent axes, independent shards."""
        x = self._make_multi_axis_input((4, 3), {"dp": S(0), "tp": R})
        w = self._make_multi_axis_input((3, 5), {"dp": R, "tp": S(1)})
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, "dp"), S(0))
        self.assertEqual(_get_axis_type(result, "tp"), S(1))


# =============================================================================
# Redistribute with S(i) types
# =============================================================================


class TestGlobalSpmdRedistribute(GlobalSpmdTestCase):
    """Test redistribute with S(i) types and its role in enabling ops."""

    def test_redistribute_s0_to_r(self):
        """redistribute(S(0), R) gathers shards to replicated."""
        x = self._make_input((4, 3), "tp", S(0))
        x_r = redistribute(x, "tp", src=S(0), dst=R)
        self.assertEqual(_get_axis_type(x_r, "tp"), R)

    def test_redistribute_r_to_s0(self):
        """redistribute(R, S(0)) shards a replicated tensor."""
        x = self._make_input((6, 3), "tp", R)
        x_s = redistribute(x, "tp", src=R, dst=S(0))
        self.assertEqual(_get_axis_type(x_s, "tp"), S(0))

    def test_redistribute_s0_to_s1(self):
        """redistribute(S(0), S(1)) all-to-all to change shard dim."""
        x = self._make_input((6, 3), "tp", S(0))
        x_s1 = redistribute(x, "tp", src=S(0), dst=S(1))
        self.assertEqual(_get_axis_type(x_s1, "tp"), S(1))

    def test_redistribute_enables_mm(self):
        """Redistribute weight from S(0) (K sharded) to R, then mm works.

        Without redistribute, mm(R, S(0)) errors because K is sharded.
        After gathering weight to R, mm(R, R) -> R succeeds.
        """
        x = self._make_input((4, 3), "tp", R)
        w = self._make_input((3, 5), "tp", S(0))

        # Direct mm fails: K dim of weight is sharded
        with self.assertRaises(SpmdTypeError):
            torch.mm(x, w)

        # Redistribute weight: S(0) -> R (all_gather)
        w_r = redistribute(w, "tp", src=S(0), dst=R)
        self.assertEqual(_get_axis_type(w_r, "tp"), R)

        # Now mm works: R @ R -> R
        result = torch.mm(x, w_r)
        self.assertEqual(_get_axis_type(result, "tp"), R)

    def test_redistribute_enables_mm_reshard(self):
        """Redistribute weight from S(0) (K sharded) to S(1) (N sharded).

        mm(R, S(1)) is valid: output N dim is sharded -> S(1).
        """
        x = self._make_input((4, 3), "tp", R)
        w = self._make_input((3, 5), "tp", S(0))

        # Redistribute weight: S(0) -> S(1) (all-to-all, K-shard to N-shard)
        w_s1 = redistribute(w, "tp", src=S(0), dst=S(1))
        self.assertEqual(_get_axis_type(w_s1, "tp"), S(1))

        # Now mm works: R @ S(1) -> S(1)
        result = torch.mm(x, w_s1)
        self.assertEqual(_get_axis_type(result, "tp"), S(1))

    def test_redistribute_fixes_contracted_dim(self):
        """Both inputs shard K. Redistribute to get a feasible strategy.

        x: [M, K@S(1)], w: [K@S(0), N].
        Fix: redistribute x from S(1)->S(0) (shard M), w from S(0)->R.
        Then mm(S(0), R) -> S(0).
        """
        x = self._make_input((4, 3), "tp", S(1))  # K sharded
        w = self._make_input((3, 5), "tp", S(0))  # K sharded

        # Direct mm gives P (contracted dim sharded)
        y = torch.mm(x, w)
        self.assertEqual(_get_axis_type(y, "tp"), P)

        # Alternative: redistribute to avoid partial
        x_s0 = redistribute(x, "tp", src=S(1), dst=S(0))
        w_r = redistribute(w, "tp", src=S(0), dst=R)
        result = torch.mm(x_s0, w_r)
        self.assertEqual(_get_axis_type(result, "tp"), S(0))


# =============================================================================
# Local/global SPMD transition via local_map
# =============================================================================


class TestLocalGlobalTransition(GlobalSpmdTestCase):
    """Test transitions between local and global SPMD via local_map.

    local_map is a higher-order function that:
    - Validates inputs match in_type
    - On entry: decays S(i) -> V for axes with S(i) in in_type
    - Runs the function body in local SPMD (only R, I, V, P propagate)
    - On exit: re-annotates outputs with S(i) from out_type

    in_type is a list of dicts, one per positional arg, specifying the
    expected global SPMD type of each input.
    """

    def test_local_map_matmul(self):
        """Full example from design: local_map wraps a matmul in local SPMD.

        Outside (global SPMD):
          x: S(0)@dp, w: R@dp
        Inside local_map (local SPMD):
          x: V@dp (S decayed), w: R@dp
          x @ w -> V@dp (local SPMD propagation)
        Outside after local_map:
          result: S(0)@dp (from out_type)
        """
        x = self._make_input((4, 3), "dp", S(0))
        w = self._make_input((3, 5), "dp", R)

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_matmul(x, w):
            # Inside local_map, S(0) has decayed to V
            self.assertIs(_get_axis_type(x, "dp"), V)
            self.assertIs(_get_axis_type(w, "dp"), R)
            l1 = x @ w
            # V @ R -> V in local SPMD
            self.assertIs(_get_axis_type(l1, "dp"), V)
            return l1

        y1 = local_matmul(x, w)
        # After local_map, out_type restores S(0)
        self.assertEqual(_get_axis_type(y1, "dp"), S(0))

    def test_local_map_preserves_non_shard_types(self):
        """R and I types are unchanged across local_map boundary."""
        x = self._make_input((4, 3), "tp", R)

        @local_map(
            in_type=[{"tp": R}],
            out_type={"tp": R},
        )
        def identity(x):
            self.assertIs(_get_axis_type(x, "tp"), R)
            return x.clone()

        y = identity(x)
        self.assertEqual(_get_axis_type(y, "tp"), R)

    def test_local_map_validates_in_type(self):
        """local_map rejects inputs that don't match in_type."""
        x = self._make_input((4, 3), "dp", S(0))
        w = self._make_input((3, 5), "dp", R)

        @local_map(
            in_type=[{"dp": S(1)}, {"dp": R}],  # S(1), but x is S(0)
            out_type={"dp": S(0)},
        )
        def local_matmul(x, w):
            return x @ w

        with self.assertRaises((SpmdTypeError, AssertionError)):
            local_matmul(x, w)

    def test_local_map_then_global_op(self):
        """After local_map returns to global SPMD, S(i) propagation works."""
        x = self._make_input((4, 3), "dp", S(0))
        w = self._make_input((3, 5), "dp", R)

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_matmul(x, w):
            return x @ w

        y1 = local_matmul(x, w)
        self.assertEqual(_get_axis_type(y1, "dp"), S(0))

        # Back in global SPMD: redistribute then do another op
        y1_r = redistribute(y1, "dp", src=S(0), dst=R)
        self.assertEqual(_get_axis_type(y1_r, "dp"), R)

        # R @ R -> R in global SPMD
        y2 = y1_r @ w
        self.assertEqual(_get_axis_type(y2, "dp"), R)

    def test_local_map_multiple_outputs(self):
        """local_map with a function returning multiple tensors."""
        x = self._make_input((4, 6), "tp", S(0))

        @local_map(
            in_type=[{"tp": S(0)}],
            out_type={"tp": S(0)},
        )
        def split_fn(x):
            a, b = torch.split(x, 3, dim=1)
            return a, b

        a, b = split_fn(x)
        self.assertEqual(_get_axis_type(a, "tp"), S(0))
        self.assertEqual(_get_axis_type(b, "tp"), S(0))

    def test_round_trip_global_local_global(self):
        """Full round trip: global -> local -> global -> redistribute -> op.

        This mirrors the sample test from the design:
        1. Start with S(0) in global SPMD
        2. Enter local_map: S(0) -> V, do local matmul
        3. Exit local_map: V -> S(0)
        4. In global SPMD: redistribute S(0) -> R
        5. Do global matmul: R @ R -> R
        """
        x = self._make_input((4, 3), "dp", S(0))
        w = self._make_input((3, 5), "dp", R)

        # Step 1-3: local SPMD matmul
        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_matmul(x, w):
            return x @ w

        y1 = local_matmul(x, w)
        self.assertEqual(_get_axis_type(y1, "dp"), S(0))

        # Step 4: redistribute S(0) -> R
        y1_r = redistribute(y1, "dp", src=S(0), dst=R)
        self.assertEqual(_get_axis_type(y1_r, "dp"), R)

        # Step 5: global matmul R @ R -> R
        y2 = y1_r @ w
        self.assertEqual(_get_axis_type(y2, "dp"), R)

    def test_global_pointwise_chain(self):
        """Chain of pointwise ops in global SPMD preserves S(0).

        R inputs must be redistributed to S(0) before pointwise ops.
        """
        x = self._make_input((4, 3), "tp", S(0))
        bias = self._make_input((4, 3), "tp", R)
        scale = self._make_input((4, 3), "tp", R)
        bias_s = redistribute(bias, "tp", src=R, dst=S(0))
        scale_s = redistribute(scale, "tp", src=R, dst=S(0))

        y = x + bias_s
        self.assertEqual(_get_axis_type(y, "tp"), S(0))
        y = y * scale_s
        self.assertEqual(_get_axis_type(y, "tp"), S(0))
        y = -y
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

    def test_global_matmul_then_pointwise(self):
        """mm(S(0), R) -> S(0), then pointwise ops preserve S(0)."""
        x = self._make_input((4, 3), "tp", S(0))
        w = self._make_input((3, 5), "tp", R)
        bias = self._make_input((4, 5), "tp", S(0))

        y = torch.mm(x, w)
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

        # S(0) + S(0) -> S(0)
        y = y + bias
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

    def test_global_contracted_allreduce_pointwise(self):
        """mm(S(1),S(0)) -> P, all_reduce P->R, then pointwise chain."""
        x = self._make_input((4, 3), "tp", S(1))
        w = self._make_input((3, 5), "tp", S(0))
        bias = self._make_input((4, 5), "tp", R)

        y = torch.mm(x, w)
        self.assertEqual(_get_axis_type(y, "tp"), P)

        y = all_reduce(y, "tp", src=P, dst=R)
        self.assertEqual(_get_axis_type(y, "tp"), R)

        # R + R -> R, then R * R -> R
        y = y + bias
        self.assertEqual(_get_axis_type(y, "tp"), R)
        y = y * bias
        self.assertEqual(_get_axis_type(y, "tp"), R)

    def test_global_two_matmuls(self):
        """Chain two matmuls: mm(S(0),R) -> S(0), mm(S(0),R) -> S(0)."""
        x = self._make_input((4, 3), "tp", S(0))
        w1 = self._make_input((3, 5), "tp", R)
        w2 = self._make_input((5, 2), "tp", R)

        h = torch.mm(x, w1)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        y = torch.mm(h, w2)
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

    def test_global_column_then_row_parallel(self):
        """Column-parallel mm(S(0),R) then row-parallel mm(R,S(1)).

        mm(S(0), R) -> S(0), redistribute S(0)->R, mm(R, S(1)) -> S(1).
        """
        x = self._make_input((4, 3), "tp", S(0))
        w1 = self._make_input((3, 6), "tp", R)
        w2 = self._make_input((6, 5), "tp", S(1))

        h = torch.mm(x, w1)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        h_r = redistribute(h, "tp", src=S(0), dst=R)
        self.assertEqual(_get_axis_type(h_r, "tp"), R)

        y = torch.mm(h_r, w2)
        self.assertEqual(_get_axis_type(y, "tp"), S(1))

    def test_local_map_pointwise_chain(self):
        """local_map wrapping a chain of pointwise ops."""
        x = self._make_input((4, 3), "tp", S(0))
        bias = self._make_input((4, 3), "tp", R)

        @local_map(
            in_type=[{"tp": S(0)}, {"tp": R}],
            out_type={"tp": S(0)},
        )
        def local_fn(x, bias):
            self.assertIs(_get_axis_type(x, "tp"), V)
            self.assertIs(_get_axis_type(bias, "tp"), R)
            y = x + bias  # V + R -> V
            y = -y  # V -> V
            y = y * bias  # V * R -> V
            return y

        result = local_fn(x, bias)
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_local_global_local(self):
        """local_map -> global ops -> local_map: two local regions around a global region.

        1. local_map: matmul inside local SPMD, output S(0)
        2. Global: redistribute bias R->S(0), pointwise S(0) + S(0)
        3. local_map: second matmul inside local SPMD, output S(0)
        """
        x = self._make_input((4, 3), "dp", S(0))
        w1 = self._make_input((3, 5), "dp", R)
        bias = self._make_input((4, 5), "dp", R)
        w2 = self._make_input((5, 2), "dp", R)

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_mm1(x, w):
            return x @ w

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_mm2(x, w):
            return x @ w

        # local region 1
        h = local_mm1(x, w1)
        self.assertEqual(_get_axis_type(h, "dp"), S(0))

        # global region: redistribute then pointwise
        bias_s = redistribute(bias, "dp", src=R, dst=S(0))
        h = h + bias_s
        self.assertEqual(_get_axis_type(h, "dp"), S(0))

        # local region 2
        y = local_mm2(h, w2)
        self.assertEqual(_get_axis_type(y, "dp"), S(0))

    def test_local_global_redistribute_local(self):
        """local -> global redistribute -> local: redistribute between two local regions.

        1. local_map: matmul, output S(0)
        2. Global: redistribute S(0) -> R
        3. local_map: second matmul with R input, output R
        """
        x = self._make_input((4, 3), "dp", S(0))
        w1 = self._make_input((3, 5), "dp", R)
        w2 = self._make_input((5, 2), "dp", R)

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_mm(x, w):
            return x @ w

        @local_map(
            in_type=[{"dp": R}, {"dp": R}],
            out_type={"dp": R},
        )
        def local_mm_replicated(x, w):
            return x @ w

        # local region 1: matmul with sharded input
        h = local_mm(x, w1)
        self.assertEqual(_get_axis_type(h, "dp"), S(0))

        # global region: redistribute
        h_r = redistribute(h, "dp", src=S(0), dst=R)
        self.assertEqual(_get_axis_type(h_r, "dp"), R)

        # local region 2: matmul with replicated input
        y = local_mm_replicated(h_r, w2)
        self.assertEqual(_get_axis_type(y, "dp"), R)

    def test_local_map_input_types_restored(self):
        """local_map restores input S types after the call."""
        x = self._make_input((4, 3), "tp", S(0))
        w = self._make_input((3, 5), "tp", R)

        @local_map(
            in_type=[{"tp": S(0)}, {"tp": R}],
            out_type={"tp": S(0)},
        )
        def local_mm(x, w):
            return x @ w

        y = local_mm(x, w)
        # x should still have S(0) after local_map returns
        self.assertEqual(_get_axis_type(x, "tp"), S(0))
        self.assertEqual(_get_axis_type(w, "tp"), R)
        # y should have the out_type
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

    def test_global_mlp_block(self):
        """Simulate an MLP block: mm -> add bias -> relu -> mm -> add bias.

        All ops propagate S(0) through the pipeline.
        """
        x = self._make_input((6, 4), "tp", S(0))
        w1 = self._make_input((4, 8), "tp", R)
        b1 = self._make_input((6, 8), "tp", S(0))
        w2 = self._make_input((8, 3), "tp", R)
        b2 = self._make_input((6, 3), "tp", S(0))

        h = torch.mm(x, w1)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        h = h + b1
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        h = torch.relu(h)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        y = torch.mm(h, w2)
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

        y = y + b2
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

    def test_global_activation_chain(self):
        """Chain of nonlinear activations: abs -> tanh -> sigmoid -> exp."""
        x = self._make_input((6, 4), "tp", S(0))

        h = torch.abs(x)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        h = torch.tanh(h)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        h = torch.sigmoid(h)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        # exp preserves S(0)
        h = torch.exp(h)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

    def test_global_transpose_matmul_pattern(self):
        """Transpose weight then matmul: a common pattern in attention.

        w is stored as S(0) on [N, K], transpose to [K, N] -> S(1),
        then mm(R, S(1)) -> S(1).
        """
        x = self._make_input((4, 6), "tp", R)
        w = self._make_input((3, 6), "tp", S(0))

        w_t = torch.t(w)  # S(0) on (3,6) -> S(1) on (6,3)
        self.assertEqual(_get_axis_type(w_t, "tp"), S(1))

        y = torch.mm(x, w_t)
        self.assertEqual(_get_axis_type(y, "tp"), S(1))

    def test_global_normalize_pattern(self):
        """Simulate normalization: x * scale + bias, redistribute R first."""
        x = self._make_input((6, 4), "tp", S(0))
        scale = self._make_input((6, 4), "tp", R)
        bias = self._make_input((6, 4), "tp", R)
        scale_s = redistribute(scale, "tp", src=R, dst=S(0))
        bias_s = redistribute(bias, "tp", src=R, dst=S(0))

        y = x * scale_s + bias_s
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

    def test_global_sum_then_allreduce(self):
        """sum(S(0)) -> P, all_reduce P->R."""
        x = self._make_input((6, 4), "tp", S(0))

        s = torch.sum(x)
        self.assertEqual(_get_axis_type(s, "tp"), P)

        s_r = all_reduce(s, "tp", src=P, dst=R)
        self.assertEqual(_get_axis_type(s_r, "tp"), R)


# =============================================================================
# Backward correctness via adjoint identity
# =============================================================================


class TestGlobalSpmdBackward(GlobalSpmdTestCase):
    """Test backward correctness for global SPMD operations.

    Autograd's C++ engine doesn't call __torch_function__, so gradient
    tensors don't receive SPMD type annotations. These tests verify that
    gradient VALUES are correct via the adjoint identity:

        <A*(g), dx>_src = <g, A(dx)>_dst

    This identity holds exactly for (multi)linear ops when one argument
    is fixed. The dual inner product <·,·>_T pairs the gradient type of T
    with T itself, respecting the multi-rank structure.
    """

    def _gradient_type(self, typ):
        """Return the gradient (dual) type for a global SPMD type.

        S(i) is locally V, and grad(V) = V, so grad(S(i)) = S(i).
        """
        if isinstance(typ, Shard):
            return typ
        return {R: P, I: I, V: V, P: R}[typ]

    def _make_random_typed(self, like_lt, typ, axis):
        """Create a random LocalTensor with the given SPMD type."""
        if isinstance(typ, Shard) or typ is V or typ is P:
            result = self.mode.rank_map(
                lambda r: torch.randn_like(like_lt._local_tensors[r])
            )
        else:  # R or I: same data on all ranks
            t = torch.randn_like(like_lt._local_tensors[0])
            result = self.mode.rank_map(lambda r: t.clone())
        assert_type(result, {axis: typ})
        return result

    def _dual_inner(self, grad, perturbation, typ):
        """Dual inner product <grad, perturbation> respecting SPMD type.

        Pairs the gradient type of T with T itself:
          R: <P_grad, R_pert> = sum_r grad_r · pert_0
          P: <R_grad, P_pert> = grad_0 · sum_r pert_r
          V/S(i): <V_grad, V_pert> = sum_r grad_r · pert_r
          I: <I_grad, I_pert> = grad_0 · pert_0
        """
        W = self.WORLD_SIZE
        ranks = range(W)
        if isinstance(typ, Shard) or typ is V:
            return sum(
                (grad._local_tensors[r] * perturbation._local_tensors[r])
                .sum()
                .item()
                for r in ranks
            )
        elif typ is R:
            grad_sum = sum(grad._local_tensors[r] for r in ranks)
            return (grad_sum * perturbation._local_tensors[0]).sum().item()
        elif typ is I:
            return (
                (grad._local_tensors[0] * perturbation._local_tensors[0])
                .sum()
                .item()
            )
        elif typ is P:
            pert_sum = sum(perturbation._local_tensors[r] for r in ranks)
            return (grad._local_tensors[0] * pert_sum).sum().item()

    def _adjoint_check(self, fn, x, axis, src_type, dst_type):
        """Verify <A*(g), dx>_src = <g, J·dx>_dst where J is the Jacobian.

        For affine maps f(x) = Ax + b, the Jacobian-vector product is
        J·dx = f(dx) - f(0), which strips the constant term.
        """
        x.requires_grad_(True)
        y = fn(x)

        g = self._make_random_typed(y, self._gradient_type(dst_type), axis)
        y.backward(g)
        grad_x = x.grad

        dx = self._make_random_typed(x, src_type, axis)
        with torch.no_grad():
            Adx = fn(dx)
            # Subtract fn(0) to extract the linear (Jacobian) part.
            # For purely linear maps fn(0) = 0, so this is a no-op.
            zero = self.mode.rank_map(
                lambda r: torch.zeros_like(x._local_tensors[r])
            )
            assert_type(zero, {axis: src_type})
            A0 = fn(zero)
            Jdx = self.mode.rank_map(
                lambda r: Adx._local_tensors[r] - A0._local_tensors[r]
            )

        lhs = self._dual_inner(grad_x, dx, src_type)
        rhs = self._dual_inner(g, Jdx, dst_type)

        torch.testing.assert_close(
            torch.tensor(lhs),
            torch.tensor(rhs),
            msg=f"Adjoint identity failed: <A*g, dx>={lhs}, <g, Jdx>={rhs}",
        )

    # -- Documenting the current limitation --

    def test_grad_is_local_tensor_without_type(self):
        """Gradients are LocalTensors but lack SPMD type annotations.

        This documents the current limitation: autograd's C++ engine
        bypasses __torch_function__, so gradients have correct VALUES
        but no type annotations.
        """
        from torch.distributed._local_tensor import LocalTensor

        x = self._make_input((4, 3), "tp", S(0))
        w = self._make_input((3, 5), "tp", R)
        x.requires_grad_(True)
        y = torch.mm(x, w)
        g = self._make_random_typed(y, S(0), "tp")
        y.backward(g)

        self.assertIsInstance(x.grad, LocalTensor)
        self.assertFalse(has_local_type(x.grad))

    # -- Direct per-rank value check --

    def test_mm_s0_r_backward_values(self):
        """Direct per-rank check: grad_x = g @ w.T, grad_w = x.T @ g."""
        x = self._make_input((4, 3), "tp", S(0))
        w = self._make_input((3, 5), "tp", R)
        x.requires_grad_(True)
        w.requires_grad_(True)
        y = torch.mm(x, w)

        g = self._make_random_typed(y, S(0), "tp")
        y.backward(g)

        for r in range(self.WORLD_SIZE):
            expected_grad_x = g._local_tensors[r] @ w._local_tensors[r].t()
            torch.testing.assert_close(x.grad._local_tensors[r], expected_grad_x)

            expected_grad_w = x._local_tensors[r].t() @ g._local_tensors[r]
            torch.testing.assert_close(w.grad._local_tensors[r], expected_grad_w)

    # -- Adjoint identity: pointwise linear ops --

    def test_add_s0_s0_backward(self):
        """add(S(0), S(0)) -> S(0): backward wrt first S(0) input."""
        x = self._make_input((4, 3), "tp", S(0))
        bias = self._make_input((4, 3), "tp", R)
        bias_s = redistribute(bias, "tp", src=R, dst=S(0))
        self._adjoint_check(lambda x: x + bias_s, x, "tp", S(0), S(0))

    def test_sub_s0_s0_backward(self):
        x = self._make_input((4, 3), "tp", S(0))
        y_input = self._make_input((4, 3), "tp", S(0))
        self._adjoint_check(lambda x: x - y_input, x, "tp", S(0), S(0))

    def test_neg_s0_backward(self):
        x = self._make_input((4, 3), "tp", S(0))
        self._adjoint_check(lambda x: -x, x, "tp", S(0), S(0))

    def test_clone_s0_backward(self):
        x = self._make_input((4, 3), "tp", S(0))
        self._adjoint_check(lambda x: x.clone(), x, "tp", S(0), S(0))

    # -- Adjoint identity: mul (multilinear, one arg fixed) --

    def test_mul_s0_s0_backward_grad_x(self):
        """mul(S(0), S(0)): backward wrt first S(0) input."""
        x = self._make_input((4, 3), "tp", S(0))
        scale = self._make_input((4, 3), "tp", R)
        scale_s = redistribute(scale, "tp", src=R, dst=S(0))
        self._adjoint_check(lambda x: x * scale_s, x, "tp", S(0), S(0))

    def test_mul_s0_s0_backward_grad_scale(self):
        """mul(S(0), S(0)): backward wrt second S(0) input."""
        x = self._make_input((4, 3), "tp", S(0))
        scale = self._make_input((4, 3), "tp", R)
        scale_s = redistribute(scale, "tp", src=R, dst=S(0))
        self._adjoint_check(lambda s: x * s, scale_s, "tp", S(0), S(0))

    # -- Adjoint identity: mm S(0)@R -> S(0) --

    def test_mm_s0_r_backward_grad_x(self):
        x = self._make_input((4, 3), "tp", S(0))
        w = self._make_input((3, 5), "tp", R)
        self._adjoint_check(lambda x: torch.mm(x, w), x, "tp", S(0), S(0))

    def test_mm_s0_r_backward_grad_w(self):
        """mm(S(0), R): backward wrt R weight (gradient type is P)."""
        x = self._make_input((4, 3), "tp", S(0))
        w = self._make_input((3, 5), "tp", R)
        self._adjoint_check(lambda w: torch.mm(x, w), w, "tp", R, S(0))

    # -- Adjoint identity: mm R@S(1) -> S(1) --

    def test_mm_r_s1_backward_grad_x(self):
        """mm(R, S(1)): backward wrt R input (gradient type is P)."""
        x = self._make_input((4, 3), "tp", R)
        w = self._make_input((3, 5), "tp", S(1))
        self._adjoint_check(lambda x: torch.mm(x, w), x, "tp", R, S(1))

    def test_mm_r_s1_backward_grad_w(self):
        """mm(R, S(1)): backward wrt S(1) weight."""
        x = self._make_input((4, 3), "tp", R)
        w = self._make_input((3, 5), "tp", S(1))
        self._adjoint_check(lambda w: torch.mm(x, w), w, "tp", S(1), S(1))

    # -- Adjoint identity: mm S(1)@S(0) -> P (contracted dim) --

    def test_mm_contracted_backward_grad_x(self):
        """mm(S(1), S(0)) -> P: backward wrt S(1) input."""
        x = self._make_input((4, 3), "tp", S(1))
        w = self._make_input((3, 5), "tp", S(0))
        self._adjoint_check(lambda x: torch.mm(x, w), x, "tp", S(1), P)

    def test_mm_contracted_backward_grad_w(self):
        """mm(S(1), S(0)) -> P: backward wrt S(0) weight."""
        x = self._make_input((4, 3), "tp", S(1))
        w = self._make_input((3, 5), "tp", S(0))
        self._adjoint_check(lambda w: torch.mm(x, w), w, "tp", S(0), P)

    # -- Backward through local_map (local/global SPMD mixture) --

    def test_local_map_mm_backward_grad_x(self):
        """local_map wrapping mm: backward wrt S(0) input."""
        x = self._make_input((4, 3), "dp", S(0))
        w = self._make_input((3, 5), "dp", R)

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_mm(x, w):
            return x @ w

        self._adjoint_check(lambda x: local_mm(x, w), x, "dp", S(0), S(0))

    def test_local_map_mm_backward_grad_w(self):
        """local_map wrapping mm: backward wrt R weight (gradient is P)."""
        x = self._make_input((4, 3), "dp", S(0))
        w = self._make_input((3, 5), "dp", R)

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_mm(x, w):
            return x @ w

        self._adjoint_check(lambda w: local_mm(x, w), w, "dp", R, S(0))

    def test_global_add_then_local_map_mm_backward(self):
        """global add(S(0), S(0)) -> local_map mm -> S(0): backward wrt x."""
        x = self._make_input((4, 3), "dp", S(0))
        bias = self._make_input((4, 3), "dp", R)
        bias_s = redistribute(bias, "dp", src=R, dst=S(0))
        w = self._make_input((3, 5), "dp", R)

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_mm(x, w):
            return x @ w

        self._adjoint_check(
            lambda x: local_mm(x + bias_s, w), x, "dp", S(0), S(0)
        )

    def test_local_map_mm_then_global_mul_backward(self):
        """local_map mm -> global mul(S(0), S(0)) -> S(0): backward wrt x."""
        x = self._make_input((4, 3), "dp", S(0))
        w = self._make_input((3, 5), "dp", R)
        scale = self._make_input((4, 5), "dp", R)
        scale_s = redistribute(scale, "dp", src=R, dst=S(0))

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_mm(x, w):
            return x @ w

        self._adjoint_check(
            lambda x: local_mm(x, w) * scale_s, x, "dp", S(0), S(0)
        )

    def test_local_global_local_chain_backward(self):
        """local_mm -> global add -> local_mm: backward through full chain.

        Verifies gradient flows correctly across two local_map boundaries
        with a global SPMD region in between.
        """
        x = self._make_input((4, 3), "dp", S(0))
        w1 = self._make_input((3, 5), "dp", R)
        bias = self._make_input((4, 5), "dp", R)
        bias_s = redistribute(bias, "dp", src=R, dst=S(0))
        w2 = self._make_input((5, 2), "dp", R)

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_mm(x, w):
            return x @ w

        def fn(x):
            h = local_mm(x, w1)
            h = h + bias_s
            return local_mm(h, w2)

        self._adjoint_check(fn, x, "dp", S(0), S(0))

    def test_global_mm_chain_backward(self):
        """mm -> add -> mm chain entirely in global SPMD: backward wrt x."""
        x = self._make_input((4, 3), "tp", S(0))
        w1 = self._make_input((3, 5), "tp", R)
        bias = self._make_input((4, 5), "tp", R)
        bias_s = redistribute(bias, "tp", src=R, dst=S(0))
        w2 = self._make_input((5, 2), "tp", R)

        def fn(x):
            h = torch.mm(x, w1)
            h = h + bias_s
            return torch.mm(h, w2)

        self._adjoint_check(fn, x, "tp", S(0), S(0))

    def test_local_map_nonlinear_backward_runs(self):
        """local_map with nonlinear ops: verify backward runs successfully.

        The adjoint identity only holds for linear maps, so for nonlinear
        ops (relu) we just verify backward produces a gradient.
        """
        x = self._make_input((4, 3), "tp", S(0))
        w1 = self._make_input((3, 5), "tp", R)
        w2 = self._make_input((5, 2), "tp", R)

        @local_map(
            in_type=[{"tp": S(0)}, {"tp": R}, {"tp": R}],
            out_type={"tp": S(0)},
        )
        def local_mlp(x, w1, w2):
            h = x @ w1
            h = torch.relu(h)
            return h @ w2

        x.requires_grad_(True)
        y = local_mlp(x, w1, w2)
        g = self._make_random_typed(y, S(0), "tp")
        y.backward(g)
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    run_tests()
