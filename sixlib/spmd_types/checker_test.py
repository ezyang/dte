"""
Tests for SPMD type checker: type inference, strict mode, error messages.

Covers: _checker.py
"""

import unittest

import expecttest
import torch
from sixlib.spmd_types import (
    I,
    P,
    R,
    S,
    V,
)
from sixlib.spmd_types._checker import (
    assert_type,
    get_axis_local_type,
    has_local_type,
    infer_local_type_for_axis,
    OpLinearity,
    SpmdTypeMode,
)
from sixlib.spmd_types._collectives import all_reduce
from sixlib.spmd_types._test_utils import LocalTensorTestCase
from sixlib.spmd_types.types import SpmdTypeError
from torch.distributed._local_tensor import LocalTensorMode


class TestEinsumTypePropagation(unittest.TestCase):
    """Test einsum type propagation rules.

    TODO: These tests depend on LTensor which was removed. They need to be
    reimplemented once the type-checking layer is rebuilt.
    """

    pass


class TestEinsumSingleOperand(unittest.TestCase):
    """Test einsum with single operand (unary operations).

    TODO: These tests depend on LTensor which was removed. They need to be
    reimplemented once the type-checking layer is rebuilt.
    """

    pass


class TestLinearTypePropagation(LocalTensorTestCase):
    """Test type propagation through F.linear (matmul + optional bias)."""

    def test_linear_r_v_gives_v(self):
        """F.linear with R weight and V input should give V output, not R.

        Regression: V is canonicalized to {} so its type was silently dropped
        during per-axis type collection, making the output R instead of V.
        """
        inp = self._generate_inputs((2, 4), "tp", V)
        weight = self._generate_inputs((3, 4), "tp", R)
        result = torch.nn.functional.linear(inp, weight)
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_matmul_r_v_gives_v(self):
        """torch.matmul with R and V should give V output."""
        x = self._generate_inputs((2, 4), "tp", R)
        y = self._generate_inputs((4, 3), "tp", V)
        result = torch.matmul(x, y)
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_add_r_v_gives_v(self):
        """torch.add with R and V should give V output."""
        x = self._generate_inputs((4,), "tp", R)
        y = self._generate_inputs((4,), "tp", V)
        result = torch.add(x, y)
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_add_p_p_gives_p(self):
        """torch.add with P and P should give P (addition is linear)."""
        x = self._generate_inputs((4,), "tp", P)
        y = self._generate_inputs((4,), "tp", P)
        result = torch.add(x, y)
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_sub_p_p_gives_p(self):
        """torch.sub with P and P should give P (subtraction is linear)."""
        x = self._generate_inputs((4,), "tp", P)
        y = self._generate_inputs((4,), "tp", P)
        result = torch.sub(x, y)
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_matmul_p_rejected(self):
        """torch.matmul with all-P is rejected (matmul is not linear in this sense)."""
        x = self._generate_inputs((2, 4), "tp", P)
        y = self._generate_inputs((4, 3), "tp", P)
        with self.assertRaises(SpmdTypeError):
            torch.matmul(x, y)

    def test_mul_p_r_gives_p(self):
        """torch.mul with P and R should give P (multilinear: P in one factor, R in other)."""
        x = self._generate_inputs((4,), "tp", P)
        y = self._generate_inputs((4,), "tp", R)
        result = torch.mul(x, y)
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_cat_p_p_gives_p(self):
        """torch.cat with P and P should give P (cat is linear)."""
        x = self._generate_inputs((4,), "tp", P)
        y = self._generate_inputs((4,), "tp", P)
        result = torch.cat([x, y])
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_matmul_p_r_gives_p(self):
        """torch.matmul with P and R should give P (multilinear: P in one factor)."""
        x = self._generate_inputs((2, 4), "tp", P)
        y = self._generate_inputs((4, 3), "tp", R)
        result = torch.matmul(x, y)
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_add_p_r_rejected(self):
        """torch.add with mixed P and R is still rejected."""
        x = self._generate_inputs((4,), "tp", P)
        y = self._generate_inputs((4,), "tp", R)
        with self.assertRaises(SpmdTypeError):
            torch.add(x, y)

    def test_add_p_scalar_rejected(self):
        """torch.add(P, scalar) is affine, not linear -- must be rejected."""
        x = self._generate_inputs((4,), "tp", P)
        with self.assertRaises(SpmdTypeError):
            torch.add(x, 1.0)

    def test_sub_p_scalar_rejected(self):
        """torch.sub(P, scalar) is affine -- must be rejected."""
        x = self._generate_inputs((4,), "tp", P)
        with self.assertRaises(SpmdTypeError):
            torch.sub(x, 1.0)

    def test_add_p_int_scalar_rejected(self):
        """torch.add(P, int_scalar) is also affine -- must be rejected."""
        x = self._generate_inputs((4,), "tp", P)
        with self.assertRaises(SpmdTypeError):
            torch.add(x, 1)

    def test_cat_p_with_dim_arg_allowed(self):
        """torch.cat([P, P], 0) -- dim is not a tensor input, should be allowed."""
        x = self._generate_inputs((4,), "tp", P)
        y = self._generate_inputs((4,), "tp", P)
        result = torch.cat([x, y], 0)
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_mul_p_scalar_allowed(self):
        """torch.mul(P, scalar) -- multilinear, scalar ignored, P propagates."""
        x = self._generate_inputs((4,), "tp", P)
        result = torch.mul(x, 2.0)
        self.assertIs(get_axis_local_type(result, "tp"), P)


class TestStrictMode(LocalTensorTestCase):
    """Test SpmdTypeMode strict mode which errors on unannotated tensors."""

    def setUp(self):
        """Enter LocalTensorMode and strict SpmdTypeMode for each test."""
        self.mode = LocalTensorMode(self.WORLD_SIZE)
        self.mode.__enter__()
        self.spmd_mode = SpmdTypeMode(strict=True)
        self.spmd_mode.__enter__()

    def test_strict_mixed_annotated_unannotated_fails(self):
        """Strict mode raises when one operand is annotated and the other is not."""
        x = self._generate_inputs((4,), "tp", R)
        y = self.mode.rank_map(lambda r: torch.randn(4))
        with self.assertRaises(SpmdTypeError):
            torch.add(x, y)

    def test_strict_mixed_cat_fails(self):
        """Strict mode catches mixed typed/untyped tensors inside lists (e.g. torch.cat)."""
        x = self._generate_inputs((4,), "tp", R)
        y = self.mode.rank_map(lambda r: torch.randn(4))
        with self.assertRaises(SpmdTypeError):
            torch.cat([x, y])

    def test_strict_all_annotated_passes(self):
        """Strict mode allows operations when all tensors are annotated."""
        x = self._generate_inputs((4,), "tp", R)
        y = self._generate_inputs((4,), "tp", R)
        result = torch.add(x, y)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_strict_all_v_annotated_mixed_with_unannotated_fails(self):
        """Strict mode catches all-V tensor (canonicalizes to {}) mixed with unannotated."""
        x = self._generate_inputs((4,), "tp", V)
        # x has _spmd_types attr (set to {} after canonicalization)
        y = self.mode.rank_map(lambda r: torch.randn(4))
        # y has no _spmd_types attr at all
        with self.assertRaises(SpmdTypeError):
            torch.add(x, y)

    def test_strict_all_unannotated_passes(self):
        """Strict mode allows operations when no tensors are annotated (no SPMD tracking)."""
        x = self.mode.rank_map(lambda r: torch.randn(4))
        y = self.mode.rank_map(lambda r: torch.randn(4))
        result = torch.add(x, y)  # Should not raise -- no typed tensors
        self.assertFalse(has_local_type(result))

    def test_strict_collective_typed_input_passes(self):
        """Strict mode allows collectives when the input tensor is typed."""
        x = self._generate_inputs((4,), "tp", P)
        result = all_reduce(x, "tp", src=P, dst=R)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_strict_collective_untyped_input_passes(self):
        """Strict mode allows collectives when no tensors are typed at all."""
        y = self.mode.rank_map(lambda r: torch.randn(4))
        result = all_reduce(y, "tp", src=P, dst=R)
        self.assertFalse(has_local_type(result))

    def test_nonstrict_mixed_passes(self):
        """Non-strict mode allows mixing annotated and unannotated tensors."""
        self.spmd_mode.__exit__(None, None, None)
        nonstrict = SpmdTypeMode(strict=False)
        nonstrict.__enter__()
        try:
            x = self._generate_inputs((4,), "tp", R)
            y = self.mode.rank_map(lambda r: torch.randn(4))
            result = torch.add(x, y)  # Should not raise
            self.assertIs(get_axis_local_type(result, "tp"), R)
        finally:
            nonstrict.__exit__(None, None, None)
            self.spmd_mode.__enter__()


class TestTypeErrorMessages(expecttest.TestCase):
    """Test that type errors include actionable fix suggestions.

    Tests call infer_local_type_for_axis directly rather than going through
    torch ops, because V (Varying) is canonicalized away in the type tracker
    and cannot appear in collected types.
    """

    def test_general_I_R(self):
        """I + R: suggest reinterpret I->R (I->V filtered: changes output from R to V)."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [I, R])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis 'tp' cannot mix with other types. Found types: [I, R]
Are you missing a collective? e.g.,
  reinterpret(tensor, 'tp', src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)""",
        )

    def test_general_I_V(self):
        """I + V: suggest reinterpret I->R or I->V."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [I, V])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis 'tp' cannot mix with other types. Found types: [I, V]
Are you missing a collective? e.g.,
  reinterpret(tensor, 'tp', src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)
  reinterpret(tensor, 'tp', src=I, dst=V) on the Invariant operand (no-op forward, all-reduce in backward)""",
        )

    def test_general_I_P(self):
        """I + P: suggest convert I->P or all_reduce P->I."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [I, P])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis 'tp' cannot mix with other types. Found types: [I, P]
Are you missing a collective? e.g.,
  all_reduce(tensor, 'tp', src=P, dst=I) on the Partial operand (all-reduce in forward, no-op backward)""",
        )

    def test_general_P_R(self):
        """P + R: suggest all_reduce P->R or convert R->P."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [P, R])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis 'tp' can only combine with partial. Found types: [P, R]
Are you missing a collective? e.g.,
  all_reduce(tensor, 'tp', src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_general_P_V(self):
        """P + V: suggest all_reduce P->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [P, V])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis 'tp' can only combine with partial. Found types: [P, V]
Are you missing a collective? e.g.,
  all_reduce(tensor, 'tp', src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_general_R_P_V(self):
        """R + P + V: suggest all_reduce P->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [R, P, V])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis 'tp' can only combine with partial. Found types: [R, P, V]
Are you missing a collective? e.g.,
  all_reduce(tensor, 'tp', src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_general_I_R_V(self):
        """I + R + V: suggest reinterpret I->R or I->V."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [I, R, V])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis 'tp' cannot mix with other types. Found types: [I, R, V]
Are you missing a collective? e.g.,
  reinterpret(tensor, 'tp', src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)
  reinterpret(tensor, 'tp', src=I, dst=V) on the Invariant operand (no-op forward, all-reduce in backward)""",
        )

    def test_mul_P_P(self):
        """P * P: suggest all_reduce P->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [P, P], linearity=OpLinearity.MULTILINEAR)
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial in multiple factors of multilinear op on axis 'tp' is forbidden. Found types: [P, P]
Are you missing a collective? e.g.,
  all_reduce(tensor, 'tp', src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_mul_P_V(self):
        """P * V: suggest all_reduce P->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [P, V], linearity=OpLinearity.MULTILINEAR)
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis 'tp' can only multiply with Replicate. Found types: [P, V]
Are you missing a collective? e.g.,
  all_reduce(tensor, 'tp', src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_mul_P_I(self):
        """P * I: suggest reinterpret I->R or all_reduce P->I."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [P, I], linearity=OpLinearity.MULTILINEAR)
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis 'tp' cannot mix with other types. Found types: [P, I]
Are you missing a collective? e.g.,
  reinterpret(tensor, 'tp', src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)
  all_reduce(tensor, 'tp', src=P, dst=I) on the Partial operand (all-reduce in forward, no-op backward)""",
        )


class TestOpLinearity(unittest.TestCase):
    """Test the OpLinearity parameter on infer_local_type_for_axis."""

    def test_all_p_linear_allowed(self):
        """All-P with LINEAR should return P."""
        result = infer_local_type_for_axis("tp", [P, P], linearity=OpLinearity.LINEAR)
        self.assertIs(result, P)

    def test_all_p_nonlinear_rejected(self):
        """All-P with NONLINEAR (default) should raise."""
        with self.assertRaises(SpmdTypeError):
            infer_local_type_for_axis("tp", [P, P])

    def test_mixed_p_r_rejected_even_linear(self):
        """Mixed P+R is rejected even with LINEAR."""
        with self.assertRaises(SpmdTypeError):
            infer_local_type_for_axis("tp", [P, R], linearity=OpLinearity.LINEAR)

    def test_all_r_unaffected_by_linearity(self):
        """All-R works regardless of linearity."""
        self.assertIs(infer_local_type_for_axis("tp", [R, R]), R)
        self.assertIs(
            infer_local_type_for_axis("tp", [R, R], linearity=OpLinearity.LINEAR), R
        )

    def test_multilinear_p_r_gives_p(self):
        """MULTILINEAR with P and R should return P."""
        result = infer_local_type_for_axis(
            "tp", [P, R], linearity=OpLinearity.MULTILINEAR
        )
        self.assertIs(result, P)

    def test_multilinear_p_p_rejected(self):
        """MULTILINEAR with P*P should raise."""
        with self.assertRaises(SpmdTypeError):
            infer_local_type_for_axis("tp", [P, P], linearity=OpLinearity.MULTILINEAR)

    def test_multilinear_p_v_rejected(self):
        """MULTILINEAR with P and V should raise."""
        with self.assertRaises(SpmdTypeError):
            infer_local_type_for_axis("tp", [P, V], linearity=OpLinearity.MULTILINEAR)


class TestAssertTypeShardSugar(unittest.TestCase):
    """Test assert_type with S(i) sugar and PartitionSpec."""

    def test_shard_sugar_sets_varying(self):
        """S(i) in types should set the axis to V in local types."""
        x = torch.randn(4)
        assert_type(x, {"tp": S(0)})
        self.assertIs(get_axis_local_type(x, "tp"), V)

    def test_shard_sugar_duplicate_dim_rejected(self):
        """S(i) entries mapping multiple axes to the same dim should be rejected."""
        x = torch.randn(4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {"dp": S(0), "tp": S(0)})

    def test_shard_sugar_negative_dim(self):
        """S(-1) should resolve to the last dimension."""
        x = torch.randn(3, 4)
        assert_type(x, {"tp": S(-1)})
        self.assertIs(get_axis_local_type(x, "tp"), V)

    def test_shard_sugar_with_partition_spec_rejected(self):
        """Cannot mix S(i) in types with an explicit partition_spec."""
        from sixlib.spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        with self.assertRaises(TypeError):
            assert_type(x, {"tp": S(1)}, PartitionSpec(None, "tp"))

    def test_partition_spec_sets_varying(self):
        """Axes in partition_spec should be treated as Varying."""
        from sixlib.spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        assert_type(x, {}, PartitionSpec(None, "tp"))
        self.assertIs(get_axis_local_type(x, "tp"), V)

    def test_partition_spec_wrong_length_rejected(self):
        """PartitionSpec length must match tensor ndim."""
        from sixlib.spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {}, PartitionSpec("tp"))

    def test_partition_spec_conflicts_with_non_varying_rejected(self):
        """Axis in partition_spec that is R/I/P in types should be rejected."""
        from sixlib.spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {"tp": R}, PartitionSpec(None, "tp"))

    def test_partition_spec_with_explicit_v_ok(self):
        """Explicit V in types for an axis in partition_spec should be allowed."""
        from sixlib.spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        assert_type(x, {"dp": V}, PartitionSpec(None, None))
        self.assertIs(get_axis_local_type(x, "dp"), V)

    def test_mixed_shard_and_local_types(self):
        """S(i) for one axis and R/I/P for another should work."""
        x = torch.randn(3, 4)
        assert_type(x, {"dp": S(0), "tp": I})
        self.assertIs(get_axis_local_type(x, "dp"), V)
        self.assertIs(get_axis_local_type(x, "tp"), I)

    def test_partition_spec_with_non_varying_types(self):
        """partition_spec with R/I axes in types should work if they don't overlap."""
        from sixlib.spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        assert_type(x, {"tp": I}, PartitionSpec("dp", None))
        self.assertIs(get_axis_local_type(x, "tp"), I)
        self.assertIs(get_axis_local_type(x, "dp"), V)

    def test_shard_sugar_out_of_bounds_positive(self):
        """S(i) with i >= ndim should be rejected."""
        x = torch.randn(3, 4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {"tp": S(2)})

    def test_shard_sugar_out_of_bounds_negative(self):
        """S(i) with i < -ndim should be rejected."""
        x = torch.randn(3, 4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {"tp": S(-3)})

    def test_shard_sugar_0d_tensor_rejected(self):
        """S(i) on a 0-d tensor should be rejected."""
        x = torch.tensor(1.0)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {"tp": S(0)})


class TestOpLinearityRegistry(LocalTensorTestCase):
    """Test that the expanded _OP_REGISTRY covers aliases, dunders, and structural ops."""

    def test_clone_functional_form(self):
        """torch.clone(P) should give P (functional form, not just Tensor.clone)."""
        x = self._generate_inputs((4,), "tp", P)
        result = torch.clone(x)
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_concat_aliases(self):
        """torch.concat and torch.concatenate should propagate P like torch.cat."""
        x = self._generate_inputs((4,), "tp", P)
        y = self._generate_inputs((4,), "tp", P)
        for fn in [torch.concat, torch.concatenate]:
            result = fn([x, y])
            self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_structural_ops_preserve_p(self):
        """Structural ops (reshape, view, transpose, etc.) should preserve P."""
        x = self._generate_inputs((2, 3), "tp", P)
        # NB: contiguous is registered but untestable here -- LocalTensorMode
        # decomposes it to per-rank calls that bypass SpmdTypeMode.
        ops = [
            ("reshape", lambda t: torch.reshape(t, (6,))),
            ("Tensor.view", lambda t: t.view(6)),
            ("transpose", lambda t: torch.transpose(t, 0, 1)),
            ("squeeze", lambda t: torch.squeeze(t.unsqueeze(0))),
            ("unsqueeze", lambda t: torch.unsqueeze(t, 0)),
            ("flatten", lambda t: torch.flatten(t)),
            ("expand", lambda t: t.unsqueeze(0).expand(2, 2, 3)),
            ("permute", lambda t: torch.permute(t, (1, 0))),
        ]
        for name, op in ops:
            result = op(x)
            self.assertIs(
                get_axis_local_type(result, "tp"),
                P,
                f"{name} did not preserve P type",
            )

    def test_neg_preserves_p(self):
        """torch.neg(P) and -P (unary neg operator) should both give P."""
        x = self._generate_inputs((4,), "tp", P)
        result_func = torch.neg(x)
        self.assertIs(get_axis_local_type(result_func, "tp"), P)
        result_op = -x
        self.assertIs(get_axis_local_type(result_op, "tp"), P)

    def test_operator_add_p_p(self):
        """P + P via the + operator should give P."""
        x = self._generate_inputs((4,), "tp", P)
        y = self._generate_inputs((4,), "tp", P)
        result = x + y
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_split_chunk_unbind_preserve_p(self):
        """Multi-output ops (split, chunk, unbind) should propagate P."""
        x = self._generate_inputs((6,), "tp", P)
        for name, fn in [
            ("split", lambda t: torch.split(t, 3)),
            ("chunk", lambda t: torch.chunk(t, 2)),
            ("unbind", lambda t: torch.unbind(t.view(2, 3))),
        ]:
            results = fn(x)
            for i, r in enumerate(results):
                self.assertIs(
                    get_axis_local_type(r, "tp"),
                    P,
                    f"{name}[{i}] did not preserve P type",
                )

    def test_inplace_add_preserves_p(self):
        """P.add_(P) should give P."""
        x = self._generate_inputs((4,), "tp", P)
        y = self._generate_inputs((4,), "tp", P)
        x.add_(y)
        self.assertIs(get_axis_local_type(x, "tp"), P)

    def test_div_p_r_gives_p(self):
        """torch.div(P, R) should give P (linear in numerator when denominator is fixed)."""
        x = self._generate_inputs((4,), "tp", P)
        y = self._generate_inputs((4,), "tp", R)
        result = torch.div(x, y)
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_div_r_p_rejected(self):
        """torch.div(R, P) should raise (P in denominator is not linear)."""
        x = self._generate_inputs((4,), "tp", R)
        y = self._generate_inputs((4,), "tp", P)
        with self.assertRaises(SpmdTypeError):
            torch.div(x, y)

    def test_div_p_p_rejected(self):
        """torch.div(P, P) should raise (P in denominator is not linear)."""
        x = self._generate_inputs((4,), "tp", P)
        y = self._generate_inputs((4,), "tp", P)
        with self.assertRaises(SpmdTypeError):
            torch.div(x, y)

    def test_div_r_v_gives_v(self):
        """torch.div(R, V) should give V (normal non-P path, no fixed_args filtering)."""
        x = self._generate_inputs((4,), "tp", R)
        y = self._generate_inputs((4,), "tp", V)
        result = torch.div(x, y)
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_all_registered_ops_are_callable(self):
        """Every key in _OP_REGISTRY should be a callable (no stale references)."""
        from sixlib.spmd_types._checker import _OP_REGISTRY

        for func in _OP_REGISTRY:
            self.assertTrue(
                callable(func),
                f"Registry entry {func!r} is not callable",
            )

    def test_all_decomp_rules_are_callable(self):
        """Every key in _DECOMP_TYPE_RULES should be a callable."""
        from sixlib.spmd_types._checker import _DECOMP_TYPE_RULES

        for func in _DECOMP_TYPE_RULES:
            self.assertTrue(
                callable(func),
                f"Decomp rule entry {func!r} is not callable",
            )


class TestAddmmTypeDecomposition(LocalTensorTestCase):
    """Test type-level decomposition for addmm and related ops.

    addmm(self, mat1, mat2) = self + mm(mat1, mat2) decomposes into a sum
    (LINEAR) of self and a matmul (MULTILINEAR) of mat1 and mat2. This catches
    bugs where e.g. addmm(P, R, R) was incorrectly accepted as P.
    """

    # --- addmm ---

    def test_addmm_P_R_R_rejected(self):
        """addmm(P, R, R) must error: R@R=R, then P+R is invalid."""
        self_t = self._generate_inputs((2, 3), "tp", P)
        mat1 = self._generate_inputs((2, 4), "tp", R)
        mat2 = self._generate_inputs((4, 3), "tp", R)
        with self.assertRaises(SpmdTypeError):
            torch.addmm(self_t, mat1, mat2)

    def test_addmm_R_P_R_rejected(self):
        """addmm(R, P, R) must error: P@R=P, then P+R is invalid."""
        self_t = self._generate_inputs((2, 3), "tp", R)
        mat1 = self._generate_inputs((2, 4), "tp", P)
        mat2 = self._generate_inputs((4, 3), "tp", R)
        with self.assertRaises(SpmdTypeError):
            torch.addmm(self_t, mat1, mat2)

    def test_addmm_R_R_P_rejected(self):
        """addmm(R, R, P) must error: R@P=P, then P+R is invalid."""
        self_t = self._generate_inputs((2, 3), "tp", R)
        mat1 = self._generate_inputs((2, 4), "tp", R)
        mat2 = self._generate_inputs((4, 3), "tp", P)
        with self.assertRaises(SpmdTypeError):
            torch.addmm(self_t, mat1, mat2)

    def test_addmm_P_P_R_gives_P(self):
        """addmm(P, P, R): mm(P,R)=P, then P+P=P."""
        self_t = self._generate_inputs((2, 3), "tp", P)
        mat1 = self._generate_inputs((2, 4), "tp", P)
        mat2 = self._generate_inputs((4, 3), "tp", R)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_addmm_P_R_P_gives_P(self):
        """addmm(P, R, P): mm(R,P)=P, then P+P=P."""
        self_t = self._generate_inputs((2, 3), "tp", P)
        mat1 = self._generate_inputs((2, 4), "tp", R)
        mat2 = self._generate_inputs((4, 3), "tp", P)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_addmm_R_R_R_gives_R(self):
        """addmm(R, R, R) -> R (no change from before)."""
        self_t = self._generate_inputs((2, 3), "tp", R)
        mat1 = self._generate_inputs((2, 4), "tp", R)
        mat2 = self._generate_inputs((4, 3), "tp", R)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    def test_addmm_V_R_R_gives_V(self):
        """addmm(V, R, R): mm(R,R)=R, then V+R=V."""
        self_t = self._generate_inputs((2, 3), "tp", V)
        mat1 = self._generate_inputs((2, 4), "tp", R)
        mat2 = self._generate_inputs((4, 3), "tp", R)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, "tp"), V)

    def test_addmm_R_P_P_rejected(self):
        """addmm(R, P, P): mm(P,P) is invalid (two P factors in multilinear)."""
        self_t = self._generate_inputs((2, 3), "tp", R)
        mat1 = self._generate_inputs((2, 4), "tp", P)
        mat2 = self._generate_inputs((4, 3), "tp", P)
        with self.assertRaises(SpmdTypeError):
            torch.addmm(self_t, mat1, mat2)

    def test_addmm_V_V_V_gives_V(self):
        """addmm(V, V, V): mm(V,V)=V, then V+V=V."""
        self_t = self._generate_inputs((2, 3), "tp", V)
        mat1 = self._generate_inputs((2, 4), "tp", V)
        mat2 = self._generate_inputs((4, 3), "tp", V)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, "tp"), V)

    # --- addmv ---

    def test_addmv_P_R_R_rejected(self):
        """addmv(P, R, R) must error."""
        self_t = self._generate_inputs((3,), "tp", P)
        mat = self._generate_inputs((3, 4), "tp", R)
        vec = self._generate_inputs((4,), "tp", R)
        with self.assertRaises(SpmdTypeError):
            torch.addmv(self_t, mat, vec)

    def test_addmv_P_P_R_gives_P(self):
        """addmv(P, P, R): mv(P,R)=P, then P+P=P."""
        self_t = self._generate_inputs((3,), "tp", P)
        mat = self._generate_inputs((3, 4), "tp", P)
        vec = self._generate_inputs((4,), "tp", R)
        result = torch.addmv(self_t, mat, vec)
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_addmv_R_R_R_gives_R(self):
        """addmv(R, R, R) -> R."""
        self_t = self._generate_inputs((3,), "tp", R)
        mat = self._generate_inputs((3, 4), "tp", R)
        vec = self._generate_inputs((4,), "tp", R)
        result = torch.addmv(self_t, mat, vec)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    # --- baddbmm ---

    def test_baddbmm_P_R_R_rejected(self):
        """baddbmm(P, R, R) must error."""
        self_t = self._generate_inputs((2, 3, 5), "tp", P)
        batch1 = self._generate_inputs((2, 3, 4), "tp", R)
        batch2 = self._generate_inputs((2, 4, 5), "tp", R)
        with self.assertRaises(SpmdTypeError):
            torch.baddbmm(self_t, batch1, batch2)

    def test_baddbmm_P_P_R_gives_P(self):
        """baddbmm(P, P, R): bmm(P,R)=P, then P+P=P."""
        self_t = self._generate_inputs((2, 3, 5), "tp", P)
        batch1 = self._generate_inputs((2, 3, 4), "tp", P)
        batch2 = self._generate_inputs((2, 4, 5), "tp", R)
        result = torch.baddbmm(self_t, batch1, batch2)
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_baddbmm_R_R_R_gives_R(self):
        """baddbmm(R, R, R) -> R."""
        self_t = self._generate_inputs((2, 3, 5), "tp", R)
        batch1 = self._generate_inputs((2, 3, 4), "tp", R)
        batch2 = self._generate_inputs((2, 4, 5), "tp", R)
        result = torch.baddbmm(self_t, batch1, batch2)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    # --- addr ---

    def test_addr_P_R_R_rejected(self):
        """addr(P, R, R) must error."""
        self_t = self._generate_inputs((3, 4), "tp", P)
        vec1 = self._generate_inputs((3,), "tp", R)
        vec2 = self._generate_inputs((4,), "tp", R)
        with self.assertRaises(SpmdTypeError):
            torch.addr(self_t, vec1, vec2)

    def test_addr_P_P_R_gives_P(self):
        """addr(P, P, R): outer(P,R)=P, then P+P=P."""
        self_t = self._generate_inputs((3, 4), "tp", P)
        vec1 = self._generate_inputs((3,), "tp", P)
        vec2 = self._generate_inputs((4,), "tp", R)
        result = torch.addr(self_t, vec1, vec2)
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_addr_R_R_R_gives_R(self):
        """addr(R, R, R) -> R."""
        self_t = self._generate_inputs((3, 4), "tp", R)
        vec1 = self._generate_inputs((3,), "tp", R)
        vec2 = self._generate_inputs((4,), "tp", R)
        result = torch.addr(self_t, vec1, vec2)
        self.assertIs(get_axis_local_type(result, "tp"), R)

    # --- sum/mean (LINEAR reductions) ---

    def test_sum_P_gives_P(self):
        """torch.sum(P) -> P (sum is linear)."""
        x = self._generate_inputs((2, 3), "tp", P)
        result = torch.sum(x)
        self.assertIs(get_axis_local_type(result, "tp"), P)

    def test_mean_P_gives_P(self):
        """torch.mean(P) -> P (mean is linear)."""
        # randn already produces float32 so torch.mean works directly
        x = self._generate_inputs((2, 3), "tp", P)
        result = torch.mean(x)
        self.assertIs(get_axis_local_type(result, "tp"), P)


if __name__ == "__main__":
    unittest.main()
