"""
Tests for SPMD type hierarchy (R, I, V, P, S) and PartitionSpec.

Covers: types.py, _mesh.py (mesh setup).
"""

import unittest

from sixlib.spmd_types import (
    get_mesh,
    I,
    Invariant,
    P,
    Partial,
    PartitionSpec,
    PerMeshAxisLocalSpmdType,
    R,
    Replicate,
    S,
    set_mesh,
    V,
    Varying,
)


class TestTypeEnum(unittest.TestCase):
    """Test that R, I, V, P are enum members and aliases work."""

    def test_enum_identity(self):
        """R, I, V, P should be PerMeshAxisLocalSpmdType enum members."""
        self.assertIs(R, PerMeshAxisLocalSpmdType.R)
        self.assertIs(I, PerMeshAxisLocalSpmdType.I)
        self.assertIs(V, PerMeshAxisLocalSpmdType.V)
        self.assertIs(P, PerMeshAxisLocalSpmdType.P)

    def test_isinstance(self):
        """Enum members should be instances of PerMeshAxisLocalSpmdType."""
        self.assertIsInstance(R, PerMeshAxisLocalSpmdType)
        self.assertIsInstance(I, PerMeshAxisLocalSpmdType)
        self.assertIsInstance(V, PerMeshAxisLocalSpmdType)
        self.assertIsInstance(P, PerMeshAxisLocalSpmdType)

    def test_backward_compat_aliases(self):
        """Backward compat aliases should be the same enum members."""
        self.assertIs(Replicate, R)
        self.assertIs(Invariant, I)
        self.assertIs(Varying, V)
        self.assertIs(Partial, P)

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
        self.assertEqual(ps, PartitionSpec())

    def test_single_axis_partition_spec(self):
        """Partition spec with single mesh axis."""
        ps = PartitionSpec("tp", None)
        self.assertEqual(ps, PartitionSpec("tp", None))

    def test_multi_axis_partition_spec(self):
        """Partition spec with multiple mesh axes on same dim."""
        ps = PartitionSpec(("dp", "tp"), None)
        self.assertEqual(ps, PartitionSpec(("dp", "tp"), None))

    def test_partition_spec_iteration(self):
        """Partition spec should be iterable."""
        ps = PartitionSpec("tp", "dp", None)
        axes = list(ps)
        self.assertEqual(axes, ["tp", "dp", None])

    def test_partition_spec_repr(self):
        """Test string representation of PartitionSpec."""
        ps = PartitionSpec()
        self.assertEqual(repr(ps), "PartitionSpec()")

        ps = PartitionSpec("tp", None)
        self.assertEqual(repr(ps), "PartitionSpec('tp', None)")

        ps = PartitionSpec(("dp", "tp"), "ep")
        self.assertEqual(repr(ps), "PartitionSpec(('dp', 'tp'), 'ep')")


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


if __name__ == "__main__":
    unittest.main()
