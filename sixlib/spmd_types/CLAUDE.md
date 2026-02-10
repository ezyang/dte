# SPMD Types

A type system for distributed (SPMD) tensor computations in PyTorch. Read the design specification in `DESIGN.md` before making changes.

## File Structure

| File | Purpose |
|------|---------|
| `types.py` | Type hierarchy (`R`, `I`, `V`, `P`, `S`), `PartitionSpec` |
| `DESIGN.md` | Design specification for the type system |
| `_checker.py` | Type tracking on tensors, type inference rules |
| `_collectives.py` | Collective operations |
| `api_test.py` | Test suite using `LocalTensorMode` for single-process distributed testing |

## Testing

```bash
pytest -x -s sixlib/spmd_types/api_test.py
```

Tests use `LocalTensorMode` with a `FakeStore` to simulate multiple ranks in a single process — no GPU or distributed backend needed.
