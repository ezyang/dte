"""Local SPMD type coercion operations: reinterpret and convert."""

import torch
from sixlib.spmd_types import _dist
from sixlib.spmd_types._mesh import _get_mesh_axis_group
from sixlib.spmd_types.types import (
    I,
    P,
    PerMeshAxisSpmdType,
    R,
    Shard,
    V,
)
from torch.distributed import ProcessGroup
from torch.distributed._local_tensor import local_tensor_mode, LocalTensor
from torch.overrides import handle_torch_function, has_torch_function_unary

# =============================================================================
# reinterpret autograd Functions
# =============================================================================


class _ReplicateToVarying(torch.autograd.Function):
    """reinterpret(R,V): R -> V, backward is reinterpret(V,P): V -> P (no-op)."""

    @staticmethod
    def forward(ctx, x, axis):
        ctx.axis = axis
        return x

    @staticmethod
    def backward(ctx, grad_out):
        # reinterpret(V,P) is a no-op in forward direction
        return grad_out, None


class _ReplicateToInvariant(torch.autograd.Function):
    """reinterpret(R,I): R -> I, backward is convert(I,P): I -> P."""

    @staticmethod
    def forward(ctx, x, axis):
        ctx.axis = axis
        return x  # no-op in forward

    @staticmethod
    def backward(ctx, grad_out):
        # backward is convert(I,P): I -> P
        # Zero out all but rank 0
        pg = _get_mesh_axis_group(ctx.axis)

        mode = local_tensor_mode()
        if mode is not None and isinstance(grad_out, LocalTensor):
            return mode.tensor_map(
                grad_out, lambda r, t: _replicate_to_partial_fwd(t, r)
            ), None
        else:
            rank = _dist.dist.get_rank(pg)
            return _replicate_to_partial_fwd(grad_out, rank), None


class _InvariantToReplicate(torch.autograd.Function):
    """reinterpret(I,R): I -> R, backward is all_reduce(I): P -> I."""

    @staticmethod
    def forward(ctx, x, axis):
        ctx.axis = axis
        return x  # no-op in forward

    @staticmethod
    def backward(ctx, grad_out):
        # backward is all_reduce(I): P -> I
        from sixlib.spmd_types._collectives import all_reduce

        return all_reduce(grad_out, ctx.axis, src=P, dst=I), None


class _VaryingToPartial(torch.autograd.Function):
    """reinterpret(V,P): V -> P, backward is reinterpret(R,V): R -> V (no-op)."""

    @staticmethod
    def forward(ctx, x, axis):
        ctx.axis = axis
        return x  # no-op in forward

    @staticmethod
    def backward(ctx, grad_out):
        # backward is reinterpret(R,V): R -> V (no-op)
        return grad_out, None


class _ReplicateToPartial(torch.autograd.Function):
    """reinterpret(R,P): R -> P, backward is reinterpret(R,P): R -> P."""

    @staticmethod
    def forward(ctx, x, axis):
        ctx.axis = axis
        return x  # no-op in forward

    @staticmethod
    def backward(ctx, grad_out):
        # backward is reinterpret(R,P): same as forward (no-op)
        return grad_out, None


def reinterpret(
    x,
    axis: "str | ProcessGroup",
    *,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
):
    """
    Coerce from one local SPMD type to another without changing the local tensor.

    Guaranteed to be a no-op in forwards, but can have nontrivial backwards
    that requires comms.

    Args:
        x: Input tensor
        axis: The mesh axis to operate on (string name or ProcessGroup)
        src: Source local SPMD type (R, I, V, P)
        dst: Target local SPMD type (R, I, V, P)

    Supported coercions:
        - reinterpret(R,I): R -> I, backward is convert(I,P): I -> P
        - reinterpret(R,V): R -> V, backward is reinterpret(V,P): V -> P
        - reinterpret(R,P): R -> P, backward is reinterpret(R,P): R -> P
        - reinterpret(I,R): I -> R, backward is all_reduce(I): P -> I
        - reinterpret(I,V): I -> V, composition of I -> R -> V
        - reinterpret(I,P): I -> P, composition of I -> R -> P
        - reinterpret(V,P): V -> P, backward is reinterpret(R,V): R -> V

    Note: This API does not support S(i) for src/dst, because the restriction
    on no local tensor change means the semantics would be the same as V.
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            reinterpret,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
        )
    # Validate no Shard types
    if isinstance(src, Shard) or isinstance(dst, Shard):
        raise ValueError(
            f"reinterpret does not support S(i). Use V instead, or use convert for "
            f"semantics-preserving conversions. Got src={src}, dst={dst}"
        )

    if src is dst:
        return x  # no-op

    if src is R and dst is V:
        return _ReplicateToVarying.apply(x, axis)
    elif src is R and dst is I:
        return _ReplicateToInvariant.apply(x, axis)
    elif src is R and dst is P:
        return _ReplicateToPartial.apply(x, axis)
    elif src is I and dst is R:
        return _InvariantToReplicate.apply(x, axis)
    elif src is I and dst is V:
        # Composition: I -> R -> V
        return _ReplicateToVarying.apply(_InvariantToReplicate.apply(x, axis), axis)
    elif src is I and dst is P:
        # Composition: I -> R -> P
        return _ReplicateToPartial.apply(_InvariantToReplicate.apply(x, axis), axis)
    elif src is V and dst is P:
        return _VaryingToPartial.apply(x, axis)
    else:
        if src is P:
            raise ValueError(
                f"reinterpret({src}, {dst}) is not supported; it is semantically ill-defined. "
                "Call all_reduce(src=P, dst=R) first to materialize the sum, "
                "then do whatever conversion you need from R."
            )
        elif src is V:
            # V -> R or V -> I
            raise ValueError(
                f"reinterpret({src}, {dst}) is not supported. "
                f"We cannot unsafely assert that varying values on all ranks are actually the same. "
                f"Ensure your source is already {dst} instead."
            )
        else:
            raise ValueError(f"reinterpret({src}, {dst}) is not supported.")


# =============================================================================
# convert helper functions
# =============================================================================


def _get_rank(pg):
    """Get rank, using LocalTensorMode's simulated rank if available."""
    mode = local_tensor_mode()
    if mode is not None:
        # We're in LocalTensorMode - rank will be set by tensor_map callback
        # This function shouldn't be called directly in that case
        return _dist.dist.get_rank(pg)
    return _dist.dist.get_rank(pg)


def _replicate_to_varying_fwd(x, world_size, split_dim, rank, *, stack):
    """Forward: split and take local portion based on rank."""
    if stack:
        return x.select(split_dim, rank).contiguous()
    chunks = torch.chunk(x, world_size, dim=split_dim)
    return chunks[rank].contiguous()


def _varying_to_partial_fwd(x, world_size, split_dim, rank, *, stack):
    """Forward: pad with zeros, place data at rank position."""
    if stack:
        pad_shape = list(x.shape)
        pad_shape.insert(split_dim, world_size)
        result = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        slices = [slice(None)] * len(pad_shape)
        slices[split_dim] = rank
        result[tuple(slices)] = x
        return result
    pad_shape = list(x.shape)
    pad_shape[split_dim] = pad_shape[split_dim] * world_size
    result = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    chunk_size = x.shape[split_dim]
    slices = [slice(None)] * len(pad_shape)
    slices[split_dim] = slice(rank * chunk_size, (rank + 1) * chunk_size)
    result[tuple(slices)] = x
    return result


def _replicate_to_partial_fwd(x, rank):
    """Forward: keep value on rank 0, zero elsewhere."""
    if rank == 0:
        return x.clone()
    else:
        return torch.zeros_like(x)


# =============================================================================
# convert autograd Functions
# =============================================================================


class _ConvertReplicateToVarying(torch.autograd.Function):
    """convert(R,V): R -> V, backward is convert(V,P): V -> P."""

    @staticmethod
    def forward(ctx, x, axis, split_dim, stack):
        ctx.axis = axis
        ctx.split_dim = split_dim
        ctx.stack = stack
        pg = _get_mesh_axis_group(axis)
        world_size = _dist.dist.get_world_size(pg)

        mode = local_tensor_mode()
        if mode is not None and isinstance(x, LocalTensor):
            return mode.tensor_map(
                x,
                lambda r, t: _replicate_to_varying_fwd(
                    t, world_size, split_dim, r, stack=stack
                ),
            )
        else:
            rank = _dist.dist.get_rank(pg)
            return _replicate_to_varying_fwd(
                x, world_size, split_dim, rank, stack=stack
            )

    @staticmethod
    def backward(ctx, grad_out):
        # backward is convert(V,P): V -> P
        pg = _get_mesh_axis_group(ctx.axis)
        world_size = _dist.dist.get_world_size(pg)

        mode = local_tensor_mode()
        if mode is not None and isinstance(grad_out, LocalTensor):
            result = mode.tensor_map(
                grad_out,
                lambda r, t: _varying_to_partial_fwd(
                    t, world_size, ctx.split_dim, r, stack=ctx.stack
                ),
            )
            return result, None, None, None
        else:
            rank = _dist.dist.get_rank(pg)
            return (
                _varying_to_partial_fwd(
                    grad_out, world_size, ctx.split_dim, rank, stack=ctx.stack
                ),
                None,
                None,
                None,
            )


class _ConvertInvariantToVarying(torch.autograd.Function):
    """convert(I,V): I -> V, backward is all_gather(I): V -> I."""

    @staticmethod
    def forward(ctx, x, axis, split_dim, stack):
        ctx.axis = axis
        ctx.split_dim = split_dim
        ctx.stack = stack
        pg = _get_mesh_axis_group(axis)
        world_size = _dist.dist.get_world_size(pg)

        mode = local_tensor_mode()
        if mode is not None and isinstance(x, LocalTensor):
            return mode.tensor_map(
                x,
                lambda r, t: _replicate_to_varying_fwd(
                    t, world_size, split_dim, r, stack=stack
                ),
            )
        else:
            rank = _dist.dist.get_rank(pg)
            return _replicate_to_varying_fwd(
                x, world_size, split_dim, rank, stack=stack
            )

    @staticmethod
    def backward(ctx, grad_out):
        # backward is all_gather(I): V -> I
        from sixlib.spmd_types._collectives import all_gather

        src = V if ctx.stack else Shard(ctx.split_dim)
        return all_gather(grad_out, ctx.axis, src=src, dst=I), None, None, None


class _ConvertReplicateToPartial(torch.autograd.Function):
    """convert(R,P): R -> P, backward is convert(R,P): R -> P."""

    @staticmethod
    def forward(ctx, x, axis):
        ctx.axis = axis
        pg = _get_mesh_axis_group(axis)

        mode = local_tensor_mode()
        if mode is not None and isinstance(x, LocalTensor):
            return mode.tensor_map(x, lambda r, t: _replicate_to_partial_fwd(t, r))
        else:
            rank = _dist.dist.get_rank(pg)
            return _replicate_to_partial_fwd(x, rank)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is same operation: convert(R,P)
        pg = _get_mesh_axis_group(ctx.axis)

        mode = local_tensor_mode()
        if mode is not None and isinstance(grad_out, LocalTensor):
            return mode.tensor_map(
                grad_out, lambda r, t: _replicate_to_partial_fwd(t, r)
            ), None
        else:
            rank = _dist.dist.get_rank(pg)
            return _replicate_to_partial_fwd(grad_out, rank), None


class _ConvertInvariantToPartial(torch.autograd.Function):
    """convert(I,P): I -> P, backward is reinterpret(R,I): R -> I (no-op)."""

    @staticmethod
    def forward(ctx, x, axis):
        ctx.axis = axis
        pg = _get_mesh_axis_group(axis)

        mode = local_tensor_mode()
        if mode is not None and isinstance(x, LocalTensor):
            return mode.tensor_map(x, lambda r, t: _replicate_to_partial_fwd(t, r))
        else:
            rank = _dist.dist.get_rank(pg)
            return _replicate_to_partial_fwd(x, rank)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is reinterpret(R,I): R -> I (no-op)
        return grad_out, None


class _ConvertVaryingToPartial(torch.autograd.Function):
    """convert(V,P): V -> P, backward is convert(R,V): R -> V."""

    @staticmethod
    def forward(ctx, x, axis, split_dim, stack):
        ctx.axis = axis
        ctx.split_dim = split_dim
        ctx.stack = stack
        pg = _get_mesh_axis_group(axis)
        world_size = _dist.dist.get_world_size(pg)

        mode = local_tensor_mode()
        if mode is not None and isinstance(x, LocalTensor):
            return mode.tensor_map(
                x,
                lambda r, t: _varying_to_partial_fwd(
                    t, world_size, split_dim, r, stack=stack
                ),
            )
        else:
            rank = _dist.dist.get_rank(pg)
            return _varying_to_partial_fwd(x, world_size, split_dim, rank, stack=stack)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is convert(R,V): R -> V (take local slice)
        pg = _get_mesh_axis_group(ctx.axis)
        world_size = _dist.dist.get_world_size(pg)

        mode = local_tensor_mode()
        if mode is not None and isinstance(grad_out, LocalTensor):
            result = mode.tensor_map(
                grad_out,
                lambda r, t: _replicate_to_varying_fwd(
                    t, world_size, ctx.split_dim, r, stack=ctx.stack
                ),
            )
            return result, None, None, None
        else:
            rank = _dist.dist.get_rank(pg)
            return (
                _replicate_to_varying_fwd(
                    grad_out, world_size, ctx.split_dim, rank, stack=ctx.stack
                ),
                None,
                None,
                None,
            )


def convert(
    x,
    axis: "str | ProcessGroup",
    *,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
    dim: int = 0,
):
    """
    Convert from one local SPMD type to another while preserving tensor semantics.

    Unlike reinterpret, convert may perform local tensor operations (but no comms).
    When a tensor is varying, we interpret it as concatenation on the specified dim.

    Args:
        x: Input tensor
        axis: The mesh axis to operate on (string name or ProcessGroup)
        src: Source local SPMD type (R, I, V, P, or S(i))
        dst: Target local SPMD type (R, I, V, P, or S(i))
        dim: The tensor dimension for split/concat operations (default: 0).
             When src or dst is S(i), the dim from S(i) is used instead.

    Supported conversions:
        - convert(R,V): R -> V, backward is convert(V,P): V -> P
        - convert(R,S(i)): R -> S(i), backward is convert(S(i),P): S(i) -> P
        - convert(I,V): I -> V, backward is all_gather(I): V -> I
        - convert(I,S(i)): I -> S(i), backward is all_gather(S(i),I): S(i) -> I
        - convert(R,P): R -> P, backward is convert(R,P): R -> P
        - convert(I,P): I -> P, backward is reinterpret(R,I): R -> I
        - convert(V,P): V -> P, backward is convert(R,V): R -> V
        - convert(S(i),P): S(i) -> P, backward is convert(R,S(i)): R -> S(i)
        - convert(R,I) and convert(I,R) are same as reinterpret
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            convert,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
            dim=dim,
        )
    # Extract dim from Shard if present
    if isinstance(src, Shard):
        dim = src.dim
    if isinstance(dst, Shard):
        dim = dst.dim

    # Normalize Shard to V for dispatch (they use the same underlying functions)
    src_base = V if isinstance(src, Shard) else src
    dst_base = V if isinstance(dst, Shard) else dst

    if src_base is dst_base:
        return x  # no-op

    if src_base is R and dst_base is V:
        stack = not isinstance(dst, Shard)
        return _ConvertReplicateToVarying.apply(x, axis, dim, stack)
    elif src_base is R and dst_base is P:
        return _ConvertReplicateToPartial.apply(x, axis)
    elif src_base is R and dst_base is I:
        # Same as reinterpret
        return _ReplicateToInvariant.apply(x, axis)
    elif src_base is I and dst_base is V:
        stack = not isinstance(dst, Shard)
        return _ConvertInvariantToVarying.apply(x, axis, dim, stack)
    elif src_base is I and dst_base is P:
        return _ConvertInvariantToPartial.apply(x, axis)
    elif src_base is I and dst_base is R:
        # Same as reinterpret
        return _InvariantToReplicate.apply(x, axis)
    elif src_base is V and dst_base is P:
        stack = not isinstance(src, Shard)
        return _ConvertVaryingToPartial.apply(x, axis, dim, stack)
    else:
        if src_base is P:
            if dst_base is R or dst_base is I:
                raise ValueError(
                    f"convert({src}, {dst}) is not supported. "
                    f"Use all_reduce(src=P, dst={dst}) to perform the reduction and get the full sum."
                )
            elif dst_base is V or isinstance(dst, Shard):
                raise ValueError(
                    f"convert({src}, {dst}) is not supported. "
                    f"Use reduce_scatter(src=P, dst={dst}) to perform the reduction and get shards of the sum."
                )
            else:
                raise ValueError(
                    f"convert({src}, {dst}) is not supported. Cannot convert out of P."
                )
        else:
            raise ValueError(f"convert({src}, {dst}) is not supported.")
