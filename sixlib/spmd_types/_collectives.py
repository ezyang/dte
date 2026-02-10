"""Distributed collective operations: all_reduce, all_gather, reduce_scatter, all_to_all, redistribute."""

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from sixlib.spmd_types._local import convert, reinterpret
from sixlib.spmd_types._mesh import _get_mesh_axis_group
from sixlib.spmd_types.types import (
    I,
    P,
    PerMeshAxisSpmdType,
    R,
    Shard,
    V,
)

# =============================================================================
# all_reduce: P -> R | I
# =============================================================================


class _AllReduce(torch.autograd.Function):
    """all_reduce: P -> R|I.

    When dst=R, backward is all_reduce(R): P -> R.
    When dst=I, backward is reinterpret(I,R): I -> R (no-op).
    """

    @staticmethod
    def forward(ctx, x, axis, dst, inplace):
        ctx.axis = axis
        ctx.dst = dst
        pg = _get_mesh_axis_group(axis)
        if inplace:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pg)
            ctx.mark_dirty(x)
            return x
        return funcol.all_reduce(x, "sum", pg).wait()

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.dst is R:
            # backward of P -> R is P -> R (same operation)
            return all_reduce(grad_out, ctx.axis, src=P, dst=R), None, None, None
        else:
            # backward of P -> I: reinterpret(I,R) is identity but sets up autograd for double backward
            return reinterpret(grad_out, ctx.axis, src=I, dst=R), None, None, None


def all_reduce(
    x,
    axis: "str | dist.ProcessGroup",
    *,
    src: PerMeshAxisSpmdType = P,
    dst: PerMeshAxisSpmdType,
    inplace: bool = False,
):
    """
    Reduce shards along the mesh axis, so every rank has the full summed value.

    Args:
        x: Input tensor with P type on the mesh axis
        axis: The mesh axis to reduce over (string name or ProcessGroup)
        src: Source type (must be P)
        dst: Target type (R or I)
        inplace: If True, perform the all-reduce in-place on the input tensor
            using ``dist.all_reduce`` instead of allocating a new output tensor.

    Returns:
        Tensor with R or I type depending on dst

    When dst=R, backward is all_reduce(R): P -> R
    When dst=I, backward is reinterpret(I,R): I -> R (no-op)
    """
    if src is not P:
        if src is V:
            x = reinterpret(x, axis, src=V, dst=P)
        elif src is R or src is I:
            raise ValueError(
                f"all_reduce src must be P, got {src}. "
                "all_reduce on replicated/invariant data is usually a bug. "
                f"If you really want to scale by mesh size, use reinterpret(src={src}, dst=P) first."
            )
        else:
            raise ValueError(f"all_reduce src must be P, got {src}")
    if dst is R or dst is I:
        return _AllReduce.apply(x, axis, dst, inplace)
    else:
        raise ValueError(f"all_reduce dst must be R or I, got {dst}")


# =============================================================================
# all_gather: V -> R | I
# =============================================================================


class _AllGather(torch.autograd.Function):
    """all_gather: V -> R|I.

    When dst=R, backward is reduce_scatter: P -> V.
    When dst=I, backward is convert(I,V): I -> V.
    """

    @staticmethod
    def forward(ctx, x, axis, dst, gather_dim, stack):
        ctx.axis = axis
        ctx.dst = dst
        ctx.gather_dim = gather_dim
        ctx.stack = stack
        pg = _get_mesh_axis_group(axis)
        world_size = dist.get_world_size(pg)
        gathered = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x, group=pg)
        if stack:
            return torch.stack(gathered, dim=gather_dim)
        return torch.cat(gathered, dim=gather_dim)

    @staticmethod
    def backward(ctx, grad_out):
        dst_type = V if ctx.stack else Shard(ctx.gather_dim)
        if ctx.dst is R:
            # backward is reduce_scatter: P -> V
            return (
                reduce_scatter(
                    grad_out, ctx.axis, src=P, dst=dst_type, scatter_dim=ctx.gather_dim
                ),
                None,
                None,
                None,
                None,
            )
        else:
            # backward is convert(I,V): I -> V
            return (
                convert(grad_out, ctx.axis, src=I, dst=dst_type, dim=ctx.gather_dim),
                None,
                None,
                None,
                None,
            )


def all_gather(
    x,
    axis: "str | dist.ProcessGroup",
    *,
    src: PerMeshAxisSpmdType = V,
    dst: PerMeshAxisSpmdType,
):
    """
    Gather shards along the mesh axis, so every rank has the full copy of data.

    ```
    [A]
    [B]  =>  [A, B, C]
    [C]
    ```

    Args:
        x: Input tensor with V or S(i) type on the mesh axis
        axis: The mesh axis to gather over (string name or ProcessGroup)
        src: Source type (V or S(i)). When V, stacks on dim 0. When S(i), concatenates on dim i.
        dst: Target type (R or I)

    Returns:
        Tensor with R or I type depending on dst

    When dst=R, backward is reduce_scatter: P -> V
    When dst=I, backward is convert(I,V): I -> V
    """
    # Validate src is V or S(i)
    if not (src is V or isinstance(src, Shard)):
        raise ValueError(f"all_gather src must be V or S(i), got {src}")

    gather_dim = src.dim if isinstance(src, Shard) else 0
    stack = src is V
    if dst is R or dst is I:
        return _AllGather.apply(x, axis, dst, gather_dim, stack)
    else:
        raise ValueError(f"all_gather dst must be R or I, got {dst}")


# =============================================================================
# reduce_scatter: P -> V
# =============================================================================


class _ReduceScatter(torch.autograd.Function):
    """reduce_scatter: P -> V, backward is all_gather(R): V -> R."""

    @staticmethod
    def forward(ctx, x, axis, scatter_dim, stack):
        ctx.axis = axis
        ctx.scatter_dim = scatter_dim
        ctx.stack = stack
        pg = _get_mesh_axis_group(axis)
        result = funcol.reduce_scatter_tensor(x, "sum", scatter_dim, pg).wait()
        if stack:
            return result.squeeze(scatter_dim)
        return result

    @staticmethod
    def backward(ctx, grad_out):
        # backward is all_gather(R): V -> R
        src_type = V if ctx.stack else Shard(ctx.scatter_dim)
        return all_gather(grad_out, ctx.axis, src=src_type, dst=R), None, None, None


def reduce_scatter(
    x,
    axis: "str | dist.ProcessGroup",
    *,
    src: PerMeshAxisSpmdType = P,
    dst: PerMeshAxisSpmdType = V,
    scatter_dim: int = 0,
):
    """
    Reduce shards along the mesh axis, but only get one shard of the result.

    ```
    +[Ax, Bx, Cx]      [Ax + Ay + Az]
    +[Ay, By, Cy]  =>  [Bx + By + Bz]
    +[Az, Bz, Cz]      [Cx + Cy + Cz]
    ```

    Args:
        x: Input tensor with P type on the mesh axis
        axis: The mesh axis to reduce-scatter over (string name or ProcessGroup)
        src: Source type (must be P)
        dst: Target type (V or S(i))
        scatter_dim: The tensor dimension to scatter along (default: 0)

    Returns:
        Tensor with V or S(i) type depending on dst

    The backward is all_gather(R): V -> R
    """
    if src is not P:
        if src is V:
            x = reinterpret(x, axis, src=V, dst=P)
        elif src is R or src is I:
            raise ValueError(
                f"reduce_scatter src must be P, got {src}. "
                "reduce_scatter on replicated/invariant data is usually a bug. "
                f"If you really want to scale by mesh size, use reinterpret(src={src}, dst=P) first."
            )
        else:
            raise ValueError(f"reduce_scatter src must be P, got {src}")
    if not (dst is V or isinstance(dst, Shard)):
        raise ValueError(f"reduce_scatter dst must be V or S(i), got {dst}")

    if isinstance(dst, Shard):
        scatter_dim = dst.dim

    stack = dst is V
    return _ReduceScatter.apply(x, axis, scatter_dim, stack)


# =============================================================================
# all_to_all: V -> V
# =============================================================================


class _AllToAll(torch.autograd.Function):
    """all_to_all: V -> V, backward is all_to_all: V -> V."""

    @staticmethod
    def forward(ctx, x, axis, split_dim, concat_dim, stack):
        ctx.axis = axis
        ctx.split_dim = split_dim
        ctx.concat_dim = concat_dim
        ctx.stack = stack
        pg = _get_mesh_axis_group(axis)
        world_size = dist.get_world_size(pg)
        # Split input
        if stack:
            input_chunks = list(torch.unbind(x, dim=split_dim))
        else:
            input_chunks = list(torch.chunk(x, world_size, dim=split_dim))
        output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]
        dist.all_to_all(output_chunks, input_chunks, group=pg)
        if stack:
            return torch.stack(output_chunks, dim=concat_dim)
        return torch.cat(output_chunks, dim=concat_dim)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is also all_to_all (transpose back: swap src/dst and dims)
        if ctx.stack:
            return (
                all_to_all(
                    grad_out,
                    ctx.axis,
                    src=V,
                    dst=V,
                    split_dim=ctx.concat_dim,
                    concat_dim=ctx.split_dim,
                ),
                None,
                None,
                None,
                None,
            )
        else:
            return (
                all_to_all(
                    grad_out,
                    ctx.axis,
                    src=Shard(ctx.concat_dim),
                    dst=Shard(ctx.split_dim),
                ),
                None,
                None,
                None,
                None,
            )


def all_to_all(
    x,
    axis: "str | dist.ProcessGroup",
    *,
    src: PerMeshAxisSpmdType = V,
    dst: PerMeshAxisSpmdType = V,
    split_dim: int = 0,
    concat_dim: int = 0,
):
    """
    Transpose a local tensor axis with the mesh axis.

    ```
    [A0, A1, A2]      [A0, B0, C0]
    [B0, B1, B2]  =>  [A1, B1, C1]
    [C0, C1, C2]      [A2, B2, C2]
    ```

    Args:
        x: Input tensor with V or S(i) type on the mesh axis
        axis: The mesh axis to transpose with (string name or ProcessGroup)
        src: Source type (V or S(i))
        dst: Target type (V or S(j))
        split_dim: The tensor dimension to split along (default: 0)
        concat_dim: The tensor dimension to concatenate along (default: 0)

    Returns:
        Tensor with V or S(j) type depending on dst

    The backward is also all_to_all: V -> V (with src/dst swapped)
    """
    # Validate src and dst are V or S(i)
    if not (src is V or isinstance(src, Shard)):
        raise ValueError(f"all_to_all src must be V or S(i), got {src}")
    if not (dst is V or isinstance(dst, Shard)):
        raise ValueError(f"all_to_all dst must be V or S(i), got {dst}")

    if isinstance(src, Shard):
        split_dim = src.dim
    if isinstance(dst, Shard):
        concat_dim = dst.dim

    stack = src is V and dst is V
    return _AllToAll.apply(x, axis, split_dim, concat_dim, stack)


# =============================================================================
# redistribute: semantics-preserving type change with comms
# =============================================================================


def redistribute(
    x,
    axis: "str | dist.ProcessGroup",
    *,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
    dim: int = 0,
):
    """
    Semantics-preserving conversion between local SPMD types, allowing comms.

    Unlike convert (which is no-comm), redistribute will perform the necessary
    collective communication to change from one type to another while preserving
    the semantic value of the tensor.

    Args:
        x: Input tensor
        axis: The mesh axis to operate on (string name or ProcessGroup)
        src: Source local SPMD type
        dst: Target local SPMD type
        dim: Tensor dimension for shard operations (default: 0).
             When src or dst is S(i), the dim from S(i) is used.

    Routes to:
        redistribute(S(i),R)    -> all_gather(S(i),R)
        redistribute(S(i),I)    -> all_gather(S(i),I)
        redistribute(P,R)       -> all_reduce(P,R)
        redistribute(P,I)       -> all_reduce(P,I)
        redistribute(P,S(i))    -> reduce_scatter(P,S(i))
        redistribute(S(i),S(j)) -> all_to_all(S(i),S(j))
        redistribute(V,R)       -> all_gather(V,R)
        redistribute(V,I)       -> all_gather(V,I)
        redistribute(P,V)       -> reduce_scatter(P,V)
        redistribute(V,V)       -> all_to_all(V,V)

    For conversions that don't require comms (R<->I, R->V, R->P, I->V, I->P, V->P),
    this function delegates to convert or reinterpret as appropriate.
    """
    # Extract dim from Shard if present
    if isinstance(src, Shard):
        dim = src.dim
    if isinstance(dst, Shard):
        dim = dst.dim

    # Normalize to base types for dispatch
    src_is_shard = isinstance(src, Shard)
    dst_is_shard = isinstance(dst, Shard)
    src_base = V if src_is_shard else src
    dst_base = V if dst_is_shard else dst

    if src_base is dst_base:
        if src_is_shard and dst_is_shard and src.dim != dst.dim:
            # S(i) -> S(j): need all_to_all
            return all_to_all(
                x, axis, src=src, dst=dst, split_dim=src.dim, concat_dim=dst.dim
            )
        return x  # no-op

    # Varying/Shard -> Replicate: all_gather
    if src_base is V and dst_base is R:
        return all_gather(x, axis, src=src, dst=R)

    # Varying/Shard -> Invariant: all_gather
    if src_base is V and dst_base is I:
        return all_gather(x, axis, src=src, dst=I)

    # Partial -> Replicate: all_reduce
    if src_base is P and dst_base is R:
        return all_reduce(x, axis, src=P, dst=R)

    # Partial -> Invariant: all_reduce
    if src_base is P and dst_base is I:
        return all_reduce(x, axis, src=P, dst=I)

    # Partial -> Varying/Shard: reduce_scatter
    if src_base is P and dst_base is V:
        return reduce_scatter(x, axis, src=P, dst=dst, scatter_dim=dim)

    # For non-comm conversions, delegate to convert
    # R -> I, I -> R, R -> V, R -> P, I -> V, I -> P, V -> P
    if src_base is R and dst_base is I:
        return convert(x, axis, src=R, dst=I)
    if src_base is I and dst_base is R:
        return convert(x, axis, src=I, dst=R)
    if src_base is R and dst_base is V:
        return convert(x, axis, src=R, dst=dst, dim=dim)
    if src_base is R and dst_base is P:
        return convert(x, axis, src=R, dst=P)
    if src_base is I and dst_base is V:
        return convert(x, axis, src=I, dst=dst, dim=dim)
    if src_base is I and dst_base is P:
        return convert(x, axis, src=I, dst=P)
    if src_base is V and dst_base is P:
        return convert(x, axis, src=src, dst=P, dim=dim)

    raise ValueError(f"redistribute({src}, {dst}) is not supported.")
