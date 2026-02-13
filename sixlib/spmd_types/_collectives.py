"""Distributed collective operations: all_reduce, all_gather, reduce_scatter, all_to_all, redistribute."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sixlib.spmd_types import _dist
from sixlib.spmd_types._local import convert, reinterpret
from sixlib.spmd_types._mesh import _get_mesh_axis_group
from sixlib.spmd_types.types import (
    _canonicalize_shard,
    I,
    P,
    PerMeshAxisSpmdType,
    R,
    Shard,
    V,
)
from torch.overrides import handle_torch_function, has_torch_function_unary

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

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
        # TODO: check if contiguous assertion is really necessary
        assert x.is_contiguous(), "all_reduce input must be contiguous"
        # TODO: check if world_size == 1 short-circuit is really necessary
        if _dist.dist.get_world_size(pg) == 1:
            return x
        if inplace:
            _dist.dist.all_reduce(x, op=_dist.dist.ReduceOp.SUM, group=pg)
            ctx.mark_dirty(x)
            return x
        else:
            result = x.clone()
            _dist.dist.all_reduce(result, op=_dist.dist.ReduceOp.SUM, group=pg)
            return result

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.dst is R:
            # backward of P -> R is P -> R (same operation)
            grad = all_reduce(grad_out, ctx.axis, src=P, dst=R)
        else:
            # backward of P -> I: reinterpret(I,R) is identity but sets up autograd for double backward
            grad = reinterpret(grad_out, ctx.axis, src=I, dst=R)
        return grad, None, None, None


def all_reduce(
    x,
    axis: str | ProcessGroup,
    *,
    src: PerMeshAxisSpmdType = P,
    dst: PerMeshAxisSpmdType,
    inplace: bool = False,
):
    """``all_reduce(x: Partial, mesh_axis, dst) -> Replicate | Invariant``

    Reduce shards along the mesh axis, so that every rank has the full summed value.

    ::

        def all_reduce_spec(x: f32[*shape]) -> f32[*shape]:
            return x  # Identity! The summation already occured on x's conversion to Partial

        +Ax
        +Ay  =>  Ax + Ay + Az
        +Az

    Args:
        x: Input tensor with P type on the mesh axis
        axis: The mesh axis to reduce over (string name or ProcessGroup)
        src: Source type (must be P)
        dst: Target type (R or I)
        inplace: If True, perform the all-reduce in-place on the input tensor
            using ``dist.all_reduce`` instead of allocating a new output tensor.

    Returns:
        Tensor with R or I type depending on dst

    **Backward cases:**

    When ``dst=R``, aka ``all_reduce(R): P -> R``, the backwards is ``all_reduce(R): P -> R``::

                          +Ax
        Ax + Ay + Az  <=  +Ay
                          +Az

    When ``dst=I``, aka ``all_reduce(I): P -> I``, the backwards is ``reinterpret(I,R): I -> R``::

        A  <=  A

    It is common to want to ``all_reduce`` on varying data; just
    ``reinterpret(V,P)`` the data as partial before calling ``all_reduce``.
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            all_reduce,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
            inplace=inplace,
        )
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
        world_size = _dist.dist.get_world_size(pg)
        gathered = [torch.empty_like(x) for _ in range(world_size)]
        _dist.dist.all_gather(gathered, x, group=pg)
        if stack:
            return torch.stack(gathered, dim=gather_dim)
        return torch.cat(gathered, dim=gather_dim)

    @staticmethod
    def backward(ctx, grad_out):
        dst_type = V if ctx.stack else Shard(ctx.gather_dim)
        if ctx.dst is R:
            # backward is reduce_scatter: P -> V
            grad = reduce_scatter(
                grad_out, ctx.axis, src=P, dst=dst_type, scatter_dim=ctx.gather_dim
            )
        else:
            # backward is convert(I,V): I -> V
            grad = convert(grad_out, ctx.axis, src=I, dst=dst_type, dim=ctx.gather_dim)
        return grad, None, None, None, None


def all_gather(
    x,
    axis: str | ProcessGroup,
    *,
    src: PerMeshAxisSpmdType = V,
    dst: PerMeshAxisSpmdType,
):
    """``all_gather(x: Varying, mesh_axis, src, dst) -> Replicate | Invariant``

    Gather shards along the mesh axis, so that every rank has the full copy of
    the data.  PyTorch's ``all_gather`` can either concat or stack inputs
    together.  When the source tensor is interpreted as a varying tensor, we
    stack the inputs together, creating a new dimension of size mesh axis.  But
    if the source tensor is a sharded tensor, we concat along the sharded
    dimension.

    ::

        def all_gather_spec(xs, src):
            # NB: dst only affects autograd
            match src:
                case V:
                    '''
                    A
                    B  =>  [A, B, C]
                    C
                    '''
                    return torch.stack(xs)

                case S(i):
                    '''
                    When i == 0:
                    [A0, A1]
                    [B0, B1]  =>  [A0, A1, B0, B1, C0, C1]
                    [C0, C1]
                    '''
                    return torch.concat(xs, i)

    Args:
        x: Input tensor with V or S(i) type on the mesh axis
        axis: The mesh axis to gather over (string name or ProcessGroup)
        src: Source type (V or S(i)). When V, stacks on dim 0. When S(i), concatenates on dim i.
        dst: Target type (R or I)

    Returns:
        Tensor with R or I type depending on dst

    **Backward cases:**

    ``all_gather(V,R): V -> R``, the backwards is ``reduce_scatter(V): P -> V``::

        Ax + Ay + Az      +[Ax, Bx, Cx]
        Bx + By + Bz  <=  +[Ay, By, Cy]
        Cx + Cy + Cz      +[Az, Bz, Cz]

    ``all_gather(V,I): V -> I``, the backwards is ``convert(I,V): I -> V``::

        A
        B  <=  [A, B, C]
        C

    ``all_gather(S(i),R): S(i) -> R``, the backwards is ``reduce_scatter(S(i)): P -> S(i)``::

        When i == 0:
        [A0x + A0y + A0z, A1x + A1y + A1z]      +[A0x, A1x, B0x, B1x, C0x, C1x]
        [B0x + B0y + B0z, B1x + B1y + B1z]  <=  +[A0y, A1y, B0y, B1y, C0y, C1y]
        [C0x + C0y + C0z, C1x + C1y + C1z]      +[A0z, A1z, B0z, B1z, C0z, C1z]

    ``all_gather(S(i),I): S(i) -> I``, the backwards is ``convert(I,S(i)): I -> S(i)``::

        [A0, A1]
        [B0, B1]  <=  [A0, A1, B0, B1, C0, C1]
        [C0, C1]
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            all_gather,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
        )
    # Canonicalize negative Shard dims
    src = _canonicalize_shard(src, x.ndim)

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
        world_size = _dist.dist.get_world_size(pg)
        if stack:
            # x stacked on dim 0: shape[0] == world_size
            result = x.new_empty([1] + list(x.shape[1:]))
            _dist.dist.reduce_scatter_tensor(
                result, x, op=_dist.dist.ReduceOp.SUM, group=pg
            )
            return result.squeeze(0)
        else:
            # reduce_scatter_tensor always scatters along dim 0, so we
            # movedim before/after when scatter_dim != 0.
            needs_permute = scatter_dim != 0
            if needs_permute:
                x = x.movedim(scatter_dim, 0).contiguous()

            output_shape = list(x.shape)
            output_shape[0] //= world_size
            result = x.new_empty(output_shape)
            _dist.dist.reduce_scatter_tensor(
                result, x, op=_dist.dist.ReduceOp.SUM, group=pg
            )

            if needs_permute:
                result = result.movedim(0, scatter_dim)
            return result

    @staticmethod
    def backward(ctx, grad_out):
        # backward is all_gather(R): V -> R
        src_type = V if ctx.stack else Shard(ctx.scatter_dim)
        return all_gather(grad_out, ctx.axis, src=src_type, dst=R), None, None, None


def reduce_scatter(
    x,
    axis: str | ProcessGroup,
    *,
    src: PerMeshAxisSpmdType = P,
    dst: PerMeshAxisSpmdType = V,
    scatter_dim: int = 0,
):
    """``reduce_scatter(x, mesh_axis, dst): Partial -> Varying``

    Reduce shards along the mesh axis, but only get one shard of the result
    (e.g., an inefficient implementation of reduce-scatter would be to
    all-reduce and then drop the data you did not need.)  Like ``all_gather``,
    ``dst`` can either be varying for stack semantics, or shard for concat
    semantics.

    ::

        def reduce_scatter_spec(x: f32[mesh_axis_size, *shape], dst) -> List[f32[*shape]]:
            # NB: The semantic summation already occured on x's conversion to Partial
            match dst:
                case V:
                    '''
                    +[Ax, Bx, Cx]      Ax + Ay + Az
                    +[Ay, By, Cy]  =>  Bx + By + Bz
                    +[Az, Bz, Cz]      Cx + Cy + Cz
                    '''
                    return x.unbind()
                case S(i):
                    '''
                    When i == 0:
                    +[A0x, A1x, B0x, B1x, C0x, C1x]      [A0x + A0y + A0z, A1x + A1y + A1z]
                    +[A0y, A1y, B0y, B1y, C0y, C1y]  =>  [B0x + B0y + B0z, B1x + B1y + B1z]
                    +[A0z, A1z, B0z, B1z, C0z, C1z]      [C0x + C0y + C0z, C1x + C1y + C1z]
                    '''
                    return x.chunk(mesh_axis_size, i)

    Args:
        x: Input tensor with P type on the mesh axis
        axis: The mesh axis to reduce-scatter over (string name or ProcessGroup)
        src: Source type (must be P)
        dst: Target type (V or S(i))
        scatter_dim: The tensor dimension to scatter along (default: 0)

    Returns:
        Tensor with V or S(i) type depending on dst

    **Backward cases:**

    ``reduce_scatter(V): P -> V``, the backwards is ``all_gather(V,R): V -> R``::

                       A
        [A, B, C]  <=  B
                       C

    ``reduce_scatter(S(i)): P -> S(i)``, the backwards is ``all_gather(S(i),R): S(i) -> R``::

        When i == 0:
                                      [A0, A1]
        [A0, A1, B0, B1, C0, C1]  <=  [B0, B1]
                                      [C0, C1]

    It is common to want to reduce-scatter on varying data; just
    ``reinterpret(V,P)`` the data as partial before calling ``reduce_scatter``.
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            reduce_scatter,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
            scatter_dim=scatter_dim,
        )
    # Canonicalize negative Shard dims
    dst = _canonicalize_shard(dst, x.ndim)

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
        world_size = _dist.dist.get_world_size(pg)
        # Split input
        if stack:
            input_chunks = list(torch.unbind(x, dim=split_dim))
        else:
            input_chunks = list(torch.chunk(x, world_size, dim=split_dim))
        output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]
        _dist.dist.all_to_all(output_chunks, input_chunks, group=pg)
        if stack:
            return torch.stack(output_chunks, dim=concat_dim)
        return torch.cat(output_chunks, dim=concat_dim)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is also all_to_all (transpose back: swap src/dst and dims)
        if ctx.stack:
            grad = all_to_all(
                grad_out,
                ctx.axis,
                src=V,
                dst=V,
                split_dim=ctx.concat_dim,
                concat_dim=ctx.split_dim,
            )
        else:
            grad = all_to_all(
                grad_out,
                ctx.axis,
                src=Shard(ctx.concat_dim),
                dst=Shard(ctx.split_dim),
            )
        return grad, None, None, None, None


def all_to_all(
    x,
    axis: str | ProcessGroup,
    *,
    src: PerMeshAxisSpmdType = V,
    dst: PerMeshAxisSpmdType = V,
    split_dim: int = 0,
    concat_dim: int = 0,
):
    """``all_to_all(x, mesh_axis, src, dst): Varying -> Varying``

    An all-to-all transposes a mesh axis with a tensor axis.

    ::

        def all_to_all_spec(xs, src, dst):
            match src, dst:
                case V, V:
                    x = torch.stack(xs)
                    x = x.transpose(0, 1)
                    return x.unbind()

                case S(i), S(j):
                    x = torch.concat(xs, i)
                    return x.chunk(mesh_axis_size, j)

    The varying and shard versions of ``all_to_all`` are pretty different (even
    though under the hood they both have an all-to-all communication pattern),
    so we describe them separately.

    * ``all_to_all(V,V)`` transposes the logical mesh axis with the dim 0 local
      tensor axis.

    * ``all_to_all(S(i),S(j))`` intuitively unshards the tensor on dim i, and
      then reshards it on dim j (but skips actually doing the all-gather).

    ::

        Diagram for all_to_all(V,V):
        [A0, A1, A2]      [A0, B0, C0]
        [B0, B1, B2]  =>  [A1, B1, C1]
        [C0, C1, C2]      [A2, B2, C2]

    Args:
        x: Input tensor with V or S(i) type on the mesh axis
        axis: The mesh axis to transpose with (string name or ProcessGroup)
        src: Source type (V or S(i))
        dst: Target type (V or S(j))
        split_dim: The tensor dimension to split along (default: 0)
        concat_dim: The tensor dimension to concatenate along (default: 0)

    Returns:
        Tensor with V or S(j) type depending on dst

    **Backward cases:**

    The forwards is ``V -> V``, the backwards is ``all_to_all: V -> V`` (with
    src/dst and split_dim/concat_dim swapped).

    ``all_to_all(V,V): V -> V``, the backwards is ``all_to_all(V,V): V -> V``::

        [A0, A1, A2]      [A0, B0, C0]
        [B0, B1, B2]  <=  [A1, B1, C1]
        [C0, C1, C2]      [A2, B2, C2]

    ``all_to_all(S(i),S(j)): S(i) -> S(j)``, the backwards is
    ``all_to_all(S(j),S(i)): S(j) -> S(i)``.
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            all_to_all,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
            split_dim=split_dim,
            concat_dim=concat_dim,
        )
    # Canonicalize negative Shard dims
    src = _canonicalize_shard(src, x.ndim)
    dst = _canonicalize_shard(dst, x.ndim)

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
    axis: str | ProcessGroup,
    *,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
    dim: int = 0,
):
    """Semantics-preserving type change that allows comms.

    It is helpful to have a version of ``convert`` that is semantics preserving
    but allows for comms.  ``redistribute`` routes to the following
    collectives::

        redistribute(S(i),R)    =   all_gather(S(i),R)
        redistribute(S(i),I)    =   all_gather(S(i),I)
        redistribute(P,R)       =   all_reduce(P,R)
        redistribute(P,I)       =   all_reduce(P,I)
        redistribute(P,S(i))    =   reduce_scatter(P,S(i))
        redistribute(S(i),S(j)) =   all_to_all(S(i),S(j))

    These only work if the mesh axis is the LAST to shard a particular tensor
    dimension.

    For conversions that don't require comms (R<->I, R->V, R->P, I->V, I->P,
    V->P), this function delegates to ``convert`` or ``reinterpret`` as
    appropriate.

    Args:
        x: Input tensor
        axis: The mesh axis to operate on (string name or ProcessGroup)
        src: Source local SPMD type
        dst: Target local SPMD type
        dim: Tensor dimension for shard operations (default: 0).
             When src or dst is S(i), the dim from S(i) is used.
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            redistribute,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
            dim=dim,
        )
    # Canonicalize negative Shard dims
    src = _canonicalize_shard(src, x.ndim)
    dst = _canonicalize_shard(dst, x.ndim)

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
