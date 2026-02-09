import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed._local_tensor import LocalTensor, local_tensor_mode

"""

This module defines a type system for distributed training code, based off of
JAX's sharding in types, but adapted for the PyTorch ecosystem.  It has four
primary design goals:

* Be restrictive enough to ensure user programs always compute mathematically
  correct gradients.

* Be permissive enough to express most common communication optimizations found
  in LLM training code today.

* The sharding of backwards is always known from the sharding of forwards;
  non-local reasoning is never necessary.

* Be optional, so that code can run without any types at runtime.  Equivalently,
  this implies that any communications are explicit in the code.

The type system can be run in two different modes:

* Local SPMD (a more permissive mode), where the semantics of non-collective
  operations is defined purely in terms of their operation on the local tensors
  on ranks.  This mode is less intrusive, as we are generally indifferent to
  the semantics of non-collective operators.

* Global SPMD (a more restrictive mode), where only programs that would give
  equivalent results when run on single device and parallelized are permitted.
  This mode is more intrusive, as we need to (1) understand how a sharded
  tensor should be interpreted as a tensor at all times, and (2) understand
  how the sharding of the tensor changes when we run non-collective operations
  on it.

For users seeking to adopt this type system with an existing Megatron-like
training codebase (plain Tensor and manual collectives everywhere), we
recommend migrating your code to local SPMD for the majority of your code and
only do global SPMD annotations at module boundaries (to specify input/output
contracts).  This minimizes the amount of code changes necessary (since your
code is already local SPMD) for maximum benefit (cross-module sharding/partial
contracts are the biggest source of potential errors in a training codebase;
the internal local SPMD typechecking give partial assurance that your
global SPMD type annotations are consistent with the code.) You may also
consider adopting global SPMD code for specific modules where confusing local
for global SPMD semantics is a persistent source of bugs (e.g., loss functions
or metrics--although, unlike an automatic partitioning system like XLA's GSPMD,
you will not be able to stop thinking about collectives, because remember, all
comms are explicit in code!)

The documentation is structured in the following way: first, we will explain
the local SPMD type system, as it is a subset of the global SPMD type system.
We'll start with describing the types, how operators propagate the types,
and finally how collectives and special functions interact with the types.

Then, we will explain how the global SPMD type system adds extra restrictions
and type refinement on top of the local SPMD type system.

# Local SPMD

What is a local SPMD type?  It describes how a tensor is distributed across
ranks on an axis of the device mesh, as well as how the gradients of the
tensor are distributed over this axis.  For each device mesh axis, a tensor can
assume four distinct local SPMD types: replicate (R), invariant (I), varying
(V) and partial (P).  The semantics:

* Both replicate and invariant mean that the data is replicated across the
  device mesh axis.  The gradient of replicate is partial, while the
  gradient of invariant is invariant.  Intuitively, tensors are typically
  only invariant when they are parameters (and you desire the gradient to
  have already been all-reduced) and replicated when being computed on
  (where you typically desire their gradients to be partial so you can delay
  the all-reduce, in case it can actually be a reduce-scatter.)  Notably,
  operations involving invariant tensors correspond directly to Megatron-style
  autograd functions like `CopyToModelParallelRegion`: when you
  `reinterpret(I,R)` a parameter for use in computation, the backward pass
  does an all-reduce (`all_reduce(I): P -> I`), giving you the synchronized
  gradient you need for the optimizer step.

* Varying means that tensor has different values across the device mesh axis.
  The gradient of varying is varying.  We don't "know" how the tensor is
  supposed to be reassembled together, just that each rank has different data.
  Semantically, we have is a list of tensors (one per rank), and operations are
  done per-element.

* Partial means that there is a pending sum on the (differing) values
  across the device mesh axis, but it hasn't happened yet.  Delaying the reduction
  can be profitable as it might be possible to do a reduce-scatter or eliminate the
  reduction entirely.  The gradient of partial is replicate.

**Note on replicate vs invariant:** these types have identical *forward* values
(each rank holds the same data), but they encode different *backward* semantics.
A replicate tensor allows each rank to contribute distinct local gradients that
must be aggregated (e.g., via all-reduce/reduce-scatter), while an invariant
tensor requires the gradient itself to be identical on every rank (no implicit
summation).  The distinction reflects how the value is *used* and how gradients
should be interpreted, not how the forward value is computed.

To summarize the forward-backward relationship of these types:

```
Forward     Backward
----------------------
Replicate   Partial
Partial     Replicate
Invariant   Invariant
Varying     Varying
```

The typing rules for non-comms ops are constrained by the fact we NEVER do
communication on these operators; so the type system forbids combinations of
types that would require comms in backwards to get correct gradients.
Specifically, here are the valid combinations of states:

```
op(Replicate..) -> Replicate
op(Invariant..) -> Invariant  # NB: this case is uncommon
op(Varying..) -> Varying
linear_op(Partial) -> Partial

# Invariant not allowed to mix with other types
op(Replicate, Varying) -> Varying
Partial + Partial -> Partial
```

For n-ary operations involving Partial, only addition (and more generally,
multilinear operations) can accept multiple Partial arguments.  This is because
addition distributes over the pending sum: (sum of a) + (sum of b) = sum of (a + b).
But multiplication does NOT distribute: (sum of a) * (sum of b) != sum of (a * b).
To see why concretely, consider two partial tensors each with value [1] on two
ranks.  Their denotations are both 1+1=2, so the product should be 4.  But if we
locally multiply and then reduce, we get 1*1 + 1*1 = 2, which is wrong.
Therefore, `Partial * Partial` is forbidden; you must all-reduce at least one
operand first.

TODO: Prove that these rules are correct

For comms ops, this type system requires three changes to the classical
distributed APIs:

* The preexisting comms APIs are augmented with src/dst arguments when needed, to
  disambiguate the local SPMD type of their input/outputs.  The mechanical
  reason for this is that the backwards of these comm functions depends on
  what the local SPMD types are, so passing this info helps us choose the
  right autograd Function!  These arguments not only take the local SPMD types,
  but they also (suggestively) take `Shard(tensor_dim)` as an argument.  Note that
  when multiple mesh axis shard the same tensor dim, operations are only valid
  for operating on the *last* mesh axis (see the global SPMD APIs for a better
  way for working in this situation.) In local SPMD, this is how you swap
  between stack/concat semantics (described in more detail on the functions.)
  For brevity, we will refer to the src/dst arguments by initial; e.g., R, P,
  V, I and S(i).  We will also suggestively use S(i) to describe the types
  when this is used, but in the local SPMD type system this is equivalent to V.

* We add a new `reinterpret` operator, which represents all situations where we
  you directly coerce from one local SPMD type to another without any change
  to the local tensor and without any comms.  Changing the local SPMD type
  of a tensor can change its semantic meaning (e.g., when [3.0] is replicated
  across a mesh axis, its denotation is [3.0]; but if you reinterpret this as
  partial, its denotation is [3.0 * mesh_axis_size]).  In general,
  reinterpreting can have a nontrivial backwards that requires comms.  In JAX,
  these operations are called `pcast`.

* We add a new `convert` operator, which represents situations where we change
  from one local SPMD type to another while preserving the semantics of the
  tensor.  Like `reinterpret`, we guarantee no communications will be
  issued when doing this; but we may need to do local tensor operations.  To
  take the previous example, to convert from replicate to partial must zero
  out all but one of the tensors on the mesh axis, so that summing them up
  results in the original value.  When a tensor is varying, it is ambiguous
  what the "semantics" of the tensor are, but by convention we interpret
  varying as concatenation on dim 0, following the natural behavior of
  collectives like all-gather and reduce-scatter.  Like `reinterpret`, these
  operations can have nontrivial backwards.

For ordinary local Torch ops (e.g., einsum, matmul, elementwise ops, and
reductions over tensor dimensions like sum), there is no cross-rank
communication, so they do not take src/dst arguments.  Their local SPMD types
propagate from their inputs according to the typing rules above, and any
nontrivial type changes must be made explicit via the primitives above.

TODO: Show local SPMD rules work even when the shapes across ranks are not the
same.  I think you need to have some special collectives in this case.

## Comms by API

Let's describe all of these operators in more detail.

To help understand the meaning of these operations, we will provide two things
for each function:

* A *specification* of the function (e.g., `all_gather_spec`), written in
  terms of explicit lists of tensors across all ranks.  This specification
  specifies what happens on the denotations of the inputs/outputs; notably,
  a partial input arrives at the specification already summed (because that's
  what semantically it represents.)

* A diagrammatic representation of this function, showing the flow of data
  across both the device mesh axis as well as a tensor dim axis.  Each of these
  operators will have a diagram that shows the action of collectives on 1D
  tensors on three ranks.  This representation accurately represents pending
  reductions.

Here is an example of the diagram:

```
[A]
[B]  =>  [A, B, C]
[C]
```

To explain in more detail:

* The double arrow `=>` shows the before and after an operation.
* Each row (line) specifies the content of a different rank.
* A variable denotes a tensor (potentially scalar).  So for example, `[A]`
  denotes at least a 1-D tensor, while `A` can be any dimensionality.  When
  thinking about the examples, it can be helpful to imagine `A` as a 0-d
  scalar tensor to avoid worrying too much about the higher dimensional case.
* If a tensor is written on only one row, all other rows have the same contents.
  (Omitting the replicated rows helps convey the meaning that there really
  is only semantically *one* value in this situation, even though there
  are several replicas.)

So this diagram says we start off with rank 0: [A], rank 1: [B] and rank 2: [C],
and after the operation all three ranks have [A, B, C] (aka an all-gather).

If the tensors have a leading plus, e.g., `+[A, B, C]`, this means there is a
pending reduction on the mesh axis, so even though there are different values
across the ranks, the denotation of this state is a single tensor you would
get from summing up all of the tensors.  So for example, these two diagrams
are semantically equivalent (and you physically get from the left to the right
with an all-gather):

```
+[A]
+[B]  ==  [A + B + C]
+[C]
```

When we write the backwards for ops, we will reverse the double arrow
(`grad_input <= grad_output`) for symmetry with the forward diagram
(`input => output`).  The shape of `input` and `grad_input` must match
(both in terms of the tensor shape, as well as whether or not there are
semantically one or many values across the mesh axis), so it's a good
way of checking if you have written the correct backwards.

When referencing operators, we may refer to them compactly by removing `x` and
`mesh_axis` from the function signature, and using R/V/P/I to abbreviate the
local SPMD type passed into src/dst arguments.

OK, without further ado!

TODO: In general, the Varying functions can be generalized to take a tensor dim
argument so that they do an action on not tensor dim 0 but somewhere else.

MAJOR TODO: Need to consider how to support collectives on multiple mesh axes at the same time

### `all_gather(x: Varying, mesh_axis, src, dst) -> Replicate | Invariant`

```
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
```

Gather shards along the mesh axis, so that every rank has the full copy of the
data.  PyTorch's `all_gather` can either concat or stack inputs together.  When
the source tensor is interpreted as a varying tensor, we stack the inputs together,
creating a new dimension of size mesh axis.  But if the source tensor is a sharded
tensor, we concat along the sharded dimension.

The backwards for each case:

`all_gather(V,R): V -> R`, the backwards is `reduce_scatter(V): P -> V`.

```
Ax + Ay + Az      +[Ax, Bx, Cx]
Bx + By + Bz  <=  +[Ay, By, Cy]
Cx + Cy + Cz      +[Az, Bz, Cz]
```

`all_gather(V,I): V -> I`, the backwards is `convert(I,V): I -> V`.

```
A
B  <=  [A, B, C]
C
```

`all_gather(S(i),R): S(i) -> R`, the backwards is `reduce_scatter(S(i)): P -> S(i)`.

```
When i == 0:
[A0x + A0y + A0z, A1x + A1y + A1z]      +[A0x, A1x, B0x, B1x, C0x, C1x]
[B0x + B0y + B0z, B1x + B1y + B1z]  <=  +[A0y, A1y, B0y, B1y, C0y, C1y]
[C0x + C0y + C0z, C1x + C1y + C1z]      +[A0z, A1z, B0z, B1z, C0z, C1z]
```

`all_gather(S(i),I): S(i) -> I`, the backwards is `convert(I,S(i)): I -> S(i)`.

```
[A0, A1]
[B0, B1]  <=  [A0, A1, B0, B1, C0, C1]
[C0, C1]
```

### `all_reduce(x: Partial, mesh_axis, dst) -> Replicate | Invariant`

```
def all_reduce_spec(x: f32[*shape]) -> f32[*shape]:
    return x  # Identity! The summation already occured on x's conversion to Partial

+Ax
+Ay  =>  Ax + Ay + Az
+Az
```

Reduce shards along the mesh axis, so that every rank has the full summed value.

When `dst='replicate'`, aka `all_reduce(R): P -> R`, the backwards is `all_reduce(R): P -> R`

```
                  +Ax
Ax + Ay + Az  <=  +Ay
                  +Az
```

When `dst='invariant'`, aka `all_reduce(I): P -> I`, the backwards is `reinterpret(I,R): I -> R`

```
A  <=  A
```

It is common to want to `all_reduce` on varying data; just `reinterpret(V,P)` the data
as partial before calling `all_reduce`.

TODO: I think with erasure we can infer this cast

### `reduce_scatter(x, mesh_axis, dst): Partial -> Varying`

```
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

```

Reduce shards along the mesh axis, but only get one shard of the result (e.g.,
an inefficient implementation of reduce-scatter would be to all-reduce and
then drop the data you did not need.)  Like `all_gather`, `dst` can either
be varying for stack semantics, or shard for concat semantics.

The backwards for each case:

`reduce_scatter(V): P -> V`, the backwards is `all_gather(V,R): V -> R`.

```
               A
[A, B, C]  <=  B
               C
```

`reduce_scatter(S(i)): P -> S(i)`, the backwards is `all_gather(S(i),R): S(i) -> R`.

```
When i == 0:
                              [A0, A1]
[A0, A1, B0, B1, C0, C1]  <=  [B0, B1]
                              [C0, C1]
```

It is common to want to reduce-scatter on varying data; just `reinterpret(V,P)`
the data as partial before calling `reduce_scatter`.

### `all_to_all(x, mesh_axis, src, dst): Varying -> Varying`

```
def all_to_all_spec(xs, src, dst):
    # An all-to-all transposes a mesh axis with a tensor axis.

    '''
    Diagram for all_to_all(V,V):
    [A0, A1, A2]      [A0, B0, C0]
    [B0, B1, B2]  =>  [A1, B1, C1]
    [C0, C1, C2]      [A2, B2, C2]

    To reason about all_to_all(S(i),S(j)), its best decompose
    it to all_gather(S(i),R) and convert(R,S(j)), or to just
    think of its local SPMD semantics as unsharding on one axis and resharding
    on another.
    '''

    match src, dst:
        case V, V:
            x = torch.stack(xs)
            x = x.transpose(0, 1)
            return x.unbind()

        case S(i), S(j):
            x = torch.concat(xs, i)
            return x.chunk(mesh_axis_size, j)

        # TODO: does V,S(j) or S(i),V make sense?
```

The varying and shard versions of `all_to_all` are pretty different (even though
under the hood they both have an all-to-all communication pattern), so we'll
describe them separately.

* `all_to_all(V,V)` transposes the logical mesh axis with the dim 0 local tensor axis
  (TODO: we can hypothetically split on not dim 0 but for the non-contiguous
  reads to be fused we'd need a special collective which NCCL doesn't
  directly support).

* `all_to_all(S(i),S(j))` intuitively unshards the tensor on dim i, and then
  reshards it on dim j (but skips actually doing the all-gather).

The forwards is `V -> V`, the backwards is `all_to_all: V -> V` (with src/dst
and split_dim/concat_dim swapped).

The backwards for each case:

`all_to_all(V,V): V -> V`, the backwards is `all_to_all(V,V): V -> V`:

```
[A0, A1, A2]      [A0, B0, C0]
[B0, B1, B2]  <=  [A1, B1, C1]
[C0, C1, C2]      [A2, B2, C2]
```

`all_to_all(S(i),S(j)): S(i) -> S(j)`, the backwards is `all_to_all(S(j),S(i)): S(j) -> S(i)`.

### `reinterpret(x, mesh_axis, src, dst)`

Coerce from one local SPMD type to another local SPMD type without changing
the local tensor.  It is guaranteed to be a no-op in forwards.

**Important:** Unlike `convert`, `reinterpret` can change the semantic value of
a tensor.  For example, `reinterpret(R,P)` treats a replicated value as if it
were partial, meaning after reduction you get N times the original value (where
N is the mesh axis size).  If you want to preserve semantics, use `convert`.

Here are the supported coercions:

`reinterpret(R,I): R -> I`, the backwards is `convert(I,P): I -> P`

```
def reinterpret_R_I_spec(x: f32[*shape]) -> f32[*shape]:
    return x

Forward:
A  =>  A

Backward:
+A
+0  <=  A
+0
```

`reinterpret(R,V): R -> V`, the backwards is `reinterpret(V,P): V -> P`

```
def reinterpret_R_V_spec(x: f32[*shape]) -> List[f32[*shape]]:
    # Makes N copies of the input
    return [x] * mesh_axis_size

Forward:
       A
A  =>  A
       A

Backward:
+A      A
+B  <=  B
+C      C
```

`reinterpret(I,R): I -> R`, the backwards is `all_reduce(I): P -> I`

```
def reinterpret_I_R_spec(x: f32[*shape]) -> f32[*shape]:
    return x

Forward:
A => A

Backward:
                  +Ax
Ax + Ay + Az  <=  +Ay
                  +Az
```

`reinterpret(V,P): V -> P`, the backwards is `reinterpret(R,V): R -> V`

```
def reinterpret_V_P_spec(xs: List[f32[*shape]]) -> f32[*shape]:
    # Semantically does a sum, even if physically it hasn't happened yet!
    return sum(xs)

Forward:
A      +A
B  =>  +B
C      +C

Backward:
A
A  <=  A
A
```

`reinterpret(R,P): R -> P`, the backwards is `reinterpret(R,P): R -> P`

```
def reinterpret_R_P(x: f32[*shape]) -> f32[*shape]:
    # Summing each replicated entry together scales the value by axis size
    return x * mesh_axis_size

Forward:
       +A
A  =>  +A
       +A

Backward:
+A
+A  <=  A
+A
```

`reinterpret(I,V): I -> V` is the composition of `I -> R -> V`.  `reinterpret(R,P): R -> P`
is the composition of `R -> V -> P`.  `reinterpret(I,P): I -> P` is the composition of
`I -> R -> P`.  Note that these reinterprets have unusual semantics: the
resulting tensor has been scaled by the mesh axis size (because you are now
obligated to sum each of the (equal) quantities of the rank together!) If you
instead wanted to *preserve* the original semantic meaning of the tensor, use
`convert`.

This API does not support shard for src/dst, because the restriction on no local tensor
change means that the local SPMD semantics would be precisely the same as the corresponding
varying operation.

Here is a table of permissible reinterprets (`-` is no-op, `X` is direct
coercion, `/` is transitive coercion.)

```
       dst
       R I V P
src R  - X X /
    I  X - / /
    V      - X
    P        -
```

### `convert(x, mesh_axis, src, dst)`

Convert from one local SPMD to another while preserving the semantics of the
tensor, without doing communications.  When src/dst is shard, this means
preserving the global SPMD semantics of the tensor per this sharding; when
src/dst is varying, this means preserving the local SPMD semantics (we simply
say that a non-varying tensor can be interpreted as varying by unbinding dim
0, following the natural behavior of collectives like all-gather and
reduce-scatter that stack/unbind on dim 0).  The shard/varying conversions
are actually exactly identical, except for being rank-preserving or not, so
in the summary tables we will only include the varying versions of operations.
However, we include both in the API description for clarity.

Here are the supported conversions:

`convert(R,V): R -> V`, the backward is `convert(V,P) : V -> P`

Input is replicated across ranks, so each rank holds the full tensor.  The
output keeps only the local shard along tensor dim 0, producing a varying
value.  This operation reduces the rank of the tensor.

```
def convert_R_V_spec(x: f32[mesh_axis_size, *shape]) -> List[f32[*shape]]:
    return x.unbind()

Forward:
               A
[A, B, C]  =>  B
               C

Backward:
+[A, 0, 0]      A
+[0, B, 0]  <=  B
+[0, 0, C]      C
```

`convert(R,S(i)): R -> S(i)`, the backward is `convert(S(i),P) : S(i) -> P`

Like above, but the rank of the tensor is not reduced, and an arbitrary tensor
dim can be specified to be sharded.

```
def convert_R_S_spec(x, i):
    return x.chunk(mesh_axis_size, i)


Forward (for i = 0):
                              [A0, A1]
[A0, A1, B0, B1, C0, C1]  =>  [B0, B1]
                              [C0, C1]

Backward (for i = 0):
+[A0, A1, 0,  0,  0,  0 ]      [A0, A1]
+[0,  0,  B0, B1, 0,  0 ]  <=  [B0, B1]
+[0,  0,  0,  0,  C0, C1]      [C0, C1]
```

`convert(I,V): I -> V`, the backwards is `all_gather(V,I): V -> I`

Input is invariant across ranks, so each rank holds the full tensor.  The
output keeps only the local slice along dim 0, producing a varying
value.  The rank of the tensor is reduced.

```
def convert_I_V_spec(x: f32[mesh_axis_size, *shape]) -> List[f32[*shape]]:
    return x.unbind()

Forward
               A
[A, B, C]  =>  B
               C

Backward:
               A
[A, B, C]  <=  B
               C
```

`convert(I,S(i)): I -> S(i)`, the backwards is `all_gather(S(i),I): S(i) -> I`

Like above, but the rank of the tensor is not reduced, and an arbitrary
tensor dim can be specified to be sharded.

```
def convert_I_S_spec(x, i):
    return x.chunk(mesh_axis_size, i)

Forward (for i = 0):
                              [A0, A1]
[A0, A1, B0, B1, C0, C1]  =>  [B0, B1]
                              [C0, C1]

Backward (for i = 0):
                              [A0, A1]
[A0, A1, B0, B1, C0, C1]  <=  [B0, B1]
                              [C0, C1]
```


`convert(R,P): R -> P`, the backwards is `convert(R,P) : R -> P`

Input is replicated across ranks.  The output keeps the same per-rank tensor
shape, but all ranks except the first are zeroed out, producing a partial value
that sums to the original tensor after a cross-rank reduction.

```
def convert_R_P_spec(x: f32[*shape]) -> f32[*shape]:
    return x

Forward:
         +[A]
[A]  =>  +[0]
         +[0]

Backward:
+[A]
+[0]  <=  [A]
+[0]
```

`convert(I,P): I -> P`, the backwards is `reinterpret(R,I): R -> I`

Input is invariant across ranks.  The output keeps the same per-rank tensor
shape, but all ranks except the first are zeroed out, producing a partial value
that sums to the original tensor after a cross-rank reduction.

```
def convert_I_P_spec(x: f32[*shape]) -> f32[*shape]:
    return x

Forward:
         +[A]
[A]  =>  +[0]
         +[0]

Backward:
[A]  <=  [A]
```

`convert_spec(V,P): V -> P`, the backwards is `convert(R,V): R -> V`

Input is varying, with each rank holding a shard or distinct value.  The output
places each rank's value into a disjoint position of a partial tensor (zeros
elsewhere) so that summing across ranks reconstructs the stacked value.
The rank of the tensor is increased.

```
def convert_V_P_spec(xs: List[f32[*shape]]) -> f32[mesh_axis_size, *shape]:
    return torch.stack(xs)

Forward:
A      +[A, 0, 0]
B  =>  +[0, B, 0]
C      +[0, 0, C]

Backward:
A
B  <=  [A, B, C]
C
```

`convert_spec(S(i),P): S(i) -> P`, the backwards is `convert(R,S(i)): R -> S(i)`

Like above, but the rank of the tensor is not reduced, and an arbitrary tensor
dim can be specified to be scattered on.

```
def convert_S_P_spec(xs, i):
    return torch.concat(xs, i)

Forward (for i = 0):
[A0, A1]      +[A0, A1, 0,  0,  0,  0 ]
[B0, B1]  =>  +[0,  0,  B0, B1, 0,  0 ]
[C0, C1]      +[0,  0,  0,  0,  C0, C1]

Backward (for i = 0):
[A0, A1]
[B0, B1]  <=  [A0, A1, B0, B1, C0, C1]
[C0, C1]
```

You cannot convert out of P: the only way to eliminate the pending reduction
is to do the actual all-reduce.

For convenience, we also support convert(R,I) and convert(I,R), which have the
same meaning as reinterpret(R,I) and reinterpret(I,R).

Here is a table of permissible converts (`-` is no-op, `O` is supported, `X` is when
the semantics is the same as reinterpret.)

```
       dst
       R I V P
src R  - X O O
    I  X - O O
    V      - O
    P        -
```

## Comms by state transition

Here is a table comprehensively organizing all of the operators above by state
transition, omitting operators which are done by composition.  Note that type
does NOT uniquely determine an operator: there can be multiple ways to get
from one type to another which have different semantics.


```
      \   DST   Replicate           Invariant           Varying             Partial
SRC    \----------------------------------------------------------------------------------------
Replicate       -                   reinterpret(R,I)    reinterpret(R,V)    reinterpret(R,P)
                                                        convert(R,V)        convert(R,P)

Invariant       reinterpret(I,R)    -                   reinterpret(I,V)    convert(I,P)
                                                        convert(I,V)

Varying         all_gather(R)       all_gather(I)       all_to_all()        reinterpret(V,P)
                                                                            convert(V,P)

Partial         all_reduce(R)       all_reduce(I)       reduce_scatter()    -
```

## Comms by forwards-backwards

Here is a table that summarizes the forward-backward relationships between all
of these operators.  Remember that R in forwards goes to P in backwards, and
vice versa.  Also remember that the backwards of a backwards is its forwards,
so you can read the table left-to-right and right-to-left (but for ease of
reading, we've included the flipped rows explicitly).

Fwd Type    Forward                 Bwd Type    Backward
----------------------------------------------------------------------------
R -> I      reinterpret(R,I)        I -> P      convert(I,P)
R -> V      reinterpret(R,V)        V -> P      reinterpret(V,P)
            convert(R,V)                        convert(V,P)
R -> P      reinterpret(R,P)        R -> P      reinterpret(R,P)
            convert(R,P)                        convert(R,P)
I -> R      reinterpret(I,R)        P -> I      all_reduce(I)
I -> V      convert(I,V)            V -> I      all_gather(I)
I -> P      convert(I,P)            R -> I      reinterpret(R,I)
V -> R      all_gather(R)           P -> V      reduce_scatter()
V -> I      all_gather(I)           I -> V      convert(I,V)
V -> V      all_to_all()            V -> V      all_to_all()
V -> P      reinterpret(V,P)        R -> V      reinterpret(R,V)
            convert(V,P)                        convert(R,V)
P -> R      all_reduce(R)           P -> R      all_reduce(R)
P -> I      all_reduce(I)           I -> R      reinterpret(I,R)
P -> V      reduce_scatter()        V -> R      all_gather(R)

# Global SPMD

What is a global SPMD type?  We take the existing local SPMD type, and augment
it with a partition spec that says how varying mesh dimensions should be
reassembled together into the full tensor.  A partition spec is a tuple
whose size matches the tensor's dimension.  For each dim, it specifies
zero, one or several mesh axes which shard that dimension.  It is common to
print the partition spec along with the shape of a tensor; so for example,
`f32[8,16@tp]` says that dim=1 of the tensor has been sharded by the "tp"
mesh axis.  It is only legal for varying mesh dimensions to occur in the
partition spec.

(Aside: already in the local SPMD API, we have also made available a Shard(i)
for expressing sharding on a per-device mesh basis.  This form can be more
convenient when doing global SPMD transformations on a per-mesh axis basis,
but it comes with a gotcha: you can only manipulate the *last* mesh axis that
shards a particular mesh dimension.  Many people find the JAX-style tensor dim
oriented partition specmore intuitive to work with, and the global SPMD APIs
will emphasize this.)

We continue to do local SPMD type checking as described above.  Our new
problem is to describe the shard propagation rules for partition spec.
Unfortunately, unlike in local SPMD, this must be done on a per operator basis
(for both non-comms and comms operators).

Local SPMD and global SPMD can seamlessly interoperate with each other.  If
you enter a local map (aka `shard_map`) region, you simply forget the
partition spec; when you want to reenter global SPMD, you simply have to
specify how the tensor should be reassembled into a full tensor.  We make the
following choices for how local and global SPMD types can interact:

1. Switching between local and global SPMD is done via a higher order function
   like `local_map` or `shard_map` (as opposed to having two kinds of tensor that
   propagate through the program, ala `from_local` and `to_local`).  Either mode
   of operation can be supported, but we think it is easier to reason about
   global versus local SPMD over an entire scope, rather than having to reason
   about whether or not a particular tensor is global or local SPMD on a
   fine-grained
   basis.

2. Global versus local SPMD semantics are done on a per mesh axis basis
   (similar to JAX).  So you can switch to local SPMD semantics for only one mesh
   axis (e.g., the EP axis) while staying in global SPMD for another (e.g., the
   EP_TP axis.)

One of our design goals is you can simply forget the partition spec, decaying
a global SPMD type into a local SPMD type, and still have a well-typed
program.  One consequence of this is, unlike DTensor, classic operations like
sum(), matmul() and einsum() will not automatically work in cases where an
operation can be done completely locally except for a pending reduction.


## Shard propagation

We can think of partition spec as a function which takes a full tensor to a
mesh of sharded tensors.  The question of shard propagation is, given an input
PartitionSpec `in_spec`, does there exist an output PartitionSpec `out_spec`
such that this equality holds:

```
map(f, in_spec(x)) == out_spec(f(x))
```

A classic shard propagation rule is the one for `einsum`.  The following conditions
must hold:

1. No contracted dimension is sharded (but see the following discussion).
2. For each mesh axis, if an output label is sharded on that mesh axis, then every
   operand either uses the label and is sharded on that axis, or doesn't use that
   label and is not sharded on that axis.
3. No repeated mesh axes are allowed on an array's spec.

When these conditions succeed, at runtime we can simply run einsum as we would
have run it in local SPMD, and know that there is an appropriate
interpretation of the output where we computed the global SPMD semantics.
When sharding propagation errors, it means that the global SPMD semantics and
local SPMD semantics align, and comms are needed to ensure we actually compute
global SPMD semantics.

Let's talk about local operators that can produce partial.  When a contracted
dimension is sharded, we can in principle do the operation entirely locally by
declaring that there is a pending reduction.  This is PyTorch DTensor's
behavior by default; it will transparently generate partial reductions from
ordinary operations when warranted.  However, we find this problematic for two
reasons:

1. First, implicitly converting the varying output to partial, when necessary
   to have correct global SPMD semantics, would mean that behavior in local
   SPMD and global SPMD regions differ.  This is doable; we simply need to
   know if we are global SPMD or local SPMD at runtime, but it is aesthetically
   displeasing because, for the most part, global SPMD is just a strict
   subset of local SPMD operations.

2. Second, in discussions with people who have traditionally programmed in local
   SPMD (that's most people in PyTorch!) and then have tried out global SPMD
   via DTensor, they have generally found that understanding when partial pops
   up and how it propagates to be quite confusing.  Usually, the story goes that
   they just wrote some code, and then they're debugging why extra collectives
   have occurred, and they only then realize that there are some partial tensors
   floating around.

So we take a different approach: you must explicitly specify when you want an
partial output on these operators.  Because this ambiguity only arises for
partial outputs, we expose this as simply as a new keyword argument
`out_partial_axes` which is a set of device mesh axes to be partial on.  In
local SPMD, the semantics of this argument is to do the local operation and
then `reinterpret` the result as partial on each of the out partial axes.
However, this must be done all in one step in global SPMD, since the
intermediate local operation without reinterprets is not valid in global SPMD.

### Worked example comparing local SPMD and global SPMD

An illustrative example highlighting the difference between local and global
SPMD is what happens when we perform the matmul in row-parallel linear
Megatron-style TP.  In this setting, both the intermediate and the weight are
sharded on the TP mesh axis (aka varying), and after this matmul we need to
perform an all-reduce to compute the real value of the linear.  To correctly
express that we want to do a *global* matrix multiply, you must run *two*
operations in local SPMD:

```
out: Varying = linear(hidden: Varying, weight: Varying)
out2: Partial = reinterpret(out: Varying, to='partial')
```

If you only run a linear in local SPMD, you have asked to perform only a local
matrix multiply per rank; the local semantics of a matmul do NOT imply a global
reduction over all ranks.  The reinterpret is necessary to indicate, "Actually, I do
want a global matmul!"

In global SPMD, we would instead error on the linear call.  Instead, you would write:

```
out2 = linear(hidden, weight, out_partial_axes='tp')
```

### Shard propagation for comms operators

In the API description for comms operators, operators could operate on both
Varying (non-rank preserving) and Shard (rank preserving) src/dst.  In global
SPMD, only the Shard variants have shard propagation rules; the operators that
operate on varying have an ambiguity on what to do with the added/removed
rank (you can use those versions by dropping into local SPMD, and then when
returning to global SPMD explicitly specifying what your new desired global SPMD
type is).

The global SPMD interpretation of convert (when it is defined for global SPMD)
is straightforward: it is the identity function.

### Per-mesh-dim redistribute

It is helpful to have a version of `convert` that is semantics preserving but
allows for comms.  We will call this `redistribute`.  It routes to the following collectives:

```
redistribute(S(i),R)    =   all_gather(S(i),R)
redistribute(S(i),I)    =   all_gather(S(i),I)
redistribute(P,R)       =   all_reduce(P,R)
redistribute(P,I)       =   all_reduce(P,I)
redistribute(P,S(i))    =   reduce_scatter(P,S(i))
redistribute(S(i),S(j)) =   all_to_all(S(i),S(j))
```

Once again, these only work if the mesh axis is the LAST to shard a particular
tensor dimension.  We will introduce a better partition spec API below.

### Partition spec redistribute

The above API has two problems:

* When multiple mesh axes shard the same dimension, you cannot freely do per mesh
  axes on it; only the *last* mesh axis can be operated on.  For example, if you want
  to reshard `f32[8@dp,tp]` to `f32[8@tp,dp]`, you have to first gather on tp,
  all-to-all to interchange dp with tp, and then shard on dp.

* It is inefficient to do redistribute on a mesh-axis by mesh-axis basis; for example,
  if we need to do an all-reduce on both "dp" and "tp" dimension, it is much better to
  do a single all-reduce on a communicator for the flattened device mesh.

So we should also support a convenience API `redistribute(src_partition_spec, dst_partition_spec)`,
which plans the sequence of collectives needed to arrive at the destination partition spec,
and will flatten collectives together as possible.

## Miscellaneous design notes

From YZ: in global spmd, can represent partial as an extra dimension, hidden with vmap

  (NB: If the type system did not need to be erasable, we could elide the src
  argument from all comms, as we could simply read out the type from the
  incoming tensor; however, due to erasure src type also has to be specified
  when its ambiguous.)

Related work: https://arxiv.org/pdf/2506.15961

"""

# TODO: There may be inaccuracies in the code below as I am still working on
# it

from dataclasses import dataclass
from typing import Union

# =============================================================================
# Per-Mesh-Axis Local SPMD Type Hierarchy
# =============================================================================


class PerMeshAxisLocalSpmdType:
    """
    Base class for local SPMD types on a single mesh axis.

    Describes how a tensor is distributed across ranks on one axis of the
    device mesh, as well as how the gradients are distributed. The four
    concrete types are: Replicate (R), Invariant (I), Varying (V), Partial (P).
    """

    def backward_type(self) -> "PerMeshAxisLocalSpmdType":
        """Return the type that gradients have in the backward pass."""
        raise NotImplementedError


class Replicate(PerMeshAxisLocalSpmdType):
    """
    Replicate type - data is replicated across ranks.

    The gradient of replicate is partial.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "R"

    def backward_type(self) -> "PerMeshAxisLocalSpmdType":
        return P


class Invariant(PerMeshAxisLocalSpmdType):
    """
    Invariant type - data is replicated across ranks, gradient is also invariant.

    Unlike replicate, the gradient is expected to already be synchronized.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "I"

    def backward_type(self) -> "PerMeshAxisLocalSpmdType":
        return I


class Varying(PerMeshAxisLocalSpmdType):
    """
    Varying type - data differs across ranks.

    The gradient of varying is also varying.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "V"

    def backward_type(self) -> "PerMeshAxisLocalSpmdType":
        return V


class Partial(PerMeshAxisLocalSpmdType):
    """
    Partial type - pending sum across ranks.

    The gradient of partial is replicate.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "P"

    def backward_type(self) -> "PerMeshAxisLocalSpmdType":
        return R


@dataclass(frozen=True)
class Shard:
    """
    Shard type - varying on a specific tensor dimension.

    Like Varying but rank-preserving; operates on a specific tensor dimension
    rather than adding/removing dimensions. Used for concat semantics in
    collectives (vs stack semantics for V).

    The gradient of Shard(i) is also Shard(i).
    """

    dim: int

    def __repr__(self):
        return f"S({self.dim})"

    def backward_type(self) -> "Shard":
        return self


# Single character singletons for ease of pattern matching
R = Replicate()
I = Invariant()
V = Varying()
P = Partial()
S = Shard  # S(i) creates a Shard with dim=i

# Type aliases
PerMeshAxisSpmdType = Union[PerMeshAxisLocalSpmdType, Shard]
LocalSpmdType = dict[str, PerMeshAxisSpmdType]
# =============================================================================
# PartitionSpec for Global SPMD
# =============================================================================


class PartitionSpec(tuple):
    """
    A partition spec describes how tensor dimensions map to mesh axes.

    Each element corresponds to a tensor dimension and specifies zero, one, or
    multiple mesh axis names that shard that dimension. For example:
        - PartitionSpec('tp', None) means dim 0 is sharded on 'tp', dim 1 is replicated
        - PartitionSpec(('dp', 'tp'), None) means dim 0 is sharded on both 'dp' and 'tp'
        - PartitionSpec() means fully replicated

    When printed with a tensor shape, we use notation like f32[8@tp,16] to show
    that dim 0 (size 8) is sharded on 'tp' while dim 1 (size 16) is replicated.
    """

    def __new__(cls, *args: str | tuple[str, ...] | None):
        return super().__new__(cls, args)

    def __repr__(self):
        if not self:
            return "PartitionSpec()"
        parts = []
        for s in self:
            if s is None:
                parts.append("None")
            elif isinstance(s, tuple):
                parts.append(f"({', '.join(repr(x) for x in s)})")
            else:
                parts.append(repr(s))
        return f"PartitionSpec({', '.join(parts)})"

    def get_mesh_axes(self) -> set[str]:
        """Return all mesh axis names mentioned in this spec."""
        axes = set()
        for s in self:
            if s is None:
                continue
            elif isinstance(s, tuple):
                axes.update(s)
            else:
                axes.add(s)
        return axes

    def is_replicated(self) -> bool:
        """Return True if all dimensions are replicated (no sharding)."""
        return all(s is None for s in self)


GlobalSpmdType = tuple[LocalSpmdType, PartitionSpec]

class LTensor:
    """
    A tensor with local SPMD type annotations.

    Tracks the local SPMD type (R, I, V, P, or S(i)) for each mesh axis.
    This enables type-checking to ensure distributed operations compute
    mathematically correct gradients.

    Attributes:
        data: The underlying torch.Tensor
        types: LocalSpmdType (dict mapping mesh axis names to PerMeshAxisSpmdType)
    """

    def __init__(self, data: torch.Tensor, types: LocalSpmdType | None = None):
        """
        Create an LTensor with optional type annotations.

        Args:
            data: The underlying tensor
            types: LocalSpmdType - dict mapping mesh axis names to
                   PerMeshAxisSpmdType (R, I, V, P, or S(i))
        """
        self.data = data
        self.types: LocalSpmdType = types if types is not None else {}
        for axis_name, typ in self.types.items():
            if not isinstance(typ, (PerMeshAxisLocalSpmdType, Shard)):
                raise TypeError(
                    f"Invalid type '{typ}' for axis '{axis_name}'. "
                    f"Must be a PerMeshAxisSpmdType (R, I, V, P, or S(i))"
                )

    def get_type(self, axis_name: str) -> PerMeshAxisSpmdType | None:
        """Get the local SPMD type for a mesh axis, or None if not tracked."""
        return self.types.get(axis_name)

    def with_type(self, axis_name: str, typ: PerMeshAxisSpmdType) -> "LTensor":
        """Return a new LTensor with an updated type for the given axis."""
        if not isinstance(typ, (PerMeshAxisLocalSpmdType, Shard)):
            raise TypeError(f"Invalid type '{typ}'. Must be a PerMeshAxisSpmdType (R, I, V, P, or S(i))")
        new_types = self.types.copy()
        new_types[axis_name] = typ
        return LTensor(self.data, new_types)


def einsum(equation: str, *operands: LTensor, out_partial_axes: set[str] | None = None) -> LTensor:
    """
    Perform einsum with local SPMD type checking.

    The typing rule for einsum is simple: all operands must have matching types
    on each mesh axis, and the output inherits those types.

    For each mesh axis:
    - If all operands are R -> output is R
    - If all operands are I -> output is I
    - If all operands are V -> output is V
    - If all operands are P -> output is P (but beware: this is only
      mathematically correct for additive operations, not contractions)
    - Mixed R/V -> output is V
    - I cannot mix with other types
    - P cannot mix with non-P types

    Args:
        equation: The einsum equation string
        *operands: LTensor operands
        out_partial_axes: Optional set of mesh axis names. When specified, the
            output is reinterpreted as partial on each of these axes. This is
            used in global SPMD to indicate that a contracted dimension was
            sharded and the result needs a pending reduction. In local SPMD,
            this does the local operation then reinterpret(V,P) for each axis.

    Returns:
        LTensor with the result and inferred types
    """
    if out_partial_axes is None:
        out_partial_axes = set()

    # Extract underlying tensors
    tensors = [op.data for op in operands]

    # Collect all mesh axes mentioned
    all_axes = set()
    for op in operands:
        all_axes.update(op.types.keys())
    all_axes.update(out_partial_axes)

    # Type check and infer output types
    output_types = {}
    for axis in all_axes:
        axis_types = []
        for op in operands:
            typ = op.types.get(axis)
            if typ is not None:
                axis_types.append(typ)

        if not axis_types:
            # Axis only in out_partial_axes, treat as partial output
            if axis in out_partial_axes:
                output_types[axis] = P
            continue

        # Check type compatibility and infer output type
        # Normalize types to their base class for comparison
        type_classes = set(type(t) for t in axis_types)

        if len(type_classes) == 1:
            # All same type
            inferred_type = axis_types[0]
        elif type_classes == {Replicate, Varying}:
            # Mixed replicate/varying -> varying
            inferred_type = V
        elif Invariant in type_classes and len(type_classes) > 1:
            raise TypeError(
                f"Invariant type on axis '{axis}' cannot mix with other types. "
                f"Found types: {axis_types}"
            )
        elif Partial in type_classes:
            # Partial can only appear alone (for linear ops) or with another partial
            # (for bilinear ops like matmul)
            non_partial = type_classes - {Partial}
            if non_partial:
                raise TypeError(
                    f"Partial type on axis '{axis}' can only combine with partial. "
                    f"Found types: {axis_types}"
                )
            inferred_type = P
        else:
            raise TypeError(
                f"Incompatible types on axis '{axis}': {axis_types}"
            )

        # Apply out_partial_axes: reinterpret V as P
        if axis in out_partial_axes:
            if inferred_type is V or isinstance(inferred_type, Shard):
                output_types[axis] = P
            elif inferred_type is R:
                # R with out_partial becomes P (via reinterpret(R,P))
                output_types[axis] = P
            elif inferred_type is P:
                # Already partial
                output_types[axis] = P
            else:
                raise TypeError(
                    f"Cannot mark axis '{axis}' as partial with type {inferred_type}. "
                    f"out_partial_axes is only valid for V, S(i), R, or P types."
                )
        else:
            output_types[axis] = inferred_type

    # Perform the einsum
    result = torch.einsum(equation, *tensors)

    return LTensor(result, output_types)


# Global device mesh storage
_global_mesh = None


def set_mesh(mesh):
    """
    Set the global device mesh for distributed operations.

    Args:
        mesh: A DeviceMesh object that maps axis names to process groups.
              Must have a `get_group(axis_name)` method.
    """
    global _global_mesh
    _global_mesh = mesh


def get_mesh():
    """
    Get the current global device mesh.

    Returns:
        The global DeviceMesh, or None if not set.
    """
    return _global_mesh


def _get_mesh_axis_group(axis_name):
    """Get the process group for a mesh axis from the global mesh."""
    if _global_mesh is None:
        raise RuntimeError(
            "No global mesh set. Call set_mesh() with a DeviceMesh before using "
            "distributed operations."
        )
    return _global_mesh.get_group(axis_name)


class _ReplicateToVarying(torch.autograd.Function):
    """reinterpret(R,V): R -> V, backward is reinterpret(V,P): V -> P (no-op)."""

    @staticmethod
    def forward(ctx, x, axis_name):
        ctx.axis_name = axis_name
        return x

    @staticmethod
    def backward(ctx, grad_out):
        # reinterpret(V,P) is a no-op in forward direction
        return grad_out, None


class _AllReduceToReplicate(torch.autograd.Function):
    """all_reduce(R): P -> R, backward is all_reduce(R): P -> R."""

    @staticmethod
    def forward(ctx, x, axis_name):
        ctx.axis_name = axis_name
        pg = _get_mesh_axis_group(axis_name)
        return funcol.all_reduce(x, "sum", pg).wait()

    @staticmethod
    def backward(ctx, grad_out):
        # backward of P -> R is P -> R (same operation)
        pg = _get_mesh_axis_group(ctx.axis_name)
        return funcol.all_reduce(grad_out, "sum", pg).wait(), None


class _AllReduceToInvariant(torch.autograd.Function):
    """all_reduce(I): P -> I, backward is reinterpret(I,R): I -> R (no-op)."""

    @staticmethod
    def forward(ctx, x, axis_name):
        ctx.axis_name = axis_name
        pg = _get_mesh_axis_group(axis_name)
        return funcol.all_reduce(x, "sum", pg).wait()

    @staticmethod
    def backward(ctx, grad_out):
        # reinterpret(I,R) is a no-op
        return grad_out, None


def all_reduce(x, axis_name, *, src: PerMeshAxisSpmdType = P, dst: PerMeshAxisSpmdType):
    """
    Reduce shards along the mesh axis, so every rank has the full summed value.

    Args:
        x: Input tensor with P type on the mesh axis
        axis_name: The mesh axis to reduce over
        src: Source type (must be P)
        dst: Target type (R or I)

    Returns:
        Tensor with R or I type depending on dst

    When dst=R, backward is all_reduce(R): P -> R
    When dst=I, backward is reinterpret(I,R): I -> R (no-op)
    """
    if src is not P:
        raise ValueError(f"all_reduce src must be P, got {src}")
    if dst is R:
        return _AllReduceToReplicate.apply(x, axis_name)
    elif dst is I:
        return _AllReduceToInvariant.apply(x, axis_name)
    else:
        raise ValueError(f"all_reduce dst must be R or I, got {dst}")


# =============================================================================
# all_gather: V -> R | I
# =============================================================================


class _AllGatherToReplicate(torch.autograd.Function):
    """all_gather(R): V -> R, backward is reduce_scatter: P -> V."""

    @staticmethod
    def forward(ctx, x, axis_name, gather_dim):
        ctx.axis_name = axis_name
        ctx.gather_dim = gather_dim
        pg = _get_mesh_axis_group(axis_name)
        # Gather tensors from all ranks
        world_size = dist.get_world_size(pg)
        gathered = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x, group=pg)
        return torch.cat(gathered, dim=gather_dim)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is reduce_scatter: P -> V
        pg = _get_mesh_axis_group(ctx.axis_name)
        world_size = dist.get_world_size(pg)
        # Split grad_out and reduce-scatter
        chunks = torch.chunk(grad_out, world_size, dim=ctx.gather_dim)
        # Stack for reduce_scatter
        stacked = torch.stack(list(chunks), dim=0)
        output = torch.empty_like(chunks[0])
        dist.reduce_scatter(output, list(chunks), op=dist.ReduceOp.SUM, group=pg)
        return output, None, None


class _AllGatherToInvariant(torch.autograd.Function):
    """all_gather(I): V -> I, backward is convert(I,V): I -> V."""

    @staticmethod
    def forward(ctx, x, axis_name, gather_dim):
        ctx.axis_name = axis_name
        ctx.gather_dim = gather_dim
        pg = _get_mesh_axis_group(axis_name)
        world_size = dist.get_world_size(pg)
        gathered = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x, group=pg)
        return torch.cat(gathered, dim=gather_dim)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is convert(I,V): I -> V (just slice out local portion)
        pg = _get_mesh_axis_group(ctx.axis_name)
        world_size = dist.get_world_size(pg)
        rank = dist.get_rank(pg)
        chunks = torch.chunk(grad_out, world_size, dim=ctx.gather_dim)
        return chunks[rank].contiguous(), None, None


def all_gather(x, axis_name, *, src: PerMeshAxisSpmdType = V, dst: PerMeshAxisSpmdType, gather_dim: int = 0):
    """
    Gather shards along the mesh axis, so every rank has the full copy of data.

    ```
    [A]
    [B]  =>  [A, B, C]
    [C]
    ```

    Args:
        x: Input tensor with V or S(i) type on the mesh axis
        axis_name: The mesh axis to gather over
        src: Source type (V or S(i))
        dst: Target type (R or I)
        gather_dim: The tensor dimension to concatenate along (default: 0)

    Returns:
        Tensor with R or I type depending on dst

    When dst=R, backward is reduce_scatter: P -> V
    When dst=I, backward is convert(I,V): I -> V
    """
    # Validate src is V or S(i)
    if not (src is V or isinstance(src, Shard)):
        raise ValueError(f"all_gather src must be V or S(i), got {src}")

    if dst is R:
        return _AllGatherToReplicate.apply(x, axis_name, gather_dim)
    elif dst is I:
        return _AllGatherToInvariant.apply(x, axis_name, gather_dim)
    else:
        raise ValueError(f"all_gather dst must be R or I, got {dst}")


# =============================================================================
# reduce_scatter: P -> V
# =============================================================================


class _ReduceScatter(torch.autograd.Function):
    """reduce_scatter: P -> V, backward is all_gather(R): V -> R."""

    @staticmethod
    def forward(ctx, x, axis_name, scatter_dim):
        ctx.axis_name = axis_name
        ctx.scatter_dim = scatter_dim
        pg = _get_mesh_axis_group(axis_name)
        # Use functional reduce_scatter_tensor which works with LocalTensorMode
        return funcol.reduce_scatter_tensor(x, "sum", scatter_dim, pg).wait()

    @staticmethod
    def backward(ctx, grad_out):
        # backward is all_gather(R): V -> R
        pg = _get_mesh_axis_group(ctx.axis_name)
        return funcol.all_gather_tensor(grad_out, ctx.scatter_dim, pg).wait(), None, None


def reduce_scatter(x, axis_name, *, src: PerMeshAxisSpmdType = P, dst: PerMeshAxisSpmdType = V, scatter_dim: int = 0):
    """
    Reduce shards along the mesh axis, but only get one shard of the result.

    ```
    +[Ax, Bx, Cx]      [Ax + Ay + Az]
    +[Ay, By, Cy]  =>  [Bx + By + Bz]
    +[Az, Bz, Cz]      [Cx + Cy + Cz]
    ```

    Args:
        x: Input tensor with P type on the mesh axis
        axis_name: The mesh axis to reduce-scatter over
        src: Source type (must be P)
        dst: Target type (V or S(i))
        scatter_dim: The tensor dimension to scatter along (default: 0)

    Returns:
        Tensor with V or S(i) type depending on dst

    The backward is all_gather(R): V -> R
    """
    if src is not P:
        raise ValueError(f"reduce_scatter src must be P, got {src}")
    if not (dst is V or isinstance(dst, Shard)):
        raise ValueError(f"reduce_scatter dst must be V or S(i), got {dst}")

    return _ReduceScatter.apply(x, axis_name, scatter_dim)


# =============================================================================
# all_to_all: V -> V
# =============================================================================


class _AllToAll(torch.autograd.Function):
    """all_to_all: V -> V, backward is all_to_all: V -> V."""

    @staticmethod
    def forward(ctx, x, axis_name, split_dim, concat_dim):
        ctx.axis_name = axis_name
        ctx.split_dim = split_dim
        ctx.concat_dim = concat_dim
        pg = _get_mesh_axis_group(axis_name)
        world_size = dist.get_world_size(pg)
        # Split input
        input_chunks = list(torch.chunk(x, world_size, dim=split_dim))
        output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]
        dist.all_to_all(output_chunks, input_chunks, group=pg)
        return torch.cat(output_chunks, dim=concat_dim)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is also all_to_all (transpose back)
        pg = _get_mesh_axis_group(ctx.axis_name)
        world_size = dist.get_world_size(pg)
        # Split on concat_dim, gather on split_dim (reverse of forward)
        input_chunks = list(torch.chunk(grad_out, world_size, dim=ctx.concat_dim))
        output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]
        dist.all_to_all(output_chunks, input_chunks, group=pg)
        return torch.cat(output_chunks, dim=ctx.split_dim), None, None, None


def all_to_all(x, axis_name, *, src: PerMeshAxisSpmdType = V, dst: PerMeshAxisSpmdType = V, split_dim: int = 0, concat_dim: int = 0):
    """
    Transpose a local tensor axis with the mesh axis.

    ```
    [A0, A1, A2]      [A0, B0, C0]
    [B0, B1, B2]  =>  [A1, B1, C1]
    [C0, C1, C2]      [A2, B2, C2]
    ```

    Args:
        x: Input tensor with V or S(i) type on the mesh axis
        axis_name: The mesh axis to transpose with
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

    return _AllToAll.apply(x, axis_name, split_dim, concat_dim)


# =============================================================================
# Additional reinterpret cases
# =============================================================================


class _ReplicateToInvariant(torch.autograd.Function):
    """reinterpret(R,I): R -> I, backward is convert(I,P): I -> P."""

    @staticmethod
    def forward(ctx, x, axis_name):
        ctx.axis_name = axis_name
        return x  # no-op in forward

    @staticmethod
    def backward(ctx, grad_out):
        # backward is convert(I,P): I -> P
        # Zero out all but rank 0
        pg = _get_mesh_axis_group(ctx.axis_name)

        mode = local_tensor_mode()
        if mode is not None and isinstance(grad_out, LocalTensor):
            return mode.tensor_map(grad_out, lambda r, t: _replicate_to_partial_fwd(t, r)), None
        else:
            rank = dist.get_rank(pg)
            return _replicate_to_partial_fwd(grad_out, rank), None


class _InvariantToReplicate(torch.autograd.Function):
    """reinterpret(I,R): I -> R, backward is all_reduce(I): P -> I."""

    @staticmethod
    def forward(ctx, x, axis_name):
        ctx.axis_name = axis_name
        return x  # no-op in forward

    @staticmethod
    def backward(ctx, grad_out):
        # backward is all_reduce(I): P -> I
        pg = _get_mesh_axis_group(ctx.axis_name)
        return funcol.all_reduce(grad_out, "sum", pg).wait(), None


class _VaryingToPartial(torch.autograd.Function):
    """reinterpret(V,P): V -> P, backward is reinterpret(R,V): R -> V (no-op)."""

    @staticmethod
    def forward(ctx, x, axis_name):
        ctx.axis_name = axis_name
        return x  # no-op in forward

    @staticmethod
    def backward(ctx, grad_out):
        # backward is reinterpret(R,V): R -> V (no-op)
        return grad_out, None


class _ReplicateToPartial(torch.autograd.Function):
    """reinterpret(R,P): R -> P, backward is reinterpret(R,P): R -> P."""

    @staticmethod
    def forward(ctx, x, axis_name):
        ctx.axis_name = axis_name
        return x  # no-op in forward

    @staticmethod
    def backward(ctx, grad_out):
        # backward is reinterpret(R,P): same as forward (no-op)
        return grad_out, None


# Update reinterpret to handle all cases
def reinterpret(x, axis_name, *, src: PerMeshAxisSpmdType, dst: PerMeshAxisSpmdType):
    """
    Coerce from one local SPMD type to another without changing the local tensor.

    Guaranteed to be a no-op in forwards, but can have nontrivial backwards
    that requires comms.

    Args:
        x: Input tensor
        axis_name: The mesh axis to operate on
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
    # Validate no Shard types
    if isinstance(src, Shard) or isinstance(dst, Shard):
        raise ValueError(
            f"reinterpret does not support S(i). Use V instead, or use convert for "
            f"semantics-preserving conversions. Got src={src}, dst={dst}"
        )

    if src is dst:
        return x  # no-op

    if src is R and dst is V:
        return _ReplicateToVarying.apply(x, axis_name)
    elif src is R and dst is I:
        return _ReplicateToInvariant.apply(x, axis_name)
    elif src is R and dst is P:
        return _ReplicateToPartial.apply(x, axis_name)
    elif src is I and dst is R:
        return _InvariantToReplicate.apply(x, axis_name)
    elif src is I and dst is V:
        # Composition: I -> R -> V
        return _ReplicateToVarying.apply(_InvariantToReplicate.apply(x, axis_name), axis_name)
    elif src is I and dst is P:
        # Composition: I -> R -> P
        return _ReplicateToPartial.apply(_InvariantToReplicate.apply(x, axis_name), axis_name)
    elif src is V and dst is P:
        return _VaryingToPartial.apply(x, axis_name)
    else:
        raise ValueError(
            f"reinterpret({src}, {dst}) is not supported. "
            "Cannot reinterpret from P, and from V only V -> P is supported."
        )


# =============================================================================
# convert: type coercion with semantic preservation
# =============================================================================


def _get_rank(pg):
    """Get rank, using LocalTensorMode's simulated rank if available."""
    mode = local_tensor_mode()
    if mode is not None:
        # We're in LocalTensorMode - rank will be set by tensor_map callback
        # This function shouldn't be called directly in that case
        return dist.get_rank(pg)
    return dist.get_rank(pg)


def _replicate_to_varying_fwd(x, world_size, split_dim, rank):
    """Forward: split and take local portion based on rank."""
    chunks = torch.chunk(x, world_size, dim=split_dim)
    return chunks[rank].contiguous()


def _varying_to_partial_fwd(x, world_size, split_dim, rank):
    """Forward: pad with zeros, place data at rank position."""
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


class _ConvertReplicateToVarying(torch.autograd.Function):
    """convert(R,V): R -> V, backward is convert(V,P): V -> P."""

    @staticmethod
    def forward(ctx, x, axis_name, split_dim):
        ctx.axis_name = axis_name
        ctx.split_dim = split_dim
        pg = _get_mesh_axis_group(axis_name)
        world_size = dist.get_world_size(pg)

        mode = local_tensor_mode()
        if mode is not None and isinstance(x, LocalTensor):
            return mode.tensor_map(
                x,
                lambda r, t: _replicate_to_varying_fwd(t, world_size, split_dim, r)
            )
        else:
            rank = dist.get_rank(pg)
            return _replicate_to_varying_fwd(x, world_size, split_dim, rank)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is convert(V,P): V -> P
        pg = _get_mesh_axis_group(ctx.axis_name)
        world_size = dist.get_world_size(pg)

        mode = local_tensor_mode()
        if mode is not None and isinstance(grad_out, LocalTensor):
            result = mode.tensor_map(
                grad_out,
                lambda r, t: _varying_to_partial_fwd(t, world_size, ctx.split_dim, r)
            )
            return result, None, None
        else:
            rank = dist.get_rank(pg)
            return _varying_to_partial_fwd(grad_out, world_size, ctx.split_dim, rank), None, None


class _ConvertInvariantToVarying(torch.autograd.Function):
    """convert(I,V): I -> V, backward is all_gather(I): V -> I."""

    @staticmethod
    def forward(ctx, x, axis_name, split_dim):
        ctx.axis_name = axis_name
        ctx.split_dim = split_dim
        pg = _get_mesh_axis_group(axis_name)
        world_size = dist.get_world_size(pg)

        mode = local_tensor_mode()
        if mode is not None and isinstance(x, LocalTensor):
            return mode.tensor_map(
                x,
                lambda r, t: _replicate_to_varying_fwd(t, world_size, split_dim, r)
            )
        else:
            rank = dist.get_rank(pg)
            return _replicate_to_varying_fwd(x, world_size, split_dim, rank)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is all_gather(I): V -> I
        pg = _get_mesh_axis_group(ctx.axis_name)
        world_size = dist.get_world_size(pg)
        gathered = [torch.empty_like(grad_out) for _ in range(world_size)]
        dist.all_gather(gathered, grad_out, group=pg)
        return torch.cat(gathered, dim=ctx.split_dim), None, None


class _ConvertReplicateToPartial(torch.autograd.Function):
    """convert(R,P): R -> P, backward is convert(R,P): R -> P."""

    @staticmethod
    def forward(ctx, x, axis_name):
        ctx.axis_name = axis_name
        pg = _get_mesh_axis_group(axis_name)

        mode = local_tensor_mode()
        if mode is not None and isinstance(x, LocalTensor):
            return mode.tensor_map(x, lambda r, t: _replicate_to_partial_fwd(t, r))
        else:
            rank = dist.get_rank(pg)
            return _replicate_to_partial_fwd(x, rank)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is same operation: convert(R,P)
        pg = _get_mesh_axis_group(ctx.axis_name)

        mode = local_tensor_mode()
        if mode is not None and isinstance(grad_out, LocalTensor):
            return mode.tensor_map(grad_out, lambda r, t: _replicate_to_partial_fwd(t, r)), None
        else:
            rank = dist.get_rank(pg)
            return _replicate_to_partial_fwd(grad_out, rank), None


class _ConvertInvariantToPartial(torch.autograd.Function):
    """convert(I,P): I -> P, backward is reinterpret(R,I): R -> I (no-op)."""

    @staticmethod
    def forward(ctx, x, axis_name):
        ctx.axis_name = axis_name
        pg = _get_mesh_axis_group(axis_name)

        mode = local_tensor_mode()
        if mode is not None and isinstance(x, LocalTensor):
            return mode.tensor_map(x, lambda r, t: _replicate_to_partial_fwd(t, r))
        else:
            rank = dist.get_rank(pg)
            return _replicate_to_partial_fwd(x, rank)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is reinterpret(R,I): R -> I (no-op)
        return grad_out, None


class _ConvertVaryingToPartial(torch.autograd.Function):
    """convert(V,P): V -> P, backward is convert(R,V): R -> V."""

    @staticmethod
    def forward(ctx, x, axis_name, split_dim):
        ctx.axis_name = axis_name
        ctx.split_dim = split_dim
        pg = _get_mesh_axis_group(axis_name)
        world_size = dist.get_world_size(pg)

        mode = local_tensor_mode()
        if mode is not None and isinstance(x, LocalTensor):
            return mode.tensor_map(
                x,
                lambda r, t: _varying_to_partial_fwd(t, world_size, split_dim, r)
            )
        else:
            rank = dist.get_rank(pg)
            return _varying_to_partial_fwd(x, world_size, split_dim, rank)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is convert(R,V): R -> V (take local slice)
        pg = _get_mesh_axis_group(ctx.axis_name)
        world_size = dist.get_world_size(pg)

        mode = local_tensor_mode()
        if mode is not None and isinstance(grad_out, LocalTensor):
            result = mode.tensor_map(
                grad_out,
                lambda r, t: _replicate_to_varying_fwd(t, world_size, ctx.split_dim, r)
            )
            return result, None, None
        else:
            rank = dist.get_rank(pg)
            return _replicate_to_varying_fwd(grad_out, world_size, ctx.split_dim, rank), None, None


def convert(x, axis_name, *, src: PerMeshAxisSpmdType, dst: PerMeshAxisSpmdType, dim: int = 0):
    """
    Convert from one local SPMD type to another while preserving tensor semantics.

    Unlike reinterpret, convert may perform local tensor operations (but no comms).
    When a tensor is varying, we interpret it as concatenation on the specified dim.

    Args:
        x: Input tensor
        axis_name: The mesh axis to operate on
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
        return _ConvertReplicateToVarying.apply(x, axis_name, dim)
    elif src_base is R and dst_base is P:
        return _ConvertReplicateToPartial.apply(x, axis_name)
    elif src_base is R and dst_base is I:
        # Same as reinterpret
        return _ReplicateToInvariant.apply(x, axis_name)
    elif src_base is I and dst_base is V:
        return _ConvertInvariantToVarying.apply(x, axis_name, dim)
    elif src_base is I and dst_base is P:
        return _ConvertInvariantToPartial.apply(x, axis_name)
    elif src_base is I and dst_base is R:
        # Same as reinterpret
        return _InvariantToReplicate.apply(x, axis_name)
    elif src_base is V and dst_base is P:
        return _ConvertVaryingToPartial.apply(x, axis_name, dim)
    else:
        raise ValueError(
            f"convert({src}, {dst}) is not supported. "
            "Cannot convert out of P."
        )


# =============================================================================
# redistribute: semantics-preserving type change with comms
# =============================================================================


def redistribute(x, axis_name, *, src: PerMeshAxisSpmdType, dst: PerMeshAxisSpmdType, dim: int = 0):
    """
    Semantics-preserving conversion between local SPMD types, allowing comms.

    Unlike convert (which is no-comm), redistribute will perform the necessary
    collective communication to change from one type to another while preserving
    the semantic value of the tensor.

    Args:
        x: Input tensor
        axis_name: The mesh axis to operate on
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
            return all_to_all(x, axis_name, src=src, dst=dst, split_dim=src.dim, concat_dim=dst.dim)
        return x  # no-op

    # Varying/Shard -> Replicate: all_gather
    if src_base is V and dst_base is R:
        return all_gather(x, axis_name, src=src, dst=R, gather_dim=dim)

    # Varying/Shard -> Invariant: all_gather
    if src_base is V and dst_base is I:
        return all_gather(x, axis_name, src=src, dst=I, gather_dim=dim)

    # Partial -> Replicate: all_reduce
    if src_base is P and dst_base is R:
        return all_reduce(x, axis_name, src=P, dst=R)

    # Partial -> Invariant: all_reduce
    if src_base is P and dst_base is I:
        return all_reduce(x, axis_name, src=P, dst=I)

    # Partial -> Varying/Shard: reduce_scatter
    if src_base is P and dst_base is V:
        return reduce_scatter(x, axis_name, src=P, dst=dst, scatter_dim=dim)

    # Varying -> Varying (but different, e.g. for all_to_all with different dims)
    # This case is handled above when both are V

    # For non-comm conversions, delegate to convert
    # R -> I, I -> R, R -> V, R -> P, I -> V, I -> P, V -> P
    if src_base is R and dst_base is I:
        return convert(x, axis_name, src=R, dst=I)
    if src_base is I and dst_base is R:
        return convert(x, axis_name, src=I, dst=R)
    if src_base is R and dst_base is V:
        return convert(x, axis_name, src=R, dst=dst, dim=dim)
    if src_base is R and dst_base is P:
        return convert(x, axis_name, src=R, dst=P)
    if src_base is I and dst_base is V:
        return convert(x, axis_name, src=I, dst=dst, dim=dim)
    if src_base is I and dst_base is P:
        return convert(x, axis_name, src=I, dst=P)
    if src_base is V and dst_base is P:
        return convert(x, axis_name, src=src, dst=P, dim=dim)

    raise ValueError(
        f"redistribute({src}, {dst}) is not supported."
    )


"""
Other notes:

Ailing: Why not explicit communication?
If sharding is part of the type, imagine we have other things in type, like
shape.  Do we do reshape/transpose to make an op happen implicitly?  We do
implicit sharding transformations seems too much.  Another example: we do have
dtype promotion in eager, but it's a debatable decision.  Sharding is similar
to shape/dtype metadata.  Device is a good example, we don't move it
automatically between devices.
"""
