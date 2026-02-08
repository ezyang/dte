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
  Unlike the shard placement in DTensor, we intentionally don't track how the
  differing values should be reassembled into a full tensor--this is by design,
  as local SPMD only needs to track enough information to ensure backwards
  correctness, not full global tensor semantics.  (However, when semantics-
  preserving operations like `convert` need a convention, we assume varying
  means concatenation on dim 0; see the `convert` documentation for details.)
  The gradient of varying is varying.

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

* The preexisting comms APIs are augmented with src/tgt arguments when needed, to
  disambiguate the local SPMD type of their input/outputs.  The mechanical
  reason for this is that the backwards of these comm functions depends on
  what the local SPMD types are, so passing this info helps us choose the
  right autograd Function!

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
communication, so they do not take src/tgt arguments.  Their local SPMD types
propagate from their inputs according to the typing rules above, and any
nontrivial type changes must be made explicit via the primitives above.

TODO: Show local SPMD rules work even when the shapes across ranks are not the
same.  I think you need to have some special collectives in this case.

## Comms by API

Let's describe all of these operators in more detail.  When reasoning about
the semantic meaning of an operation in a distributed setting, it's helpful
to look at the flow of data across both the device mesh axis as well as
a tensor dim axis.  Each of these operators will have a diagram that shows the action of
collectives on 1D tensors on three ranks.  Here is an example:

```
[A]
[B]  =>  [A, B, C]
[C]
```

To explain in more detail:

* The double arrow `=>` shows the before and after an operation.
* Each row (line) specifies the content of a different rank.
* Each bracketed expression `[A, B, C]` indicates a 1-D tensor.
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
local SPMD type passed into src/tgt arguments.

OK, without further ado!

### `all_gather(x: Varying, mesh_axis, tgt) -> Replicate | Invariant`

```
[A]
[B]  =>  [A, B, C]
[C]
```

Gather shards along the mesh axis, so that every rank has the full copy of the data.

When `tgt='replicate'`, aka `all_gather(R): V -> R`, the backwards is `reduce_scatter: P -> V`.

```
[A0 + A1 + A2]      +[A0, B0, C0]
[B0 + B1 + B2]  <=  +[A1, B1, C1]
[C0 + C1 + C2]      +[A2, B2, C2]
```

When `tgt='invariant'`, aka `all_gather(I): V -> I`, the backwards is `convert(I,V): I -> V`.

```
[A]
[B]  <=  [A, B, C]
[C]
```

### `all_reduce(x: Partial, mesh_axis, src, tgt) -> Replicate | Invariant`

Reduce shards along the mesh axis, so that every rank has the full summed value.

For partial input, the forwards is:

```
+[A0]
+[A1]  =>  [A0 + A1 + A2]
+[A2]
```

When `tgt='replicate'`, aka `all_reduce(R): P -> R`, the backwards is `all_reduce(R): P -> R`

```
                    +[A0]
[A0 + A1 + A2]  <=  +[A1]
                    +[A2]
```

When `tgt='invariant'`, aka `all_reduce(I): P -> I`, the backwards is `reinterpret(I,R): I -> R`

```
[A]  <=  [A]
```

It is common to want to `all_reduce` on varying data; just `reinterpret(V,P)` the data
as partial before calling `all_reduce`.

TODO: I think with erasure we can infer this cast

### `reduce_scatter(x, mesh_axis): Partial -> Varying`

```
+[A0, B0, C0]      [A0 + A1 + A2]
+[A1, B1, C1]  =>  [B0 + B1 + B2]
+[A2, B2, C2]      [C0 + C1 + C2]
```

Reduce shards along the mesh axis, but only get one shard of the result (e.g.,
an inefficient implementation of reduce-scatter would be to all-reduce and
then drop the data you did not need.)  Unlike `all_reduce`, this always
produces a varying output, so it doesn't accept a tgt argument.

The forwards is `P -> V`, the backwards is `all_gather: V -> R`

It is common to want to `reduce_scatter` on varying data; just `reinterpret(V,P)`
the data as partial before calling `reduce_scatter`.

```
               [A]
[A, B, C]  <=  [B]
               [C]
```

### `all_to_all(x, mesh_axis): Varying -> Varying`

```
[A0, A1, A2]      [A0, B0, C0]
[B0, B1, B2]  =>  [A1, B1, C1]
[C0, C1, C2]      [A2, B2, C2]
```

Transpose a local tensor axis with the mesh axis.

The forwards is `V -> V`, the backwards is `all_to_all: V -> V`.

```
[A0, A1, A2]      [A0, B0, C0]
[B0, B1, B2]  <=  [A1, B1, C1]
[C0, C1, C2]      [A2, B2, C2]
```

### `reinterpret(x, mesh_axis, src, tgt)`

Coerce from one local SPMD type to another local SPMD type without changing
the local tensor.  It is guaranteed to be a no-op in forwards.

**Important:** Unlike `convert`, `reinterpret` can change the semantic value of
a tensor.  For example, `reinterpret(R,P)` treats a replicated value as if it
were partial, meaning after reduction you get N times the original value (where
N is the mesh axis size).  If you want to preserve semantics, use `convert`.

Here are the supported coercions:

`reinterpret(R,I): R -> I`, the backwards is `convert(I,P): I -> P`

```
Forward:
[A] => [A]

Backward:
+[A]
+[0]  <=  [A]
+[0]
```

`reinterpret(R,V): R -> V`, the backwards is `reinterpret(V,P): V -> P`

```
Forward:
         [A]
[A]  =>  [A]
         [A]

Backward:
+[A0]      [A0]
+[A1]  <=  [A1]
+[A2]      [A2]
```

`reinterpret(I,R): I -> R`, the backwards is `all_reduce(I): P -> I`

```
Forward:
[A] => [A]

Backward:
                    +[A0]
[A0 + A1 + A2]  <=  +[A1]
                    +[A2]
```

`reinterpret(V,P): V -> P`, the backwards is `reinterpret(R,V): R -> V`

```
Forward:
[A0]      +[A0]
[A1]  =>  +[A1]
[A2]      +[A2]

Backward:
[A]
[A]  <=  [A]
[A]
```

`reinterpret(R,P): R -> P`, the backwards is `reinterpret(R,P): R -> P`

```
Forward:
         +[A]
[A]  =>  +[A]
         +[A]

Backward:
+[A]
+[A]  <=  [A]
+[A]
```

`reinterpret(I,V): I -> V` is the composition of `I -> R -> V`.  `reinterpret(R,P): R -> P`
is the composition of `R -> V -> P`.  `reinterpret(I,P): I -> P` is the composition of
`I -> R -> P`.  Note that these reinterprets have unusual semantics: the
resulting tensor has been scaled by the mesh axis size (because you are now
obligated to sum each of the (equal) quantities of the rank together!) If you
instead wanted to *preserve* the original semantic meaning of the tensor, use
`convert`.

Here is a table of permissible reinterprets (`-` is no-op, `X` is direct
coercion, `/` is transitive coercion.)

```
       tgt
       R I V P
src R  - X X /
    I  X - / /
    V      - X
    P        -
```

### `convert(x, mesh_axis, src, tgt)`

Convert from one local SPMD to another while preserving the semantics of the
tensor.  When a tensor is varying, it is ambiguous what the "semantics" of
the tensor are; by convention we interpret varying as concatenation on dim=0,
following the natural behavior of collectives like all-gather and reduce-scatter
that split/combine along the outermost tensor dimension.  Here are the supported
conversions:

`convert(R,V): R -> V`, the backward is `convert(V,P) : V -> P`

Input is replicated across ranks, so each rank holds the full tensor.  The
output keeps only the local shard along the specified dim, producing a varying
value.

```
Forward:
               [A]
[A, B, C]  =>  [B]
               [C]

Backward:
+[A, 0, 0]      [A]
+[0, B, 0]  <=  [B]
+[0, 0, C]      [C]
```

`convert(I,V): I -> V`, the backwards is `all_gather(I): V -> I`

Input is invariant across ranks, so each rank holds the full tensor.  The
output keeps only the local shard along the specified dim, producing a varying
value.

```
Forward
               [A]
[A, B, C]  =>  [B]
               [C]

Backward:
               [A]
[A, B, C]  <=  [B]
               [C]
```

`convert(R,P): R -> P`, the backwards is `convert(R,P) : R -> P`

Input is replicated across ranks.  The output keeps the same per-rank tensor
shape, but all ranks except the first are zeroed out, producing a partial value
that sums to the original tensor after a cross-rank reduction.

```
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
Forward:
         +[A]
[A]  =>  +[0]
         +[0]

Backward:
[A]  <=  [A]
```

`convert(V,P): V -> P`, the backwards is `convert(R,V): R -> V`

Input is varying, with each rank holding a shard or distinct value.  The output
places each rank's value into a disjoint position of a partial tensor (zeros
elsewhere) so that summing across ranks reconstructs the stacked value.

```
Forward:
[A]      +[A, 0, 0]
[B]  =>  +[0, B, 0]
[C]      +[0, 0, C]

Backward:
[A]
[B]  <=  [A, B, C]
[C]
```

You cannot convert out of P: the only way to eliminate the pending reduction
is to do the actual all-reduce.

For convenience, we also support convert(R,I) and convert(I,R), which have the
same meaning as reinterpret(R,I) and reinterpret(I,R).

Here is a table of permissible converts (`-` is no-op, `O` is supported, `X` is when
the semantics is the same as reinterpret.)

```
       tgt
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

(Aside: another way we could have defined the global SPMD type is by replacing
Varying with Shard(tensor dim), which is indeed how PyTorch DTensor defines
sharding.  However, most people find the JAX-style tensor dim more intuitive
to work with, and it is nice to be able to factor the global SPMD type
into a local SPMD type plus something extra!)

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

All local SPMD comms operators are also valid in global SPMD.  We simply need to describe
how they propagate sharding.  Intuitively, all of these comms operators simply replace
Varying with Shard(0) (you can translate this into partition spec form using the standard
device mesh dim to tensor dim translation).  Here are all of the affected operators:

```
all_gather(R): Shard(0) -> Replicate
all_gather(I): Shard(0) -> Invariant
reduce_scatter: Partial -> Shard(0)
all_to_all: Shard(0) -> Shard(0)
reinterpret(R,V): Replicate -> Shard(0)
reinterpret(V,P): Shard(0) -> Partial
convert(R,V): Replicate -> Shard(0)
convert(I,V): Replicate -> Shard(0)
convert(V,P): Replicate -> Shard(0)
```

Do we have to only accept sharding on tensor dim 0?  Many collectives take an
argument saying which tensor dim to operate on (e.g., an all-gather on tensor
dim 1 instead of tensor dim 0); in which case the tensor dim Shard references
changes accordingly.

TODO: reinterpret can probably usefully take a tensor dim argument

The global SPMD interpretation of convert is straightforward: it is the identity function.
However, the global SPMD interpretations of reinterpret are sometimes counter-intuitive.
Here, we write "single device" versions of these functions to make it clear what their
action is, given sharding on tensor dim 0:

```
def reinterpret_R_I(x):
    return x

def reinterpret_R_S0(x):
    repeats = [mesh_axis_size] + [1] * (x.ndim - 1)
    return x.repeat(repeats)

def reinterpret_I_R(x):
    return x

def reinterpret_S0_P(x):
    shape = (mesh_axis_size, x.shape[0] // mesh_axis_size) + x.shape[1:]
    return x.view(shape).sum(0)

def reinterpret_R_P(x):
    return x * mesh_axis_size
```

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

# NB: partial here denotes partial summation
ALLOWED_PCAST_STATES = {'partial', 'replicate', 'varying', 'invariant'}


class LTensor:
    """
    A tensor with local SPMD type annotations.

    Tracks the local SPMD type (replicate, invariant, varying, partial) for each
    mesh axis. This enables type-checking to ensure distributed operations
    compute mathematically correct gradients.

    Attributes:
        data: The underlying torch.Tensor
        types: Dict mapping mesh axis names to local SPMD types
    """

    def __init__(self, data: torch.Tensor, types: dict[str, str] | None = None):
        """
        Create an LTensor with optional type annotations.

        Args:
            data: The underlying tensor
            types: Dict mapping mesh axis names to types
                   ('replicate', 'invariant', 'varying', 'partial')
        """
        self.data = data
        self.types = types if types is not None else {}
        for axis_name, typ in self.types.items():
            if typ not in ALLOWED_PCAST_STATES:
                raise ValueError(
                    f"Invalid type '{typ}' for axis '{axis_name}'. "
                    f"Must be one of {ALLOWED_PCAST_STATES}"
                )

    def get_type(self, axis_name: str) -> str | None:
        """Get the local SPMD type for a mesh axis, or None if not tracked."""
        return self.types.get(axis_name)

    def with_type(self, axis_name: str, typ: str) -> "LTensor":
        """Return a new LTensor with an updated type for the given axis."""
        if typ not in ALLOWED_PCAST_STATES:
            raise ValueError(f"Invalid type '{typ}'. Must be one of {ALLOWED_PCAST_STATES}")
        new_types = self.types.copy()
        new_types[axis_name] = typ
        return LTensor(self.data, new_types)


def einsum(equation: str, *operands: LTensor) -> LTensor:
    """
    Perform einsum with local SPMD type checking.

    The typing rule for einsum is simple: all operands must have matching types
    on each mesh axis, and the output inherits those types.

    For each mesh axis:
    - If all operands are Replicate -> output is Replicate
    - If all operands are Invariant -> output is Invariant
    - If all operands are Varying -> output is Varying
    - If operand is Partial -> only valid for linear operations on single Partial input
    - Mixed Replicate/Varying -> output is Varying
    - Invariant cannot mix with other types

    Args:
        equation: The einsum equation string
        *operands: LTensor operands

    Returns:
        LTensor with the result and inferred types
    """
    # Extract underlying tensors
    tensors = [op.data for op in operands]

    # Collect all mesh axes mentioned
    all_axes = set()
    for op in operands:
        all_axes.update(op.types.keys())

    # Type check and infer output types
    output_types = {}
    for axis in all_axes:
        axis_types = []
        for op in operands:
            typ = op.types.get(axis)
            if typ is not None:
                axis_types.append(typ)

        if not axis_types:
            continue

        # Check type compatibility and infer output type
        unique_types = set(axis_types)

        if len(unique_types) == 1:
            # All same type
            output_types[axis] = axis_types[0]
        elif unique_types == {'replicate', 'varying'}:
            # Mixed replicate/varying -> varying
            output_types[axis] = 'varying'
        elif 'invariant' in unique_types and len(unique_types) > 1:
            raise TypeError(
                f"Invariant type on axis '{axis}' cannot mix with other types. "
                f"Found types: {axis_types}"
            )
        elif 'partial' in unique_types:
            # Partial can only appear alone (for linear ops) or with another partial
            # (for bilinear ops like matmul)
            non_partial = unique_types - {'partial'}
            if non_partial:
                raise TypeError(
                    f"Partial type on axis '{axis}' can only combine with partial. "
                    f"Found types: {axis_types}"
                )
            output_types[axis] = 'partial'
        else:
            raise TypeError(
                f"Incompatible types on axis '{axis}': {axis_types}"
            )

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


def all_reduce(x, axis_name, *, src: str = 'partial', tgt: str):
    """
    Reduce shards along the mesh axis, so every rank has the full summed value.

    Args:
        x: Input tensor with Partial type on the mesh axis
        axis_name: The mesh axis to reduce over
        src: Source type (must be 'partial')
        tgt: Target type ('replicate' or 'invariant')

    Returns:
        Tensor with Replicate or Invariant type depending on tgt

    When tgt='replicate', backward is all_reduce(R): P -> R
    When tgt='invariant', backward is reinterpret(I,R): I -> R (no-op)
    """
    if src != 'partial':
        raise ValueError(f"all_reduce src must be 'partial', got {src}")
    if tgt == 'replicate':
        return _AllReduceToReplicate.apply(x, axis_name)
    elif tgt == 'invariant':
        return _AllReduceToInvariant.apply(x, axis_name)
    else:
        raise ValueError(f"all_reduce tgt must be 'replicate' or 'invariant', got {tgt}")


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


def all_gather(x, axis_name, *, tgt: str, gather_dim: int = 0):
    """
    Gather shards along the mesh axis, so every rank has the full copy of data.

    ```
    [A]
    [B]  =>  [A, B, C]
    [C]
    ```

    Args:
        x: Input tensor with Varying type on the mesh axis
        axis_name: The mesh axis to gather over
        tgt: Target type ('replicate' or 'invariant')
        gather_dim: The tensor dimension to concatenate along (default: 0)

    Returns:
        Tensor with Replicate or Invariant type depending on tgt

    When tgt='replicate', backward is reduce_scatter: P -> V
    When tgt='invariant', backward is convert(I,V): I -> V
    """
    if tgt == 'replicate':
        return _AllGatherToReplicate.apply(x, axis_name, gather_dim)
    elif tgt == 'invariant':
        return _AllGatherToInvariant.apply(x, axis_name, gather_dim)
    else:
        raise ValueError(f"all_gather tgt must be 'replicate' or 'invariant', got {tgt}")


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


def reduce_scatter(x, axis_name, *, scatter_dim: int = 0):
    """
    Reduce shards along the mesh axis, but only get one shard of the result.

    ```
    +[A0, B0, C0]      [A0 + A1 + A2]
    +[A1, B1, C1]  =>  [B0 + B1 + B2]
    +[A2, B2, C2]      [C0 + C1 + C2]
    ```

    Args:
        x: Input tensor with Partial type on the mesh axis
        axis_name: The mesh axis to reduce-scatter over
        scatter_dim: The tensor dimension to scatter along (default: 0)

    Returns:
        Tensor with Varying type

    The backward is all_gather(R): V -> R
    """
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


def all_to_all(x, axis_name, *, split_dim: int = 0, concat_dim: int = 0):
    """
    Transpose a local tensor axis with the mesh axis.

    ```
    [A0, A1, A2]      [A0, B0, C0]
    [B0, B1, B2]  =>  [A1, B1, C1]
    [C0, C1, C2]      [A2, B2, C2]
    ```

    Args:
        x: Input tensor with Varying type on the mesh axis
        axis_name: The mesh axis to transpose with
        split_dim: The tensor dimension to split along (default: 0)
        concat_dim: The tensor dimension to concatenate along (default: 0)

    Returns:
        Tensor with Varying type

    The backward is also all_to_all: V -> V
    """
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
        return dist.all_reduce(grad_out, op=dist.ReduceOp.SUM, group=pg, async_op=False), None


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
def reinterpret(x, axis_name, *, src: str, tgt: str):
    """
    Coerce from one local SPMD type to another without changing the local tensor.

    Guaranteed to be a no-op in forwards, but can have nontrivial backwards
    that requires comms.

    Args:
        x: Input tensor
        axis_name: The mesh axis to operate on
        src: Source local SPMD type ('replicate', 'invariant', 'varying', 'partial')
        tgt: Target local SPMD type ('replicate', 'invariant', 'varying', 'partial')

    Supported coercions:
        - reinterpret(R,I): R -> I, backward is convert(I,P): I -> P
        - reinterpret(R,V): R -> V, backward is reinterpret(V,P): V -> P
        - reinterpret(R,P): R -> P, backward is reinterpret(R,P): R -> P
        - reinterpret(I,R): I -> R, backward is all_reduce(I): P -> I
        - reinterpret(V,P): V -> P, backward is reinterpret(R,V): R -> V
    """
    if src not in ALLOWED_PCAST_STATES:
        raise ValueError(f"Invalid src state: {src}. Must be one of {ALLOWED_PCAST_STATES}")
    if tgt not in ALLOWED_PCAST_STATES:
        raise ValueError(f"Invalid tgt state: {tgt}. Must be one of {ALLOWED_PCAST_STATES}")

    if src == tgt:
        return x  # no-op

    if src == 'replicate' and tgt == 'varying':
        return _ReplicateToVarying.apply(x, axis_name)
    elif src == 'replicate' and tgt == 'invariant':
        return _ReplicateToInvariant.apply(x, axis_name)
    elif src == 'replicate' and tgt == 'partial':
        return _ReplicateToPartial.apply(x, axis_name)
    elif src == 'invariant' and tgt == 'replicate':
        return _InvariantToReplicate.apply(x, axis_name)
    elif src == 'invariant' and tgt == 'varying':
        # Composition: I -> R -> V
        return _ReplicateToVarying.apply(_InvariantToReplicate.apply(x, axis_name), axis_name)
    elif src == 'invariant' and tgt == 'partial':
        # Composition: I -> R -> P
        return _ReplicateToPartial.apply(_InvariantToReplicate.apply(x, axis_name), axis_name)
    elif src == 'varying' and tgt == 'partial':
        return _VaryingToPartial.apply(x, axis_name)
    else:
        raise ValueError(
            f"reinterpret({src}, {tgt}) is not supported. "
            "Cannot reinterpret from partial or to varying/replicate from varying."
        )


# Update the alias
pcast = reinterpret


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


def convert(x, axis_name, *, src: str, tgt: str, dim: int = 0):
    """
    Convert from one local SPMD type to another while preserving tensor semantics.

    Unlike reinterpret, convert may perform local tensor operations (but no comms).
    When a tensor is varying, we interpret it as concatenation on the specified dim.

    Args:
        x: Input tensor
        axis_name: The mesh axis to operate on
        src: Source local SPMD type
        tgt: Target local SPMD type
        dim: The tensor dimension for split/concat operations (default: 0)

    Supported conversions:
        - convert(R,V): R -> V, backward is convert(V,P): V -> P
        - convert(I,V): I -> V, backward is all_gather(I): V -> I
        - convert(R,P): R -> P, backward is convert(R,P): R -> P
        - convert(I,P): I -> P, backward is reinterpret(R,I): R -> I
        - convert(V,P): V -> P, backward is convert(R,V): R -> V
        - convert(R,I) and convert(I,R) are same as reinterpret
    """
    if src not in ALLOWED_PCAST_STATES:
        raise ValueError(f"Invalid src state: {src}. Must be one of {ALLOWED_PCAST_STATES}")
    if tgt not in ALLOWED_PCAST_STATES:
        raise ValueError(f"Invalid tgt state: {tgt}. Must be one of {ALLOWED_PCAST_STATES}")

    if src == tgt:
        return x  # no-op

    if src == 'replicate' and tgt == 'varying':
        return _ConvertReplicateToVarying.apply(x, axis_name, dim)
    elif src == 'replicate' and tgt == 'partial':
        return _ConvertReplicateToPartial.apply(x, axis_name)
    elif src == 'replicate' and tgt == 'invariant':
        # Same as reinterpret
        return _ReplicateToInvariant.apply(x, axis_name)
    elif src == 'invariant' and tgt == 'varying':
        return _ConvertInvariantToVarying.apply(x, axis_name, dim)
    elif src == 'invariant' and tgt == 'partial':
        return _ConvertInvariantToPartial.apply(x, axis_name)
    elif src == 'invariant' and tgt == 'replicate':
        # Same as reinterpret
        return _InvariantToReplicate.apply(x, axis_name)
    elif src == 'varying' and tgt == 'partial':
        return _ConvertVaryingToPartial.apply(x, axis_name, dim)
    else:
        raise ValueError(
            f"convert({src}, {tgt}) is not supported. "
            "Cannot convert out of partial."
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
