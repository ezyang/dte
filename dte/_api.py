import torch
# TODO[claude]: fix imports

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
recommend migrating your code local SPMD for the majority of your code and
only do global SPMD annotations at module boundaries (to specify input/output
contracts).  This minimizes the amount of code changes necessary (since your
code is already local SPMD) for maximum benefit (cross-module sharding/partial
contracts are the biggest source of potential errors in a training codebase;
the internal local SPMD typechecking give partial assurance that your
global SPMD type annotations are consistent with the code.) You may also
consider adopting global SPMD code for specific modules where confusing local
for global SPMD semantics is a persistent source of bugs (e.g., loss functions
or metrics--although, unlike a GSPMD like system, you will not be able to stop
thinking about collectives, because remember, all comms are explicit in code!)

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
  only invariant when they are parameters (and you desire to the gradient to
  have already been all-reduced) and replicated when being computed on
  (where you typically desire their gradients to be partial so you can delay
  the all-reduce, in case it can actually be a reduce-scatter.)  Notably,
  operations involving invariant tensors correspond directly to Megatron-style
  autograd functions like `CopyToModelParallelRegion`.

* Varying means that tensor has different values across the device mesh axis.
  (Unlike the shard placement in DTensor, we don't know exactly how the differing
  values across the mesh should be reassembled into a full tensor.)  The
  gradient of varying is varying.

* Partial means that there is a pending sum on the (differing) values
  across the device mesh axis, but it hasn't happened yet.  Delaying the reduction
  can be profitable as it might be possible to do a reduce-scatter or eliminate the
  reduction entirely.  The gradient of partial is replicate.

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
linear_op(Partial) -> Partial  # The n-ary op rule is more complicated

# Invariant not allowed to mix with other types
op(Replicate, Varying) -> Varying
Partial + Partial -> Partial  # only bilinear functions can take multiple Partial arguments
```

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
  what the "semantics" of the tensor are, but by convention we assume that a
  varying tensor is concatenated on tensor dim 0.  Like `reinterpret`, these
  operations can have nontrivial backwards.

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

So this diagram says we start off with rank 0: [A], rank 1: [B] and rank 3: [C],
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

TODO: I'm not sure reversing the arrow actually makes things easier to understand.

When referencing operators, we may refer to them compactly by removing `x` and
`mesh_axis` from the function signature, and using R/V/P/I to abbreviate the
local SPMD type passed into src/dst arguments.

OK, without further ado!

### `all_gather(x: Varying, mesh_axis, tgt) -> Replicate | Invariant`

```
[A]
[B]  =>  [A, B, C]
[C]
```

Gather shards along the mesh axis, so that every rank has the full copy of the data.

When `dst='replicate'`, aka `all_gather(R): V -> R`, the backwards is `reduce_scatter: P -> V`.

```
[A0 + A1 + A2]      +[A0, B0, C0]
[B0 + B1 + B2]  <=  +[A1, B1, C1]
[C0 + C1 + C2]      +[A2, B2, C2]
```

When `dst='invariant'`, aka `all_gather(I): V -> I`, the backwards is `convert(I,V): I -> V`.

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

It is common to want to `all_reduce` on varying data; just `pinterpret(V,P)` the data
as partial before calling `all_reduce`.

TODO: I think with erasure we can infer this cast

### `reduce_scatter(x, mesh_axis): Varying | Partial -> Varying`

```
+[A0, B0, C0]      [A0 + A1 + A2]
+[A1, B1, C1]  =>  [B0 + B1 + B2]
+[A2, B2, C2]      [C0 + C1 + C2]
```

Reduce shards along the mesh axis, but only get one shard of the result (e.g.,
an inefficient implementation of reduce-scatter would be to all-reduce and
then drop the data you did not need.)  Like `all_reduce`, you can also run
this on a varying answer (the backwards gets an extra call to `pcast: R ->
V`); unlike `all_reduce`, this always produces a varying output, so it doesn't
accept a tgt argument.

The forwards is `P -> V`, the backwards is `all_gather: V -> R`

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
the local tensor.  It is guaranteed to be a no-op in forwards.  Here
are the supported coercions:

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

`reinterpret(I,V): I -> V`, the backwards is `all_reduce(I): V -> I`

```
Forward:
         [A]
[A]  =>  [A]
         [A]

Backward:
                    [A0]
[A0 + A1 + A2]  <=  [A1]
                    [A2]
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

TODO[claude]: Do these compositions below directly, like the ones above

`reinterpret(R,P): R -> P`, is the composition of `R -> V -> P`.  `reinterpret(I,P): I -> P`, is the composition
of `I -> V -> P`.  Note that this reinterpret has unusual semantics: the
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
    I  X - X /
    V      - X
    P        -
```

### `convert(x, mesh_axis, src, tgt)`

Convert from one local SPMD to another while preserving the semantics of the
tensor.  The semantics of a varying tensor in this setting is interpreted to
be the concatenation of all the tensors in the mesh on dim=0 (matching the
semantics of all-gather).  Here are the supported conversions:

`convert(R,V): R -> V`, the backward is `convert(V,P) : V -> P`

```
Forward:
               [A]
[A, B, C]  =>  [B]
               [C]

Backward:
+[A, 0, 0]      [A]
+[0, B, 0]  <=  [B]
+[0, 0, C]      [C]
````

`convert(I,V): I -> V`, the backwards is `all_gather(I): V -> I`

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

```
Forward:
         +[A]
[A]  =>  +[0]
         +[0]

Backward:
[A]  <=  [A]
```

`convert(V,P): V -> P`, the backwards is `convert(R,V): R -> V`

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
same meaning as reinterpret(R,I) and reinterpret(R,I).

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
transition.  Note that type does NOT uniquely determine an operator: there can
be multiple ways to get from one type to another which have different
semantics.


```
      \   DST   Replicate           Invariant           Varying             Partial
SRC    \----------------------------------------------------------------------------------------
Replicate       -                   reinterpret(R,I)    reinterpret(R,V)    reinterpret(R,P)
                                                        convert(R,V)        convert(R,P)

Invariant       reinterpret(I,R)    -                   reinterpret(I,V)    reinterpret(I,P)
                                                        convert(I,V)        convert(I,P)

Varying         all_gather(R)       all_gather(I)       all_to_all()        reinterpret(V,P)

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
I -> V      reinterpret(I,V)        V -> I      all_reduce(I) . reinterpret(V,P)
            convert(I,V)                        all_gather(I)
I -> P      reinterpret(I,P)        R -> I      all_reduce(I) . reinterpret(R,P)
            convert(I,P)                        reinterpret(R,I)
V -> R      all_gather(R)           P -> V      reduce_scatter()
V -> I      all_gather(I)           I -> V      convert(I,V)
V -> V      all_to_all()            V -> V      all_to_all()
V -> P      reinterpret(V,P)        R -> V      reinterpret(R,V)
            convert(V,P)                        convert(R,V)
P -> R      all_reduce(R)           P -> R      all_reduce(R)
P -> I      all_reduce(I)           I -> R      reinterpret(I,R)
P -> V      reduce_scatter()        V -> R      all_gather(R)

# Global SPMD

An illustrative example highlighting the difference between local and global
SPMD is what happens when we perform the matmul in row-parallel linear
Megatron-style TP.  In this setting, both the intermediate and the weight are
sharded on the TP mesh axis (aka varying), and after this matmul we need to
perform an all-reduce to compute the real value of the linear.  To correctly
express that we want to do a *global* matrix multiply, you must run *two*
operations in local SPMD:

```
out: Varying = linear(hidden: Varying, weight: Varying)
out2: Partial = pcast(out: Varying, to='partial')
```

If you only run a linear in local SPMD, you have asked to perform only a local
matrix multiply per rank; the local semantics of a matmul do NOT imply a global
reduction over all ranks.  The pcast is necessary to indicate, "Actually, I do
want a global matmul!"  In global SPMD, we would instead error on the linear call.

TODO: Write the rest

TODO from YZ: in global spmd, can represent partial as an extra dimension, hidden with vmap

## Miscellaneous design notes

  (NB: If the type system did not need to be erasable, we could elide the src
  argument from all comms, as we could simply read out the type from the
  incoming tensor; however, due to erasure src type also has to be specified
  when its ambiguous.)

Related work: https://arxiv.org/pdf/2506.15961

"""

# TODO: There may be inaccuracies in the code below as I am still working on
# it

# TODO[claude]: for all of the public (non-prefix underscore API), we need to
# implement a separate type-checking mode, that keeps track of partial,
# replicate and varying on a per-mesh axis basis.  We will implement just the
# rule for einsum, which is simple.
class LTensor:
    pass


# TODO: this would need to be overrideable by the user to get PGs from their
# preferred spot; e.g., parallel_state or a (global) device mesh
def _get_mesh_axis_group(axis_name):
    # TODO[claude]: Introduce a concept of a global device mesh ala set_mesh
    # and then retrieve the PG from that mesh
    assert False

# NB: partial here denotes partial summation
ALLOWED_PCAST_STATES = {'partial', 'replicate', 'varying', 'invariant'}

class _ReplicateToVarying(torch.autograd.Function):
    def forward(ctx, x, axis_name):
        self.axis_name = axis_name
        return x

    def backward(ctx, grad_out):
        return psum_invariant(grad_out, self.axis_name)


# NB: Something is a pcast only if it is a no-op in forwards.  But pcasts can
# become collectives in backwards!
# TODO[claude]: replace the asserts with proper if-raise errors
# TODO[claude]: add docstrings for functions
def pcast(x, axis_name, *, src: str, dst: str):
    # NB: no pytree support on x
    # TODO: is 'replicated' more symmetric
    # TODO: support 'unreduced' for more compat with JAX?
    # TODO: is 'varying' more symmetric with 'partial'/'replicate'?
    assert src in ALLOWED_PCAST_STATES, src
    assert dst in ALLOWED_PCAST_STATES, dst
    if src == 'replicate' and dst == 'varying':
        return _ReplicateToVarying.apply(x, axis_name)

class _AllReduceFromVary(torch.autograd.Function):
    # varying -> replicate
    def forward(ctx, x, axis_name):
        self.axis_name = axis_name
        pg = _get_mesh_axis_group(axis_name)
        return dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pg, async_op=False)

    # partial -> vary
    def backward(ctx, grad_out):
        return pcast(grad_out, axis_name, from='partial', to='varying')

# TODO[claude]: the typechecking version will only accept vary
def all_reduce(x, axis_name):
    return _AllReduceFromVary.apply(x, axis_name)

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
