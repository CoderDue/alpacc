# Fused Lookback Monoid via Semiring Composition

## Background

The parallel lexer performs a single-pass prefix scan over input blocks using
decoupled lookback. Before this change the lexer ran **two** independent
inter-block lookback rounds per chunk:

1. An **endomorphism scan** — composes DFA transition functions to find the
   global state at every position.
2. A **max/add pair scan** — propagates the running token-start position (max)
   and token count (add) so each block knows its output offset.

The goal was to fuse these into **one** lookback round using the monoid
composition rule from the
[semiring composition blog post](https://williamdue.github.io/blog/monoid-composition/).

---

## The Semiring Composition Rule

Given a non-commutative semiring $(A, +, \cdot, e_+, e_\cdot)$, one can form a
composed monoid on $A^2$ by:

$$
(a_1, b_1) \star (a_2, b_2) \;:=\; (a_1 \cdot a_2,\; b_1 + a_1 \cdot b_2)
$$

with identity $(e_\cdot, e_+)$. Associativity follows from the semiring axioms
(distributivity of $\cdot$ over $+$, associativity of both).

The key insight is that the $b$ component is a **scan feeding into a scan**: the
$a$-prefix $a_1$ acts on $b_2$ before accumulating into $b_1$. This lets a
single pass compute a prefix over $a$ and simultaneously compute a $b$-reduction
that depends on the $a$-prefix at each position.

---

## Identifying the Semiring

We need to find $A$, $+$, and $\cdot$ such that the three things we want to
propagate — an endomorphism, a max, and an add — all live inside one semiring
element.

The endomorphism $\cdot$ already forms a monoid under function composition. We
want the $b$ component to carry both a max-value and an add-value, dependent on
the endomorphism prefix. Specifically:

- After the endo prefix scan, each position knows the DFA starting state $q$
  that threads into it from the left.
- The max-token-start and add-token-count for a block both depend on what
  starting state $q$ arrives at that block.

So we need $b$ to encode the answer **for all possible starting states** $q$,
i.e. $b$ is a function $S \to \mathbb{N}$ where $S$ is the DFA state set.

The action of an endomorphism $f \in A$ on such a function $b : S \to \mathbb{N}$
is precomposition:

$$
f \cdot b \;:=\; b \circ f, \qquad (f \cdot b)[q] = b[f(q)]
$$

This means "look up what state $f$ maps $q$ to, then evaluate $b$ there" —
exactly threading the endo prefix through $b$.

For the $+$ on $B = S \to \mathbb{N}$ we can use **pointwise max** (for the
token-start value) or **pointwise addition** (for the token count). Both
distribute over the action:

$$
f \cdot (b_1 \oplus b_2) = (f \cdot b_1) \oplus (f \cdot b_2)
$$

because $(b_1 \oplus b_2)[f(q)] = b_1[f(q)] \oplus b_2[f(q)]$.

So we use two applications of the composition rule in parallel, giving a triple
$(f, B_{\max}, B_{\mathrm{add}})$ with operation:

$$
(f_1, B_1^{\max}, B_1^{+}) \;\star\; (f_2, B_2^{\max}, B_2^{+})
\;=\;
\Bigl(
  f_1 \circ f_2,\quad
  B_1^{\max} + f_1 \cdot B_2^{\max},\quad
  B_1^{+} + f_1 \cdot B_2^{+}
\Bigr)
$$

Expanding the pointwise operations and the action:

$$
= \Bigl(
  f_1 \circ f_2,\quad
  q \mapsto \max\!\bigl(B_1^{\max}[q],\; B_2^{\max}[f_1(q)]\bigr),\quad
  q \mapsto B_1^{+}[q] + B_2^{+}[f_1(q)]
\Bigr)
$$

with identity $(e,\; q \mapsto 0,\; q \mapsto 0)$ where $e$ is the identity
endomorphism.

### Associativity

Follows directly from applying the blog post's proof twice — once for
$(A, B^{\max}, \max, \circ)$ and once for $(A, B^{+}, +, \circ)$. Both uses
share the same $a_1 = f_1$, so the composed triple is also associative.

Concretely, for the max component:

$$
\bigl((f_1,B_1) \star (f_2,B_2)\bigr) \star (f_3,B_3)
\;\text{ at }q
\;=\; \max\!\bigl(B_1[q],\, B_2[f_1(q)],\, B_3[f_1(f_2(q))]\bigr)
$$

$$
(f_1,B_1) \star \bigl((f_2,B_2) \star (f_3,B_3)\bigr)
\;\text{ at }q
\;=\; \max\!\bigl(B_1[q],\, \max(B_2[f_1(q)],\, B_3[f_1(f_2(q))])\bigr)
$$

Equal by associativity of $\max$. The add component is analogous with $+$.

---

## Shared Memory Savings

Naively the $B$ tables (each `NUM_STATES × uint32_t`) would need to live in
shared memory during the warp scan, costing $2 \times |S| \times 4 \times 32$
bytes per warp — about 7 KB for the JSON grammar — roughly halving achievable
IPT.

Instead, only the endo component is scanned in the warp. After the endo scan,
**lane 0 alone** iterates the 32 predecessor entries sequentially: it loads each
block's $B$ table from device memory, applies the running endo prefix $f_{\text{pfx}}$
to get the query state $f_{\text{pfx}}(q)$ for each starting state $q$, and
accumulates into register arrays `bmax_pfx[NUM_STATES]` and
`badd_pfx[NUM_STATES]`.

This keeps shared memory to `endo_t[WARP] + Status[WARP] + 3 scalars` — 304
bytes for JSON — the same order as before.

---

## Result

Two inter-block barrier rounds collapse to one, with no increase in shared
memory pressure. The $B$ tables are published to device memory per block
(`FusedStates<I>` allocates `num_blocks × NUM_STATES × uint32_t` for each of
`bmax_aggregates`, `bmax_prefixes`, `badd_aggregates`, `badd_prefixes`), and
lane 0 reads them sequentially during lookback.
