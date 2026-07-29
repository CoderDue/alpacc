// ---------------------------------------------------------------------------
// Compile-time shared-memory accounting for the lexer kernel.
//
// The kernel's per-block shmem footprint has three parts:
//   1. IPT-scaling storage: `states[SHMEM_STRIDE * BLOCK_SIZE]` and the
//      `exch` union `[IPT * BLOCK_SIZE]`.  These dominate at large IPT.
//   2. cub::BlockScan TempStorage for the state scan (state_t) and the two
//      SoA scans (u32, shared between Max and Add).  cub gives us an exact
//      sizeof() for each, so we can query it as a constexpr.
//   3. Small fixed scalars: `next_block_first_state`, `last_start`,
//      `num_sel_sh`, and the lookbackPrefixPair warp buffers.
//
// max_items_per_thread() below scans IPT from 1 to some cap (1024) and
// returns the largest value that keeps the total under `usable * SHMEM /
// 100` (default 90%).  Because it's constexpr the search folds away at
// compile time; the returned IPT feeds straight into the kernel template.
template<typename I, typename state_t, typename J, typename length_t, typename terminal_t>
constexpr size_t exch_elem_bytes() {
  size_t t = sizeof(terminal_t);
  size_t j = sizeof(J);
  size_t l = sizeof(length_t);
  return (t > j ? (t > l ? t : l) : (j > l ? j : l));
}

template<typename I, typename state_t, uint32_t BLOCK_SIZE>
__host__ __device__ constexpr size_t shmem_pad_stride(uint32_t items_per_thread) {
  if (sizeof(state_t) >= 8u) {
    // Each element occupies sizeof(state_t)/4 bank slots (4-byte banks, sm_75+).
    // Stride in banks = items_per_thread * (sizeof(state_t)/4).
    // GCD(stride_banks, 32) gives the conflict factor; we want it == 2 (minimum
    // achievable for even-sized types).  For 8-byte types, stride_banks = IPT*2;
    // GCD is 2 iff IPT is odd.  Add 1 pad when IPT is even to make stride odd.
    uint32_t banks_per_elem = (uint32_t)(sizeof(state_t) / 4u);
    (void)(items_per_thread * banks_per_elem); // stride_banks: used only in comment above
    // Pad until stride_banks is odd * banks_per_elem (i.e. stride_banks/banks_per_elem is odd)
    uint32_t pad = (items_per_thread % 2u == 0u) ? 1u : 0u;
    return (size_t)(items_per_thread + pad);
  }
  uint32_t shmem_mod    = 8u / (uint32_t)sizeof(state_t);
  uint32_t shmem_target = 4u / (uint32_t)sizeof(state_t);
  uint32_t shmem_rem    = items_per_thread % shmem_mod;
  uint32_t shmem_raw    = (shmem_target - shmem_rem + shmem_mod) % shmem_mod;
  uint32_t shmem_pad    = (shmem_raw == 0) ? shmem_mod : shmem_raw;
  return (size_t)(items_per_thread + shmem_pad);
}

template<typename I, typename state_t, typename J, typename length_t, typename terminal_t, uint32_t BLOCK_SIZE>
constexpr size_t lexer_shmem_variable(uint32_t items_per_thread) {
  size_t states_bytes = sizeof(state_t) * shmem_pad_stride<I, state_t, BLOCK_SIZE>(items_per_thread) * BLOCK_SIZE;
  size_t exch_bytes   = exch_elem_bytes<I, state_t, J, length_t, terminal_t>() * items_per_thread * BLOCK_SIZE;
  return states_bytes + exch_bytes;
}

template<typename I, typename state_t, uint32_t BLOCK_SIZE>
constexpr size_t lexer_shmem_fixed() {
  size_t cub_state_temp = sizeof(typename cub::BlockScan<state_t, BLOCK_SIZE>::TempStorage);
  size_t cub_u32_temp   = sizeof(typename cub::BlockScan<uint32_t,  BLOCK_SIZE>::TempStorage);
  // fusedLookback: single endo_t broadcast slot (thread 0 walks serially).
  size_t lookback = sizeof(state_t);
  // Fused B_max/B_add aggregate + prefix in shared memory (NUM_STATES-wide).
  size_t b_tables = 4 * (size_t)NUM_STATES * sizeof(uint32_t);
  size_t fixed_scalars = sizeof(state_t)    // next_block_first_state
                        + sizeof(I)          // last_start
                        + sizeof(I);         // num_sel_sh
  return cub_state_temp + cub_u32_temp + lookback + b_tables + fixed_scalars;
}

// Largest ITEMS_PER_THREAD ≤ HARD_CAP whose per-block shmem footprint fits
// in floor(SHARED_MEMORY * USABLE_PCT / 100) bytes.  Reserving 10% by
// default (USABLE_PCT = 90) leaves headroom for cub internals and any small
// implicit allocations we haven't modelled.
template<typename I, typename state_t, typename J, typename length_t, typename terminal_t,
         uint32_t BLOCK_SIZE, uint32_t SHARED_MEMORY,
         uint32_t HARD_CAP = 1024, uint32_t USABLE_PCT = 90>
constexpr uint32_t max_items_per_thread() {
  size_t usable = (size_t)SHARED_MEMORY * USABLE_PCT / 100u;
  size_t fixed = lexer_shmem_fixed<I, state_t, BLOCK_SIZE>();
  uint32_t best = 1;
  for (uint32_t ipt = 1; ipt <= HARD_CAP; ipt++) {
    size_t total = fixed + lexer_shmem_variable<I, state_t, J, length_t, terminal_t, BLOCK_SIZE>(ipt);
    if (total <= usable) best = ipt;
    else break;
  }
  return best;
}

// ---------------------------------------------------------------------------
// Per-arch tuning table (CUB-style).
//
// CUB stores a nominal IPT in 4-byte-work units per (arch, algorithm) and
// scales at instantiation by `NOMINAL_ITEMS_PER_THREAD_4B * 4 / sizeof(T)`.
// We mirror that pattern: the table holds `nominal_ipt_4B` and
// `block_size`, and the lexer takes ELEM_BYTES = sizeof(index_t) as the
// type-size bucket — the max/add scans over start codes and produce
// flags are index_t-sized and drive the per-item register pressure at
// the block boundary.  state_t affects an earlier scan whose shmem cost
// is folded into `max_items_per_thread()` as a clamp.
//
// Values marked `[measured]` come from
// `benchmarks/sweep-cuda-lexer.sh` on the JSON grammar for the given
// arch and are the fastest observed setting at ELEM_BYTES = 4
// (state_t = u16, index_t = i32).  Values marked `[cub]` are copied
// directly from CUB's DeviceScan `NOMINAL_ITEMS_PER_THREAD_4B` and are
// starting points until we have measurements.  Rows with
// nominal_ipt_4B == 0 mean "unknown arch; fall back to the shmem-based
// search".  Replace `[cub]` with `[measured]` as sweep data comes in.
template<int SM_ARCH, size_t ELEM_BYTES>
struct alpacc_ipt_tuning {
  static constexpr uint32_t nominal_ipt_4B = 0;
  static constexpr uint32_t block_size     = 256;
};

// Pascal (sm_60, sm_61) — Tesla P100 / GP102          [cub]
template<size_t ELEM> struct alpacc_ipt_tuning<60, ELEM> {
  static constexpr uint32_t nominal_ipt_4B = 15;
  static constexpr uint32_t block_size     = 128;
};
template<size_t ELEM> struct alpacc_ipt_tuning<61, ELEM> {
  static constexpr uint32_t nominal_ipt_4B = 15;
  static constexpr uint32_t block_size     = 128;
};

// Volta (sm_70) — V100                                 [cub]
template<size_t ELEM> struct alpacc_ipt_tuning<70, ELEM> {
  static constexpr uint32_t nominal_ipt_4B = 15;
  static constexpr uint32_t block_size     = 128;
};

// Turing (sm_75) — T4, 1660 Ti, RTX 20xx               [measured]
// Sweep on the JSON grammar (endo_t = u64, index_t = i32, 10M-token dataset)
// with `benchmarks/sweep-cuda-lexer.sh` picked BS=256, IPT=8 as the fastest
// (5374 μs); IPT=12 was slower and IPT≥16 hit shmem overflow.
// endo_t is now u64 (was u16 state_t), so shmem budget is exhausted sooner.
// nominal_ipt_4B = 8 * 4 / 4 = 8.
template<size_t ELEM> struct alpacc_ipt_tuning<75, ELEM> {
  static constexpr uint32_t nominal_ipt_4B = 8;
  static constexpr uint32_t block_size     = 256;
};

// Ampere data-centre (sm_80) — A100                    [cub, needs re-sweep]
// Previous measurement used old u16 state_t; endo_t is now u64, needs re-sweep.
template<size_t ELEM> struct alpacc_ipt_tuning<80, ELEM> {
  static constexpr uint32_t nominal_ipt_4B = 8;
  static constexpr uint32_t block_size     = 256;
};

// Ampere consumer (sm_86) — RTX 30xx, A40              [cub]
template<size_t ELEM> struct alpacc_ipt_tuning<86, ELEM> {
  static constexpr uint32_t nominal_ipt_4B = 12;
  static constexpr uint32_t block_size     = 128;
};

// Ada Lovelace (sm_89) — RTX 40xx, L4/L40              [cub]
template<size_t ELEM> struct alpacc_ipt_tuning<89, ELEM> {
  static constexpr uint32_t nominal_ipt_4B = 12;
  static constexpr uint32_t block_size     = 128;
};

// Hopper (sm_90) — H100                                [cub]
template<size_t ELEM> struct alpacc_ipt_tuning<90, ELEM> {
  static constexpr uint32_t nominal_ipt_4B = 15;
  static constexpr uint32_t block_size     = 128;
};

// Blackwell (sm_100) — B100/B200                       [cub]
template<size_t ELEM> struct alpacc_ipt_tuning<100, ELEM> {
  static constexpr uint32_t nominal_ipt_4B = 15;
  static constexpr uint32_t block_size     = 128;
};

// Type-size bucket driver: sizeof(index_t).
// The tuning table entries are measured at a fixed endo_t width (8 bytes,
// one uint64_t word).  For wider endo_t the table no longer applies and
// arch_ipt() returns 0, falling back to the shmem-budget search.
// index_t drives register arrays (starts[], local_offs[]) and the SoA
// lookback buffers; it is the right axis for the table when endo_t is narrow.
template<typename endo_t, typename J>
constexpr size_t elem_bytes() {
  return sizeof(J);
}

// Table-driven IPT (0 if the arch is unknown or endo_t is wider than the
// measured baseline, in which case callers fall back to max_items_per_thread<>()).
template<int SM_ARCH, typename endo_t, typename J>
constexpr uint32_t arch_ipt() {
  // Table entries were measured with sizeof(endo_t) == 8 (one uint64_t word).
  // For wider endomorphisms the table is unreliable; return 0 to use the
  // shmem-budget search instead.
  if (sizeof(endo_t) > 8u) return 0;
  constexpr size_t bytes = elem_bytes<endo_t, J>();
  constexpr uint32_t nominal = alpacc_ipt_tuning<SM_ARCH, bytes>::nominal_ipt_4B;
  if (nominal == 0) return 0;
  // Scale from 4B-work units to the actual per-thread element size.
  uint32_t scaled = nominal * 4u / (uint32_t)bytes;
  return scaled == 0 ? 1 : scaled;
}

template<int SM_ARCH, typename endo_t, typename J>
constexpr uint32_t arch_block_size() {
  constexpr size_t bytes = elem_bytes<endo_t, J>();
  return alpacc_ipt_tuning<SM_ARCH, bytes>::block_size;
}

// ---------------------------------------------------------------------------
// Endomorphism helpers.
//
// endo_t is a struct { uint64_t w[ENDO_WORDS]; } encoding up to MAX_IMAGE_SIZE
// (in, out) pairs, each PAIR_BITS = 2*STATE_BITS bits wide, packed from
// bit 0 of w[0] across word boundaries.
// Bit 63 of w[ENDO_WORDS-1] is the identity flag. ENDO_IDENTITY has only this bit set.
// Char endomorphisms always have this bit clear (pair data never reaches it).
//
// Produce detection uses d_produce_matrix: after the inclusive prefix scan,
// the transition at position i is (s_before, s_after) where
//   s_before = eval_endo(prefix[i-1], INIT_STATE)
//   s_after  = eval_endo(prefix[i],   INIT_STATE)
// and is_produce = (d_produce_matrix[s_before] >> s_after) & 1.
//
// STATE_BITS, PAIR_BITS, STATE_MASK, MAX_IMAGE_SIZE, INIT_STATE, ENDO_WORDS,
// ENDO_IDENTITY and the tables h_endo / h_accept / h_produce_matrix are all
// baked in by the Haskell code generator above.
// ---------------------------------------------------------------------------

__device__ __constant__ uint64_t d_produce_matrix[NUM_STATES];

__device__ __host__ __forceinline__
bool endo_is_identity(const endo_t& e) {
  return (e.w[ENDO_WORDS - 1] >> 63) & 1;
}

__device__ __host__ __forceinline__
uint64_t endo_extract(const endo_t& e, int off, int bits) {
  int word = off / 64;
  int bit  = off % 64;
  uint64_t lo = e.w[word] >> bit;
  uint64_t hi = (bit + bits > 64 && word + 1 < ENDO_WORDS)
              ? (e.w[word + 1] << (64 - bit)) : 0;
  uint64_t mask = (bits < 64) ? ((uint64_t(1) << bits) - 1) : ~uint64_t(0);
  return (lo | hi) & mask;
}

__device__ __host__ __forceinline__
void endo_insert(endo_t& e, int off, int bits, uint64_t val) {
  int word = off / 64;
  int bit  = off % 64;
  e.w[word] |= val << bit;
  if (bit + bits > 64 && word + 1 < ENDO_WORDS)
    e.w[word + 1] |= val >> (64 - bit);
}

__device__ __host__ __forceinline__
endo_t endo_compose(const endo_t& f, const endo_t& g) {
  if (endo_is_identity(f)) return g;
  if (endo_is_identity(g)) return f;
  endo_t result;
  for (int k = 0; k < ENDO_WORDS; k++) result.w[k] = 0;
  int out = 0;
#pragma unroll
  for (int fi = 0; fi < MAX_IMAGE_SIZE; fi++) {
    int off_f  = fi * PAIR_BITS;
    int fi_in  = (int)endo_extract(f, off_f,              STATE_BITS);
    int fi_out = (int)endo_extract(f, off_f + STATE_BITS, STATE_BITS);
    if (fi_in == 0 && fi_out == 0) break;
#pragma unroll
    for (int gi = 0; gi < MAX_IMAGE_SIZE; gi++) {
      int off_g   = gi * PAIR_BITS;
      int gi_in   = (int)endo_extract(g, off_g,              STATE_BITS);
      int gi_out  = (int)endo_extract(g, off_g + STATE_BITS, STATE_BITS);
      if (gi_in == 0 && gi_out == 0) break;
      if (gi_in == fi_out) {
        int off_r = out * PAIR_BITS;
        endo_insert(result, off_r,              STATE_BITS, (uint64_t)fi_in);
        endo_insert(result, off_r + STATE_BITS, STATE_BITS, (uint64_t)gi_out);
        out++;
        break;
      }
    }
  }
  return result;
}

__device__ __host__ __forceinline__
int eval_endo(const endo_t& e, int query) {
  if (endo_is_identity(e)) return query;
#pragma unroll
  for (int i = 0; i < MAX_IMAGE_SIZE; i++) {
    int off = i * PAIR_BITS;
    int in  = (int)endo_extract(e, off,              STATE_BITS);
    int out = (int)endo_extract(e, off + STATE_BITS, STATE_BITS);
    if (in == 0 && out == 0) break;
    if (in == query) return out;
  }
  return 0;
}

__device__ __forceinline__
terminal_t get_terminal(endo_t e) {
  return static_cast<terminal_t>(h_terminal[eval_endo(e, INIT_STATE)]);
}

// Returns true if the transition s_before -> s_after produces a token.
__device__ __forceinline__
bool is_produce_transition(int s_before, int s_after) {
  return (d_produce_matrix[s_before] >> s_after) & 1;
}

// ---------------------------------------------------------------------------
// Fused (endo, max, add) decoupled lookback via monoid composition.
//
// Defined here (after endo_compose / eval_endo) so those helpers are in scope.
// scan.cu (emitted earlier, before grammar defines) contains only fully-generic
// helpers and must not reference endo_t / NUM_STATES / ENDO_WORDS.
//
// Elements: (f, B_max[NUM_STATES], B_add[NUM_STATES])
//   Monoid: (f1,B1) ★ (f2,B2) = ( f1∘f2,
//                                   q ↦ max(B1_max[q], B2_max[f1[q]]),
//                                   q ↦ B1_add[q]  + B2_add[f1[q]]  )
//
// Per-block state buffer: a single SoA lookback buffer carrying the endo
// aggregate/prefix plus NUM_STATES-wide B_max and B_add tables (aggregate
// and prefix, both u32).  One status atomic per block covers all three
// components — endo, B_max, B_add — via a single release/acquire pair.
// ---------------------------------------------------------------------------

struct FusedStates {
  volatile endo_t*   endo_aggregates = nullptr;
  volatile endo_t*   endo_prefixes   = nullptr;
  volatile uint32_t* bmax_aggregates = nullptr;   // [num_blocks * NUM_STATES]
  volatile uint32_t* bmax_prefixes   = nullptr;
  volatile uint32_t* badd_aggregates = nullptr;
  volatile uint32_t* badd_prefixes   = nullptr;
  AtomicStatus*      statuses        = nullptr;
  uint32_t num_blocks = 0;

  FusedStates() {}

  FusedStates(uint32_t nb) : num_blocks(nb) {
    size_t b_bytes = (size_t)nb * NUM_STATES * sizeof(uint32_t);
    cudaMalloc((void**)&endo_aggregates, nb * sizeof(endo_t));
    cudaMalloc((void**)&endo_prefixes,   nb * sizeof(endo_t));
    cudaMalloc((void**)&bmax_aggregates, b_bytes);
    cudaMalloc((void**)&bmax_prefixes,   b_bytes);
    cudaMalloc((void**)&badd_aggregates, b_bytes);
    cudaMalloc((void**)&badd_prefixes,   b_bytes);
    cudaMalloc((void**)&statuses,        nb * sizeof(AtomicStatus));
    cudaMemset((void*)statuses, Invalid, nb * sizeof(AtomicStatus));
  }

  void reset() {
    if (statuses) cudaMemset((void*)statuses, Invalid, num_blocks * sizeof(AtomicStatus));
  }

  void cleanUp() {
    if (endo_aggregates) cudaFree((void*)endo_aggregates);
    if (endo_prefixes)   cudaFree((void*)endo_prefixes);
    if (bmax_aggregates) cudaFree((void*)bmax_aggregates);
    if (bmax_prefixes)   cudaFree((void*)bmax_prefixes);
    if (badd_aggregates) cudaFree((void*)badd_aggregates);
    if (badd_prefixes)   cudaFree((void*)badd_prefixes);
    if (statuses)        cudaFree((void*)statuses);
    endo_aggregates = nullptr; endo_prefixes = nullptr;
    bmax_aggregates = nullptr; bmax_prefixes = nullptr;
    badd_aggregates = nullptr; badd_prefixes = nullptr;
    statuses = nullptr;
  }
};

struct LexerScanStates {
  FusedStates fused;

  LexerScanStates() {}
  LexerScanStates(uint32_t nb) : fused(nb) {}
  void reset()    { fused.reset(); }
  void cleanUp()  { fused.cleanUp(); }
};

struct EndoOp {
  __device__ __forceinline__
  endo_t operator()(const endo_t& a, const endo_t& b) const {
    return endo_compose(a, b);
  }
  __device__ __forceinline__
  endo_t operator()(const volatile endo_t& a, const volatile endo_t& b) const {
    endo_t na, nb;
    for (int k = 0; k < ENDO_WORDS; k++) { na.w[k] = a.w[k]; nb.w[k] = b.w[k]; }
    return endo_compose(na, nb);
  }
};

// Fused decoupled lookback: publishes this block's (endo, B_max, B_add)
// aggregate under one status atomic, then walks predecessors oldest-first
// via a plain serial loop on thread 0, implementing the semiring
// composition monoid
//   (a1,b1) ★ (a2,b2) := (a1∘a2,  b1 ⊕ a1·b2)
// from docs/fused-lookback-monoid.md.
//
// This is the simplest correct version: no warp parallelism, no early
// termination — thread 0 waits for each predecessor's status to become
// Prefix, then folds it into the running (F, C) via ★ in oldest → newest
// order.  When the walk hits a predecessor whose status is Prefix, that
// predecessor's `*_prefixes` arrays already absorb everything to its left,
// so we start the walk at the *highest* p with Prefix (thread 0 scans
// backwards to find it first) and then fold p, p+1, …, dyn_idx-1 in order.
__device__ inline void
fusedLookback(FusedStates states,
              uint32_t dyn_idx,
              const endo_t& endo_agg,
              const uint32_t* bmax_agg,   // shmem [NUM_STATES]
              const uint32_t* badd_agg,   // shmem [NUM_STATES]
              uint32_t* bmax_pfx_sh,      // shmem [NUM_STATES] — running C
              uint32_t* badd_pfx_sh,      // shmem [NUM_STATES] — running C
              endo_t& out_endo_pfx,
              uint32_t& out_bmax_scalar,  // = running_C[INIT_STATE]
              uint32_t& out_badd_scalar) {
  volatile __shared__ endo_t shmem_endo_pfx;
  const bool is_first = threadIdx.x == 0;

  if (is_first) {
    states.statuses[dyn_idx].store(Invalid, cuda::memory_order_relaxed);
  }
  __syncthreads();

  // Publish this block's aggregate: endo (thread 0), B_max/B_add (all threads
  // cooperate across NUM_STATES).
  if (is_first) {
    for (int k = 0; k < ENDO_WORDS; k++)
      states.endo_aggregates[dyn_idx].w[k] = endo_agg.w[k];
  }
  {
    const uint32_t base = dyn_idx * (uint32_t)NUM_STATES;
    for (int q = threadIdx.x; q < NUM_STATES; q += blockDim.x) {
      states.bmax_aggregates[base + q] = bmax_agg[q];
      states.badd_aggregates[base + q] = badd_agg[q];
    }
  }

  // Zero the running B prefix.
  for (int q = threadIdx.x; q < NUM_STATES; q += blockDim.x) {
    bmax_pfx_sh[q] = 0u;
    badd_pfx_sh[q] = 0u;
  }

  // Block 0: aggregate = prefix; publish and mark Prefix directly.
  if (dyn_idx == 0) {
    if (is_first) {
      for (int k = 0; k < ENDO_WORDS; k++)
        states.endo_prefixes[dyn_idx].w[k] = endo_agg.w[k];
    }
    const uint32_t base = dyn_idx * (uint32_t)NUM_STATES;
    for (int q = threadIdx.x; q < NUM_STATES; q += blockDim.x) {
      states.bmax_prefixes[base + q] = bmax_agg[q];
      states.badd_prefixes[base + q] = badd_agg[q];
    }
    __threadfence();
    if (is_first) states.statuses[dyn_idx].store(Prefix, cuda::memory_order_release);

    if (is_first) shmem_endo_pfx = d_ENDO_IDENTITY;
    __syncthreads();
    endo_t r;
    for (int k = 0; k < ENDO_WORDS; k++) r.w[k] = shmem_endo_pfx.w[k];
    out_endo_pfx = r;
    out_bmax_scalar = 0u;
    out_badd_scalar = 0u;
    return;
  }

  // Non-block-0: mark Aggregate, then thread 0 does a plain serial walk.
  __threadfence();
  if (is_first) states.statuses[dyn_idx].store(Aggregate, cuda::memory_order_release);
  __syncthreads();

  // Thread 0: find the highest p < dyn_idx whose status is Prefix (spin
  // until each candidate resolves).  Walk backwards from dyn_idx-1 down;
  // the first Prefix we see is our start.  If no p in the whole range has
  // Prefix (only possible when dyn_idx == 0, already handled), we'd start
  // at 0 — but there's always at least block 0, which eventually reaches
  // Prefix.
  if (is_first) {
    // Step A: wait for every predecessor's status to be at least Aggregate
    // (which means its endo/B aggregates are visible), then find the
    // highest Prefix predecessor by scanning back.  Simplest: scan back
    // from dyn_idx-1, and for each p spin until status != Invalid.  Stop
    // at first Prefix.
    int start_p = 0;
    for (int p = (int)dyn_idx - 1; p >= 0; p--) {
      Status s;
      do {
        s = states.statuses[p].load(cuda::memory_order_acquire);
      } while (s == Invalid);
      if (s == Prefix) { start_p = p; break; }
      // else Aggregate — keep scanning back.
    }

    // Step B: fold predecessors start_p, start_p+1, …, dyn_idx-1 in ★
    // order (oldest→newest).  At start_p, if its status is Prefix use its
    // *_prefixes (absorbs everything ≤ start_p); at later p's, we spin
    // until status resolves to at least Aggregate.
    endo_t running_F = d_ENDO_IDENTITY;
    for (int p = start_p; p < (int)dyn_idx; p++) {
      Status s;
      if (p == start_p) {
        // Already known Prefix from Step A (or start_p == 0 with any status).
        s = states.statuses[p].load(cuda::memory_order_acquire);
      } else {
        do {
          s = states.statuses[p].load(cuda::memory_order_acquire);
        } while (s == Invalid);
      }
      bool use_pfx = (p == start_p) && (s == Prefix);
      const uint32_t base = (uint32_t)p * (uint32_t)NUM_STATES;
      volatile uint32_t* bmax_src = use_pfx ? states.bmax_prefixes : states.bmax_aggregates;
      volatile uint32_t* badd_src = use_pfx ? states.badd_prefixes : states.badd_aggregates;
      volatile endo_t*   endo_src = use_pfx ? states.endo_prefixes : states.endo_aggregates;
      // running_C[q] ⊕= B_p[running_F(q)]
      for (int q = 0; q < NUM_STATES; q++) {
        int qp = eval_endo(running_F, q);
        uint32_t vm = bmax_src[base + qp];
        uint32_t va = badd_src[base + qp];
        if (vm > bmax_pfx_sh[q]) bmax_pfx_sh[q] = vm;
        badd_pfx_sh[q] += va;
      }
      // running_F = running_F ∘ f_p
      endo_t e_p;
      for (int k = 0; k < ENDO_WORDS; k++) e_p.w[k] = endo_src[p].w[k];
      running_F = endo_compose(running_F, e_p);
    }
    for (int k = 0; k < ENDO_WORDS; k++) shmem_endo_pfx.w[k] = running_F.w[k];
  }
  __syncthreads();

  endo_t f_pfx_bcast;
  for (int k = 0; k < ENDO_WORDS; k++) f_pfx_bcast.w[k] = shmem_endo_pfx.w[k];
  out_endo_pfx = f_pfx_bcast;

  // Publish this block's prefix: prefix = pfx ★ agg per the monoid rule.
  //   endo_prefix[q]  = f_pfx ∘ endo_agg
  //   bmax_prefix[q]  = max(bmax_pfx[q], bmax_agg[f_pfx(q)])
  //   badd_prefix[q]  = badd_pfx[q]  +  badd_agg[f_pfx(q)]
  if (is_first) {
    endo_t combined = endo_compose(f_pfx_bcast, endo_agg);
    for (int k = 0; k < ENDO_WORDS; k++)
      states.endo_prefixes[dyn_idx].w[k] = combined.w[k];
  }
  {
    const uint32_t base = dyn_idx * (uint32_t)NUM_STATES;
    for (int q = threadIdx.x; q < NUM_STATES; q += blockDim.x) {
      int qp = eval_endo(f_pfx_bcast, q);
      uint32_t m  = bmax_pfx_sh[q];
      uint32_t am = bmax_agg[qp];
      uint32_t pm = (m > am) ? m : am;
      uint32_t pa = badd_pfx_sh[q] + badd_agg[qp];
      states.bmax_prefixes[base + q] = pm;
      states.badd_prefixes[base + q] = pa;
    }
  }
  __threadfence();
  if (is_first) states.statuses[dyn_idx].store(Prefix, cuda::memory_order_release);

  // Extract the scalar prefixes at INIT_STATE for this block.
  out_bmax_scalar = bmax_pfx_sh[INIT_STATE];
  out_badd_scalar = badd_pfx_sh[INIT_STATE];
}

template<typename I, typename J>
struct LexerCtx {

private:
  J offset = 0;
  endo_t* d_endo;                       // device copy of h_endo[256]
  volatile uint32_t* d_dyn_block_index;
  volatile endo_t* d_new_last_endo;
  volatile endo_t* d_old_last_endo;
  I* d_new_size;
  volatile J* d_new_last_start;
  volatile J* d_old_last_start;
  volatile uint32_t* d_len_overflow;

  void swapLastStart() {
    J h_last_start;
    gpuAssert(cudaMemcpy(&h_last_start, (const void*) d_new_last_start, sizeof(J), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy((void *) d_new_last_start, (const void*) d_old_last_start, sizeof(J), cudaMemcpyDeviceToDevice));
    gpuAssert(cudaMemcpy((void *) d_old_last_start, &h_last_start, sizeof(J), cudaMemcpyHostToDevice));
  }

  void swapLastEndo() {
    endo_t h_last_endo;
    gpuAssert(cudaMemcpy(&h_last_endo, (const void*) d_new_last_endo, sizeof(endo_t), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy((void *) d_new_last_endo, (const void*) d_old_last_endo, sizeof(endo_t), cudaMemcpyDeviceToDevice));
    gpuAssert(cudaMemcpy((void *) d_old_last_endo, &h_last_endo, sizeof(endo_t), cudaMemcpyHostToDevice));
  }

  void resetDynamicIndex() const {
    cudaMemset((void*)d_dyn_block_index, 0, sizeof(uint32_t));
  }

  void updateOffset() {
    offset += CHUNK_SIZE;
  }

  void resetNewSize() const {
    cudaMemset(d_new_size, 0, sizeof(I));
  }

public:
  const I CHUNK_SIZE;
  LexerScanStates d_scan_states;

  LexerCtx(const I chunk_size,
           const I block_size,
           const I items_per_thread) : CHUNK_SIZE(chunk_size) {
    I num_blocks = numBlocks(chunk_size, block_size, items_per_thread);
    gpuAssert(cudaMalloc(&d_endo, sizeof(h_endo)));
    cudaMemcpy(d_endo, h_endo, sizeof(h_endo), cudaMemcpyHostToDevice);
    d_scan_states = LexerScanStates(num_blocks);

    gpuAssert(cudaMalloc((void**)&d_dyn_block_index, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_new_last_endo, sizeof(endo_t)));
    gpuAssert(cudaMalloc((void**)&d_old_last_endo, sizeof(endo_t)));
    gpuAssert(cudaMalloc((void**)&d_new_last_start, sizeof(J)));
    gpuAssert(cudaMalloc((void**)&d_old_last_start, sizeof(J)));
    gpuAssert(cudaMalloc((void**)&d_len_overflow, sizeof(uint32_t)));

    cudaMemset((void*)d_dyn_block_index, 0, sizeof(uint32_t));
    cudaMemset((void*)d_new_size, 0, sizeof(I));
    endo_t identity = ENDO_IDENTITY;
    cudaMemcpyToSymbol(d_ENDO_IDENTITY, &identity, sizeof(endo_t));
    cudaMemcpyToSymbol(d_produce_matrix, h_produce_matrix, sizeof(h_produce_matrix));
    cudaMemcpy((void*)d_new_last_endo, &identity, sizeof(endo_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_old_last_endo, &identity, sizeof(endo_t), cudaMemcpyHostToDevice);
    cudaMemset((void*)d_new_last_start, 0, sizeof(J));
    cudaMemset((void*)d_old_last_start, 0, sizeof(J));
    cudaMemset((void*)d_len_overflow, 0, sizeof(uint32_t));
  }

  void reset() {
    offset = 0;
    cudaMemset((void*)d_dyn_block_index, 0, sizeof(uint32_t));
    cudaMemset((void*)d_new_size, 0, sizeof(I));
    endo_t identity = ENDO_IDENTITY;
    cudaMemcpy((void*)d_new_last_endo, &identity, sizeof(endo_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_old_last_endo, &identity, sizeof(endo_t), cudaMemcpyHostToDevice);
    cudaMemset((void*)d_new_last_start, 0, sizeof(J));
    cudaMemset((void*)d_old_last_start, 0, sizeof(J));
    cudaMemset((void*)d_len_overflow, 0, sizeof(uint32_t));
    d_scan_states.reset();
  }

  void cleanUp() {
    if (d_endo) cudaFree(d_endo);
    if (d_new_last_start) cudaFree((void*)d_new_last_start);
    if (d_old_last_start) cudaFree((void*)d_old_last_start);
    if (d_dyn_block_index) cudaFree((void*)d_dyn_block_index);
    if (d_new_size) cudaFree((void*)d_new_size);
    if (d_new_last_endo) cudaFree((void*)d_new_last_endo);
    if (d_old_last_endo) cudaFree((void*)d_old_last_endo);
    if (d_len_overflow) cudaFree((void*)d_len_overflow);
    d_scan_states.cleanUp();
  }

  __device__ __host__ __forceinline__
  endo_t operator()(const endo_t& a, const endo_t& b) const {
    return endo_compose(a, b);
  }

  __device__ __host__ __forceinline__
  endo_t operator()(const volatile endo_t& a, const volatile endo_t& b) const {
    endo_t na, nb;
    for (int k = 0; k < ENDO_WORDS; k++) { na.w[k] = a.w[k]; nb.w[k] = b.w[k]; }
    return endo_compose(na, nb);
  }

  __device__ __host__ __forceinline__
  endo_t toState(const uint8_t &a) const {
#ifdef __CUDA_ARCH__
    endo_t r;
    for (int k = 0; k < ENDO_WORDS; k++)
      r.w[k] = __ldg(&d_endo[a].w[k]);
    return r;
#else
    return d_endo[a];
#endif
  }

  __device__ __host__ __forceinline__
  J addOffset(I i) const {
    return i + offset;
  }

  __device__ __forceinline__
  uint32_t getDynamicIndex() const {
    return dynamicIndex(d_dyn_block_index);
  }

  __device__ __host__ __forceinline__
  void setLastState(endo_t e) const {
    for (int k = 0; k < ENDO_WORDS; k++) d_new_last_endo->w[k] = e.w[k];
  }

  __device__ __host__ __forceinline__
  endo_t getLastState() const {
    endo_t r;
    for (int k = 0; k < ENDO_WORDS; k++) r.w[k] = d_old_last_endo->w[k];
    return r;
  }

  __device__ __host__ __forceinline__
  void setNewSize(I size) const {
    *d_new_size = size;
  }

  __device__ __host__ __forceinline__
  void setLastStart(J i) const {
    *d_new_last_start = i;
  }

  __device__ __host__ __forceinline__
  J getLastStart() const {
    return *d_old_last_start;
  }

  __device__ __forceinline__
  void signalLengthOverflow() const {
    if (*d_len_overflow == 0u)
      atomicOr((uint32_t*)d_len_overflow, 1u);
  }

  bool isOverflow() const {
    uint32_t overflow = 0;
    gpuAssert(cudaMemcpy(&overflow, (const void*) d_len_overflow, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return overflow != 0;
  }

  bool isAccept() const {
    endo_t h_last_endo;
    gpuAssert(cudaMemcpy(&h_last_endo, (const void*) d_new_last_endo, sizeof(endo_t), cudaMemcpyDeviceToHost));
    return h_accept[eval_endo(h_last_endo, INIT_STATE)];
  }

  I terminalsSize() const {
    I h_new_size = I();
    gpuAssert(cudaMemcpy(&h_new_size, (const void*) d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    return h_new_size;
  }

  void update() {
    resetDynamicIndex();
    swapLastStart();
    swapLastEndo();
    updateOffset();
  }
};

// Variant G: __launch_bounds__ removed. Nvcc picks regs/thread freely; if
// the scatter's register capture pushes past 64 regs/thread, occupancy is
// register-limited rather than shmem-limited.
template<typename I, typename J, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
lexer(LexerCtx<I, J> ctx, uint8_t* d_string, terminal_t* d_terminals, J* d_starts, length_t* d_lengths, const I size, const bool is_last_chunk) {
  constexpr I SHMEM_STRIDE =
      (I)shmem_pad_stride<I, endo_t, BLOCK_SIZE>(ITEMS_PER_THREAD);
  __shared__ endo_t states[SHMEM_STRIDE * BLOCK_SIZE];
  // Exchange buffer for the two-phase scatter on dense tiles.
  // exch_t (terminals), exch_j (starts), and exch_l (lengths) are never live
  // simultaneously, so they share one shmem region via a union.
  constexpr I EXCH_ELEMS = ITEMS_PER_THREAD * BLOCK_SIZE;
  union {
    terminal_t as_t[EXCH_ELEMS];
    J          as_j[EXCH_ELEMS];
    length_t   as_l[EXCH_ELEMS];
  } __shared__ exch;
  terminal_t* exch_t = exch.as_t;
  J*          exch_j = exch.as_j;
  length_t*   exch_l = exch.as_l;
  __shared__ endo_t next_block_first_state;

  // Phase A reads directly from ctx.d_to_state via __ldg() into the
  // states[] tile.

  // Main slots: ceil(ITEMS_PER_THREAD / 8) uint64_t registers per thread.
  // One extra slot is added to cover the single byte past the block boundary
  // needed for the boundary produce check (same as the original +1 trick).
  constexpr uint32_t VPT = vecPerThread<uint8_t, uint64_t, ITEMS_PER_THREAD>() + 1;
  uint64_t copy_reg[VPT];
  uint8_t* chars_reg = reinterpret_cast<uint8_t*>(copy_reg);
  uint32_t is_produce_state = 0;

  uint32_t dyn_index = ctx.getDynamicIndex();
  I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;

  if (threadIdx.x == I()) {
    next_block_first_state = d_ENDO_IDENTITY;
  }

  // Vectorized global → registers.  glbToReg covers the first VPT-1 slots
  // (ITEMS_PER_THREAD bytes).  The last slot is loaded separately to reach the
  // one byte immediately past the block boundary.
  glbToReg<uint8_t, uint64_t, BLOCK_SIZE, ITEMS_PER_THREAD>(glb_offs, size, d_string, copy_reg);
  {
    constexpr uint32_t EPV = (uint32_t)sizeof(uint64_t);
    constexpr uint32_t v   = VPT - 1;
    I elem0       = (I)(v * BLOCK_SIZE + threadIdx.x) * EPV;
    I n_remaining = size - glb_offs;
    if (elem0 + EPV <= n_remaining) {
      copy_reg[v] = *reinterpret_cast<const uint64_t*>(d_string + glb_offs + elem0);
    } else {
      uint8_t* bytes = reinterpret_cast<uint8_t*>(&copy_reg[v]);
#pragma unroll
      for (uint32_t b = 0; b < EPV; b++) {
        I gid = elem0 + (I)b;
        bytes[b] = (gid < n_remaining) ? d_string[glb_offs + gid] : 0;
      }
    }
  }

  // Phase A: byte -> state_t written directly into the states[] tile via
  // ctx.toState() (__ldg-cached on the 256-entry d_to_state table).
  {
#pragma unroll
    for (uint32_t i = 0; i < VPT; i++) {
      I lid = (I)i * BLOCK_SIZE + threadIdx.x;
      I _gid = glb_offs + (I)sizeof(uint64_t) * lid;
      for (uint32_t j = 0; j < sizeof(uint64_t); j++) {
        I gid = _gid + (I)j;
        I lid_off = (I)sizeof(uint64_t) * lid + (I)j;
        uint32_t reg_off = sizeof(uint64_t) * i + j;
        bool is_in_block = lid_off < (I)(ITEMS_PER_THREAD * BLOCK_SIZE);
        if (is_in_block) {
          endo_t s = (gid < size) ? ctx.toState(chars_reg[reg_off]) : d_ENDO_IDENTITY;
          I shmem_idx = (lid_off / ITEMS_PER_THREAD) * SHMEM_STRIDE
                      + (lid_off % ITEMS_PER_THREAD);
          states[shmem_idx] = s;
        } else if (lid_off == (I)(ITEMS_PER_THREAD * BLOCK_SIZE) && gid < size) {
          // First byte of the next block for boundary produce test.
          next_block_first_state = ctx.toState(chars_reg[reg_off]);
        }
      }
    }
  }
  __syncthreads();

  // Phase B: block-local endo scan (no inter-block prefix applied yet).
  // st[i] becomes the inclusive local scan; f_agg is the block aggregate.
  // Phase B: block-local endo scan, B_agg computation, fused lookback,
  // produce detection, and block-local u32 scans — all in flat scope.
  endo_t st[ITEMS_PER_THREAD];
  uint32_t start_codes[ITEMS_PER_THREAD];
  uint32_t produce_flags[ITEMS_PER_THREAD];
  I max_prefix = I();
  I prefix     = I();
  {
    const I off = threadIdx.x * SHMEM_STRIDE;
    bool is_chunk_first = (glb_offs == 0) && (threadIdx.x == 0);
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
      st[i] = states[off + i];
      if (is_chunk_first && i == 0)
        st[i] = ctx(ctx.getLastState(), st[i]);
    }
    using BlockScanE = cub::BlockScan<endo_t, BLOCK_SIZE>;
    __shared__ typename BlockScanE::TempStorage endo_temp;
    endo_t f_agg;
    BlockScanE(endo_temp).InclusiveScan(st, st, ctx, f_agg);

    // Write local endo scan back to shmem so thread 0 can iterate all items
    // to compute B_max[q] and B_add[q] aggregates for each starting state q.
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
      states[threadIdx.x * SHMEM_STRIDE + i] = st[i];
    __syncthreads();

    // Build B_max[q], B_add[q] aggregates in shmem via shmem atomics.
    // Each thread walks its own IPT items and does atomicMax / atomicAdd
    // into bmax_agg_sh[q] / badd_agg_sh[q].  For numfmt (NUM_STATES=458)
    // the serial thread-0 loop would be ~1M ops on one thread; distributing
    // across 256 threads via shmem atomics is a much shorter critical path.
    __shared__ uint32_t bmax_agg_sh[NUM_STATES];
    __shared__ uint32_t badd_agg_sh[NUM_STATES];
    __shared__ uint32_t bmax_pfx_sh[NUM_STATES];
    __shared__ uint32_t badd_pfx_sh[NUM_STATES];
    for (int q = threadIdx.x; q < NUM_STATES; q += BLOCK_SIZE) {
      bmax_agg_sh[q] = 0u;
      badd_agg_sh[q] = 0u;
    }
    __syncthreads();

    // Thread 0 iterates all items in the block to compute B_agg[q] for all q.
    // (Simple serial version; parallelise later once correctness is established.)
    if (threadIdx.x == 0) {
      for (I lid = 0; lid < (I)(BLOCK_SIZE * ITEMS_PER_THREAD); lid++) {
        I gid = glb_offs + lid;
        if (gid >= size) break;
        I ti = lid / ITEMS_PER_THREAD;
        I ii = lid % ITEMS_PER_THREAD;
        endo_t cur = states[ti * SHMEM_STRIDE + ii];
        endo_t prev_endo = (lid > 0)
            ? states[(lid-1)/ITEMS_PER_THREAD * SHMEM_STRIDE + (lid-1)%ITEMS_PER_THREAD]
            : d_ENDO_IDENTITY;
        endo_t nxt;
        bool has_next;
        if (lid == (I)(BLOCK_SIZE * ITEMS_PER_THREAD) - 1) {
          nxt = ctx(cur, next_block_first_state);
          has_next = true;
        } else {
          I ti_n = (lid + 1) / ITEMS_PER_THREAD;
          I ii_n = (lid + 1) % ITEMS_PER_THREAD;
          nxt = (gid + 1 < size) ? states[ti_n * SHMEM_STRIDE + ii_n] : cur;
          has_next = (gid + 1 < size);
        }
#ifdef IGNORE_TOKEN
        bool cur_is_ignore = (get_terminal(cur) == IGNORE_TOKEN);
#endif
        for (int q = 0; q < NUM_STATES; q++) {
          int s_cur  = eval_endo(cur, q);
          int s_prev = (lid == 0) ? q : eval_endo(prev_endo, q);
          if (is_produce_transition(s_prev, s_cur)) {
            if ((uint32_t)(gid + 1) > bmax_agg_sh[q])
              bmax_agg_sh[q] = (uint32_t)(gid + 1);
          }
          bool is_next_p = false;
          if (has_next) {
            int s_next = eval_endo(nxt, q);
            is_next_p = is_produce_transition(s_cur, s_next);
          }
          if (is_last_chunk && gid == size - 1) is_next_p = true;
#ifdef IGNORE_TOKEN
          if (is_next_p && cur_is_ignore) is_next_p = false;
#endif
          if (is_next_p) badd_agg_sh[q] += 1u;
        }
      }
    }
    __syncthreads();

    // Wrap shmem aggregates as raw pointers (uniform read across threads).
    // Note the fused-lookback API takes `const uint32_t*`; shmem arrays
    // satisfy this once __syncthreads() above has published the writes.
    endo_t f_pfx;
    uint32_t raw_max_prefix, raw_add_prefix;
    fusedLookback(ctx.d_scan_states.fused,
                  dyn_index,
                  f_agg,
                  bmax_agg_sh, badd_agg_sh,
                  bmax_pfx_sh, badd_pfx_sh,
                  f_pfx,
                  raw_max_prefix, raw_add_prefix);
    max_prefix = (I)raw_max_prefix;
    prefix     = (I)raw_add_prefix;

    // Apply endo prefix to shmem for produce detection and scatter.
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
      states[threadIdx.x * SHMEM_STRIDE + i] = ctx(f_pfx, st[i]);
    __syncthreads();
  } // end Phase B

  // Produce detection over post-prefix shmem endos.
#pragma unroll
  for (I i = 0; i < ITEMS_PER_THREAD; i++) {
    I lid = threadIdx.x * ITEMS_PER_THREAD + i;
    I gid = glb_offs + lid;
    bool is_next_produce = false;
    uint32_t start_code = 0;
    I shmem_cur  = (I)threadIdx.x * SHMEM_STRIDE + i;
    I shmem_next = (i < ITEMS_PER_THREAD - 1)
                   ? shmem_cur + 1
                   : ((I)threadIdx.x + 1) * SHMEM_STRIDE;
    I shmem_prev = (i > 0)
                   ? shmem_cur - 1
                   : (threadIdx.x > 0)
                     ? (I)(threadIdx.x - 1) * SHMEM_STRIDE + (ITEMS_PER_THREAD - 1)
                     : (I)-1;
    if (gid < size) {
      endo_t state = states[shmem_cur];
      int s_cur = eval_endo(state, INIT_STATE);
#ifdef IGNORE_TOKEN
      bool is_not_ignore = get_terminal(state) != IGNORE_TOKEN;
#else
      bool is_not_ignore = true;
#endif
      if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
        int s_next = eval_endo(ctx(state, next_block_first_state), INIT_STATE);
        is_next_produce = is_produce_transition(s_cur, s_next);
      } else {
        int s_next = eval_endo(states[shmem_next], INIT_STATE);
        is_next_produce = is_produce_transition(s_cur, s_next);
      }
      if (is_last_chunk) {
        is_next_produce |= gid == size - 1;
        is_next_produce &= is_not_ignore;
      } else {
        is_next_produce &= is_not_ignore;
      }
      int s_prev = (lid == 0) ? INIT_STATE : eval_endo(states[shmem_prev], INIT_STATE);
      bool this_produce = is_produce_transition(s_prev, s_cur);
      start_code = this_produce ? (uint32_t)(gid + 1) : 0u;
    }
    is_produce_state |= is_next_produce << i;
    start_codes[i]   = start_code;
    produce_flags[i] = is_next_produce ? 1u : 0u;
  }

  // Block-local u32 scans (inter-block prefix already in max_prefix/prefix).
  using BlockScan32 = cub::BlockScan<uint32_t, BLOCK_SIZE>;
  __shared__ typename BlockScan32::TempStorage scan_temp;
  scanRegLocal<uint32_t, I, Max<uint32_t>, ITEMS_PER_THREAD, BLOCK_SIZE>(
      start_codes, scan_temp, Max<uint32_t>());
  scanRegLocal<uint32_t, I, Add<uint32_t>, ITEMS_PER_THREAD, BLOCK_SIZE>(
      produce_flags, scan_temp, Add<uint32_t>());

  I starts[ITEMS_PER_THREAD];
  I local_offs[ITEMS_PER_THREAD];
  __shared__ I last_start;
  __shared__ I num_sel_sh;

#pragma unroll
  for (I i = 0; i < ITEMS_PER_THREAD; i++) {
    I lid = threadIdx.x * ITEMS_PER_THREAD + i;
    I gid = glb_offs + lid;
    starts[i] = max(max_prefix, (I)start_codes[i]);
    local_offs[i] = ((is_produce_state >> i) & 1) ? (I)produce_flags[i] - 1 : I();
    if (gid == size - 1) {
      last_start = starts[i];
    }
  }

  if (threadIdx.x == BLOCK_SIZE - 1) {
    num_sel_sh = (I)produce_flags[ITEMS_PER_THREAD - 1];
  }
  __syncthreads();

  const I num_sel = num_sel_sh;

  if (dyn_index == gridDim.x - 1 && threadIdx.x == blockDim.x - 1) {
    ctx.setNewSize(Add<I>()(prefix, num_sel));
    ctx.setLastState(states[(BLOCK_SIZE - 1) * SHMEM_STRIDE + (ITEMS_PER_THREAD - 1)]);  // stores endo_t

    if (last_start != I()) {
      ctx.setLastStart(ctx.addOffset(last_start - 1));
    } else {
      ctx.setLastStart(ctx.getLastStart());
    }
  }

  if (num_sel > BLOCK_SIZE) {
    // Dense tile: two-phase scatter for terminals, starts, and lengths.
    // Each array is compacted into the shmem exchange at tile-local offsets,
    // then written out as coalesced wide stores.
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
      if ((is_produce_state >> i) & 1) {
        I shmem_cur = (I)threadIdx.x * (SHMEM_STRIDE) + i;
        exch_t[local_offs[i]] = get_terminal(states[shmem_cur]);
      }
    }
    __syncthreads();
    shmemToGlbVec<terminal_t, uint64_t, BLOCK_SIZE, I>(prefix, num_sel, d_terminals, exch_t);
    __syncthreads();

    // Compute the tok_start values into per-thread registers *and* into
    // exch_j in the same pass.  Keeping them in registers means the length
    // loop below reads its own thread's register value directly instead of
    // going through shmem — that avoids the extra sync we'd otherwise need
    // between shmemToGlbVec's exch_j reads and the exch_l writes below
    // (exch_l aliases exch_j via the union).
    J tok_starts[ITEMS_PER_THREAD];
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
      if ((is_produce_state >> i) & 1) {
        I offset = local_offs[i];
        J v;
        if (Add<I>()(prefix, offset) == I() && starts[i] == I()) {
          v = ctx.getLastStart();
        } else {
          v = ctx.addOffset(starts[i] - 1);
        }
        tok_starts[i] = v;
        exch_j[offset] = v;
      }
    }
    __syncthreads();
    shmemToGlbVec<J, uint64_t, BLOCK_SIZE, I>(prefix, num_sel, d_starts, exch_j);
    __syncthreads();

    // Compute lengths directly from the per-thread tok_starts registers
    // and write straight to exch_l.  Since we already have tok_starts[i]
    // in a register, no shmem read is needed here.  With no exch_j read
    // to protect, we can write exch_l (aliased to exch_j) as soon as the
    // shmemToGlbVec above is fenced by the sync above — one fewer sync
    // than the previous three-shmem-pass scheme.
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
      if ((is_produce_state >> i) & 1) {
        I gid = glb_offs + (I)threadIdx.x * ITEMS_PER_THREAD + (I)i;
        J tok_len = ctx.addOffset(gid + 1) - tok_starts[i];
        if ((length_t)(tok_len) != tok_len) ctx.signalLengthOverflow();
        exch_l[local_offs[i]] = (length_t)tok_len;
      }
    }
    __syncthreads();
    shmemToGlbVec<length_t, uint64_t, BLOCK_SIZE, I>(prefix, num_sel, d_lengths, exch_l);
  } else {
    // Sparse tile: direct scatter.
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
      I lid = threadIdx.x * ITEMS_PER_THREAD + i;
      I gid = glb_offs + lid;
      if ((is_produce_state >> i) & 1) {
        I shmem_cur = (I)threadIdx.x * (SHMEM_STRIDE) + i;
        I offset = Add<I>()(prefix, local_offs[i]);
        d_terminals[offset] = get_terminal(states[shmem_cur]);
        J tok_start, tok_end, tok_len;
        if (offset == I() && starts[i] == I()) {
          tok_start = ctx.getLastStart();
        } else {
          tok_start = ctx.addOffset(starts[i] - 1);
        }
        tok_end = ctx.addOffset(gid + 1);
        tok_len = tok_end - tok_start;
        if ((length_t)(tok_len) != tok_len) ctx.signalLengthOverflow();
        d_starts[offset]  = tok_start;
        d_lengths[offset] = (length_t)tok_len;
      }
    }
  }
}

