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
    uint32_t stride_banks   = items_per_thread * banks_per_elem;
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
  // s_boundary[BLOCK_SIZE+1]: one int per thread plus one for next-block state.
  size_t boundary_bytes = sizeof(int) * (BLOCK_SIZE + 1);
  // states[] and exch overlap in time; s_boundary[] lives after exch within states[].
  // Total = max(states_bytes, exch_bytes + boundary_bytes).
  size_t overlap = states_bytes > (exch_bytes + boundary_bytes)
                 ? states_bytes : (exch_bytes + boundary_bytes);
  return overlap;
}

template<typename I, typename state_t, uint32_t BLOCK_SIZE>
constexpr size_t lexer_shmem_fixed() {
  size_t cub_state_temp = sizeof(typename cub::BlockScan<state_t, BLOCK_SIZE>::TempStorage);
  size_t cub_u32_temp   = sizeof(typename cub::BlockScan<uint32_t,  BLOCK_SIZE>::TempStorage);
  size_t lookback = 2u * sizeof(I) * WARP + sizeof(uint8_t) * WARP + 2u * sizeof(I);
  size_t fixed_scalars = sizeof(state_t)    // next_block_first_state
                        + sizeof(I)          // last_start
                        + sizeof(I);         // num_sel_sh
  return cub_state_temp + cub_u32_temp + lookback + fixed_scalars;
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
// Bit 63 of w[0] is the identity flag. ENDO_IDENTITY has only this bit set.
// Char endomorphisms always have bit 63 of w[0] clear.
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
  return (e.w[0] >> 63) & 1;
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
  States<I, endo_t> d_state_states;
  PairStates<I, I, I> d_maxadd_states;

  LexerCtx(const I chunk_size,
           const I block_size,
           const I items_per_thread) : CHUNK_SIZE(chunk_size) {
    I num_blocks = numBlocks(chunk_size, block_size, items_per_thread);
    gpuAssert(cudaMalloc(&d_endo, sizeof(h_endo)));
    cudaMemcpy(d_endo, h_endo, sizeof(h_endo), cudaMemcpyHostToDevice);
    d_maxadd_states = PairStates<I, I, I>(num_blocks);
    d_state_states = States<I, endo_t>(num_blocks);

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
    d_maxadd_states.reset();
    d_state_states.reset();
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
    d_maxadd_states.cleanUp();
    d_state_states.cleanUp();
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

  // Phase A tile: striped layout, written then read in blocked layout for scan.
  // After the scan the full endo shmem is no longer needed — we keep scanned
  // endos in registers and only exchange one int (evaluated state) per thread
  // boundary via s_boundary[].  The endo tile and exch union therefore overlap
  // in time; we put the exch union first so it aliases the endo tile memory.
  //
  // Layout in shmem (all fields at the same address, used at different times):
  //
  //  [0 .. EXCH_BYTES)        exch union  (terminals / starts / lengths)
  //  [0 .. ENDO_TILE_BYTES)   states[]    (only during phase A + scan)
  //  [EXCH_BYTES .. +BS*4)    s_boundary[]  (int per thread, after scan)
  //
  // ENDO_TILE_BYTES = sizeof(endo_t) * SHMEM_STRIDE * BLOCK_SIZE
  // EXCH_BYTES      = max(sizeof(terminal_t), sizeof(J), sizeof(length_t))
  //                   * ITEMS_PER_THREAD * BLOCK_SIZE
  //
  // Since ENDO_TILE_BYTES >= EXCH_BYTES for all grammars with ENDO_WORDS >= 1
  // and max(exch_elem) <= 8 = sizeof(endo_t), we declare states[] and overlay
  // the exch pointers onto it.  s_boundary[] lives right after EXCH_BYTES.

  __shared__ endo_t states[SHMEM_STRIDE * BLOCK_SIZE];

  // Overlay exch union onto states[] memory — safe because the endo tile is
  // only live during phase A and the scan; exch is only live after the scan.
  constexpr I EXCH_ELEMS = ITEMS_PER_THREAD * BLOCK_SIZE;
  terminal_t* exch_t = reinterpret_cast<terminal_t*>(states);
  J*          exch_j = reinterpret_cast<J*>(states);
  length_t*   exch_l = reinterpret_cast<length_t*>(states);

  // s_boundary[t] holds eval_endo(st[IPT-1], INIT_STATE) for thread t —
  // the last evaluated state of each thread, needed by thread t+1 for s_prev
  // at i==0, and by thread t-1 for is_next_produce at i==IPT-1.
  // Also s_boundary[BLOCK_SIZE] holds the next-block boundary state.
  // Placed after the exch region to avoid aliasing conflicts during scatter.
  constexpr size_t EXCH_BYTES = sizeof(J) * EXCH_ELEMS;  // J dominates exch elem size
  // Align s_boundary to int boundary after EXCH_BYTES within the states[] shmem.
  // We need BLOCK_SIZE+1 ints: [0..BS-1] = last state of each thread,
  // [BS] = first state of next block (for the last thread's is_next_produce).
  int* s_boundary = reinterpret_cast<int*>(
      reinterpret_cast<char*>(states) + EXCH_BYTES);

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

  // Phase A: byte -> endo_t written into the states[] tile (striped layout
  // from glbToReg, re-indexed to blocked layout for the scan).
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

  // Phase B/C: load blocked slice into registers, scan, keep in registers.
  // After scanReg, st[i] holds the inclusive prefix endo for position
  // threadIdx.x*IPT+i.  We evaluate each to a DFA state in registers and
  // exchange only the boundary states (one int per thread) via s_boundary[],
  // avoiding writing the full endo tile back to shmem.
  endo_t st[ITEMS_PER_THREAD];
  int    s[ITEMS_PER_THREAD];   // s[i] = eval_endo(st[i], INIT_STATE)
  endo_t last_endo;             // st[IPT-1] of last thread of last block

  {
    const I off = threadIdx.x * SHMEM_STRIDE;
    bool is_first = (glb_offs == 0) && (threadIdx.x == 0);
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
      st[i] = states[off + i];
      if (is_first && i == 0)
        st[i] = ctx(ctx.getLastState(), st[i]);
    }
    const endo_t pfx = scanReg<endo_t, I, LexerCtx<I, J>, ITEMS_PER_THREAD, BLOCK_SIZE>(
        st, ctx.d_state_states, ctx, d_ENDO_IDENTITY, dyn_index);
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
      st[i] = ctx(pfx, st[i]);
      s[i]  = eval_endo(st[i], INIT_STATE);
    }
    last_endo = st[ITEMS_PER_THREAD - 1];
  }

  // Publish each thread's last evaluated state so adjacent threads can read
  // cross-boundary s_prev / s_next without re-reading the full endo tile.
  // Also publish the next-block boundary state for the last thread.
  s_boundary[threadIdx.x] = s[ITEMS_PER_THREAD - 1];
  if (threadIdx.x == BLOCK_SIZE - 1) {
    s_boundary[BLOCK_SIZE] = eval_endo(ctx(last_endo, next_block_first_state), INIT_STATE);
  }
  __syncthreads();

  // Produce detection and start-code computation from register s[] and
  // s_boundary[], no endo shmem reads needed.
  uint32_t start_codes[ITEMS_PER_THREAD];
  uint32_t produce_flags[ITEMS_PER_THREAD];

#pragma unroll
  for (I i = 0; i < ITEMS_PER_THREAD; i++) {
    I lid = threadIdx.x * ITEMS_PER_THREAD + i;
    I gid = glb_offs + lid;
    bool is_next_produce = false;
    uint32_t start_code = 0;

    if (gid < size) {
      int s_cur = s[i];
#ifdef IGNORE_TOKEN
      bool is_not_ignore = (terminal_t)h_terminal[s_cur] != IGNORE_TOKEN;
#else
      bool is_not_ignore = true;
#endif
      // s_next: next position's state (register if within thread, else s_boundary)
      int s_next;
      if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
        s_next = s_boundary[BLOCK_SIZE];
      } else if (i < ITEMS_PER_THREAD - 1) {
        s_next = s[i + 1];
      } else {
        s_next = s_boundary[threadIdx.x + 1];
      }
      is_next_produce = is_produce_transition(s_cur, s_next);

      if (is_last_chunk) {
        is_next_produce |= gid == size - 1;
        is_next_produce &= is_not_ignore;
      } else {
        is_next_produce &= is_not_ignore;
      }

      // s_prev: previous position's state
      int s_prev;
      if (lid == 0) {
        s_prev = INIT_STATE;
      } else if (i > 0) {
        s_prev = s[i - 1];
      } else {
        s_prev = s_boundary[threadIdx.x - 1];
      }
      bool this_produce = is_produce_transition(s_prev, s_cur);
      start_code = this_produce ? (uint32_t)(gid + 1) : 0u;
    }
    is_produce_state |= is_next_produce << i;
    start_codes[i]   = start_code;
    produce_flags[i] = is_next_produce ? 1u : 0u;
  }

  // Two independent block-local scans on scalar u32 monoids.  Both cub
  // BlockScans share one TempStorage — they run sequentially in the same
  // thread, so the storage is only live during one at a time.
  using BlockScan32 = cub::BlockScan<uint32_t, BLOCK_SIZE>;
  __shared__ typename BlockScan32::TempStorage scan_temp;
  const uint32_t max_agg =
      scanRegLocal<uint32_t, I, Max<uint32_t>, ITEMS_PER_THREAD, BLOCK_SIZE>(
          start_codes, scan_temp, Max<uint32_t>());
  const uint32_t add_agg =
      scanRegLocal<uint32_t, I, Add<uint32_t>, ITEMS_PER_THREAD, BLOCK_SIZE>(
          produce_flags, scan_temp, Add<uint32_t>());

  // Single fused decoupled-lookback round over the SoA PairStates buffer.
  I max_prefix, prefix;
  lookbackPrefixPair<I, I, I, Max<I>, Add<I>>(
      ctx.d_maxadd_states, Max<I>(), Add<I>(), (I)0, (I)0, dyn_index,
      (I)max_agg, (I)add_agg,
      max_prefix, prefix);

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
    ctx.setLastState(last_endo);

    if (last_start != I()) {
      ctx.setLastStart(ctx.addOffset(last_start - 1));
    } else {
      ctx.setLastStart(ctx.getLastStart());
    }
  }

  // Scatter: terminals from register s[], starts/lengths from register arrays.
  // The exch union aliases the states[] shmem (safe: endo tile is no longer needed).
  if (num_sel > BLOCK_SIZE) {
    // Dense tile: two-phase scatter.
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
      if ((is_produce_state >> i) & 1) {
        exch_t[local_offs[i]] = (terminal_t)h_terminal[s[i]];
      }
    }
    __syncthreads();
    shmemToGlbVec<terminal_t, uint64_t, BLOCK_SIZE, I>(prefix, num_sel, d_terminals, exch_t);
    __syncthreads();

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
    // Sparse tile: direct scatter from registers, no shmem needed.
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
      I lid = threadIdx.x * ITEMS_PER_THREAD + i;
      I gid = glb_offs + lid;
      if ((is_produce_state >> i) & 1) {
        I offset = Add<I>()(prefix, local_offs[i]);
        d_terminals[offset] = (terminal_t)h_terminal[s[i]];
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

