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
constexpr size_t shmem_pad_stride(uint32_t items_per_thread) {
  // Mirror the SHMEM_PAD calculation from the kernel body: STRIDE picked so
  // (STRIDE * sizeof(state_t)) ≡ 4 (mod 8), yielding conflict-free reads for
  // the blocked state layout.  Padding depends only on state_t's byte width
  // and IPT.
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
  // cub TempStorage: one for the state scan (state_t) and one shared between
  // the two SoA scans (u32).  sizeof gives us the exact per-instantiation
  // size cub picks for this (T, BLOCK_SIZE) pair.
  size_t cub_state_temp = sizeof(typename cub::BlockScan<state_t, BLOCK_SIZE>::TempStorage);
  size_t cub_u32_temp   = sizeof(typename cub::BlockScan<uint32_t,  BLOCK_SIZE>::TempStorage);
  // lookbackPrefixPair shmem: two warp-sized value arrays (I each) + one
  // status array + two shmem prefix scalars.
  size_t lookback = 2u * sizeof(I) * WARP + sizeof(uint8_t) * WARP + 2u * sizeof(I);
  size_t fixed_scalars = sizeof(state_t)      // next_block_first_state
                        + sizeof(I)           // last_start
                        + sizeof(I);          // num_sel_sh
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
// Sweep on the JSON grammar (index_t = i32, 10M-token dataset) with
// `benchmarks/sweep-cuda-lexer.sh` picked BS=256, IPT=12 as the
// fastest (2832 μs kernel-only vs 2929 μs at CUB's IPT=8).
// nominal_ipt_4B = 12 * 4 / 4 = 12.
template<size_t ELEM> struct alpacc_ipt_tuning<75, ELEM> {
  static constexpr uint32_t nominal_ipt_4B = 12;
  static constexpr uint32_t block_size     = 256;
};

// Ampere data-centre (sm_80) — A100                    [cub]
template<size_t ELEM> struct alpacc_ipt_tuning<80, ELEM> {
  static constexpr uint32_t nominal_ipt_4B = 20;
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

// Type-size bucket driver: sizeof(index_t).  The block-local max and add
// scans run on I = uint32_t but their downstream register arrays
// (`starts[IPT]`, `local_offs[IPT]`) and the SoA lookback buffers are
// index_t-sized, which is what drives the scan's per-item register
// pressure at the block boundary.  state_t affects a separate scan that
// runs first; its impact on IPT is captured indirectly through the shmem
// clamp in `max_items_per_thread()`.  length_t and terminal_t are always
// narrower and don't move the optimum in practice.
template<typename state_t, typename J>
constexpr size_t elem_bytes() {
  (void)sizeof(state_t);  // silence unused-template-parameter warnings
  return sizeof(J);
}

// Table-driven IPT (0 if the arch is unknown, in which case callers fall
// back to max_items_per_thread<>()).
template<int SM_ARCH, typename state_t, typename J>
constexpr uint32_t arch_ipt() {
  constexpr size_t bytes = elem_bytes<state_t, J>();
  constexpr uint32_t nominal = alpacc_ipt_tuning<SM_ARCH, bytes>::nominal_ipt_4B;
  if (nominal == 0) return 0;
  // Scale from 4B-work units to the actual per-thread element size.
  uint32_t scaled = nominal * 4u / (uint32_t)bytes;
  return scaled == 0 ? 1 : scaled;
}

template<int SM_ARCH, typename state_t, typename J>
constexpr uint32_t arch_block_size() {
  constexpr size_t bytes = elem_bytes<state_t, J>();
  return alpacc_ipt_tuning<SM_ARCH, bytes>::block_size;
}

__device__ __host__ __forceinline__
state_t get_index(state_t state) {
  return (state & ENDO_MASK) >> ENDO_OFFSET;
}

__device__ __host__ __forceinline__
terminal_t get_terminal(state_t state) {
  return static_cast<terminal_t>((state & TERMINAL_MASK) >> TERMINAL_OFFSET);
}

__device__ __host__ __forceinline__
bool is_produce(state_t state) {
  return (state & PRODUCE_MASK) >> PRODUCE_OFFSET;
}

// CPU-only versions (if you still need them separately)
state_t get_index_cpu(state_t state) {
  return (state & ENDO_MASK) >> ENDO_OFFSET;
}

terminal_t get_terminal_cpu(state_t state) {
  return static_cast<terminal_t>((state & TERMINAL_MASK) >> TERMINAL_OFFSET);
}

bool is_produce_cpu(state_t state) {
  return (state & PRODUCE_MASK) >> PRODUCE_OFFSET;
}

template<typename I, typename J>
struct LexerCtx {

private:
  J offset = 0;
  state_t* d_to_state;
  state_t* d_compose;
  volatile uint32_t* d_dyn_block_index;
  volatile state_t* d_new_last_state;
  volatile state_t* d_old_last_state;
  I* d_new_size;
  volatile J* d_new_last_start;
  volatile J* d_old_last_start;
  volatile uint32_t* d_len_overflow;  // set to 1 by kernel on length_t overflow

  void swapLastStart() {
    J h_last_start;
    gpuAssert(cudaMemcpy(&h_last_start, (const void*) d_new_last_start, sizeof(J), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy((void *) d_new_last_start, (const void*) d_old_last_start, sizeof(J), cudaMemcpyDeviceToDevice));
    gpuAssert(cudaMemcpy((void *) d_old_last_start, &h_last_start, sizeof(J), cudaMemcpyHostToDevice));
  }

  void swapLastState() {
  state_t h_last_state;
  gpuAssert(cudaMemcpy(&h_last_state, (const void*) d_new_last_state, sizeof(state_t), cudaMemcpyDeviceToHost));
  gpuAssert(cudaMemcpy((void *) d_new_last_state, (const void*) d_old_last_state, sizeof(state_t), cudaMemcpyDeviceToDevice));
  gpuAssert(cudaMemcpy((void *) d_old_last_state, &h_last_state, sizeof(state_t), cudaMemcpyHostToDevice));
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
  States<I, state_t> d_state_states;
  // SoA inter-block buffer for the (max token-start, +produce-count) scan.
  // Both components are I (u32); keeping them in separate arrays avoids the
  // former packed-u64 layout that dragged 8-byte traffic through the warp
  // scan for two 4-byte values, and matches how the block-local scan runs
  // them as independent scalar reductions.
  PairStates<I, I, I> d_maxadd_states;

  LexerCtx(const I chunk_size,
           const I block_size,
           const I items_per_thread) : CHUNK_SIZE(chunk_size) {
    I num_blocks = numBlocks(chunk_size, block_size, items_per_thread);
    gpuAssert(cudaMalloc(&d_to_state, sizeof(h_to_state)));
    cudaMemcpy(d_to_state, h_to_state, sizeof(h_to_state),
                 cudaMemcpyHostToDevice);
    gpuAssert(cudaMalloc(&d_compose, sizeof(h_compose)));
    cudaMemcpy(d_compose, h_compose, sizeof(h_compose),
                 cudaMemcpyHostToDevice);

    d_maxadd_states = PairStates<I, I, I>(num_blocks);
    d_state_states = States<I, state_t>(num_blocks);

    gpuAssert(cudaMalloc((void**)&d_dyn_block_index, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_new_last_state, sizeof(state_t)));
    gpuAssert(cudaMalloc((void**)&d_old_last_state, sizeof(state_t)));
    gpuAssert(cudaMalloc((void**)&d_new_last_start, sizeof(J)));
    gpuAssert(cudaMalloc((void**)&d_old_last_start, sizeof(J)));
    gpuAssert(cudaMalloc((void**)&d_len_overflow, sizeof(uint32_t)));

    cudaMemset((void*)d_dyn_block_index, 0, sizeof(uint32_t));
    cudaMemset((void*)d_new_size, I(), sizeof(I));
    cudaMemset((void*)d_new_last_state, IDENTITY, sizeof(state_t));
    cudaMemset((void*)d_old_last_state, IDENTITY, sizeof(state_t));
    cudaMemset((void*)d_new_last_start, J(), sizeof(J));
    cudaMemset((void*)d_old_last_start, J(), sizeof(J));
    cudaMemset((void*)d_len_overflow, 0, sizeof(uint32_t));
  }

  void reset() {
    offset = 0;
    cudaMemset((void*)d_dyn_block_index, 0, sizeof(uint32_t));
    cudaMemset((void*)d_new_size, 0, sizeof(I));
    cudaMemset((void*)d_new_last_state, IDENTITY, sizeof(state_t));
    cudaMemset((void*)d_old_last_state, IDENTITY, sizeof(state_t));
    cudaMemset((void*)d_new_last_start, 0, sizeof(J));
    cudaMemset((void*)d_old_last_start, 0, sizeof(J));
    cudaMemset((void*)d_len_overflow, 0, sizeof(uint32_t));
    d_maxadd_states.reset();
    d_state_states.reset();
  }

  void cleanUp() {
    if (d_to_state) cudaFree(d_to_state);
    if (d_new_last_start) cudaFree((void*)d_new_last_start);
    if (d_old_last_start) cudaFree((void*)d_old_last_start);
    if (d_compose) cudaFree(d_compose);
    if (d_dyn_block_index) cudaFree((void*)d_dyn_block_index);
    if (d_new_size) cudaFree((void*)d_new_size);
    if (d_new_last_state) cudaFree((void*)d_new_last_state);
    if (d_old_last_state) cudaFree((void*)d_old_last_state);
    if (d_len_overflow) cudaFree((void*)d_len_overflow);
    d_maxadd_states.cleanUp();
    d_state_states.cleanUp();
  }

  __device__ __host__ __forceinline__
  state_t operator()(const state_t &a, const state_t &b) const {
#ifdef __CUDA_ARCH__
    return __ldg(&d_compose[get_index(b) * NUM_STATES + get_index(a)]);
#else
    return d_compose[get_index(b) * NUM_STATES + get_index(a)];
#endif
  }

  // Volatile-arg overload retained for scanWarp in lookbackPrefix, which
  // still holds its intermediates in volatile shmem (small buffers, ordered
  // by __syncwarp).  Dropping volatile in the lexer body — the state tile
  // and exchange union — has no effect here.
  __device__ __host__ __forceinline__
  state_t operator()(const volatile state_t &a, const volatile state_t &b) const {
#ifdef __CUDA_ARCH__
    return __ldg(&d_compose[get_index(b) * NUM_STATES + get_index(a)]);
#else
    return d_compose[get_index(b) * NUM_STATES + get_index(a)];
#endif
  }

  __device__ __host__ __forceinline__
  state_t toState(const uint8_t &a) const {
#ifdef __CUDA_ARCH__
    return __ldg(&d_to_state[a]);
#else
    return d_to_state[a];
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
  void setLastState(state_t state) const {
    *d_new_last_state = state;
  }

  __device__ __host__ __forceinline__
  state_t getLastState() const {
    return *d_old_last_state;
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
    state_t h_last_state;
    gpuAssert(cudaMemcpy(&h_last_state, (const void*) d_new_last_state, sizeof(state_t), cudaMemcpyDeviceToHost));
    return h_accept[get_index_cpu(h_last_state)];
  }

  I terminalsSize() const {
    I h_new_size = I();
    gpuAssert(cudaMemcpy(&h_new_size, (const void*) d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    return h_new_size;
  }

  void update() {
    resetDynamicIndex();
    swapLastStart();
    swapLastState();
    updateOffset();
  }
};

// __launch_bounds__ caps register allocation so at least 1024 threads' worth
// of blocks fit per SM (clamped to the 16-blocks/SM hardware limit): without
// it the scatter's register capture pushes past 64 regs/thread and occupancy
// drops from ~99% to ~74%, which costs more than the coalescing wins.
template<typename I, typename J, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
__launch_bounds__(BLOCK_SIZE, (1024 / BLOCK_SIZE) < 16 ? (1024 / BLOCK_SIZE) : 16)
lexer(LexerCtx<I, J> ctx, uint8_t* d_string, terminal_t* d_terminals, J* d_starts, length_t* d_lengths, const I size, const bool is_last_chunk) {
  // Bank-conflict-free padding: we need (STRIDE * sizeof(state_t)) to be
  // ≡ 4 (mod 8) so the stride in 4-byte banks is odd (coprime with 32).
  // Required: STRIDE ≡ 4/sizeof(state_t) (mod 8/sizeof(state_t)), clamped to ≥ 1.
  // Works for state_t ∈ {u8, u16, u32, u64}.
  static_assert(sizeof(state_t) == 1 || sizeof(state_t) == 2 ||
                sizeof(state_t) == 4 || sizeof(state_t) == 8, "unexpected state_t");
  constexpr I SHMEM_MOD    = 8 / (I)sizeof(state_t);
  constexpr I SHMEM_TARGET = 4 / (I)sizeof(state_t);
  constexpr I SHMEM_REM    = (ITEMS_PER_THREAD % SHMEM_MOD);
  constexpr I SHMEM_RAW    = (SHMEM_TARGET - SHMEM_REM + SHMEM_MOD) % SHMEM_MOD;
  constexpr I SHMEM_PAD    = (SHMEM_RAW == 0) ? SHMEM_MOD : SHMEM_RAW;
  constexpr I SHMEM_STRIDE = ITEMS_PER_THREAD + SHMEM_PAD;
  // Non-volatile: every read/write of the state tile is fenced by an
  // explicit __syncthreads() at the boundaries; cub::BlockScan handles its
  // own barriers internally.  Dropping volatile lets nvcc merge adjacent
  // loads and hoist them across independent compute, eliminating one of
  // the top stalls reported by ncu (~30% est speedup on shmem stores,
  // ~19% on shmem loads at HEAD).
  __shared__ state_t states[SHMEM_STRIDE * BLOCK_SIZE];
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
  __shared__ state_t next_block_first_state;
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
    next_block_first_state = IDENTITY;
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

  // Write toState results into blocked shmem (thread t owns
  // states[t*SHMEM_STRIDE .. t*SHMEM_STRIDE+IPT)).  The register layout
  // populated by glbToReg is *striped*: thread t's copy_reg[v] holds
  // global bytes [v*BLOCK_SIZE*EPV + t*EPV, +EPV).  So the outer loop
  // iterates striped positions (v, j) and computes both the destination
  // shmem index and the source register byte from (v, j) — the register
  // byte reg_off = v*EPV + j is guaranteed to belong to this thread.
  // Iterating in blocked order (item i of thread t) is only safe when
  // ITEMS_PER_THREAD == EPV, because thread t's blocked bytes t*IPT..
  // t*IPT+IPT-1 only coincide with its striped bytes t*EPV..t*EPV+EPV-1
  // in that case; for smaller IPT thread t's blocked items live in a
  // different thread's register slice.  See .claude-artifacts/
  // cuda-lexer-ipt-bug.md for details.
#pragma unroll
  for (uint32_t i = 0; i < VPT; i++) {
    I lid = (I)i * BLOCK_SIZE + threadIdx.x;
    I _gid = glb_offs + (I)sizeof(uint64_t) * lid;
    for (uint32_t j = 0; j < sizeof(uint64_t); j++) {
      I gid = _gid + (I)j;
      I lid_off = (I)sizeof(uint64_t) * lid + (I)j;
      uint32_t reg_off = sizeof(uint64_t) * i + j;
      bool is_in_block = lid_off < (I)(ITEMS_PER_THREAD * BLOCK_SIZE);
      // Blocked layout with SHMEM_STRIDE padding to break u16 bank conflicts.
      // lid_off/IPT = thread index, lid_off%IPT = item index within thread,
      // matching the read pattern states[t*SHMEM_STRIDE + item_i].  With
      // ITEMS_PER_THREAD a power-of-two template constant the division and
      // modulo compile down to shifts and masks.
      I shmem_idx = (lid_off / ITEMS_PER_THREAD) * SHMEM_STRIDE + (lid_off % ITEMS_PER_THREAD);
      if (gid < size && is_in_block) {
          if (gid == 0) {
            states[shmem_idx] = ctx(ctx.getLastState(), reinterpret_cast<state_t>(ctx.toState(chars_reg[reg_off])));
          } else {
            states[shmem_idx] = ctx.toState(chars_reg[reg_off]);
          }
      } else if (is_in_block) {
          states[shmem_idx] = IDENTITY;
      } else if (lid_off == (I)(ITEMS_PER_THREAD * BLOCK_SIZE) && gid < size) {
          // First byte of the next block, needed for the boundary produce
          // test.  Guarded by gid < size (not is_last_chunk): interior block
          // boundaries exist in the last chunk too, and for the final block
          // the byte is out of range regardless of chunk position.
          next_block_first_state = ctx.toState(chars_reg[reg_off]);
      }
    }
  }

  __syncthreads();

  // State scan: read from shmem into registers, run scanReg (no TempStorage
  // in shmem for state_t), then write inclusive results back so the
  // produce-flag loop below can read states[lid] / states[lid+1].
  {
    state_t st[ITEMS_PER_THREAD];
    const I off = threadIdx.x * (SHMEM_STRIDE);
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
      st[i] = states[off + i];
    const state_t pfx = scanReg<state_t, I, LexerCtx<I, J>, ITEMS_PER_THREAD, BLOCK_SIZE>(
        st, ctx.d_state_states, ctx, IDENTITY, dyn_index);
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++)
      states[off + i] = ctx(pfx, st[i]);
    __syncthreads();
  }

  // Split (max, +) block-local scans over u32 registers in blocked layout
  // (thread t owns tile positions [t*IPT, (t+1)*IPT)): the token-start Max
  // scan runs over start codes (0 = "no start seen", produce at gid encodes
  // gid + 1) and the compaction Add scan runs over produce flags.  The two
  // block-local scans are independent so we run them separately with scalar
  // operators — the former fused u64 MaxAdd combine had ~4 dependent
  // operations per reduction step (shift/mask/max/add threading through
  // cub's binary tree), whereas the split u32 scalar operators run at one
  // instruction per combine and give the compiler ILP across the two scans.
  // The inter-block handshake then runs *once* over the SoA PairStates
  // buffer via lookbackPrefixPair, so we don't pay for two lookback rounds.
  // All loops below use blocked indexing so registers, the produce bitmask,
  // and shmem stay consistent.
  uint32_t start_codes[ITEMS_PER_THREAD];
  uint32_t produce_flags[ITEMS_PER_THREAD];

#pragma unroll
  for (I i = 0; i < ITEMS_PER_THREAD; i++) {
    I lid = threadIdx.x * ITEMS_PER_THREAD + i;
    I gid = glb_offs + lid;
    bool is_next_produce = false;
    uint32_t start_code = 0;
    // Padded shmem indices for this item and the next global item.
    I shmem_cur  = (I)threadIdx.x * (SHMEM_STRIDE) + i;
    I shmem_next = (i < ITEMS_PER_THREAD - 1)
                   ? shmem_cur + 1
                   : ((I)threadIdx.x + 1) * (SHMEM_STRIDE);
    if (gid < size) {
      state_t state = states[shmem_cur];
#ifdef IGNORE_TOKEN
      bool is_not_ignore = get_terminal(state) != IGNORE_TOKEN;
#else
      bool is_not_ignore = true;
#endif
      if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
        is_next_produce = is_produce(ctx(state, next_block_first_state));
      } else {
        is_next_produce = is_produce(states[shmem_next]);
      }

      if (is_last_chunk) {
        is_next_produce |= gid == size - 1;
        is_next_produce &= is_not_ignore;
      } else {
        is_next_produce &= is_not_ignore;
      }

      start_code = is_produce(state) ? (uint32_t)(gid + 1) : 0u;
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
  // The two component aggregates live in separate arrays and the block
  // status is shared, so the inter-block warp scan reads both components in
  // parallel with one status atomic per predecessor tile — same handshake
  // cost as the former packed-u64 layout but without paying for a u64 load
  // when u32 will do.
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
    ctx.setLastState(states[(BLOCK_SIZE - 1) * SHMEM_STRIDE + (ITEMS_PER_THREAD - 1)]);

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

#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
      if ((is_produce_state >> i) & 1) {
        I offset = local_offs[i];
        if (Add<I>()(prefix, offset) == I() && starts[i] == I()) {
          exch_j[offset] = ctx.getLastStart();
        } else {
          exch_j[offset] = ctx.addOffset(starts[i] - 1);
        }
      }
    }
    __syncthreads();
    shmemToGlbVec<J, uint64_t, BLOCK_SIZE, I>(prefix, num_sel, d_starts, exch_j);
    __syncthreads();

    // Read starts from exch_j (which shmemToGlbVec left intact) and compute
    // lengths into per-thread registers *before* any thread writes exch_l:
    // exch_l aliases exch_j via the union, so an interleaved write would
    // corrupt another thread's read from exch_j.
    length_t tok_lens[ITEMS_PER_THREAD];
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
      if ((is_produce_state >> i) & 1) {
        I gid = glb_offs + (I)threadIdx.x * ITEMS_PER_THREAD + (I)i;
        J tok_len = ctx.addOffset(gid + 1) - (J)exch_j[local_offs[i]];
        if ((length_t)(tok_len) != tok_len) ctx.signalLengthOverflow();
        tok_lens[i] = (length_t)tok_len;
      }
    }
    __syncthreads();
#pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
      if ((is_produce_state >> i) & 1) {
        exch_l[local_offs[i]] = tok_lens[i];
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

