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
  States<I, uint64_t> d_maxadd_states;

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

    d_maxadd_states = States<I, uint64_t>(num_blocks);
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
  volatile __shared__ state_t states[SHMEM_STRIDE * BLOCK_SIZE];
  // Exchange buffer for the two-phase scatter on dense tiles.
  // exch_t (terminals), exch_j (starts), and exch_l (lengths) are never live
  // simultaneously, so they share one shmem region via a union.
  constexpr I EXCH_ELEMS = ITEMS_PER_THREAD * BLOCK_SIZE;
  union {
    terminal_t as_t[EXCH_ELEMS];
    J          as_j[EXCH_ELEMS];
    length_t   as_l[EXCH_ELEMS];
  } __shared__ exch;
  volatile terminal_t* exch_t = exch.as_t;
  volatile J*          exch_j = exch.as_j;
  volatile length_t*   exch_l = exch.as_l;
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
  // states[t*SHMEM_STRIDE .. t*SHMEM_STRIDE+IPT)).  Iterating in blocked
  // order (outer loop over items 0..IPT-1) makes shmem_idx = t*SHMEM_STRIDE+i
  // with no integer division.  Thread t's item i is byte i of copy_reg[0]
  // (for IPT <= EPV=8) or byte i%EPV of copy_reg[i/EPV] in general.
  {
    constexpr uint32_t EPV = sizeof(uint64_t);
    const I t_base_gid = glb_offs + (I)threadIdx.x * ITEMS_PER_THREAD;
    const I shmem_base = (I)threadIdx.x * SHMEM_STRIDE;
#pragma unroll
    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
      I gid = t_base_gid + (I)i;
      uint32_t reg_off = i;  // item i is chars_reg[i] in blocked layout
      if (gid < size) {
        if (gid == 0) {
          states[shmem_base + i] = ctx(ctx.getLastState(),
              reinterpret_cast<state_t>(ctx.toState(chars_reg[reg_off])));
        } else {
          states[shmem_base + i] = ctx.toState(chars_reg[reg_off]);
        }
      } else {
        states[shmem_base + i] = IDENTITY;
      }
    }
    // The byte immediately past this block (lid_off == IPT*BLOCK_SIZE) lives
    // at striped position i=IPT/EPV, j=0 for thread 0 — i.e. chars_reg[IPT]
    // for the thread whose striped slot covers that position.  In blocked
    // layout it's always thread 0's reg byte at offset IPT (= EPV * (IPT/EPV)
    // + IPT%EPV), which lives in copy_reg[IPT/EPV] byte IPT%EPV.
    // Equivalently: for thread t, the striped extra slot (copy_reg[VPT-1])
    // covers global bytes starting at (VPT-1)*BLOCK_SIZE*EPV + t*EPV.
    // The boundary byte is at global offset BLOCK_SIZE*IPT, so it belongs to
    // the thread where (VPT-1)*BLOCK_SIZE*EPV + t*EPV == BLOCK_SIZE*IPT,
    // i.e. t = (BLOCK_SIZE*IPT - (VPT-1)*BLOCK_SIZE*EPV) / EPV.
    // For IPT==EPV (the common case): VPT-1=1, t=0, byte index 0.
    constexpr uint32_t BOUNDARY_THREAD =
        (BLOCK_SIZE * ITEMS_PER_THREAD - (VPT - 1) * BLOCK_SIZE * EPV) / EPV;
    constexpr uint32_t BOUNDARY_BYTE   =
        (BLOCK_SIZE * ITEMS_PER_THREAD - (VPT - 1) * BLOCK_SIZE * EPV) % EPV;
    if (threadIdx.x == BOUNDARY_THREAD) {
      I boundary_gid = glb_offs + (I)(BLOCK_SIZE * ITEMS_PER_THREAD);
      if (boundary_gid < size) {
        uint8_t* extra = reinterpret_cast<uint8_t*>(&copy_reg[VPT - 1]);
        next_block_first_state = ctx.toState(extra[BOUNDARY_BYTE]);
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

  // Fused (max, +) scan over u64 pairs held in registers, in blocked layout
  // (thread t owns tile positions [t*IPT, (t+1)*IPT)): the token-start Max
  // scan rides in the high word (0 = "no start seen", produce at gid encodes
  // gid + 1) and the compaction Add scan in the low word (produce flag), so
  // one lookback round replaces the former two.  All loops below use blocked
  // indexing so registers, the produce bitmask, and shmem stay consistent.
  uint64_t pair[ITEMS_PER_THREAD];

#pragma unroll
  for (I i = 0; i < ITEMS_PER_THREAD; i++) {
    I lid = threadIdx.x * ITEMS_PER_THREAD + i;
    I gid = glb_offs + lid;
    bool is_next_produce = false;
    uint64_t start_code = 0;
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

      start_code = is_produce(state) ? (uint64_t)(gid + 1) : 0;
    }
    is_produce_state |= is_next_produce << i;
    pair[i] = (start_code << 32) | (uint64_t)(is_next_produce ? 1 : 0);
  }

  const uint64_t bprefix =
      scanReg<uint64_t, I, MaxAdd, ITEMS_PER_THREAD, BLOCK_SIZE>(pair, ctx.d_maxadd_states, MaxAdd(), 0, dyn_index);
  const I prefix = (I)(uint32_t)bprefix;
  const I max_prefix = (I)(bprefix >> 32);

  I starts[ITEMS_PER_THREAD];
  I local_offs[ITEMS_PER_THREAD];
  volatile __shared__ I last_start;
  __shared__ I num_sel_sh;

#pragma unroll
  for (I i = 0; i < ITEMS_PER_THREAD; i++) {
    I lid = threadIdx.x * ITEMS_PER_THREAD + i;
    I gid = glb_offs + lid;
    starts[i] = max(max_prefix, (I)(pair[i] >> 32));
    local_offs[i] = ((is_produce_state >> i) & 1) ? (I)(uint32_t)pair[i] - 1 : I();
    if (gid == size - 1) {
      last_start = starts[i];
    }
  }

  if (threadIdx.x == BLOCK_SIZE - 1) {
    num_sel_sh = (I)(uint32_t)pair[ITEMS_PER_THREAD - 1];
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

