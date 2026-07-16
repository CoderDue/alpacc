#include <cuda/atomic>
#include <cub/cub.cuh>

template<typename T, typename I, typename OP>
__device__ inline T
scanWarp(volatile T* shmem,
         OP op,
         const uint8_t lane) {
  uint8_t h;

#pragma unroll
  for (uint8_t d = 0; d < LG_WARP; d++) {
    if ((h = 1 << d) <= lane)
      shmem[threadIdx.x] = op(shmem[threadIdx.x - h], shmem[threadIdx.x]);
    __syncwarp();
  }

  return shmem[threadIdx.x];
}

template<typename T, typename I, typename OP>
__device__ inline void
scanBlock(volatile T* shmem,
          OP op) {
  const uint8_t lane = threadIdx.x & (WARP - 1);
  const I warpid = threadIdx.x >> LG_WARP;

  T res = scanWarp<T, I, OP>(shmem, op, lane);
  __syncthreads();

  if (lane == (WARP - 1))
    shmem[warpid] = res;
  __syncthreads();

  if (warpid == 0)
    scanWarp<T, I, OP>(shmem, op, lane);
  __syncthreads();

  if (warpid > 0)
    res = op(shmem[warpid-1], res);
  __syncthreads();

  shmem[threadIdx.x] = res;
  __syncthreads();
}

__device__ inline uint32_t dynamicIndex(volatile uint32_t* dyn_idx_ptr) {
  volatile __shared__ uint32_t dyn_idx;

  if (threadIdx.x == 0)
    dyn_idx = atomicAdd(const_cast<uint32_t*>(dyn_idx_ptr), 1);

  __syncthreads();
  return dyn_idx;
}

enum Status: uint8_t {
  Invalid = 0,
  Aggregate = 1,
  Prefix = 2,
};

// Alias for device-scope atomic status (acquire/release semantics).
using AtomicStatus = cuda::atomic<Status, cuda::thread_scope_device>;

template<typename I, typename T>
struct States {
  volatile T*   aggregates = nullptr;
  volatile T*   prefixes   = nullptr;
  AtomicStatus* statuses   = nullptr;
  I num_blocks = 0;

  States(I num_blocks) : num_blocks(num_blocks) {
    cudaMalloc((void**)&aggregates, num_blocks * sizeof(T));
    cudaMalloc((void**)&prefixes, num_blocks * sizeof(T));
    cudaMalloc((void**)&statuses, num_blocks * sizeof(AtomicStatus));
    cudaMemset((void*) statuses, Invalid, num_blocks * sizeof(AtomicStatus));
  }

  States() {
  }

  void reset() {
    if (statuses) cudaMemset((void*) statuses, Invalid, num_blocks * sizeof(AtomicStatus));
  }

  void cleanUp() {
    if (aggregates) cudaFree((void*) aggregates);
    if (prefixes) cudaFree((void*) prefixes);
    if (statuses) cudaFree((void*) statuses);
    aggregates = nullptr; prefixes = nullptr; statuses = nullptr;
  }
};

__device__ inline Status
combine(Status a, Status b) {
  if (a == Invalid || b == Invalid)
    return Invalid;
  if (b == Aggregate)
    return a;
  return b;
}

template<typename T, typename I, typename OP>
__device__ inline void
scanWarp(volatile T* values,
         volatile Status* statuses,
         OP op,
         const uint8_t lane) {
  uint8_t h;
  const I tid = threadIdx.x;

#pragma unroll
  for (uint8_t d = 0; d < LG_WARP; d++) {
    if ((h = 1 << d) <= lane) {
      bool is_not_aggregate = statuses[tid] != Aggregate;
      values[tid] = is_not_aggregate ? values[tid] : op(values[tid - h], values[tid]);
      statuses[tid] = combine(statuses[tid - h], statuses[tid]);
    }
    __syncwarp();
  }
}

// Computes this block's exclusive prefix via decoupled lookback, given the
// block-wide aggregate (valid in all threads). Returns the prefix to all
// threads.
template<typename T, typename I, typename OP>
__device__ inline T
lookbackPrefix(States<I, T> states,
               OP op,
               T ne,
               uint32_t dyn_idx,
               T aggregate) {
  volatile __shared__ T values[WARP];
  volatile __shared__ Status statuses[WARP];
  volatile __shared__ T shmem_prefix;
  const uint8_t lane = threadIdx.x & (WARP - 1);
  const bool is_first = threadIdx.x == 0;

  if (is_first) {
    // relaxed store is fine: the __syncthreads() below acts as a barrier
    // ensuring all threads in this block see Invalid before we proceed.
    states.statuses[dyn_idx].store(Invalid, cuda::memory_order_relaxed);
  }
  __syncthreads();

  if (is_first) {
    states.aggregates[dyn_idx] = aggregate;
  }

  if (dyn_idx == 0 && is_first) {
    states.prefixes[dyn_idx] = aggregate;
  }
  // Release store: memory_order_release makes aggregate/prefix writes visible
  // to all threads that subsequently load this status with memory_order_acquire.
  if (dyn_idx == 0 && is_first) {
    states.statuses[dyn_idx].store(Prefix, cuda::memory_order_release);
  } else if (is_first) {
    states.statuses[dyn_idx].store(Aggregate, cuda::memory_order_release);
  }

  T prefix = ne;
  if (threadIdx.x < WARP && dyn_idx != 0) {
    I lookback_idx = threadIdx.x + dyn_idx;
    I lookback_warp = WARP;
    Status status = Aggregate;
    do {
      if (lookback_warp <= lookback_idx) {
        I idx = lookback_idx - lookback_warp;
        // Acquire load: pairs with the writer's memory_order_release store,
        // ensuring aggregate/prefix writes are visible before we read them.
        Status s = states.statuses[idx].load(cuda::memory_order_acquire);
        statuses[threadIdx.x] = s;
        values[threadIdx.x] = s == Prefix ? states.prefixes[idx] : states.aggregates[idx];
      } else {
        statuses[threadIdx.x] = Aggregate;
        values[threadIdx.x] = ne;
      }

      scanWarp<T, I, OP>(values, statuses, op, lane);

      T result = values[WARP - 1];
      status = statuses[WARP - 1];

      if (status == Invalid)
        continue;

      if (is_first) {
        prefix = op(result, prefix);
      }

      lookback_warp += WARP;
    } while (status != Prefix);
  }

  if (is_first) {
    shmem_prefix = prefix;
  }

  __syncthreads();

  if (is_first) {
    states.prefixes[dyn_idx] = op(prefix, aggregate);
    states.statuses[dyn_idx].store(Prefix, cuda::memory_order_release);
  }

  return shmem_prefix;
}

// Single-pass device scan over one block's tile: cub::BlockScan (register
// scan, warp shuffles) for the block-level phase + decoupled lookback for
// the inter-block prefix. The tile lives in shared memory in blocked layout
// (thread t owns [t*ITEMS_PER_THREAD, (t+1)*ITEMS_PER_THREAD)); the caller
// must __syncthreads() between writing the tile and calling this.
template<typename T, typename I, typename OP, I ITEMS_PER_THREAD, I BLOCK_SIZE>
__device__ inline T
scan(volatile T* shmem,
     States<I, T> states,
     OP op,
     T ne,
     uint32_t dyn_idx,
     bool write_back = true) {
  using BlockScanT = cub::BlockScan<T, BLOCK_SIZE>;
  __shared__ typename BlockScanT::TempStorage temp_storage;

  T items[ITEMS_PER_THREAD];
  const I offset = threadIdx.x * ITEMS_PER_THREAD;
#pragma unroll
  for (I i = 0; i < ITEMS_PER_THREAD; i++) {
    items[i] = shmem[offset + i];
  }

  T aggregate;
  BlockScanT(temp_storage).InclusiveScan(items, items, op, aggregate);

  const T prefix = lookbackPrefix<T, I, OP>(states, op, ne, dyn_idx, aggregate);

#pragma unroll
  for (I i = 0; i < ITEMS_PER_THREAD; i++) {
    shmem[offset + i] = write_back ? op(prefix, items[i]) : items[i];
  }
  __syncthreads();

  return prefix;
}

template<typename I>
struct Add {
  __device__ __forceinline__ I operator()(I a, I b) const {
    return a + b;
  }
};
