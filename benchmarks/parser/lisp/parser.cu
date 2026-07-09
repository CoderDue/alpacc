// Compile: nvcc -O3 -std=c++17 -arch=native <this-file>.cu -o <output>
// Terminal Encoding: 
// atom: 0
// "(": 1
// ")": 2
// 
// Production Encoding: 
// E [E_0] = atom: 0
// E [E_1] = "(" E0 ")": 1
// E0 [E0_2] = : 2
// E0 [E0_3] = E E0: 3

#define HAS_LEXER
#define HAS_PARSER
#define HAS_RAW_INPUT
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <unistd.h>
#define gpuAssert(x) _gpuAssert(x, __FILE__, __LINE__)
#define numBlocks(size, block_size, items_per_thread) std::max<size_t>(1, (size + block_size * items_per_thread - 1) / (block_size * items_per_thread))


int _gpuAssert(cudaError_t code, const char *fname, int lineno) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s, File: %s, Line: %i\n", cudaGetErrorString(code), fname, lineno);
    fflush(stdout);
    return -1;
  }
  return 0;
}

void compute_descriptors(float* measurements, size_t size, size_t bytes) {
  double sample_mean = 0;
  double sample_variance = 0;
  double sample_gbps = 0;
  double factor = bytes / (1000 * size);

  for (size_t i = 0; i < size; i++) {
    double diff = max(1e3 * measurements[i], 0.5);
    sample_mean += diff / size;
    sample_variance += (diff * diff) / size;
    sample_gbps += factor / diff;
  }
  double sample_std = sqrt(sample_variance);
  double bound = (0.95 * sample_std) / sqrt(size);

  printf("%.0lfμs ", sample_mean);
  printf("(95%% CI: [%.1lfμs, %.1lfμs]); ", sample_mean - bound, sample_mean + bound);
  printf("%.0lfGB/s\n", sample_gbps);
}

const uint8_t LG_WARP = 5;
const uint8_t WARP = 1 << LG_WARP;

// ---------------------------------------------------------------------------
// Blocked I/O helpers: vectorized global↔register↔shared transfers.
//
// The key optimisation for sub-64-bit element types (e.g. uint8_t): each
// thread issues one uint64_t load per VEC_PER_THREAD register slot instead
// of ITEMS_PER_THREAD single-byte loads, reducing instruction count and
// making coalescing explicit.
//
// Naming convention:
//   glbToReg      – global → registers (packed as VEC = uint64_t)
//   regToShmem    – registers (packed) → shared memory (unpacked as T)
//   glbToShmem    – global → shared (vectorized, combined; adds __syncthreads)
//   shmemToGlb    – shared → global (element-wise; adds __syncthreads)
//   glbToShmemCpy – naive scalar fallback (kept for types where sizeof(T)==sizeof(VEC))
//
// Template parameters:
//   T              – element type (e.g. uint8_t, uint32_t)
//   VEC            – vector load type (e.g. uint64_t); must be >= sizeof(T)
//   BLOCK_SIZE     – threads per block (compile-time)
//   ITEMS_PER_THREAD – logical elements per thread
//   I              – index type (e.g. uint32_t)
//
// VEC_PER_THREAD = ceil(ITEMS_PER_THREAD * sizeof(T) / sizeof(VEC)):
//   number of VEC-sized register slots per thread.
// ---------------------------------------------------------------------------

// Number of VEC slots needed per thread to hold ITEMS_PER_THREAD elements of T.
template<typename T, typename VEC, uint32_t ITEMS_PER_THREAD>
__host__ __device__ constexpr uint32_t vecPerThread() {
    return (ITEMS_PER_THREAD * (uint32_t)sizeof(T) + (uint32_t)sizeof(VEC) - 1)
           / (uint32_t)sizeof(VEC);
}

// Load ITEMS_PER_THREAD elements of T per thread from global memory into
// VEC-sized register slots (reg[VEC_PER_THREAD]).  Out-of-bounds positions
// leave the corresponding bytes in `reg` uninitialised (callers guard with
// `gid < size` before use, or use the shmem variant which fills `ne`).
template<typename T, typename VEC, uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD, typename I>
__device__ __forceinline__ void
glbToReg(I glb_offs, I size, const T* __restrict__ d_src,
          VEC reg[vecPerThread<T, VEC, ITEMS_PER_THREAD>()])
{
    constexpr uint32_t VPT    = vecPerThread<T, VEC, ITEMS_PER_THREAD>();
    constexpr uint32_t EPV    = (uint32_t)sizeof(VEC) / (uint32_t)sizeof(T); // elements per VEC
    const T* __restrict__ src = d_src + glb_offs;
    const I n_remaining       = size - glb_offs;    // elements left from our start
#pragma unroll
    for (uint32_t v = 0; v < VPT; v++) {
        // Global index of the first element in this VEC slot for this thread.
        I elem0 = (I)(v * BLOCK_SIZE + threadIdx.x) * EPV;
        if (elem0 + EPV <= n_remaining) {
            // Full vector load: all EPV elements are in bounds.
            reg[v] = *reinterpret_cast<const VEC*>(src + elem0);
        } else {
            // Partial: byte-by-byte, guarded.
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&reg[v]);
#pragma unroll
            for (uint32_t b = 0; b < sizeof(VEC); b++) {
                I gid = elem0 + b / (uint32_t)sizeof(T);
                bytes[b] = (gid < n_remaining)
                    ? reinterpret_cast<const uint8_t*>(src + elem0)[b]
                    : 0;
            }
        }
    }
}

// Unpack VEC register slots into shared memory as T elements.
// Writes ITEMS_PER_THREAD entries at logical positions [threadIdx.x*IPT ..
// threadIdx.x*IPT + IPT) in a striped layout compatible with glbToReg above.
// No __syncthreads: caller must synchronise before reading shmem.
template<typename T, typename VEC, uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__device__ __forceinline__ void
regToShmem(const VEC reg[vecPerThread<T, VEC, ITEMS_PER_THREAD>()],
            volatile T shmem[BLOCK_SIZE * ITEMS_PER_THREAD])
{
    constexpr uint32_t VPT = vecPerThread<T, VEC, ITEMS_PER_THREAD>();
    constexpr uint32_t EPV = (uint32_t)sizeof(VEC) / (uint32_t)sizeof(T);
#pragma unroll
    for (uint32_t v = 0; v < VPT; v++) {
        const T* elems = reinterpret_cast<const T*>(&reg[v]);
#pragma unroll
        for (uint32_t e = 0; e < EPV; e++) {
            uint32_t lid = (v * BLOCK_SIZE + threadIdx.x) * EPV + e;
            if (lid < BLOCK_SIZE * ITEMS_PER_THREAD)
                shmem[lid] = elems[e];
        }
    }
}

// Vectorized global → shared: combines glbToReg + regToShmem, fills
// out-of-bounds positions with `ne`, then __syncthreads.
template<typename T, typename VEC, uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD, typename I>
__device__ __forceinline__ void
glbToShmem(I glb_offs, I size, T ne,
            const T* __restrict__ d_src,
            volatile T shmem[BLOCK_SIZE * ITEMS_PER_THREAD])
{
    constexpr uint32_t VPT = vecPerThread<T, VEC, ITEMS_PER_THREAD>();
    VEC reg[VPT];
    glbToReg<T, VEC, BLOCK_SIZE, ITEMS_PER_THREAD>(glb_offs, size, d_src, reg);
    regToShmem<T, VEC, BLOCK_SIZE, ITEMS_PER_THREAD>(reg, shmem);
    // Fill out-of-bounds with ne (only the partial tail thread needs this).
#pragma unroll
    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
        uint32_t lid = i * BLOCK_SIZE + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid >= size) shmem[lid] = ne;
    }
    __syncthreads();
}

// Scalar global → shared (naive, for types where sizeof(T) == sizeof(VEC)
// or when vectorization is not beneficial).  Adds __syncthreads.
template<typename T, typename I, uint32_t ITEMS_PER_THREAD>
__device__ __forceinline__ void
glbToShmemCpy(I glb_offs, I size, T ne,
               const T* __restrict__ d_src,
               volatile T* shmem)
{
#pragma unroll
    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
        uint32_t lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        shmem[lid] = (gid < size) ? d_src[gid] : ne;
    }
    __syncthreads();
}

// Shared → global (element-wise, no packing needed).  Adds __syncthreads.
template<typename T, typename I, uint32_t ITEMS_PER_THREAD>
__device__ __forceinline__ void
shmemToGlbCpy(I glb_offs, I size,
               T* d_dst,
               const volatile T* shmem)
{
#pragma unroll
    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
        uint32_t lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size) d_dst[gid] = shmem[lid];
    }
    __syncthreads();
}

#include <cuda/atomic>

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline void
scanThread(volatile T* shmem,
           volatile T* shmem_aux,
           OP op) {
  const I offset = threadIdx.x * ITEMS_PER_THREAD;
  const I upper = offset + ITEMS_PER_THREAD;
  T acc = shmem[offset];
#pragma unroll
  for (I lid = offset + 1; lid < upper; lid++) {
    T tmp = shmem[lid];
    acc = op(acc, tmp);
    shmem[lid] = acc;
  }
  shmem_aux[threadIdx.x] = acc;
  __syncthreads();
}

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

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline void
addAuxBlockScan(volatile T* shmem,
                volatile T* shmem_aux,
                OP op) {
  if (threadIdx.x > 0) {
    const I offset = threadIdx.x * ITEMS_PER_THREAD;
    const I upper = offset + ITEMS_PER_THREAD;
    const T val = shmem_aux[threadIdx.x - 1];
#pragma unroll
    for (I lid = offset; lid < upper; lid++) {
      shmem[lid] = op(val, shmem[lid]);
    }
  }
  __syncthreads();
}

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline void
scanBlock(volatile T* block,
          volatile T* block_aux,
          OP op) {
  scanThread<T, I, OP, ITEMS_PER_THREAD>(block, block_aux, op);

  scanBlock<T, I, OP>(block_aux, op);

  addAuxBlockScan<T, I, OP, ITEMS_PER_THREAD>(block, block_aux, op);
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

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline T
decoupledLookbackScan(States<I, T> states,
                      volatile T* shmem,
                      OP op,
                      T ne,
                      uint32_t dyn_idx,
                      bool write_back = true) {
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

  T aggregate = shmem[ITEMS_PER_THREAD * blockDim.x - 1];
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

  prefix = shmem_prefix;
  if (write_back) {
    const I offset = threadIdx.x * ITEMS_PER_THREAD;
    const I upper = offset + ITEMS_PER_THREAD;
#pragma unroll
    for (I lid = offset; lid < upper; lid++) {
      shmem[lid] = op(prefix, shmem[lid]);
    }
  }
  __syncthreads();
  return prefix;
}

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline T
scan(volatile T* block,
     volatile T* block_aux,
     States<I, T> states,
     OP op,
     T ne,
     uint32_t dyn_idx,
     bool write_back = true) {

  scanBlock<T, I, OP, ITEMS_PER_THREAD>(block, block_aux, op);

  return decoupledLookbackScan<T, I, OP, ITEMS_PER_THREAD>(states, block, op, ne, dyn_idx, write_back);
}

template<typename I>
struct Add {
  __device__ __forceinline__ I operator()(I a, I b) const {
    return a + b;
  }
};


using index_t = int64_t;
enum terminal_t : uint8_t {atom,literal_1,literal_2,ignore,empty_4};

using state_t = uint16_t;

const size_t NUM_STATES = 12;
const size_t NUM_TRANS = 256;
#define IGNORE_TOKEN 3
const state_t ENDO_MASK = 15;
const state_t ENDO_OFFSET = 0;
const state_t TERMINAL_MASK = 112;
const state_t TERMINAL_OFFSET = 4;
const state_t PRODUCE_MASK = 128;
const state_t PRODUCE_OFFSET = 7;
const state_t IDENTITY = 64;

const state_t h_to_state[NUM_TRANS] =
  {65, 65, 65, 65, 65, 65, 65, 65, 65, 50, 50, 65, 65, 50, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 50, 65, 65, 65, 65, 65, 65, 65, 19, 36, 65, 65, 65, 65, 65, 65, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 65, 65, 65, 65, 65, 65, 65, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 65, 65, 65, 65, 65, 65, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65};

const state_t h_compose[NUM_STATES * NUM_STATES] =
  {64, 65, 50, 19, 36, 5, 54, 151, 168, 137, 186, 11, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 50, 65, 54, 186, 186, 186, 54, 186, 186, 186, 54, 186, 19, 65, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 36, 65, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 5, 65, 137, 137, 137, 11, 137, 137, 137, 11, 137, 11, 54, 65, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 151, 65, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 168, 65, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 137, 65, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 186, 65, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 11, 65, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11};

const bool h_accept[NUM_STATES] =
  {false, false, true, true, true, true, true, true, true, true, true, true};// Calculate fixed overhead (independent of ITEMS_PER_THREAD)
template<typename I, typename state_t, uint32_t BLOCK_SIZE>
constexpr size_t calculate_lexer_fixed_overhead() {
  return sizeof(I) * BLOCK_SIZE +           // indices_aux
          sizeof(state_t) +                  // next_block_first_state
          sizeof(I) +                        // last_start
          sizeof(I) * WARP +                 // values
          sizeof(Status) * WARP +            // statuses
          sizeof(I);                         // shmem_prefix
}

// Calculate memory cost dependent on ITEMS_PER_THREAD (runtime parameter version)
template<typename I, typename state_t, uint32_t BLOCK_SIZE>
constexpr size_t calculate_lexer_variable_cost(uint32_t items_per_thread) {
  size_t states_bytes = sizeof(state_t) * items_per_thread * BLOCK_SIZE;
  size_t indices_bytes = sizeof(I) * items_per_thread * BLOCK_SIZE;
  size_t states_aux_bytes = sizeof(state_t) * BLOCK_SIZE;
  size_t buffer_bytes = (indices_bytes > states_aux_bytes) ? indices_bytes : states_aux_bytes;
  
  return states_bytes + buffer_bytes;
}

// Total shared memory usage
template<typename I, typename state_t, uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
constexpr size_t calculate_lexer_shared_memory_usage() {
  return calculate_lexer_fixed_overhead<I, state_t, BLOCK_SIZE>() +
         calculate_lexer_variable_cost<I, state_t, BLOCK_SIZE>(ITEMS_PER_THREAD);
}

// Compile-time calculation of maximum ITEMS_PER_THREAD
template<typename I, typename state_t, uint32_t BLOCK_SIZE, uint32_t SHARED_MEMORY>
constexpr uint32_t calculate_lexer_max_items_per_thread() {
  constexpr size_t usable_shmem = static_cast<size_t>(SHARED_MEMORY * 0.9);
  constexpr size_t fixed_overhead = calculate_lexer_fixed_overhead<I, state_t, BLOCK_SIZE>();
  
  uint32_t max_items = 1;
  for (uint32_t items = 1; items <= 1024; items++) {
    size_t total = fixed_overhead + calculate_lexer_variable_cost<I, state_t, BLOCK_SIZE>(items);
    
    if (total <= usable_shmem) {
      max_items = items;
    } else {
      break;
    }
  }
  
  return max_items;
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

template<typename T>
struct TakeRight {
  const T identity = std::numeric_limits<T>::max();

  __device__ __forceinline__ T operator()(T a, T b) const {
    if (b == identity) {
      return a;
    }

    return b;
  }
};

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
  States<I, I> d_index_states;
  States<I, I> d_take_right_states;
  TakeRight<I> take_right = TakeRight<I>();

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

    d_index_states = States<I, I>(num_blocks);
    d_take_right_states = States<I, I>(num_blocks);
    d_state_states = States<I, state_t>(num_blocks);

    gpuAssert(cudaMalloc((void**)&d_dyn_block_index, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_new_last_state, sizeof(state_t)));
    gpuAssert(cudaMalloc((void**)&d_old_last_state, sizeof(state_t)));
    gpuAssert(cudaMalloc((void**)&d_new_last_start, sizeof(J)));
    gpuAssert(cudaMalloc((void**)&d_old_last_start, sizeof(J)));

    cudaMemset((void*)d_dyn_block_index, 0, sizeof(uint32_t));
    cudaMemset((void*)d_new_size, I(), sizeof(I));
    cudaMemset((void*)d_new_last_state, IDENTITY, sizeof(state_t));
    cudaMemset((void*)d_old_last_state, IDENTITY, sizeof(state_t));
    cudaMemset((void*)d_new_last_start, J(), sizeof(J));
    cudaMemset((void*)d_old_last_start, J(), sizeof(J));
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
    d_index_states.cleanUp();
    d_state_states.cleanUp();
    d_take_right_states.cleanUp();
  }

  __device__ __host__ __forceinline__
  state_t operator()(const state_t &a, const state_t &b) const {
    return d_compose[get_index(b) * NUM_STATES + get_index(a)];
  }

  __device__ __host__ __forceinline__
  state_t operator()(const volatile state_t &a, const volatile state_t &b) const {
    return d_compose[get_index(b) * NUM_STATES + get_index(a)];
  }

  __device__ __host__ __forceinline__
  state_t toState(const uint8_t &a) const {
    return d_to_state[a];
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

template<typename I, typename J, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
lexer(LexerCtx<I, J> ctx, uint8_t* d_string, terminal_t* d_terminals, J* d_starts, J* d_ends, const I size, const bool is_last_chunk) {
  constexpr size_t indices_bytes = ITEMS_PER_THREAD * BLOCK_SIZE * sizeof(I);
  constexpr size_t states_aux_bytes = BLOCK_SIZE * sizeof(state_t);
  constexpr size_t max_bytes = (indices_bytes > states_aux_bytes) ? indices_bytes : states_aux_bytes;
  volatile __shared__ state_t states[ITEMS_PER_THREAD * BLOCK_SIZE];
  volatile __shared__ I indices_aux[BLOCK_SIZE];
  volatile __shared__ uint8_t shared_buffer[max_bytes];
  volatile I* indices = (volatile I*) shared_buffer;
  volatile state_t* states_aux = (volatile state_t*) shared_buffer;
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

#pragma unroll
  for (uint32_t i = 0; i < VPT; i++) {
    I lid = (I)i * BLOCK_SIZE + threadIdx.x;
    I _gid = glb_offs + (I)sizeof(uint64_t) * lid;
    for (uint32_t j = 0; j < sizeof(uint64_t); j++) {
      I gid = _gid + (I)j;
      I lid_off = (I)sizeof(uint64_t) * lid + (I)j;
      uint32_t reg_off = sizeof(uint64_t) * i + j;
      bool is_in_block = lid_off < (I)(ITEMS_PER_THREAD * BLOCK_SIZE);
      if (gid < size && is_in_block) {
          if (gid == 0) {
            states[lid_off] = ctx(ctx.getLastState(), reinterpret_cast<state_t>(ctx.toState(chars_reg[reg_off])));
          } else {
            states[lid_off] = ctx.toState(chars_reg[reg_off]);
          }
      } else if (is_in_block) {
          states[lid_off] = IDENTITY;
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

  scan<state_t, I, LexerCtx<I, J>, ITEMS_PER_THREAD>(states, states_aux, ctx.d_state_states, ctx, IDENTITY, dyn_index);

#pragma unroll
  for (I i = 0; i < ITEMS_PER_THREAD; i++) {
    I lid = i * blockDim.x + threadIdx.x;
    I gid = glb_offs + lid;
    bool is_next_produce = false;
    if (gid < size) {
      state_t state = states[lid];
#ifdef IGNORE_TOKEN
      bool is_not_ignore = get_terminal(state) != IGNORE_TOKEN;
#else
      bool is_not_ignore = true;
#endif
      if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
        is_next_produce = is_produce(ctx(state, next_block_first_state));
      } else {
        is_next_produce = is_produce(states[lid + 1]);
      }

      if (is_last_chunk) {
        is_next_produce |= gid == size - 1;
        is_next_produce &= is_not_ignore;
      } else {
        is_next_produce &= is_not_ignore;
      }

      indices[lid] = is_produce(state) ? gid : ctx.take_right.identity;
    } else {
      indices[lid] = ctx.take_right.identity;
    }
    is_produce_state |= is_next_produce << i;
  }

  __syncthreads();

  scan<I, I, TakeRight<I>, ITEMS_PER_THREAD>(indices, indices_aux, ctx.d_take_right_states, ctx.take_right, ctx.take_right.identity, dyn_index);

  I starts[ITEMS_PER_THREAD];
  volatile __shared__ I last_start;

#pragma unroll
  for (I i = 0; i < ITEMS_PER_THREAD; i++) {
    I lid = i * blockDim.x + threadIdx.x;
    I gid = glb_offs + lid;

    if (gid < size) {
      starts[i] = indices[lid];
      indices[lid] = (is_produce_state >> i) & 1;

      if (gid == size - 1) {
        last_start = starts[i];
      }
      
    } else {
      indices[lid] = 0;
    }
  }

  __syncthreads();

  I prefix = scan<I, I, Add<I>, ITEMS_PER_THREAD>(indices, indices_aux, ctx.d_index_states, Add<I>(), I(), dyn_index, false);

  #pragma unroll
  for (I i = 0; i < ITEMS_PER_THREAD; i++) {
    I lid = blockDim.x * i + threadIdx.x;
    I gid = glb_offs + lid;
    if (gid < size && ((is_produce_state >> i) & 1)) {
      I offset = Add<I>()(prefix, indices[lid]) - 1;
      if (offset == I() && starts[i] == ctx.take_right.identity) {
        d_starts[offset] = ctx.getLastStart();
      } else {
        d_starts[offset] = ctx.addOffset(starts[i]);
      }
      d_ends[offset] = ctx.addOffset(gid + 1);
      d_terminals[offset] = get_terminal(states[lid]);
    }
  }

  if (dyn_index == gridDim.x - 1 && threadIdx.x == blockDim.x - 1) {
    I new_size = Add<I>()(prefix, indices[ITEMS_PER_THREAD * BLOCK_SIZE - 1]);
    ctx.setNewSize(new_size);
    ctx.setLastState(states[ITEMS_PER_THREAD * BLOCK_SIZE - 1]);
    
    if (last_start != ctx.take_right.identity) {
      ctx.setLastStart(ctx.addOffset(last_start));
    } else {
      ctx.setLastStart(ctx.getLastStart());
    }
  }
}


struct WriteBinary {
  void operator()(index_t i, index_t j, terminal_t s) const {
    uint8_t buffer[2 * sizeof(index_t) + sizeof(terminal_t)];
    memcpy(buffer, &i, sizeof(index_t));
    memcpy(buffer + sizeof(index_t), &j, sizeof(index_t));
    memcpy(buffer + 2 * sizeof(index_t), &s, sizeof(terminal_t));
    fwrite(buffer, 2 * sizeof(index_t) + sizeof(terminal_t), 1, stdout);
  }
};

struct WriteAscii {
  void operator()(index_t i, index_t j, terminal_t s) const {
    printf("%lld %lld %lu\n", (long long)i, (long long)j, (size_t)s);
  }
};


struct NoWrite {
  void operator()(index_t i, index_t j, terminal_t s) const {
    // No operation
  }
};

bool read_chunk(FILE* file, uint8_t* buffer, size_t chunk_size, size_t* bytes_read) {
    *bytes_read = fread(buffer, sizeof(uint8_t), chunk_size, file);
    
    bool is_not_done = (*bytes_read == chunk_size) && !feof(file);
    
    if (*bytes_read == chunk_size) {
      int next_char = fgetc(file);
      if (next_char != EOF) {
        ungetc(next_char, file);
        is_not_done = true;
      } else {
        is_not_done = false;
      }
    }
    
    return is_not_done;
}

template<typename PRINT, uint32_t CHUNK_SIZE, uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
int lexer_stream(PRINT print, bool timeit = false) {
  
  uint8_t* h_string = (uint8_t*) malloc(CHUNK_SIZE * sizeof(uint8_t));
  terminal_t* h_terminals = (terminal_t*) malloc(CHUNK_SIZE * sizeof(terminal_t));
  index_t* h_starts = (index_t*) malloc(CHUNK_SIZE * sizeof(index_t));
  index_t* h_ends = (index_t*) malloc(CHUNK_SIZE * sizeof(index_t));
  assert(h_string != NULL);
  assert(h_terminals != NULL);
  assert(h_starts != NULL);
  assert(h_ends != NULL);

  constexpr size_t required_shmem =
      calculate_lexer_shared_memory_usage<uint32_t, state_t, BLOCK_SIZE, ITEMS_PER_THREAD>();

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  assert(required_shmem <= prop.sharedMemPerBlock &&
          "Kernel requires more shared memory than available on device!");

  uint8_t* d_string;
  terminal_t* d_terminals;
  index_t* d_starts;
  index_t* d_ends;
  gpuAssert(cudaMalloc((void**)&d_string, CHUNK_SIZE * sizeof(uint8_t)));
  gpuAssert(cudaMalloc((void**)&d_terminals, CHUNK_SIZE * sizeof(terminal_t)));
  gpuAssert(cudaMalloc((void**)&d_starts, CHUNK_SIZE * sizeof(index_t)));
  gpuAssert(cudaMalloc((void**)&d_ends, CHUNK_SIZE * sizeof(index_t)));

  assert(WARP <= BLOCK_SIZE);
  LexerCtx ctx = LexerCtx<uint32_t, index_t>(CHUNK_SIZE, BLOCK_SIZE, ITEMS_PER_THREAD);

  size_t new_size = 0;
  size_t final_size = 0;
  float time = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  bool is_not_done = true;
  while (is_not_done) {
    size_t bytes;
    is_not_done = read_chunk(stdin, h_string, CHUNK_SIZE, &bytes);
    final_size += bytes;

    gpuAssert(cudaMemcpy(d_string, h_string, bytes, cudaMemcpyHostToDevice));

    const uint32_t num_blocks = numBlocks(bytes, BLOCK_SIZE, ITEMS_PER_THREAD);
    cudaEventRecord(start, 0);
    lexer<uint32_t, index_t, BLOCK_SIZE, ITEMS_PER_THREAD><<<num_blocks, BLOCK_SIZE>>>(ctx, d_string, d_terminals, d_starts, d_ends, bytes, !is_not_done);
    gpuAssert(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float temp = 0;
    cudaEventElapsedTime(&temp, start, stop);
    gpuAssert(cudaPeekAtLastError());
    time += temp;

    new_size = ctx.terminalsSize();

    cudaMemcpy(h_terminals, d_terminals, new_size * sizeof(terminal_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_starts, d_starts, new_size * sizeof(index_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ends, d_ends, new_size * sizeof(index_t), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < new_size; i++) {
      print(h_starts[i], h_ends[i], h_terminals[i]);
    }

    ctx.update();
  }

  if (timeit) {
    fprintf(stderr, "Time: %.2fms\n", time);
  }

  fflush(stdout);

  int success = ctx.isAccept() ? 0 : -1;
  
  ctx.cleanUp();

  free(h_string);
  free(h_terminals);
  free(h_starts);
  cudaFree(d_string);
  cudaFree(d_terminals);
  cudaFree(d_starts);
  return success;
}

template<uint32_t CHUNK_SIZE, uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
bool lexer_full(
  LexerCtx<uint32_t, index_t> ctx,
  uint8_t* d_string,
  terminal_t* d_terminals,
  index_t* d_starts,
  index_t* d_ends,
  size_t size,
  size_t* new_size) {
  assert(size != 0);
  assert(d_string != NULL);
  assert(d_terminals == NULL);
  assert(d_starts == NULL);
  assert(d_ends == NULL);
  assert(WARP <= BLOCK_SIZE);
  assert(ITEMS_PER_THREAD > 1);

  constexpr size_t required_shmem = 
      calculate_lexer_shared_memory_usage<uint32_t, state_t, BLOCK_SIZE, ITEMS_PER_THREAD>();
  
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  
  assert(required_shmem <= prop.sharedMemPerBlock && 
          "Kernel requires more shared memory than available on device!");

  cudaMalloc((void**)&d_terminals, CHUNK_SIZE * sizeof(terminal_t));
  cudaMalloc((void**)&d_starts, CHUNK_SIZE * sizeof(index_t));
  cudaMalloc((void**)&d_ends, CHUNK_SIZE * sizeof(index_t));

  size_t prev_index = 0;
  size_t temp_new_size = 0;
  size_t alloc_size = CHUNK_SIZE;
  float time = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (size_t offset = 0; offset < size; offset+=CHUNK_SIZE) {
    uint32_t bytes = min((size_t) CHUNK_SIZE, size - offset);

    const uint32_t NUM_BLOCKS = numBlocks(bytes, BLOCK_SIZE, ITEMS_PER_THREAD);
    cudaEventRecord(start, 0);
    lexer<uint32_t, index_t, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_BLOCKS, BLOCK_SIZE>>>(ctx, d_string, d_terminals, d_starts, d_ends, bytes, offset < size - CHUNK_SIZE);
    gpuAssert(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float temp = 0;
    cudaEventElapsedTime(&temp, start, stop);
    gpuAssert(cudaPeekAtLastError());
    time += temp;

    temp_new_size += ctx.terminalsSize();

    if (alloc_size < temp_new_size + CHUNK_SIZE) {
      while (alloc_size < temp_new_size + CHUNK_SIZE) {
        alloc_size *= 2;
      }
      cudaMalloc((void**)&d_terminals, alloc_size * sizeof(terminal_t));
      cudaMalloc((void**)&d_starts, alloc_size * sizeof(index_t));
      cudaMalloc((void**)&d_ends, alloc_size * sizeof(index_t));
    }

    ctx.update();
  }

  *new_size = temp_new_size;
  
  cudaMalloc((void**)&d_terminals, temp_new_size * sizeof(terminal_t));
  cudaMalloc((void**)&d_starts, temp_new_size * sizeof(index_t));
  cudaMalloc((void**)&d_ends, temp_new_size * sizeof(index_t));

  return ctx.isAccept();
}

// pse.cu — previous smaller element (PSE) primitive, vendored from the APSEP
// project (https://github.com/CoderDue/APSEP, src/apsep.cuh, SPT kernel).
//
// Computes, for every i, the nearest j < i with in[j] < in[i] (INCL=false)
// or in[j] <= in[i] (INCL=true), else -1.  The LLP parser uses INCL=true for
// bracket matching (over depths) and parent vectors (over the arity scan).
//
// The algorithm lives in apsepDeviceSPT, a __device__ function taking the
// cooperative grid handle, so it can run either standalone (apsepKernelSPT
// wrapper, one cooperative launch) or as a phase of a larger cooperative
// kernel (the fused LLP parser in parser.cu).
//
// Single cooperative launch (grid.sync()), three phases:
//   Phase 1: blocked layout, in-register sequential ANSV per thread, warp
//            prefix-min + pointer-jumping chain, unresolved bitmask d_unres.
//   Phase 2: segment min-tree over block mins, one level per grid.sync(),
//            then per-block exclusive prefix-min by tree ascent.
//   Phase 3: bitmask-driven warp-cooperative tree ascent+descent lookups,
//            chunk-min + leaf ballots, dense -1-fill fast path.
//
// Grid must fit on-chip for grid.sync(): launch exactly
// cudaOccupancyMaxActiveBlocksPerMultiprocessor x num_SMs physical blocks
// via cudaLaunchCooperativeKernel (allocSPTScratch queries this).
//
// Constraints (static_asserted): IPT == 4, 4-byte T, B/32 <= 32.
// Measured on GTX 1660 Ti at N=32M, int keys: 98/182/174 GB/s useful
// bandwidth on random/descending/ascending input; INCL=true costs nothing
// (compile-time predicate) and is ~1.5x faster than strict on tie-heavy
// input, since ties then resolve intra-block.

#include <limits.h>
#include <float.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <typename T> struct ApsepInfinity {
    __host__ __device__ static T value();
};
template <> struct ApsepInfinity<int> {
    __host__ __device__ static int value() { return INT_MAX; }
};
template <> struct ApsepInfinity<unsigned> {
    __host__ __device__ static unsigned value() { return UINT_MAX; }
};
template <> struct ApsepInfinity<long> {
    __host__ __device__ static long value() { return LONG_MAX; }
};
template <> struct ApsepInfinity<long long> {
    __host__ __device__ static long long value() { return LLONG_MAX; }
};
template <> struct ApsepInfinity<float> {
    __host__ __device__ static float value() { return FLT_MAX; }
};
template <> struct ApsepInfinity<double> {
    __host__ __device__ static double value() { return DBL_MAX; }
};

static inline int nextPow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

template <typename T, int BLOCK_SIZE, int IPT, bool INCL = false>
__device__
void apsepDeviceSPT(
        cg::grid_group          grid,
        const T* __restrict__   d_in,
        T*                      d_out,
        int                     n,
        int                     num_blocks,
        int                     M,            // next pow2 >= num_blocks
        int                     leaf_offset,  // M - 1
        unsigned* __restrict__  d_unres,      // 1 bit/element: needs inter-block lookup
        T* __restrict__         d_block_mins,
        T* __restrict__         d_block_warp_mins,
        T* __restrict__         d_tree,       // 2*M-1 nodes
        T* __restrict__         d_prefix_min) // prefix_min[b] = min(block_min[0..b-1]), INF for b=0
{
    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    constexpr int W         = B / 32;
    const     T   INF       = ApsepInfinity<T>::value();

    const int phys_bid   = (int)blockIdx.x;
    const int num_phys   = (int)gridDim.x;
    const int lane       = threadIdx.x & 31;
    const int warp_id    = threadIdx.x >> 5;

    __shared__ T s_elems[B];
    __shared__ T s_tmin[BLOCK_SIZE];
    __shared__ T s_warp_min[NUM_WARPS];

    static_assert(IPT == 4, "blocked Phase 1 assumes IPT == 4");
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "blocked Phase 1 assumes 4- or 8-byte T");
    static_assert(W <= 32, "warp-cooperative Phase 3 assumes W <= 32");

    // lt(a, b): "a qualifies as the answer for query value b".  INCL=false
    // is APSEP (strict previous-smaller); INCL=true is PSE(<=) (previous
    // smaller or equal, e.g. LLP bracket matching / parent vectors).  Every
    // query comparison in all three phases must go through this predicate;
    // INCL is a compile-time constant, so the branch folds away.
    // (Macro; #undef'd at the end of the kernel.)
    #define lt(a, b) (INCL ? ((a) <= (b)) : ((a) < (b)))

    // -------------------------------------------------------------------------
    // Phase 1 (blocked layout): each thread owns IPT consecutive elements
    // (one int4 load), resolves within-thread matches sequentially in
    // registers, and runs the warp machinery (prefix-min scan, pointer-
    // jumping ANSV chain) over per-thread mins — 4x fewer shuffles than the
    // striped layout, and descending runs resolve with no shuffles at all.
    // -------------------------------------------------------------------------
    for (int block_id = phys_bid; block_id < num_blocks; block_id += num_phys) {
        const int glb_offs = block_id * B;
        const int tbase    = IPT * threadIdx.x;   // block-relative
        const bool full    = (glb_offs + B <= n);

        T v[IPT];
        if (full) {
            if constexpr (sizeof(T) == 4) {
                int4 raw = *reinterpret_cast<const int4*>(d_in + glb_offs + tbase);
                v[0] = raw.x; v[1] = raw.y; v[2] = raw.z; v[3] = raw.w;
                *reinterpret_cast<int4*>(s_elems + tbase) = raw;
            } else {
                longlong2 r0 = *reinterpret_cast<const longlong2*>(d_in + glb_offs + tbase);
                longlong2 r1 = *reinterpret_cast<const longlong2*>(d_in + glb_offs + tbase + 2);
                v[0] = (T)r0.x; v[1] = (T)r0.y; v[2] = (T)r1.x; v[3] = (T)r1.y;
                *reinterpret_cast<longlong2*>(s_elems + tbase)     = r0;
                *reinterpret_cast<longlong2*>(s_elems + tbase + 2) = r1;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < IPT; i++) {
                int gid = glb_offs + tbase + i;
                v[i] = (gid < n) ? d_in[gid] : INF;
                s_elems[tbase + i] = v[i];
            }
        }

        // In-thread sequential ANSV (block-relative result, -1 if none)
        int res[IPT];
        res[0] = -1;
        res[1] = lt(v[0], v[1]) ? tbase     : -1;
        res[2] = lt(v[1], v[2]) ? tbase + 1 : lt(v[0], v[2]) ? tbase : -1;
        res[3] = lt(v[2], v[3]) ? tbase + 2 : lt(v[1], v[3]) ? tbase + 1
                                            : lt(v[0], v[3]) ? tbase : -1;

        const T tmin = min(min(v[0], v[1]), min(v[2], v[3]));
        s_tmin[threadIdx.x] = tmin;

        // Warp inclusive prefix-min over thread mins
        T c = tmin;
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            T nb = __shfl_up_sync(0xffffffff, c, step);
            if (lane >= step) c = min(c, nb);
        }
        const T carry = __shfl_up_sync(0xffffffff, c, 1);  // valid for lane>0
        if (lane == 31) s_warp_min[warp_id] = c;

        // Chunk mins (32 consecutive elements = 8 threads) for Phase 3
        {
            T om = tmin;
            om = min(om, __shfl_xor_sync(0xffffffff, om, 1));
            om = min(om, __shfl_xor_sync(0xffffffff, om, 2));
            om = min(om, __shfl_xor_sync(0xffffffff, om, 4));
            if ((lane & 7) == 0)
                d_block_warp_mins[(size_t)block_id * W + warp_id * (32/8) + (lane >> 3)] = om;
        }

        __syncthreads();  // covers cross-thread s_elems/s_tmin/s_warp_min reads

        // Thread-level ANSV chain over tmins (pointer jumping; each lane's
        // query is its own tmin, so the gap-bound argument of the original
        // element-level algorithm carries over verbatim).
        int chain = -1;
        {
            bool has = (lane > 0) && lt(carry, tmin);
            if (__any_sync(0xffffffff, has)) {
                int g = has ? lane - 1 : -1;
                while (true) {
                    int src = (g >= 0) ? g : 0;
                    T   tg  = __shfl_sync(0xffffffff, tmin, src);
                    int gg  = __shfl_sync(0xffffffff, g,    src);
                    bool jump = (g >= 0) && !lt(tg, tmin);
                    if (!__any_sync(0xffffffff, jump)) break;
                    if (jump) g = gg;
                }
                chain = g;
            }
        }

        // Per-element resolution.  Locally-unresolved elements of a thread
        // are its prefix minima (non-increasing), so the spine walk position
        // `cur` is monotone across i and can be reused.
        unsigned bal[IPT];
        int cur = lane - 1;
        #pragma unroll
        for (int i = 0; i < IPT; i++) {
            const bool active = full || (glb_offs + tbase + i < n);
            const T val = v[i];
            bool pending = active && (res[i] < 0) && (lane > 0) && lt(carry, val);

            if (__any_sync(0xffffffff, pending)) {
                // Walk cur along the chain while !lt(tmin[cur], val).  Threads
                // strictly between cur and lane never satisfy lt(tmin, val), so
                // the walk cannot skip the answer; pending guarantees the
                // answer exists, so cur lands on it (never -1).
                while (true) {
                    int src = (pending && cur >= 0) ? cur : 0;
                    T   tg  = __shfl_sync(0xffffffff, tmin,  src);
                    int cg  = __shfl_sync(0xffffffff, chain, src);
                    bool step = pending && (cur >= 0) && !lt(tg, val);
                    if (!__any_sync(0xffffffff, step)) break;
                    if (step) cur = cg;
                }
                if (pending) {
                    int tt = warp_id * 32 + cur;
                    const T* e = s_elems + IPT * tt;
                    int k = lt(e[3], val) ? 3 : lt(e[2], val) ? 2 : lt(e[1], val) ? 1 : 0;
                    res[i] = IPT * tt + k;
                }
            }

            // Cross-warp fallback: warp min, then thread min, then exact.
            if (active && res[i] < 0) {
                for (int w = warp_id - 1; w >= 0 && res[i] < 0; w--) {
                    if (lt(s_warp_min[w], val)) {
                        for (int tt = 32 * w + 31; tt >= 32 * w; tt--) {
                            if (lt(s_tmin[tt], val)) {
                                const T* e = s_elems + IPT * tt;
                                int k = lt(e[3], val) ? 3 : lt(e[2], val) ? 2
                                      : lt(e[1], val) ? 1 : 0;
                                res[i] = IPT * tt + k;
                                break;
                            }
                        }
                    }
                }
            }

            if (active && res[i] >= 0) d_out[glb_offs + tbase + i] = (T)(glb_offs + res[i]);
            bal[i] = __ballot_sync(0xffffffff, active && res[i] < 0);
        }

        // Publish unresolved bits (Phase 3 writes each unresolved d_out byte
        // exactly once).  The word for chunk c of this warp packs the 4
        // ballots' bytes c — element 4j+i of the chunk maps to bit 8i+j; the
        // Phase 3 reader uses the same permutation (mybit).
        if (lane < 32/8) {
            unsigned word = ((bal[0] >> (8 * lane)) & 0xffu)
                          | (((bal[1] >> (8 * lane)) & 0xffu) << 8)
                          | (((bal[2] >> (8 * lane)) & 0xffu) << 16)
                          | (((bal[3] >> (8 * lane)) & 0xffu) << 24);
            d_unres[(unsigned)glb_offs / 32 + warp_id * (32/8) + lane] = word;
        }

        if (threadIdx.x == 0) {
            T bmin = s_warp_min[0];
            #pragma unroll
            for (int w = 1; w < NUM_WARPS; w++) bmin = min(bmin, s_warp_min[w]);
            d_block_mins[block_id] = bmin;
        }
        __syncthreads();  // shared reused next iteration
    }

    // -------------------------------------------------------------------------
    // grid.sync(): all block_mins are now populated
    // -------------------------------------------------------------------------
    grid.sync();

    // -------------------------------------------------------------------------
    // Phase 2: build segment min-tree bottom-up
    // Fill leaves: d_tree[leaf_offset + i] = (i < num_blocks) ? block_mins[i] : INF
    // -------------------------------------------------------------------------
    for (int i = phys_bid * BLOCK_SIZE + threadIdx.x; i < M; i += num_phys * BLOCK_SIZE)
        d_tree[leaf_offset + i] = (i < num_blocks) ? d_block_mins[i] : INF;

    // Reduce level by level
    {
        int level_size  = M / 2;
        int level_start = M / 2 - 1;
        while (level_size > 0) {
            grid.sync();
            for (int i = phys_bid * BLOCK_SIZE + threadIdx.x;
                 i < level_size;
                 i += num_phys * BLOCK_SIZE) {
                int node = level_start + i;
                d_tree[node] = min(d_tree[2 * node + 1], d_tree[2 * node + 2]);
            }
            level_size  >>= 1;
            level_start  = (level_start - 1) / 2;
        }
    }

    // -------------------------------------------------------------------------
    // grid.sync(): tree is fully built
    // -------------------------------------------------------------------------
    grid.sync();

    // Exclusive prefix-min: d_prefix_min[b] = min(block_min[0..b-1]).
    // Computed per block by ascending the already-built min-tree and taking
    // the min over left siblings on the leaf-to-root path (their subtrees
    // cover exactly leaves [0, b)).  O(log M) L2-cached reads per block and
    // a single pass, replacing the Hillis-Steele scan's log2(M) grid.sync
    // passes and 8 MB of DRAM traffic.
    for (int b = phys_bid * BLOCK_SIZE + threadIdx.x; b < num_blocks; b += num_phys * BLOCK_SIZE) {
        T pm = INF;
        int node = leaf_offset + b;
        while (node > 0) {
            if (node % 2 == 0)
                pm = min(pm, __ldg(&d_tree[node - 1]));
            node = (node - 1) / 2;
        }
        d_prefix_min[b] = pm;
    }
    grid.sync();

    // -------------------------------------------------------------------------
    // Phase 3: resolve elements flagged in d_unres, driven directly by the
    // bitmask words (one word = 32 elements), warps grid-striding over words.
    // No per-block sweep, no shared queue, no __syncthreads: a zero word is
    // skipped after one cached read, so the fixed cost is the N/8-byte
    // bitmask read instead of a B-element scan per logical block (measured
    // ~1.2 ms of Phase 3 overhead on random/ascending at N=32M).
    // -------------------------------------------------------------------------
    const int num_words   = (n + 31) >> 5;
    const int warps_total = num_phys * NUM_WARPS;
    const int mybit       = 8 * (lane & 3) + (lane >> 2);  // blocked-P1 bit permutation

    // Each warp takes an aligned group of 32 consecutive words (one coalesced
    // load; a group spans exactly 2 logical blocks, so the dense first-word-
    // of-block bits distribute evenly across warps — a bare word stride of
    // warps_total is a multiple of W and gave 1/W of the warps ALL the
    // lookup work).  Nonzero words are then processed one per warp.
    for (int wbase = (phys_bid * NUM_WARPS + warp_id) * 32;
         wbase < num_words;
         wbase += warps_total * 32) {
        const int wl = wbase + lane;
        const unsigned um_l = (wl < num_words) ? __ldg(&d_unres[wl]) : 0u;

        // Per-lane metadata for this lane's word, loaded once per group and
        // shuffled into the per-word loop (guard lanes are clamped; their
        // words are never processed since um_l == 0).  eo_l is the
        // whole-block early-out: unresolved elements within a block form a
        // non-increasing sequence (were an earlier one smaller, the later
        // one would have resolved intra-block), and the block's first
        // element is always unresolved — so it is the max of them all.  If
        // prefix_min fails lt() against it, every unresolved element in the
        // block resolves to -1 (holds for both < and <= semantics).
        const int  bid_l = min(wl / W, num_blocks - 1);
        const T    pm_l  = __ldg(&d_prefix_min[bid_l]);
        const bool eo_l  = !lt(pm_l, __ldg(&d_in[(size_t)bid_l * B]));

        // Dense fast path (descending worst case): every word in the group
        // fully unresolved and early-out — the whole span is -1.
        // Coalesced 16-byte vector stores, no per-word loop.
        if (__all_sync(0xffffffff, eo_l && um_l == 0xffffffffu)) {
            if constexpr (sizeof(T) == 4) {
                int4* out4 = reinterpret_cast<int4*>(d_out) + (size_t)wbase * 8 + lane;
                const int4 m1 = make_int4(-1, -1, -1, -1);
                #pragma unroll
                for (int i = 0; i < 8; i++) out4[i * 32] = m1;
            } else {
                // sizeof(T)==8: each longlong2 covers 2 elements; need 16 stores
                // to cover the same 32 words × 32 elements/word = 1024 elements.
                longlong2* out2 = reinterpret_cast<longlong2*>(d_out) + (size_t)wbase * 16 + lane;
                const longlong2 m1 = make_longlong2(-1LL, -1LL);
                #pragma unroll
                for (int i = 0; i < 16; i++) out2[i * 32] = m1;
            }
            continue;
        }

        unsigned nz = __ballot_sync(0xffffffff, um_l != 0);

    while (nz) {
        const int j = __ffs(nz) - 1;
        nz &= nz - 1;
        const unsigned um = __shfl_sync(0xffffffff, um_l, j);
        const int w = wbase + j;

        const int  block_id     = __shfl_sync(0xffffffff, bid_l, j);
        const T    prefix_min_b = __shfl_sync(0xffffffff, pm_l, j);
        const bool eo           = __shfl_sync(0xffffffff, (int)eo_l, j);
        const int  base         = w * 32;
        const bool mine         = (um >> mybit) & 1u;

        if (eo) {
            if (mine) d_out[base + lane] = (T)-1;
            continue;
        }

        const int gid = base + lane;
        T val = mine ? __ldg(&d_in[gid]) : INF;    // coalesced masked load
        const bool need = mine && lt(prefix_min_b, val);
        if (mine && !need) d_out[gid] = (T)-1;

        // Warp-cooperative lookup for each remaining bit, one at a time.
        unsigned pend = __ballot_sync(0xffffffff, need);
        while (pend) {
            const int bit = __ffs(pend) - 1;
            pend &= pend - 1;
            const T qval = __shfl_sync(0xffffffff, val, bit);

            // Ascent+descent on the block-level tree, redundant on all 32
            // lanes (identical addresses broadcast from L2).
            int node = leaf_offset + block_id;
            int found_block = -1;
            while (node > 0) {
                bool is_right = (node % 2 == 0);
                if (is_right) {
                    int left_sib = node - 1;
                    if (lt(__ldg(&d_tree[left_sib]), qval)) {
                        node = left_sib;
                        while (node < leaf_offset) {
                            int rc = 2 * node + 2;
                            node = lt(__ldg(&d_tree[rc]), qval) ? rc : (2 * node + 1);
                        }
                        found_block = node - leaf_offset;
                        break;
                    }
                }
                node = (node - 1) / 2;
            }

            int result = -1;
            if (found_block >= 0) {
                // lt(found_block's block_min, qval), so both ballots are nonzero.
                const T* wm = d_block_warp_mins + (size_t)found_block * W;
                T wv = (lane < W) ? __ldg(&wm[lane]) : INF;
                // lane < W guard: under INCL, lt(INF, INT_MAX-valued qval)
                // is true, so padding lanes must not enter the ballot.
                unsigned wmask = __ballot_sync(0xffffffff, lane < W && lt(wv, qval));
                int wstar = 31 - __clz(wmask);
                // found_block < block_id, so all B of its elements are < n:
                // reading d_in here is safe and identical to the old leaves.
                const T* bl = d_in + (size_t)found_block * B + wstar * 32;
                unsigned lmask = __ballot_sync(0xffffffff, lt(__ldg(&bl[lane]), qval));
                result = found_block * B + wstar * 32 + (31 - __clz(lmask));
            }
            if (lane == 0) d_out[base + bit] = (T)result;
        }
    }
    }
    #undef lt
}

// Thin standalone wrapper: one cooperative launch running only the PSE.
template <typename T, int BLOCK_SIZE, int IPT, bool INCL = false>
__global__
void apsepKernelSPT(
        const T* __restrict__   d_in,
        T*                      d_out,
        int                     n,
        int                     num_blocks,
        int                     M,
        int                     leaf_offset,
        unsigned* __restrict__  d_unres,
        T* __restrict__         d_block_mins,
        T* __restrict__         d_block_warp_mins,
        T* __restrict__         d_tree,
        T* __restrict__         d_prefix_min)
{
    apsepDeviceSPT<T, BLOCK_SIZE, IPT, INCL>(
        cg::this_grid(), d_in, d_out, n, num_blocks, M, leaf_offset,
        d_unres, d_block_mins, d_block_warp_mins, d_tree, d_prefix_min);
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
struct SPTScratch {
    unsigned* d_unres       = nullptr;  // 1 bit/element: needs inter-block lookup
    T*    d_block_mins      = nullptr;
    T*    d_block_warp_mins = nullptr;
    T*    d_tree            = nullptr;
    T*    d_prefix_min      = nullptr;  // prefix_min[b] = min(block_min[0..b-1])
    int   num_blocks        = 0;
    int   M                 = 0;
    int   leaf_offset       = 0;
    int   num_phys          = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, bool INCL = false>
SPTScratch<T, BLOCK_SIZE, IPT> allocSPTScratch(int n) {
    constexpr int B = BLOCK_SIZE * IPT;
    constexpr int W = B / 32;

    SPTScratch<T, BLOCK_SIZE, IPT> s;
    s.num_blocks  = (n + B - 1) / B;
    s.M           = nextPow2(s.num_blocks);
    s.leaf_offset = s.M - 1;

    // Determine physical block count: occupancy x SM count
    int blocks_per_sm = 0;
    gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        apsepKernelSPT<T, BLOCK_SIZE, IPT, INCL>,
        BLOCK_SIZE, 0));
    cudaDeviceProp prop;
    gpuAssert(cudaGetDeviceProperties(&prop, 0));
    s.num_phys = blocks_per_sm * prop.multiProcessorCount;

    gpuAssert(cudaMalloc(&s.d_unres,           (size_t)s.num_blocks * (B / 32) * sizeof(unsigned)));
    gpuAssert(cudaMalloc(&s.d_block_mins,      (size_t)s.num_blocks      * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_block_warp_mins, (size_t)s.num_blocks * W  * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_tree,            (size_t)(2 * s.M - 1)     * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_prefix_min,      (size_t)s.num_blocks      * sizeof(T)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
void freeSPTScratch(SPTScratch<T, BLOCK_SIZE, IPT>& s) {
    cudaFree(s.d_unres);
    cudaFree(s.d_block_mins);
    cudaFree(s.d_block_warp_mins);
    cudaFree(s.d_tree);
    cudaFree(s.d_prefix_min);
    s = SPTScratch<T, BLOCK_SIZE, IPT>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, bool INCL = false>
void runSPT(const T* d_in, T* d_out, int n, SPTScratch<T, BLOCK_SIZE, IPT>& s) {
    if (n <= 0) return;

    void* args[] = {
        (void*)&d_in, (void*)&d_out, (void*)&n,
        (void*)&s.num_blocks, (void*)&s.M, (void*)&s.leaf_offset,
        (void*)&s.d_unres, (void*)&s.d_block_mins,
        (void*)&s.d_block_warp_mins, (void*)&s.d_tree,
        (void*)&s.d_prefix_min
    };
    gpuAssert(cudaLaunchCooperativeKernel(
        (void*)apsepKernelSPT<T, BLOCK_SIZE, IPT, INCL>,
        s.num_phys, BLOCK_SIZE, args, 0, nullptr));
    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, bool INCL = false>
void launchSPT(const T* d_in, T* d_out, int n) {
    if (n <= 0) return;
    SPTScratch<T, BLOCK_SIZE, IPT> s = allocSPTScratch<T, BLOCK_SIZE, IPT, INCL>(n);
    runSPT<T, BLOCK_SIZE, IPT, INCL>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeSPTScratch<T, BLOCK_SIZE, IPT>(s);
}
using production_t = uint8_t;
using bracket_t = uint8_t;
const int64_t Q = 1;
const int64_t K = 1;
const terminal_t EMPTY_TERMINAL = (terminal_t) 4;
const terminal_t START_TERMINAL = (terminal_t) 5;
const terminal_t END_TERMINAL = (terminal_t) 6;
const int64_t NUMBER_OF_PRODUCTIONS = 9;
__device__ const terminal_t PRODUCTION_TO_TERMINAL[NUMBER_OF_PRODUCTIONS] =
  {(terminal_t) 0, (terminal_t) 0, (terminal_t) 0, (terminal_t) 0, (terminal_t) 3, (terminal_t) 0, (terminal_t) 1, (terminal_t) 2, (terminal_t) 0};
__device__ const bool PRODUCTION_TO_TERMINAL_IS_VALID[NUMBER_OF_PRODUCTIONS] =
  {false, false, false, false, true, true, true, true, false};
__device__ const uint8_t PRODUCTION_TO_ARITY[NUMBER_OF_PRODUCTIONS] =
  {1, 3, 0, 2, 0, 0, 0, 0, 1};
const int64_t HASH_TABLE_SIZE = 30;
const int64_t MAX_ITERS = 7;
const int64_t PRODUCTIONS_SIZE = 28;
const int64_t STACKS_SIZE = 32;
const int64_t MAX_BRACKETS_PER_POSITION = 4;
const int64_t MAX_PRODS_PER_POSITION = 3;
__device__ const bracket_t STACKS[STACKS_SIZE] =
  {12, 140, 12, 140, 138, 140, 12, 10, 6, 12, 140, 12, 140, 138, 140, 12, 10, 12, 140, 12, 140, 138, 140, 12, 10, 6, 134, 139, 11, 11, 138, 140};
__device__ const production_t PRODUCTIONS[PRODUCTIONS_SIZE] =
  {3, 0, 5, 3, 1, 6, 2, 7, 3, 0, 5, 3, 1, 6, 2, 7, 3, 0, 5, 3, 1, 6, 2, 7, 0, 5, 1, 6};
__device__ const bool HASH_TABLE_IS_VALID[HASH_TABLE_SIZE] =
  {false, false, false, false, true, true, true, true, true, false, false, false, false, true, false, false, false, true, true, true, true, true, true, true, true, true, false, false, false, false};
__device__ const terminal_t HASH_TABLE_KEYS[HASH_TABLE_SIZE][Q + K] =
  {{(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 5, (terminal_t) 0}, {(terminal_t) 5, (terminal_t) 1}, {(terminal_t) 1, (terminal_t) 2}, {(terminal_t) 1, (terminal_t) 1}, {(terminal_t) 1, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 6}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 2, (terminal_t) 6}, {(terminal_t) 2, (terminal_t) 1}, {(terminal_t) 2, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 2}, {(terminal_t) 2, (terminal_t) 2}, {(terminal_t) 4, (terminal_t) 5}, {(terminal_t) 6, (terminal_t) 4}, {(terminal_t) 0, (terminal_t) 1}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 0}, {(terminal_t) 0, (terminal_t) 0}};
__device__ const int32_t HASH_TABLE_STACKS_SPAN[HASH_TABLE_SIZE][2] =
  {{0, 2}, {0, 2}, {0, 2}, {0, 2}, {28, 29}, {29, 32}, {15, 17}, {11, 15}, {9, 11}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, {8, 9}, {0, 2}, {0, 2}, {0, 2}, {25, 26}, {19, 23}, {17, 19}, {6, 8}, {23, 25}, {26, 28}, {32, 32}, {2, 6}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, {0, 2}};
__device__ const int32_t HASH_TABLE_PRODUCTIONS_SPAN[HASH_TABLE_SIZE][2] =
  {{0, 3}, {0, 3}, {0, 3}, {0, 3}, {24, 26}, {26, 28}, {14, 16}, {11, 14}, {8, 11}, {0, 3}, {0, 3}, {0, 3}, {0, 3}, {8, 8}, {0, 3}, {0, 3}, {0, 3}, {24, 24}, {19, 22}, {16, 19}, {6, 8}, {22, 24}, {24, 24}, {28, 28}, {3, 6}, {0, 3}, {0, 3}, {0, 3}, {0, 3}, {0, 3}};// parser.cu — fused single-kernel LLP parser, appended by the code generator
// after common.cu, scan.cu, pse.cu, and the grammar constants.
//
// The whole pipeline runs in ONE cooperative kernel (parserFusedKernel),
// launched via cudaLaunchCooperativeKernel with grid.sync() between phases:
//
//   Phase A: per-position hash-table key lookup (FNV-1a + linear probe)
//            → per-position stack/production lengths and spans.
//   Phase B: cooperative exclusive scans of both length arrays
//            (tile-hierarchical: per-tile block scan → single-block scan of
//            tile aggregates → add prefixes) → offsets and totals.
//   Phase C: segmented copies of STACKS spans (brackets + ±1 deltas) and
//            PRODUCTIONS spans into their compacted output arrays.
//   Phase D: cooperative inclusive scan of the deltas → depths; validity
//            check (no negative prefix, last must be 0).
//   Phase E: PSE(<=) over depths via apsepDeviceSPT (pse.cu), which is
//            itself structured as grid.sync() phases.
//   Phase F: bracket symbol check of each right bracket against its match.
//
// Combined (lexer+parser) mode additionally computes the CST parent vector
// and terminal-node spans (mirrors `parents` / `parse_int` in parser.fut),
// still inside the same cooperative kernel:
//   Phase G: exclusive scan of (arity - 1) over the productions.
//   Phase H: PSE(<=) over that scan → parent vector (parents[0] = 0).
//   Phase I: scan of is-terminal flags → lexeme index per terminal
//            production; node id/span assembly (terminal_offsets + scatter).
//
// Compacted layout: buffer capacity is bounded by
// m * MAX_BRACKETS_PER_POSITION / m * MAX_PRODS_PER_POSITION (the maximal
// hash-table span sizes emitted by the generator), but all per-element work
// runs over the exact totals computed in Phase B.
//
// Binary I/O (same protocol as c/parser.c and `alpacc test compare --parser`):
//   inputs:  u64 BE num_tests; per test: u64 BE n + n×u64 BE terminal ids
//   outputs: u64 BE num_tests; per test: 1 byte validity; if valid:
//            u64 BE num_prods + num_prods×u64 BE production ids

#include <vector>
#include <limits>
#include <cstdint>
#include <cstring>
#include <climits>
#include <algorithm>

// ---------------------------------------------------------------------------
// Fused kernel configuration
// ---------------------------------------------------------------------------

constexpr uint32_t FUSED_BS  = 256;  // block size (also the scan tile size)
constexpr int      FUSED_IPT = 4;    // SPT items per thread (logical B = 1024)

// ---------------------------------------------------------------------------
// Bracket helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool bracket_is_left(bracket_t b) {
    return (b >> (8 * (int)sizeof(bracket_t) - 1)) & 1;
}

__device__ __forceinline__ bracket_t bracket_unpack(bracket_t b) {
    return b & (bracket_t)~((bracket_t)1 << (8 * (int)sizeof(bracket_t) - 1));
}

// ---------------------------------------------------------------------------
// Phase A helper: hash-table lookup for position i.
//
// Builds the Q+K window key with sentinel-extended addressing:
//   d_arr[0] = START_TERMINAL, d_arr[1..n] = tokens, d_arr[n+1] = END_TERMINAL.
// FNV-1a hash, linear probe up to MAX_ITERS slots.
// Returns true iff a key with valid spans was found.
// ---------------------------------------------------------------------------

__device__ bool
lookupSpans(const terminal_t* __restrict__ d_arr, index_t m, index_t i,
            int32_t& ss, int32_t& se, int32_t& ps, int32_t& pe)
{
    terminal_t key[Q + K];
#pragma unroll
    for (int64_t j = 0; j < Q + K; j++) {
        int64_t idx = i + j - Q;
        key[j] = (idx < 0 || idx >= m) ? EMPTY_TERMINAL : d_arr[idx];
    }

    uint64_t h = 14695981039346656037ULL;
    for (int64_t j = 0; j < Q + K; j++)
        h = (h ^ (uint64_t)key[j]) * 1099511628211ULL;
    h %= (uint64_t)HASH_TABLE_SIZE;

    ss = se = ps = pe = -1;
    for (int64_t it = 0; it < MAX_ITERS; it++) {
        int64_t slot = (int64_t)h;
        if (HASH_TABLE_IS_VALID[slot]) {
            bool match = true;
#pragma unroll
            for (int64_t j = 0; j < Q + K; j++) {
                if (HASH_TABLE_KEYS[slot][j] != key[j]) { match = false; break; }
            }
            if (match) {
                ss = HASH_TABLE_STACKS_SPAN[slot][0];
                se = HASH_TABLE_STACKS_SPAN[slot][1];
                ps = HASH_TABLE_PRODUCTIONS_SPAN[slot][0];
                pe = HASH_TABLE_PRODUCTIONS_SPAN[slot][1];
                return ss >= 0 && se >= 0 && ps >= 0 && pe >= 0;
            }
        }
        h = (h + 1) % (uint64_t)HASH_TABLE_SIZE;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Cooperative scan building blocks (tile-hierarchical, grid.sync() driven).
//
// A tile is FUSED_BS consecutive elements, one tile per block iteration.
// Pass 1: per-tile inclusive block scan of d_in into d_out (tile-local
//         values only) and the tile aggregate into d_agg[tile].
//         Safe with d_in == d_out (in-place).
// Pass 2: block 0 turns d_agg into its inclusive scan, chunk by chunk with a
//         sequential carry (num_tiles is small: n / FUSED_BS).
// The caller grid.sync()s between and after the passes and then combines
// tile-local values with d_agg[tile-1] element-wise.
// ---------------------------------------------------------------------------

template<typename T>
__device__ void
coopScanPass1(const T* __restrict__ d_in, T* d_out, T* d_agg, index_t n)
{
    __shared__ volatile T sh[FUSED_BS];
    const index_t num_tiles = (n + (index_t)FUSED_BS - (index_t)1) / (index_t)FUSED_BS;
    for (index_t tile = (index_t)blockIdx.x; tile < num_tiles; tile += (index_t)gridDim.x) {
        index_t gid = tile * (index_t)FUSED_BS + (index_t)threadIdx.x;
        sh[threadIdx.x] = (gid < n) ? d_in[gid] : (T)0;
        __syncthreads();
        scanBlock<T, uint32_t, Add<T>>(sh, Add<T>());
        if (gid < n) d_out[gid] = sh[threadIdx.x];
        if (threadIdx.x == FUSED_BS - 1) d_agg[tile] = sh[threadIdx.x];
        __syncthreads();
    }
}

template<typename T>
__device__ void
coopScanPass2(T* d_agg, index_t num_tiles)
{
    if (blockIdx.x != 0) return;
    __shared__ volatile T sh[FUSED_BS];
    T carry = (T)0;
    for (index_t base = (index_t)0; base < num_tiles; base += (index_t)FUSED_BS) {
        index_t idx = base + (index_t)threadIdx.x;
        sh[threadIdx.x] = (idx < num_tiles) ? d_agg[idx] : (T)0;
        __syncthreads();
        scanBlock<T, uint32_t, Add<T>>(sh, Add<T>());
        if (idx < num_tiles) d_agg[idx] = carry + sh[threadIdx.x];
        T chunk_total = sh[FUSED_BS - 1];
        __syncthreads();
        carry += chunk_total;
    }
}

// ---------------------------------------------------------------------------
// Fused kernel parameter block (passed by value through
// cudaLaunchCooperativeKernel).  Capacities:
//   max_bk = m * MAX_BRACKETS_PER_POSITION   (brackets, deltas, match)
//   max_pr = m * MAX_PRODS_PER_POSITION      (productions)
// ---------------------------------------------------------------------------

struct FusedBufs {
    const terminal_t* d_arr;        // [m] sentinel-extended tokens
    index_t           m;

    index_t* d_slens;               // [m] per-position stack lengths
    index_t* d_plens;               // [m] per-position production lengths
    int32_t* d_ss;                  // [m] STACKS span begin (end = begin + slen)
    int32_t* d_ps;                  // [m] PRODUCTIONS span begin (end = begin + plen)
    index_t* d_soffsets;            // [m] exclusive bracket offsets
    index_t* d_poffsets;            // [m] exclusive production offsets
    index_t* d_agg_s;               // [ceil(m/FUSED_BS)] scan scratch
    index_t* d_agg_p;               // [ceil(m/FUSED_BS)] scan scratch
    index_t* d_totals;              // [2] num_brackets, num_prods

    bracket_t* d_brackets;          // [max_bk]
    index_t*   d_scan;              // [max(max_bk,max_pr)] deltas → inclusive scan → depths
    index_t*   d_agg_d;             // [ceil(max(max_bk,max_pr)/FUSED_BS)] scan scratch
    index_t*   d_match;             // [max(max_bk,max_pr)] PSE result (index_t; -1 = no match)

    production_t* d_productions;    // [max_pr]

    int* d_valid;

    // SPT (PSE) scratch, sized for max_bk elements (combined mode:
    // max(max_bk, max_pr) — reused by the parents PSE).
    unsigned* d_unres;
    index_t*  d_block_mins;
    index_t*  d_block_warp_mins;
    index_t*  d_tree;
    index_t*  d_prefix_min;

#ifdef HAS_LEXER
    // Combined mode (parse_int): parents + CST node assembly.
    // d_scan/d_agg_d/d_match are reused for the parents scan and PSE;
    // d_match holds the parent vector after Phase H/I.
    bool           with_parents;    // runtime: skip G/H/I on token-only paths
    index_t        num_lexemes;
    const index_t* d_lex_starts;   // [num_lexemes] lexeme spans (lexer output)
    const index_t* d_lex_ends;
    uint8_t*       d_node_is_term;  // [max_pr]
    production_t*  d_node_ids;     // [max_pr] terminal or production id (both fit in production_t)
    index_t*       d_node_starts;  // [max_pr]
    index_t*       d_node_ends;    // [max_pr]
#endif
};

// ---------------------------------------------------------------------------
// The fused cooperative kernel
// ---------------------------------------------------------------------------

__global__ void
parserFusedKernel(FusedBufs b)
{
    cg::grid_group grid = cg::this_grid();
    const index_t m     = b.m;
    const index_t grank = (index_t)grid.thread_rank();
    const index_t gsz   = (index_t)grid.size();

    // ---- Phase A: key lookup + span materialisation ----
    for (index_t i = grank; i < m; i += gsz) {
        int32_t ss, se, ps, pe;
        bool valid = lookupSpans(b.d_arr, m, i, ss, se, ps, pe);
        if (!valid) atomicAnd(b.d_valid, 0);
        b.d_slens[i] = valid ? (index_t)(se - ss) : (index_t)0;
        b.d_plens[i] = valid ? (index_t)(pe - ps) : (index_t)0;
        b.d_ss[i] = ss;
        b.d_ps[i] = ps;
    }
    grid.sync();

    // ---- Phase B: cooperative scans of both length arrays ----
    const index_t num_tiles_m = (m + (index_t)FUSED_BS - (index_t)1) / (index_t)FUSED_BS;
    coopScanPass1<index_t>(b.d_slens, b.d_soffsets, b.d_agg_s, m);
    coopScanPass1<index_t>(b.d_plens, b.d_poffsets, b.d_agg_p, m);
    grid.sync();
    coopScanPass2<index_t>(b.d_agg_s, num_tiles_m);
    coopScanPass2<index_t>(b.d_agg_p, num_tiles_m);
    grid.sync();

    const index_t num_brackets = b.d_agg_s[num_tiles_m - 1];
    const index_t num_prods    = b.d_agg_p[num_tiles_m - 1];
    if (grank == 0) {
        b.d_totals[0] = num_brackets;
        b.d_totals[1] = num_prods;
    }

    // ---- Phase C: segmented copies (brackets + deltas, productions) ----
    // Exclusive offsets computed on the fly from the inclusive per-tile scan
    // values and the tile aggregate prefix (saves a full fixup sweep + sync).
    for (index_t i = grank; i < m; i += gsz) {
        index_t tile  = i / (index_t)FUSED_BS;
        index_t slen  = b.d_slens[i];
        index_t plen  = b.d_plens[i];
        index_t soff  = ((tile > 0) ? b.d_agg_s[tile - 1] : (index_t)0) + b.d_soffsets[i] - slen;
        int32_t ss = b.d_ss[i];
        for (index_t j = 0; j < slen; j++) {
            bracket_t bk = STACKS[ss + (int32_t)j];
            index_t k = soff + j;
            b.d_brackets[k] = bk;
            b.d_scan[k]     = bracket_is_left(bk) ? (index_t)1 : (index_t)-1;
        }
        index_t poff  = ((tile > 0) ? b.d_agg_p[tile - 1] : (index_t)0) + b.d_poffsets[i] - plen;
        int32_t ps = b.d_ps[i];
        for (index_t j = 0; j < plen; j++)
            b.d_productions[poff + j] = PRODUCTIONS[ps + (int32_t)j];
    }
    grid.sync();

    if (num_brackets != (index_t)0) {   // grid-uniform
        // ---- Phase D: inclusive delta scan → depths + validity ----
        const index_t num_tiles_b = (num_brackets + (index_t)FUSED_BS - (index_t)1) / (index_t)FUSED_BS;
        coopScanPass1<index_t>(b.d_scan, b.d_scan, b.d_agg_d, num_brackets);
        grid.sync();
        coopScanPass2<index_t>(b.d_agg_d, num_tiles_b);
        grid.sync();
        for (index_t i = grank; i < num_brackets; i += gsz) {
            index_t tile = i / (index_t)FUSED_BS;
            index_t s = b.d_scan[i] + ((tile > 0) ? b.d_agg_d[tile - 1] : (index_t)0);
            if (s < (index_t)0) atomicAnd(b.d_valid, 0);
            if (i == num_brackets - (index_t)1 && s != (index_t)0) atomicAnd(b.d_valid, 0);
            b.d_scan[i] = s - (bracket_is_left(b.d_brackets[i]) ? (index_t)1 : (index_t)0);  // depth
        }
        grid.sync();

        // ---- Phase E: PSE(<=) over depths (apsepDeviceSPT, pse.cu) ----
        {
            constexpr int B = (int)FUSED_BS * FUSED_IPT;
            const int nb         = (int)num_brackets;   // host guarantees ≤ INT_MAX
            const int spt_blocks = (nb + B - 1) / B;
            int M = 1;
            while (M < spt_blocks) M <<= 1;
            apsepDeviceSPT<index_t, (int)FUSED_BS, FUSED_IPT, true>(
                grid, b.d_scan, b.d_match, nb, spt_blocks, M, M - 1,
                b.d_unres, b.d_block_mins, b.d_block_warp_mins,
                b.d_tree, b.d_prefix_min);
        }
        grid.sync();

        // ---- Phase F: bracket symbol check ----
        for (index_t i = grank; i < num_brackets; i += gsz) {
            if (bracket_is_left(b.d_brackets[i])) continue;
            index_t j = b.d_match[i];
            if (j < (index_t)0 || bracket_unpack(b.d_brackets[j]) != bracket_unpack(b.d_brackets[i]))
                atomicAnd(b.d_valid, 0);
        }
    }

#ifdef HAS_LEXER
    // ---- Combined mode: parents + CST node assembly (parse_int) ----
    if (!b.with_parents) return;   // grid-uniform (kernel parameter)
    grid.sync();   // d_scan32/d_agg_d/d_match are reused below

    if (num_prods == (index_t)0) {
        // safe_zip: no productions must mean no lexemes
        if (grank == 0 && b.num_lexemes != (index_t)0) atomicAnd(b.d_valid, 0);
        return;
    }

    // ---- Phase G: exclusive scan of (arity - 1) over the productions ----
    const index_t num_tiles_p = (num_prods + (index_t)FUSED_BS - (index_t)1) / (index_t)FUSED_BS;
    for (index_t i = grank; i < num_prods; i += gsz)
        b.d_scan[i] = (index_t)PRODUCTION_TO_ARITY[b.d_productions[i]] - (index_t)1;
    grid.sync();
    coopScanPass1<index_t>(b.d_scan, b.d_scan, b.d_agg_d, num_prods);
    grid.sync();
    coopScanPass2<index_t>(b.d_agg_d, num_tiles_p);
    grid.sync();
    for (index_t i = grank; i < num_prods; i += gsz) {
        index_t tile  = i / (index_t)FUSED_BS;
        index_t incl  = b.d_scan[i] + ((tile > 0) ? b.d_agg_d[tile - 1] : (index_t)0);
        index_t delta = (index_t)PRODUCTION_TO_ARITY[b.d_productions[i]] - (index_t)1;
        b.d_scan[i] = incl - delta;   // exclusive scan value
    }
    grid.sync();

    // ---- Phase H: PSE(<=) over the scan → parent vector (in d_match) ----
    {
        constexpr int B = (int)FUSED_BS * FUSED_IPT;
        const int np         = (int)num_prods;      // host guarantees ≤ INT_MAX
        const int spt_blocks = (np + B - 1) / B;
        int M = 1;
        while (M < spt_blocks) M <<= 1;
        apsepDeviceSPT<index_t, (int)FUSED_BS, FUSED_IPT, true>(
            grid, b.d_scan, b.d_match, np, spt_blocks, M, M - 1,
            b.d_unres, b.d_block_mins, b.d_block_warp_mins,
            b.d_tree, b.d_prefix_min);
    }
    grid.sync();

    // ---- Phase I: parents fixup + terminal-node assembly ----
    for (index_t i = grank; i < num_prods; i += gsz) {
        if (i == (index_t)0 || b.d_match[i] < (index_t)0) b.d_match[i] = (index_t)0;
        bool is_term = PRODUCTION_TO_TERMINAL_IS_VALID[b.d_productions[i]];
        b.d_node_is_term[i] = is_term ? 1 : 0;
        b.d_scan[i]         = is_term ? (index_t)1 : (index_t)0;
    }
    grid.sync();
    coopScanPass1<index_t>(b.d_scan, b.d_scan, b.d_agg_d, num_prods);
    grid.sync();
    coopScanPass2<index_t>(b.d_agg_d, num_tiles_p);
    grid.sync();
    // safe_zip: #terminal productions must equal #lexemes
    if (grank == 0 && b.d_agg_d[num_tiles_p - 1] != b.num_lexemes)
        atomicAnd(b.d_valid, 0);
    for (index_t i = grank; i < num_prods; i += gsz) {
        production_t prod = b.d_productions[i];
        if (b.d_node_is_term[i]) {
            index_t tile = i / (index_t)FUSED_BS;
            index_t lex  = b.d_scan[i] + ((tile > 0) ? b.d_agg_d[tile - 1] : (index_t)0) - (index_t)1;
            bool ok = lex < b.num_lexemes;
            if (!ok) atomicAnd(b.d_valid, 0);
            b.d_node_ids[i]    = (production_t)PRODUCTION_TO_TERMINAL[prod];
            b.d_node_starts[i] = ok ? b.d_lex_starts[lex] : (index_t)0;
            b.d_node_ends[i]   = ok ? b.d_lex_ends[lex]   : (index_t)0;
        } else {
            b.d_node_ids[i]    = (production_t)prod;
            b.d_node_starts[i] = (index_t)0;
            b.d_node_ends[i]   = (index_t)0;
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// Host side: allocation, launch, readback
// ---------------------------------------------------------------------------

template<typename T>
struct DevBuf {
    T* ptr = nullptr;
    DevBuf() = default;
    explicit DevBuf(size_t n) {
        if (n > 0) gpuAssert(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~DevBuf() { if (ptr) cudaFree(ptr); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    T* get() { return ptr; }
    const T* get() const { return ptr; }
};

struct ParserFused {
    FusedBufs   bufs{};
    terminal_t* d_arr   = nullptr;   // owned; bufs.d_arr aliases it
    uint32_t    P       = 0;         // cooperative grid size
    index_t     max_m   = 0;
    index_t     max_bk  = 0;
    index_t     max_pr  = 0;
};

static ParserFused allocParserFused(index_t max_m) {
    constexpr index_t B = (index_t)FUSED_BS * FUSED_IPT;
    constexpr index_t W = B / 32;

    ParserFused p;
    p.max_m  = max_m;
    p.max_bk = std::max<index_t>(max_m * (index_t)MAX_BRACKETS_PER_POSITION, 1);
    p.max_pr = std::max<index_t>(max_m * (index_t)MAX_PRODS_PER_POSITION, 1);

    // Combined mode reuses d_scan32/d_agg_d/d_match and the SPT scratch for
    // the parents phases, which run over up to max_pr elements.
#ifdef HAS_LEXER
    const index_t scan_n = std::max(p.max_bk, p.max_pr);
#else
    const index_t scan_n = p.max_bk;
#endif
    const index_t tiles_m    = (max_m + (index_t)FUSED_BS - 1) / (index_t)FUSED_BS;
    const index_t tiles_sc   = (scan_n + (index_t)FUSED_BS - 1) / (index_t)FUSED_BS;
    const index_t spt_blocks = (scan_n + B - 1) / B;
    const index_t M          = (index_t)nextPow2((int)spt_blocks);

    FusedBufs& b = p.bufs;
    gpuAssert(cudaMalloc(&p.d_arr,        (size_t)max_m * sizeof(terminal_t)));
    gpuAssert(cudaMalloc(&b.d_slens,      (size_t)max_m * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_plens,      (size_t)max_m * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_ss,         (size_t)max_m * sizeof(int32_t)));
    gpuAssert(cudaMalloc(&b.d_ps,         (size_t)max_m * sizeof(int32_t)));
    gpuAssert(cudaMalloc(&b.d_soffsets,   (size_t)max_m * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_poffsets,   (size_t)max_m * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_agg_s,      (size_t)tiles_m * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_agg_p,      (size_t)tiles_m * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_totals,     2 * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_brackets,   (size_t)p.max_bk * sizeof(bracket_t)));
    gpuAssert(cudaMalloc(&b.d_scan,       (size_t)scan_n * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_agg_d,      (size_t)tiles_sc * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_match,      (size_t)scan_n * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_productions, (size_t)p.max_pr * sizeof(production_t)));
    gpuAssert(cudaMalloc(&b.d_valid,      sizeof(int)));
    gpuAssert(cudaMalloc(&b.d_unres,      (size_t)spt_blocks * (size_t)(B / 32) * sizeof(unsigned)));
    gpuAssert(cudaMalloc(&b.d_block_mins, (size_t)spt_blocks * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_block_warp_mins, (size_t)spt_blocks * (size_t)W * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_tree,       (size_t)(2 * M - 1) * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_prefix_min, (size_t)spt_blocks * sizeof(index_t)));
    b.d_arr = p.d_arr;

#ifdef HAS_LEXER
    gpuAssert(cudaMalloc(&b.d_node_is_term, (size_t)p.max_pr * sizeof(uint8_t)));
    gpuAssert(cudaMalloc(&b.d_node_ids,     (size_t)p.max_pr * sizeof(production_t)));
    gpuAssert(cudaMalloc(&b.d_node_starts,  (size_t)p.max_pr * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_node_ends,    (size_t)p.max_pr * sizeof(index_t)));
    b.with_parents = false;
    b.num_lexemes  = 0;
    b.d_lex_starts = nullptr;
    b.d_lex_ends   = nullptr;
#endif

    int bps = 0, sms = 0;
    gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bps, parserFusedKernel, FUSED_BS, 0));
    gpuAssert(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    index_t needed   = std::max<index_t>(std::max(tiles_m, tiles_sc), 1);
    index_t resident = std::max<index_t>((index_t)bps * (index_t)sms, 1);
    p.P = (uint32_t)std::min(resident, needed);
    return p;
}

static void freeParserFused(ParserFused& p) {
    FusedBufs& b = p.bufs;
    cudaFree(p.d_arr);
    cudaFree(b.d_slens);    cudaFree(b.d_plens);
    cudaFree(b.d_ss);
    cudaFree(b.d_ps);
    cudaFree(b.d_soffsets); cudaFree(b.d_poffsets);
    cudaFree(b.d_agg_s);    cudaFree(b.d_agg_p);
    cudaFree(b.d_totals);
    cudaFree(b.d_brackets); cudaFree(b.d_scan);
    cudaFree(b.d_agg_d);    cudaFree(b.d_match);
    cudaFree(b.d_productions);
    cudaFree(b.d_valid);
    cudaFree(b.d_unres);
    cudaFree(b.d_block_mins);
    cudaFree(b.d_block_warp_mins);
    cudaFree(b.d_tree);
    cudaFree(b.d_prefix_min);
#ifdef HAS_LEXER
    cudaFree(b.d_node_is_term);
    cudaFree(b.d_node_ids);
    cudaFree(b.d_node_starts);
    cudaFree(b.d_node_ends);
#endif
    p = ParserFused{};
}

// Launch the fused kernel for m ≤ max_m positions (input already on device).
static void launchParserFused(ParserFused& p, index_t m) {
    p.bufs.m = m;
    void* args[] = { (void*)&p.bufs };
    gpuAssert(cudaLaunchCooperativeKernel(
        (void*)parserFusedKernel, p.P, FUSED_BS, args, 0, nullptr));
    gpuAssert(cudaGetLastError());
}

// Copy input, run, read back validity and productions.
static bool runParserFused(ParserFused& p, const terminal_t* h_arr, index_t m,
                           std::vector<uint64_t>& out_prods)
{
    gpuAssert(cudaMemcpy(p.d_arr, h_arr, (size_t)m * sizeof(terminal_t), cudaMemcpyHostToDevice));
    int one = 1;
    gpuAssert(cudaMemcpy(p.bufs.d_valid, &one, sizeof(int), cudaMemcpyHostToDevice));

#ifdef HAS_LEXER
    p.bufs.with_parents = false;
#endif
    launchParserFused(p, m);
    gpuAssert(cudaDeviceSynchronize());

    int h_valid = 0;
    gpuAssert(cudaMemcpy(&h_valid, p.bufs.d_valid, sizeof(int), cudaMemcpyDeviceToHost));
    out_prods.clear();
    if (!h_valid) return false;

    index_t totals[2];
    gpuAssert(cudaMemcpy(totals, p.bufs.d_totals, 2 * sizeof(index_t), cudaMemcpyDeviceToHost));
    index_t num_prods = totals[1];
    if (num_prods > (index_t)0) {
        std::vector<production_t> h_prods((size_t)num_prods);
        gpuAssert(cudaMemcpy(h_prods.data(), p.bufs.d_productions,
                             (size_t)num_prods * sizeof(production_t), cudaMemcpyDeviceToHost));
        out_prods.assign(h_prods.begin(), h_prods.end());
    }
    return true;
}

// ---------------------------------------------------------------------------
// One-shot pipeline used by the CLI
// ---------------------------------------------------------------------------

static bool runParserPipeline(const uint64_t* h_tokens_u64, uint64_t n,
                              std::vector<uint64_t>& out_prods)
{
    if (n > (uint64_t)(std::numeric_limits<index_t>::max() - 2)) return false;
    index_t m = (index_t)(n + 2);

    // PSE indices (and SPT sizes) are ints: reject inputs whose bracket
    // capacity bound cannot be guaranteed to fit.
    if ((uint64_t)m * (uint64_t)MAX_BRACKETS_PER_POSITION > (uint64_t)INT_MAX)
        return false;

    std::vector<terminal_t> h_arr((size_t)m);
    h_arr[0] = START_TERMINAL;
    for (index_t i = 0; i < (index_t)n; i++) h_arr[(size_t)(i + 1)] = (terminal_t)h_tokens_u64[(size_t)i];
    h_arr[(size_t)m - 1] = END_TERMINAL;

    ParserFused p = allocParserFused(m);
    bool ok = runParserFused(p, h_arr.data(), m, out_prods);
    freeParserFused(p);
    return ok;
}

#ifdef HAS_LEXER
// ---------------------------------------------------------------------------
// Combined mode: run the fused kernel with the parents/parse_int phases over
// lexer output that already lives on the device.
// ---------------------------------------------------------------------------

struct BothNodes {
    std::vector<uint8_t>  is_term;
    std::vector<uint64_t> parents;
    std::vector<uint64_t> ids;      // terminal id for terminal nodes, else production id
    std::vector<uint64_t> starts;
    std::vector<uint64_t> ends;
};

static bool runBothFused(ParserFused& p,
                         const terminal_t* d_tokens, index_t n,
                         const index_t* d_lex_starts, const index_t* d_lex_ends,
                         BothNodes& out)
{
    index_t m = n + (index_t)2;
    terminal_t sent = START_TERMINAL;
    gpuAssert(cudaMemcpy(p.d_arr, &sent, sizeof(terminal_t), cudaMemcpyHostToDevice));
    if (n > (index_t)0)
        gpuAssert(cudaMemcpy(p.d_arr + 1, d_tokens, (size_t)n * sizeof(terminal_t), cudaMemcpyDeviceToDevice));
    sent = END_TERMINAL;
    gpuAssert(cudaMemcpy(p.d_arr + (size_t)(m - 1), &sent, sizeof(terminal_t), cudaMemcpyHostToDevice));
    int one = 1;
    gpuAssert(cudaMemcpy(p.bufs.d_valid, &one, sizeof(int), cudaMemcpyHostToDevice));

    p.bufs.with_parents = true;
    p.bufs.num_lexemes  = n;
    p.bufs.d_lex_starts = d_lex_starts;
    p.bufs.d_lex_ends   = d_lex_ends;

    launchParserFused(p, m);
    gpuAssert(cudaDeviceSynchronize());

    int h_valid = 0;
    gpuAssert(cudaMemcpy(&h_valid, p.bufs.d_valid, sizeof(int), cudaMemcpyDeviceToHost));
    if (!h_valid) return false;

    index_t totals[2];
    gpuAssert(cudaMemcpy(totals, p.bufs.d_totals, 2 * sizeof(index_t), cudaMemcpyDeviceToHost));
    const index_t np = totals[1];

    out.is_term.resize((size_t)np);
    out.parents.resize((size_t)np);
    out.ids.resize((size_t)np);
    out.starts.resize((size_t)np);
    out.ends.resize((size_t)np);
    if (np > (index_t)0) {
        std::vector<index_t>      h_par((size_t)np);
        std::vector<production_t> h_ids((size_t)np);
        std::vector<index_t>      h_starts((size_t)np), h_ends((size_t)np);
        gpuAssert(cudaMemcpy(out.is_term.data(), p.bufs.d_node_is_term,
                             (size_t)np * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(h_par.data(), p.bufs.d_match,
                             (size_t)np * sizeof(index_t), cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(h_ids.data(), p.bufs.d_node_ids,
                             (size_t)np * sizeof(production_t), cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(h_starts.data(), p.bufs.d_node_starts,
                             (size_t)np * sizeof(index_t), cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(h_ends.data(), p.bufs.d_node_ends,
                             (size_t)np * sizeof(index_t), cudaMemcpyDeviceToHost));
        for (index_t i = 0; i < np; i++) {
            out.parents[(size_t)i] = (uint64_t)h_par[(size_t)i];
            out.ids[(size_t)i]     = (uint64_t)h_ids[(size_t)i];
            out.starts[(size_t)i]  = (uint64_t)h_starts[(size_t)i];
            out.ends[(size_t)i]    = (uint64_t)h_ends[(size_t)i];
        }
    }
    return true;
}
#endif // HAS_LEXER

// cli.cu — generated program entry point.
//
// Appended last (after common.cu, scan.cu, [lexer.cu], pse.cu, [parser.cu]
// and the grammar constants).  Provides a unified CLI for all three modes:
//   Lex-only:    reads raw bytes from stdin/file, emits token spans
//   Parse-only:  reads binary token-ID frames, emits production IDs
//   Both:        default → raw-byte test frames → lexer → fused parser with
//                parents/parse_int phases → CST nodes (C backend format);
//                --server / --benchmark keep the token-ID frame protocol
//
// Flags (all optional):
//   -i FILE              input file (default: stdin)
//   -o FILE              output file (default: stdout)
//   --block-size  N      128 or 256 (default: 256)
//   --items-per-thread N 2, 4, or 8 (default: auto from shared memory)
//   --shared-memory N    shared memory budget in bytes (default: device query)
//   --timeit             print kernel elapsed time to stderr
//   --server             length-prefixed loop: read u64-BE frame-length then
//                        that many bytes, process, write result, flush, repeat
//   --raw-input          (Both mode only) input is raw bytes; run full pipeline

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  -i FILE              input file (default: stdin)\n"
        "  -o FILE              output file (default: stdout)\n"
        "  --block-size N       128 or 256 (default: 256)\n"
        "  --items-per-thread N 2, 4, or 8 (default: auto)\n"
        "  --shared-memory N    shared memory budget bytes (default: device)\n"
        "  --timeit             print kernel time to stderr\n"
        "  --server             length-prefixed binary frame loop mode\n"
        "  --benchmark N        time N runs (GPU-only, pre-alloc, no I/O in loop)\n"
        "  --warmup N           warmup runs before timing (default: 3)\n"
#ifdef HAS_RAW_INPUT
        "  --raw-input          raw bytes -> full lexer+parser pipeline\n"
#endif
        , prog);
}

struct CliArgs {
    const char* input_file   = nullptr;   // null → stdin
    const char* output_file  = nullptr;   // null → stdout
    uint32_t    block_size   = 256;
    uint32_t    ipt          = 0;         // 0 = auto
    uint32_t    shared_mem   = 0;         // 0 = device query
    bool        timeit       = false;
    bool        server       = false;
    bool        raw_input    = false;
    uint32_t    benchmark    = 0;    // 0 = off; >0 = number of timed runs
    uint32_t    warmup       = 3;    // warmup runs before timing (used when benchmark > 0)
};

static bool parse_uint32(const char* s, uint32_t* out) {
    char* end;
    unsigned long v = strtoul(s, &end, 10);
    if (*end != '\0' || v > UINT32_MAX) return false;
    *out = (uint32_t)v;
    return true;
}

static CliArgs parse_args(int argc, char* argv[]) {
    CliArgs a;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); exit(0);
        } else if (strcmp(argv[i], "--timeit") == 0) {
            a.timeit = true;
        } else if (strcmp(argv[i], "--server") == 0) {
            a.server = true;
        } else if (strcmp(argv[i], "--raw-input") == 0) {
            a.raw_input = true;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            a.input_file = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            a.output_file = argv[++i];
        } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            if (!parse_uint32(argv[++i], &a.block_size) ||
                (a.block_size != 128 && a.block_size != 256)) {
                fprintf(stderr, "error: --block-size must be 128 or 256\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--items-per-thread") == 0 && i + 1 < argc) {
            if (!parse_uint32(argv[++i], &a.ipt) ||
                (a.ipt != 2 && a.ipt != 4 && a.ipt != 8)) {
                fprintf(stderr, "error: --items-per-thread must be 2, 4, or 8\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--shared-memory") == 0 && i + 1 < argc) {
            if (!parse_uint32(argv[++i], &a.shared_mem)) {
                fprintf(stderr, "error: --shared-memory must be a positive integer\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc) {
            if (!parse_uint32(argv[++i], &a.benchmark) || a.benchmark == 0) {
                fprintf(stderr, "error: --benchmark must be a positive integer\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            if (!parse_uint32(argv[++i], &a.warmup)) {
                fprintf(stderr, "error: --warmup must be a non-negative integer\n");
                exit(1);
            }
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
            usage(argv[0]); exit(1);
        }
    }
    return a;
}

// ---------------------------------------------------------------------------
// Device query helpers
// ---------------------------------------------------------------------------

static uint32_t device_shared_mem() {
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    return (uint32_t)prop.sharedMemPerBlock;
}

// Given a shared memory budget and block size, find the largest IPT in
// {2,4,8} that fits, or 2 if nothing fits (caller will assert later).
static uint32_t auto_ipt(uint32_t shmem_budget, uint32_t bs) {
    // Use the lexer's shmem formula (parser kernels use far less).
    // Try 8, 4, 2 in descending order.
    constexpr uint32_t candidates[] = {8, 4, 2};
    for (uint32_t ipt : candidates) {
        // Approximate: fixed_overhead + state_t*ipt*bs + max(I*ipt*bs, state_t*bs)
        // Use conservative upper bound with state_t = 4, I = 4.
        size_t est = (size_t)4 * bs                   // indices_aux
                   + 4                                // next_block_first_state
                   + 4                                // last_start
                   + 4 * WARP + WARP                  // values + statuses
                   + 4                                // shmem_prefix
                   + (size_t)4 * ipt * bs             // states
                   + (size_t)4 * ipt * bs;            // indices (max of indices, states_aux)
        if (est <= (size_t)shmem_budget * 9 / 10)
            return ipt;
    }
    return 2;
}

// ---------------------------------------------------------------------------
// I/O helpers (reused across modes)
// ---------------------------------------------------------------------------

static uint64_t read_u64_be(FILE* f) {
    uint8_t p[8];
    if (fread(p, 1, 8, f) != 8) return (uint64_t)-1;
    return ((uint64_t)p[0]<<56)|((uint64_t)p[1]<<48)|((uint64_t)p[2]<<40)|
           ((uint64_t)p[3]<<32)|((uint64_t)p[4]<<24)|((uint64_t)p[5]<<16)|
           ((uint64_t)p[6]<< 8)|((uint64_t)p[7]);
}

static void write_u64_be(FILE* f, uint64_t v) {
    uint8_t p[8];
    p[0]=(uint8_t)(v>>56); p[1]=(uint8_t)(v>>48);
    p[2]=(uint8_t)(v>>40); p[3]=(uint8_t)(v>>32);
    p[4]=(uint8_t)(v>>24); p[5]=(uint8_t)(v>>16);
    p[6]=(uint8_t)(v>> 8); p[7]=(uint8_t)(v);
    fwrite(p, 1, 8, f);
}

// Read an entire file into a heap buffer; caller frees.
static uint8_t* slurp(FILE* f, size_t* out_len) {
    size_t cap = 1 << 20, len = 0;
    uint8_t* buf = (uint8_t*)malloc(cap);
    size_t r;
    while ((r = fread(buf + len, 1, cap - len, f)) > 0) {
        len += r;
        if (len == cap) { cap *= 2; buf = (uint8_t*)realloc(buf, cap); }
    }
    *out_len = len;
    return buf;
}

// ---------------------------------------------------------------------------
// Dispatch table for (BLOCK_SIZE, ITEMS_PER_THREAD) combinations
// ---------------------------------------------------------------------------
//
// DISPATCH_BS_IPT(bs, ipt, EXPR) expands EXPR with template args <bs_val,ipt_val>
// appended, covering BS∈{128,256} × IPT∈{2,4,8} (6 combinations).
// Usage: DISPATCH_BS_IPT(bs, ipt, fn)(args...)
//   expands to fn<128,2>(args...) or fn<128,4>(args...) etc.

#define DISPATCH_BS_IPT(bs, ipt, fn, ...)                                        \
    ([&]() -> int {                                                               \
        if      ((bs)==128 && (ipt)==2) return fn<128,2>(__VA_ARGS__);           \
        else if ((bs)==128 && (ipt)==4) return fn<128,4>(__VA_ARGS__);           \
        else if ((bs)==128 && (ipt)==8) return fn<128,8>(__VA_ARGS__);           \
        else if ((bs)==256 && (ipt)==2) return fn<256,2>(__VA_ARGS__);           \
        else if ((bs)==256 && (ipt)==4) return fn<256,4>(__VA_ARGS__);           \
        else if ((bs)==256 && (ipt)==8) return fn<256,8>(__VA_ARGS__);           \
        else {                                                                    \
            fprintf(stderr, "unsupported (block_size=%u, ipt=%u)\n", bs, ipt);  \
            return 1;                                                             \
        }                                                                         \
    }())

// ---------------------------------------------------------------------------
// Lexer-only mode
//
// Reads raw bytes from `in`, emits ASCII token spans to `out`.
// One call per input (file mode / single-shot server frame).
// ---------------------------------------------------------------------------

#ifdef HAS_LEXER

template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
static int run_lexer_stream_impl(bool timeit) {
    constexpr uint32_t CHUNK_SIZE = 100u * (1u << 20);  // 100 MiB
    return lexer_stream<WriteAscii, CHUNK_SIZE, BLOCK_SIZE, ITEMS_PER_THREAD>(
        WriteAscii(), timeit);
}

#endif // HAS_LEXER

// ---------------------------------------------------------------------------
// Parser-only mode (binary token-ID frames)
//
// Protocol (same as c/parser.c and testcuda.sh):
//   Input:  u64 BE num_tests; per test: u64 BE n + n×u64 BE token ids
//   Output: u64 BE num_tests; per test: u8 valid; if valid: u64 BE np + np×u64 BE prod ids
//
// In server mode the framing differs: each "frame" is a single test
// (no outer num_tests count; the server loop provides that).
// ---------------------------------------------------------------------------

#ifdef HAS_PARSER

// Forward declarations from parser.cu (already defined above this file):
// static bool runParserPipeline(const uint64_t*, uint64_t, std::vector<uint64_t>&);

static void run_one_parser_test(const uint64_t* tokens, uint64_t n,
                                FILE* out) {
    std::vector<uint64_t> prods;
    bool ok = runParserPipeline(tokens, n, prods);
    if (ok) {
        fputc(1, out);
        write_u64_be(out, (uint64_t)prods.size());
        for (uint64_t p : prods) write_u64_be(out, p);
    } else {
        fputc(0, out);
    }
}

// Batch mode: read num_tests-prefixed stream
// Decode a big-endian u64 from a byte pointer (no alignment requirement).
static inline uint64_t decode_be64(const uint8_t* p) {
    return ((uint64_t)p[0]<<56)|((uint64_t)p[1]<<48)|((uint64_t)p[2]<<40)|
           ((uint64_t)p[3]<<32)|((uint64_t)p[4]<<24)|((uint64_t)p[5]<<16)|
           ((uint64_t)p[6]<< 8)|((uint64_t)p[7]);
}

// Bulk-read the entire input into memory, then decode, to avoid per-token fread overhead.
static int parser_batch(FILE* in, FILE* out) {
    size_t buf_len = 0;
    uint8_t* buf = slurp(in, &buf_len);
    if (!buf || buf_len < 8) {
        free(buf);
        fprintf(stderr, "error: truncated input\n"); return 1;
    }
    const uint8_t* p   = buf;
    const uint8_t* end = buf + buf_len;

    uint64_t num_tests = decode_be64(p); p += 8;
    write_u64_be(out, num_tests);

    std::vector<uint64_t> tokens;
    for (uint64_t t = 0; t < num_tests; t++) {
        if (p + 8 > end) { free(buf); fprintf(stderr, "truncated\n"); return 1; }
        uint64_t n = decode_be64(p); p += 8;
        if (p + n * 8 > end) { free(buf); fprintf(stderr, "truncated\n"); return 1; }
        tokens.resize(n);
        for (uint64_t i = 0; i < n; i++, p += 8)
            tokens[i] = decode_be64(p);
        run_one_parser_test(tokens.data(), n, out);
    }
    free(buf);
    fflush(out);
    return 0;
}

// Server mode: loop on length-prefixed frames.
// Each frame: u64 BE frame_byte_length, then frame_byte_length bytes
// containing u64 BE n + n×u64 BE token ids.
static int parser_server(FILE* in, FILE* out) {
    std::vector<uint64_t> tokens;
    std::vector<uint8_t>  frame;
    for (;;) {
        uint64_t frame_len = read_u64_be(in);
        if (feof(in)) break;
        if (frame_len == (uint64_t)-1 || frame_len < 8) {
            fprintf(stderr, "error: bad frame length\n"); return 1;
        }
        frame.resize(frame_len);
        if (fread(frame.data(), 1, frame_len, in) != frame_len) {
            fprintf(stderr, "error: truncated frame\n"); return 1;
        }
        const uint8_t* p = frame.data();
        uint64_t n = ((uint64_t)p[0]<<56)|((uint64_t)p[1]<<48)|
                     ((uint64_t)p[2]<<40)|((uint64_t)p[3]<<32)|
                     ((uint64_t)p[4]<<24)|((uint64_t)p[5]<<16)|
                     ((uint64_t)p[6]<< 8)|((uint64_t)p[7]);
        p += 8;
        if (frame_len != 8 + 8 * n) {
            fprintf(stderr, "error: frame length mismatch\n"); return 1;
        }
        tokens.resize(n);
        for (uint64_t i = 0; i < n; i++) {
            tokens[i] = ((uint64_t)p[0]<<56)|((uint64_t)p[1]<<48)|
                        ((uint64_t)p[2]<<40)|((uint64_t)p[3]<<32)|
                        ((uint64_t)p[4]<<24)|((uint64_t)p[5]<<16)|
                        ((uint64_t)p[6]<< 8)|((uint64_t)p[7]);
            p += 8;
        }
        run_one_parser_test(tokens.data(), n, out);
        fflush(out);
    }
    return 0;
}

// Benchmark mode: read all tests from `in`, pre-allocate GPU buffers once,
// run warmup passes, then time `n_runs` passes with CUDA events.
// Only the first test in the input file is used (single long test).
// Reports mean/stddev/min/max GPU time and throughput to stderr.
static int parser_benchmark(FILE* in, uint32_t warmup_runs, uint32_t n_runs) {
    // Slurp input
    size_t buf_len = 0;
    uint8_t* buf = slurp(in, &buf_len);
    if (!buf || buf_len < 16) {
        free(buf);
        fprintf(stderr, "error: benchmark input too short\n");
        return 1;
    }
    const uint8_t* p = buf;
    // uint64_t num_tests = decode_be64(p);  // use only first test
    p += 8;
    uint64_t n = decode_be64(p); p += 8;
    if (buf_len < 16 + n * 8) {
        free(buf); fprintf(stderr, "error: input truncated\n"); return 1;
    }

    index_t ni = (index_t)n;
    index_t m  = ni + (index_t)2;

    // Build host-side extended token array (terminal_t)
    std::vector<terminal_t> h_arr((size_t)m);
    h_arr[0] = START_TERMINAL;
    for (index_t i = 0; i < ni; i++) h_arr[i + 1] = (terminal_t)decode_be64(p + i * 8);
    h_arr[(size_t)m - 1] = END_TERMINAL;
    free(buf);

    fprintf(stderr, "Benchmark: %zu tokens, m=%zu\n", (size_t)ni, (size_t)m);
    fprintf(stderr, "  MAX_BRACKETS_PER_POSITION=%zu  MAX_PRODS_PER_POSITION=%zu\n",
            (size_t)MAX_BRACKETS_PER_POSITION, (size_t)MAX_PRODS_PER_POSITION);

    if ((uint64_t)m * (uint64_t)MAX_BRACKETS_PER_POSITION > (uint64_t)INT_MAX) {
        fprintf(stderr, "error: bracket capacity bound exceeds INT_MAX\n");
        return 1;
    }

    // Pre-allocate GPU buffers and upload the input once
    ParserFused pre = allocParserFused(m);
    gpuAssert(cudaMemcpy(pre.d_arr, h_arr.data(),
                         (size_t)m * sizeof(terminal_t), cudaMemcpyHostToDevice));
    fprintf(stderr, "  Cooperative grid: %u blocks x %u threads\n",
            pre.P, (uint32_t)FUSED_BS);

    // Warmup (full runs, checks the parse succeeds)
    fprintf(stderr, "  Warmup (%u runs)...\n", warmup_runs);
    std::vector<uint64_t> dummy_prods;
    for (uint32_t i = 0; i < warmup_runs; i++) {
        bool ok = runParserFused(pre, h_arr.data(), m, dummy_prods);
        fprintf(stderr, "    run %u: %s\n", i + 1, ok ? "OK" : "PARSE FAILED");
    }

    // Timed runs using CUDA events (kernel only)
    fprintf(stderr, "  Timing (%u runs)...\n", n_runs);
    std::vector<float> times_ms(n_runs);
    cudaEvent_t ev0, ev1;
    gpuAssert(cudaEventCreate(&ev0));
    gpuAssert(cudaEventCreate(&ev1));

    for (uint32_t i = 0; i < n_runs; i++) {
        int one = 1;
        gpuAssert(cudaMemcpy(pre.bufs.d_valid, &one, sizeof(int), cudaMemcpyHostToDevice));

        gpuAssert(cudaEventRecord(ev0));
        launchParserFused(pre, m);
        gpuAssert(cudaEventRecord(ev1));
        gpuAssert(cudaEventSynchronize(ev1));
        gpuAssert(cudaEventElapsedTime(&times_ms[i], ev0, ev1));
    }

    gpuAssert(cudaEventDestroy(ev0));
    gpuAssert(cudaEventDestroy(ev1));
    freeParserFused(pre);

    // Statistics
    double sum = 0, mn = times_ms[0], mx = times_ms[0];
    for (float t : times_ms) {
        sum += t;
        if (t < mn) mn = t;
        if (t > mx) mx = t;
    }
    double mean = sum / n_runs;
    double var = 0;
    for (float t : times_ms) var += (t - mean) * (t - mean);
    double stddev = n_runs > 1 ? sqrt(var / (n_runs - 1)) : 0;

    fprintf(stderr, "\n");
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, " --benchmark results (GPU time only, pre-alloc buffers)\n");
    fprintf(stderr, " Tokens: %zu   m: %zu   Warmup: %u   Runs: %u\n",
            (size_t)ni, (size_t)m, warmup_runs, n_runs);
    fprintf(stderr, "------------------------------------------------------------\n");
    fprintf(stderr, " Mean:       %.3f ms\n", mean);
    fprintf(stderr, " Stddev:     %.3f ms\n", stddev);
    fprintf(stderr, " Min:        %.3f ms\n", mn);
    fprintf(stderr, " Max:        %.3f ms\n", mx);
    fprintf(stderr, " Throughput: %.0f Mtok/s\n", (double)ni / (mean * 1e-3) / 1e6);
    fprintf(stderr, "============================================================\n");

    return 0;
}

#endif // HAS_PARSER

// ---------------------------------------------------------------------------
// Both mode: full pipeline (raw bytes → lexer → fused parser with parents)
//
// Input:  u64 BE num_tests; per test: u64 BE n + n raw bytes
// Output: u64 BE num_tests; per test: u8 valid; if valid:
//         u64 BE num_nodes; per node: u8 is_terminal, u64 BE parent,
//         u64 BE id (terminal id for terminal nodes, else production id),
//         u64 BE span start, u64 BE span end (0, 0 for nonterminal nodes)
// Same format as the generated C backend's combined mode.
// ---------------------------------------------------------------------------

#if defined(HAS_LEXER) && defined(HAS_PARSER)

template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
static void run_one_both_test(const uint8_t* bytes, uint64_t n, FILE* out) {
    bool valid = true;
    uint32_t num_lex = 0;
    uint8_t*    d_string    = nullptr;
    terminal_t* d_terminals = nullptr;
    index_t*    d_starts    = nullptr;
    index_t*    d_ends      = nullptr;

    if (n > 0) {
        gpuAssert(cudaMalloc(&d_string,    n * sizeof(uint8_t)));
        gpuAssert(cudaMalloc(&d_terminals, n * sizeof(terminal_t)));
        gpuAssert(cudaMalloc(&d_starts,    n * sizeof(index_t)));
        gpuAssert(cudaMalloc(&d_ends,      n * sizeof(index_t)));
        gpuAssert(cudaMemcpy(d_string, bytes, n, cudaMemcpyHostToDevice));

        // Single-chunk launch on a fresh context (no streaming carry-over).
        LexerCtx<uint32_t, index_t> ctx((uint32_t)n, BLOCK_SIZE, ITEMS_PER_THREAD);
        const uint32_t num_blocks = numBlocks((uint32_t)n, BLOCK_SIZE, ITEMS_PER_THREAD);
        lexer<uint32_t, index_t, BLOCK_SIZE, ITEMS_PER_THREAD><<<num_blocks, BLOCK_SIZE>>>(
            ctx, d_string, d_terminals, d_starts, d_ends, (uint32_t)n, true);
        gpuAssert(cudaDeviceSynchronize());
        gpuAssert(cudaPeekAtLastError());
        num_lex = ctx.terminalsSize();
        valid   = ctx.isAccept();
        ctx.cleanUp();
    }

    BothNodes nodes;
    if (valid) {
        index_t m = (index_t)num_lex + (index_t)2;
        if ((uint64_t)m * (uint64_t)MAX_BRACKETS_PER_POSITION > (uint64_t)INT_MAX ||
            (uint64_t)m * (uint64_t)MAX_PRODS_PER_POSITION    > (uint64_t)INT_MAX) {
            valid = false;
        } else {
            ParserFused p = allocParserFused(m);
            valid = runBothFused(p, d_terminals, (index_t)num_lex,
                                 d_starts, d_ends, nodes);
            freeParserFused(p);
        }
    }

    if (d_string)    cudaFree(d_string);
    if (d_terminals) cudaFree(d_terminals);
    if (d_starts)    cudaFree(d_starts);
    if (d_ends)      cudaFree(d_ends);

    if (!valid) { fputc(0, out); return; }
    fputc(1, out);
    write_u64_be(out, (uint64_t)nodes.ids.size());
    for (size_t i = 0; i < nodes.ids.size(); i++) {
        fputc(nodes.is_term[i] ? 1 : 0, out);
        write_u64_be(out, nodes.parents[i]);
        write_u64_be(out, nodes.ids[i]);
        write_u64_be(out, nodes.starts[i]);
        write_u64_be(out, nodes.ends[i]);
    }
}

template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
static int both_batch_impl(FILE* in, FILE* out) {
    size_t buf_len = 0;
    uint8_t* buf = slurp(in, &buf_len);
    if (!buf || buf_len < 8) {
        free(buf);
        fprintf(stderr, "error: truncated input\n"); return 1;
    }
    const uint8_t* p   = buf;
    const uint8_t* end = buf + buf_len;

    uint64_t num_tests = decode_be64(p); p += 8;
    write_u64_be(out, num_tests);

    for (uint64_t t = 0; t < num_tests; t++) {
        if (p + 8 > end) { free(buf); fprintf(stderr, "truncated\n"); return 1; }
        uint64_t n = decode_be64(p); p += 8;
        if (n > (uint64_t)(end - p)) { free(buf); fprintf(stderr, "truncated\n"); return 1; }
        run_one_both_test<BLOCK_SIZE, ITEMS_PER_THREAD>(p, n, out);
        p += n;
    }
    free(buf);
    fflush(out);
    return 0;
}

#endif // HAS_LEXER && HAS_PARSER

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    CliArgs a = parse_args(argc, argv);

    // Open files
    FILE* in  = stdin;
    FILE* out = stdout;
    if (a.input_file)  { in  = fopen(a.input_file,  "rb"); if (!in)  { perror(a.input_file);  return 1; } }
    if (a.output_file) { out = fopen(a.output_file, "wb"); if (!out) { perror(a.output_file); return 1; } }

    // Resolve shared memory budget
    uint32_t shmem = a.shared_mem ? a.shared_mem : device_shared_mem();

    // Resolve IPT
    uint32_t ipt = a.ipt ? a.ipt : auto_ipt(shmem, a.block_size);
    uint32_t bs  = a.block_size;

    // Validate: pick the largest supported IPT ≤ requested that fits in shmem
    // (if user overrode IPT, trust them; the kernel will assert on bad shmem).

#if defined(HAS_LEXER) && !defined(HAS_PARSER)
    // ---- Lex-only mode ----
    if (a.raw_input) {
        fprintf(stderr, "error: --raw-input requires both lexer and parser\n");
        return 1;
    }
    // lexer_stream reads from stdin; redirect fd if user gave -i FILE
    if (in != stdin) {
        if (dup2(fileno(in), STDIN_FILENO) < 0) { perror("dup2"); return 1; }
        fclose(in); in = stdin;
    }
    return DISPATCH_BS_IPT(bs, ipt, run_lexer_stream_impl, a.timeit);

#elif defined(HAS_PARSER) && !defined(HAS_LEXER)
    // ---- Parse-only mode ----
    if (a.raw_input) {
        fprintf(stderr, "error: --raw-input requires both lexer and parser\n");
        return 1;
    }
    int ret;
    if (a.benchmark > 0)
        ret = parser_benchmark(in, a.warmup, a.benchmark);
    else if (a.server)
        ret = parser_server(in, out);
    else
        ret = parser_batch(in, out);
    if (in  != stdin)  fclose(in);
    if (out != stdout) fclose(out);
    return ret;

#elif defined(HAS_LEXER) && defined(HAS_PARSER)
    // ---- Both mode: raw bytes → lexer → parser → CST nodes (default) ----
    // --server and --benchmark keep the token-ID protocol of parse-only mode.
    int ret;
    if (a.benchmark > 0)
        ret = parser_benchmark(in, a.warmup, a.benchmark);
    else if (a.server)
        ret = parser_server(in, out);
    else
        ret = DISPATCH_BS_IPT(bs, ipt, both_batch_impl, in, out);
    if (in  != stdin)  fclose(in);
    if (out != stdout) fclose(out);
    return ret;
#else
    fprintf(stderr, "error: no lexer or parser compiled in\n");
    return 1;
#endif
}

