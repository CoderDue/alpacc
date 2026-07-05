// parser.cu — staged LLP parser kernels (Stages 1–3), appended by the code
// generator after common.cu, scan.cu, pse.cu, and the grammar constants.
//
// Stage 1: per-position hash-table key lookup (FNV-1a + linear probe),
//          produces (stack_len, prod_len) pairs and a validity flag.
//          Two separate inclusive scans (one per component) give offsets.
// Stage 2: segmented copy of STACKS spans (brackets), ±1 depth inclusive
//          scan with validity check, PSE(<=) via runSPT for bracket matching,
//          bracket symbol comparison.
// Stage 3: segmented copy of PRODUCTIONS spans into the output array.
//
// All kernels are templated on an index type I (uint32_t or size_t).
// Span values (HASH_TABLE_{STACKS,PRODUCTIONS}_SPAN) are int64_t to match
// the -1 sentinel used by the Futhark and C backends.
//
// Binary I/O (same protocol as c/parser.c and `alpacc test compare --parser`):
//   inputs:  u64 BE num_tests; per test: u64 BE n + n×u64 BE terminal ids
//   outputs: u64 BE num_tests; per test: 1 byte validity; if valid:
//            u64 BE num_prods + num_prods×u64 BE production ids

#include <vector>
#include <limits>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// Stage 1: combined key-lookup + span materialisation kernel
//
// One thread per position i in [0, m), m = n + 2.
// Builds the Q+K window key with sentinel-extended addressing:
//   d_arr[0] = START_TERMINAL, d_arr[1..n] = tokens, d_arr[n+1] = END_TERMINAL.
// FNV-1a hash, linear probe up to MAX_ITERS slots.
// Writes stack_len into d_slens[i], prod_len into d_plens[i], and the raw
// int64_t spans d_ss/d_se/d_ps/d_pe (needed for Stages 2 & 3 segmented copies).
// Sets *d_valid = 0 atomically on any miss or invalid span.
// ---------------------------------------------------------------------------

template<typename I>
__global__ void
parserKeysSpans(const terminal_t* __restrict__ d_arr,
                I                              m,
                I*                             d_slens,   // stack segment lengths
                I*                             d_plens,   // prod segment lengths
                int64_t* __restrict__          d_ss,
                int64_t* __restrict__          d_se,
                int64_t* __restrict__          d_ps,
                int64_t* __restrict__          d_pe,
                int*                           d_valid)
{
    I i = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x;
    if (i >= m) return;

    // Build key with sentinel padding
    terminal_t key[Q + K];
#pragma unroll
    for (size_t j = 0; j < Q + K; j++) {
        int64_t idx = (int64_t)i + (int64_t)j - (int64_t)Q;
        key[j] = (idx < 0 || idx >= (int64_t)m) ? EMPTY_TERMINAL : d_arr[(I)idx];
    }

    // FNV-1a
    uint64_t h = 14695981039346656037ULL;
    for (size_t j = 0; j < Q + K; j++)
        h = (h ^ (uint64_t)key[j]) * 1099511628211ULL;
    h %= (uint64_t)HASH_TABLE_SIZE;

    // Linear probe
    bool    found = false;
    int64_t ss = -1, se = -1, ps = -1, pe = -1;
    for (size_t it = 0; it < MAX_ITERS; it++) {
        size_t slot = (size_t)h;
        if (HASH_TABLE_IS_VALID[slot]) {
            bool match = true;
#pragma unroll
            for (size_t j = 0; j < Q + K; j++) {
                if (HASH_TABLE_KEYS[slot][j] != key[j]) { match = false; break; }
            }
            if (match) {
                ss = HASH_TABLE_STACKS_SPAN[slot][0];
                se = HASH_TABLE_STACKS_SPAN[slot][1];
                ps = HASH_TABLE_PRODUCTIONS_SPAN[slot][0];
                pe = HASH_TABLE_PRODUCTIONS_SPAN[slot][1];
                found = true;
                break;
            }
        }
        h = (h + 1) % (uint64_t)HASH_TABLE_SIZE;
    }

    bool valid = found && ss >= 0 && se >= 0 && ps >= 0 && pe >= 0;
    if (!valid) atomicAnd(d_valid, 0);

    d_slens[i] = valid ? (I)(se - ss) : (I)0;
    d_plens[i] = valid ? (I)(pe - ps) : (I)0;
    d_ss[i] = ss; d_se[i] = se;
    d_ps[i] = ps; d_pe[i] = pe;
}

// ---------------------------------------------------------------------------
// Stage 1: inclusive scan of I-typed lengths using scan.cu's Add<I> operator.
// One kernel per array (stack lengths and prod lengths scanned separately).
// ---------------------------------------------------------------------------

// inclusiveScanIKernel: scan T-typed data using uint32_t for block indices.
// T is the element type (uint32_t or size_t), must be compatible with Add<T>
// and assignable through volatile pointers (primitives only).
template<typename T, uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__global__ void
inclusiveScanIKernel(T*                          d_data,
                     States<uint32_t, T>          states,
                     volatile uint32_t*            d_dyn_idx,
                     uint64_t                      total64)
{
    constexpr uint32_t N = BLOCK_SIZE * ITEMS_PER_THREAD;
    __shared__ volatile T shmem[N];
    __shared__ volatile T shmem_aux[BLOCK_SIZE];

    uint32_t dyn_idx = dynamicIndex(d_dyn_idx);
    uint64_t glb_offs = (uint64_t)dyn_idx * N;

    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
        uint32_t lid = i * BLOCK_SIZE + threadIdx.x;
        uint64_t gid = glb_offs + lid;
        shmem[lid] = (gid < total64) ? d_data[gid] : (T)0;
    }
    __syncthreads();

    scan<T, uint32_t, Add<T>, (uint32_t)ITEMS_PER_THREAD>(
        shmem, shmem_aux, states, Add<T>(), (T)0, dyn_idx);

    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
        uint32_t lid = i * BLOCK_SIZE + threadIdx.x;
        uint64_t gid = glb_offs + lid;
        if (gid < total64) d_data[gid] = shmem[lid];
    }
}

// Specialisation for uint32_t (used by runInclusiveScanI32 for depths scan)
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
__global__ void
inclusiveScanI32Kernel(int32_t*                    d_data,
                       States<uint32_t, int32_t>   states,
                       volatile uint32_t*           d_dyn_idx,
                       uint32_t                     total)
{
    constexpr uint32_t N = BLOCK_SIZE * ITEMS_PER_THREAD;
    __shared__ volatile int32_t shmem[N];
    __shared__ volatile int32_t shmem_aux[BLOCK_SIZE];

    uint32_t dyn_idx = dynamicIndex(d_dyn_idx);
    uint32_t glb_offs = dyn_idx * N;

    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
        uint32_t lid = i * BLOCK_SIZE + threadIdx.x;
        uint32_t gid = glb_offs + lid;
        shmem[lid] = (gid < total) ? d_data[gid] : (int32_t)0;
    }
    __syncthreads();

    scan<int32_t, uint32_t, Add<int32_t>, (uint32_t)ITEMS_PER_THREAD>(
        shmem, shmem_aux, states, Add<int32_t>(), (int32_t)0, dyn_idx);

    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
        uint32_t lid = i * BLOCK_SIZE + threadIdx.x;
        uint32_t gid = glb_offs + lid;
        if (gid < total) d_data[gid] = shmem[lid];
    }
}

// ---------------------------------------------------------------------------
// Stage 1: convert inclusive scan → exclusive offsets, capture total
// ---------------------------------------------------------------------------

template<typename I>
__global__ void
inclusiveToExclusive(const I* __restrict__ d_inc,
                     I*                    d_exc,
                     I                     total,
                     I*                    d_last)
{
    I i = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x;
    if (i >= total) return;
    d_exc[i] = (i == (I)0) ? (I)0 : d_inc[i - (I)1];
    if (i == total - (I)1) *d_last = d_inc[i];
}

// ---------------------------------------------------------------------------
// Stage 2a: segmented copy of STACKS spans → d_brackets
//
// One thread per position i; copies STACKS[ss..se) to d_brackets[offset..].
// ---------------------------------------------------------------------------

template<typename I>
__global__ void
parserStacks(const I* __restrict__          d_soffsets,  // exclusive scan of stack lens
             const int64_t* __restrict__    d_ss,
             const int64_t* __restrict__    d_se,
             bracket_t*                     d_brackets,
             I                              m)
{
    I i = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x;
    if (i >= m) return;
    I   off = d_soffsets[i];
    int64_t ss = d_ss[i], se = d_se[i];
    for (int64_t j = ss; j < se; j++)
        d_brackets[off + (I)(j - ss)] = STACKS[j];
}

// ---------------------------------------------------------------------------
// Stage 2b: map brackets → ±1 depth deltas
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool bracket_is_left(bracket_t b) {
    return (b >> (8 * (int)sizeof(bracket_t) - 1)) & 1;
}

__device__ __forceinline__ bracket_t bracket_unpack(bracket_t b) {
    return b & (bracket_t)~((bracket_t)1 << (8 * (int)sizeof(bracket_t) - 1));
}

template<typename I>
__global__ void
bracketsToDeltas(const bracket_t* __restrict__ d_brackets,
                 int32_t*                      d_deltas,
                 I                             num_brackets)
{
    I i = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x;
    if (i >= num_brackets) return;
    d_deltas[i] = bracket_is_left(d_brackets[i]) ? (int32_t)1 : (int32_t)-1;
}

// ---------------------------------------------------------------------------
// Stage 2b: compute depths + validity from inclusive scan
//   depth_i = inclusive_scan_i - is_left_i
//   validity: any prefix < 0, or last != 0 → invalid
// ---------------------------------------------------------------------------

template<typename I>
__global__ void
depthsAndValidate(const int32_t* __restrict__  d_scan,
                  const bracket_t* __restrict__ d_brackets,
                  int32_t*                      d_depths,
                  int*                          d_valid,
                  I                             num_brackets)
{
    I i = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x;
    if (i >= num_brackets) return;
    int32_t s = d_scan[i];
    if (s < 0) atomicAnd(d_valid, 0);
    if (i == num_brackets - (I)1 && s != 0) atomicAnd(d_valid, 0);
    d_depths[i] = s - (bracket_is_left(d_brackets[i]) ? 1 : 0);
}

// ---------------------------------------------------------------------------
// Stage 2c: bracket symbol match check using PSE result
// ---------------------------------------------------------------------------

template<typename I>
__global__ void
checkBracketMatches(const bracket_t* __restrict__ d_brackets,
                    const int* __restrict__        d_match,
                    int*                           d_valid,
                    I                              num_brackets)
{
    I i = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x;
    if (i >= num_brackets) return;
    if (bracket_is_left(d_brackets[i])) return;
    int j = d_match[i];
    if (j < 0 || bracket_unpack(d_brackets[(I)j]) != bracket_unpack(d_brackets[i]))
        atomicAnd(d_valid, 0);
}

// ---------------------------------------------------------------------------
// Stage 3: segmented copy of PRODUCTIONS spans → d_productions
// ---------------------------------------------------------------------------

template<typename I>
__global__ void
parserProductions(const I* __restrict__          d_poffsets,  // exclusive scan of prod lens
                  const int64_t* __restrict__    d_ps,
                  const int64_t* __restrict__    d_pe,
                  production_t*                  d_productions,
                  I                              m)
{
    I i = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x;
    if (i >= m) return;
    I   off = d_poffsets[i];
    int64_t ps = d_ps[i], pe = d_pe[i];
    for (int64_t j = ps; j < pe; j++)
        d_productions[off + (I)(j - ps)] = PRODUCTIONS[j];
}

// ---------------------------------------------------------------------------
// RAII wrapper for a device allocation
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


// ---------------------------------------------------------------------------
// Helper: run inclusive scan of I in-place on a device array of length m.
// Uses States<uint32_t, I> so the number of scan blocks must fit in uint32_t.
// ---------------------------------------------------------------------------

template<typename T>
static void runInclusiveScanI(T* d_data, uint64_t m) {
    constexpr uint32_t BS  = 256;
    constexpr uint32_t IPT = 4;
    if (m == 0) return;
    uint64_t nb64 = (m + (uint64_t)(BS * IPT) - 1) / (uint64_t)(BS * IPT);
    uint32_t nb = (uint32_t)nb64;  // guarded by caller: num scan blocks fits in uint32_t

    DevBuf<uint32_t> dyn(1);
    gpuAssert(cudaMemset(dyn.get(), 0, sizeof(uint32_t)));

    States<uint32_t, T> st(nb);
    inclusiveScanIKernel<T, BS, IPT>
        <<<nb, BS>>>(d_data, st, (volatile uint32_t*)dyn.get(), m);
    gpuAssert(cudaGetLastError());
    gpuAssert(cudaDeviceSynchronize());
    st.cleanUp();
}

// ---------------------------------------------------------------------------
// Helper: run inclusive scan of int32_t in-place (for bracket depths)
// ---------------------------------------------------------------------------

static void runInclusiveScanI32(int32_t* d_data, uint32_t n) {
    constexpr uint32_t BS  = 256;
    constexpr uint32_t IPT = 4;
    if (n == 0) return;
    uint32_t nb = (n + BS * IPT - 1) / (BS * IPT);

    DevBuf<uint32_t> dyn(1);
    gpuAssert(cudaMemset(dyn.get(), 0, sizeof(uint32_t)));

    States<uint32_t, int32_t> st(nb);
    inclusiveScanI32Kernel<BS, IPT>
        <<<nb, BS>>>(d_data, st, (volatile uint32_t*)dyn.get(), n);
    gpuAssert(cudaGetLastError());
    gpuAssert(cudaDeviceSynchronize());
    st.cleanUp();
}

// ---------------------------------------------------------------------------
// Full pipeline, templated on index type I
// ---------------------------------------------------------------------------

template<typename I>
static bool runParserPipeline(const uint64_t* h_tokens_u64, uint64_t n,
                              std::vector<uint64_t>& out_prods)
{
    constexpr uint32_t BS = 256;

    if (n > (uint64_t)(std::numeric_limits<I>::max() - 2)) return false;
    I m = (I)(n + 2);

    // ---- Build extended token array on device ----
    DevBuf<terminal_t> d_arr(m);
    {
        std::vector<terminal_t> h_arr(m);
        h_arr[0] = START_TERMINAL;
        for (uint64_t i = 0; i < n; i++) h_arr[i + 1] = (terminal_t)h_tokens_u64[i];
        h_arr[m - 1] = END_TERMINAL;
        gpuAssert(cudaMemcpy(d_arr.get(), h_arr.data(), m * sizeof(terminal_t),
                             cudaMemcpyHostToDevice));
    }

    // ---- Validity flag ----
    DevBuf<int> d_valid(1);
    {
        int one = 1;
        gpuAssert(cudaMemcpy(d_valid.get(), &one, sizeof(int), cudaMemcpyHostToDevice));
    }

    // ---- Stage 1a: key lookup + span materialisation ----
    DevBuf<I>       d_slens(m), d_plens(m);
    DevBuf<int64_t> d_ss(m), d_se(m), d_ps(m), d_pe(m);
    {
        uint32_t g = (uint32_t)((m + (I)(BS - 1)) / (I)BS);
        parserKeysSpans<I><<<g, BS>>>(
            d_arr.get(), m,
            d_slens.get(), d_plens.get(),
            d_ss.get(), d_se.get(), d_ps.get(), d_pe.get(),
            d_valid.get());
        gpuAssert(cudaGetLastError());
        gpuAssert(cudaDeviceSynchronize());
    }
    {
        int h_valid;
        gpuAssert(cudaMemcpy(&h_valid, d_valid.get(), sizeof(int), cudaMemcpyDeviceToHost));
        if (!h_valid) return false;
    }

    // ---- Stage 1b: inclusive scan of stack and prod lengths separately ----
    runInclusiveScanI<I>(d_slens.get(), (uint64_t)m);
    runInclusiveScanI<I>(d_plens.get(), (uint64_t)m);

    // ---- Stage 1c: inclusive → exclusive offsets, capture totals ----
    DevBuf<I> d_soffsets(m), d_poffsets(m);
    I num_brackets = 0, num_prods = 0;
    {
        DevBuf<I> d_stotal(1), d_ptotal(1);
        uint32_t g = (uint32_t)((m + (I)(BS - 1)) / (I)BS);
        inclusiveToExclusive<I><<<g, BS>>>(d_slens.get(), d_soffsets.get(), m, d_stotal.get());
        inclusiveToExclusive<I><<<g, BS>>>(d_plens.get(), d_poffsets.get(), m, d_ptotal.get());
        gpuAssert(cudaGetLastError());
        gpuAssert(cudaMemcpy(&num_brackets, d_stotal.get(), sizeof(I), cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(&num_prods,    d_ptotal.get(), sizeof(I), cudaMemcpyDeviceToHost));
    }
    // d_slens / d_plens no longer needed; freed at end of scope.

    // ---- Stage 2: brackets ----
    {
        DevBuf<bracket_t> d_brackets(num_brackets > (I)0 ? (size_t)num_brackets : 1);
        DevBuf<int32_t>   d_scan32(num_brackets > (I)0 ? (size_t)num_brackets : 1);
        DevBuf<int32_t>   d_depths(num_brackets > (I)0 ? (size_t)num_brackets : 1);
        DevBuf<int>       d_match(num_brackets > (I)0 ? (size_t)num_brackets : 1);

        if (num_brackets > (I)0) {
            // 2a: segmented copy of STACKS spans
            {
                uint32_t g = (uint32_t)((m + (I)(BS - 1)) / (I)BS);
                parserStacks<I><<<g, BS>>>(
                    d_soffsets.get(), d_ss.get(), d_se.get(),
                    d_brackets.get(), m);
                gpuAssert(cudaGetLastError());
                gpuAssert(cudaDeviceSynchronize());
            }

            // 2b: ±1 delta map
            {
                uint32_t g = (uint32_t)((num_brackets + (I)(BS - 1)) / (I)BS);
                bracketsToDeltas<I><<<g, BS>>>(d_brackets.get(), d_scan32.get(), num_brackets);
                gpuAssert(cudaGetLastError());
                gpuAssert(cudaDeviceSynchronize());
            }

            // 2b: inclusive scan of ±1 deltas
            if ((uint64_t)num_brackets > (uint64_t)INT_MAX) return false;
            runInclusiveScanI32(d_scan32.get(), (uint32_t)num_brackets);

            // 2b: depths + validity
            {
                uint32_t g = (uint32_t)((num_brackets + (I)(BS - 1)) / (I)BS);
                depthsAndValidate<I><<<g, BS>>>(
                    d_scan32.get(), d_brackets.get(), d_depths.get(),
                    d_valid.get(), num_brackets);
                gpuAssert(cudaGetLastError());
                gpuAssert(cudaDeviceSynchronize());
            }
            {
                int h_valid;
                gpuAssert(cudaMemcpy(&h_valid, d_valid.get(), sizeof(int), cudaMemcpyDeviceToHost));
                if (!h_valid) return false;
            }

            // 2c: PSE(<=) over depths for bracket matching
            {
                SPTScratch<int32_t, 128, 4> spt =
                    allocSPTScratch<int32_t, 128, 4, true>((int)num_brackets);
                runSPT<int32_t, 128, 4, true>(
                    d_depths.get(), d_match.get(), (int)num_brackets, spt);
                gpuAssert(cudaDeviceSynchronize());
                freeSPTScratch<int32_t, 128, 4>(spt);
            }

            // 2d: bracket symbol check
            {
                uint32_t g = (uint32_t)((num_brackets + (I)(BS - 1)) / (I)BS);
                checkBracketMatches<I><<<g, BS>>>(
                    d_brackets.get(), d_match.get(), d_valid.get(), num_brackets);
                gpuAssert(cudaGetLastError());
                gpuAssert(cudaDeviceSynchronize());
            }
            {
                int h_valid;
                gpuAssert(cudaMemcpy(&h_valid, d_valid.get(), sizeof(int), cudaMemcpyDeviceToHost));
                if (!h_valid) return false;
            }
        }
    }  // bracket buffers freed here

    // ---- Stage 3: productions ----
    if (num_prods > (I)0) {
        DevBuf<production_t> d_productions(num_prods);
        {
            uint32_t g = (uint32_t)((m + (I)(BS - 1)) / (I)BS);
            parserProductions<I><<<g, BS>>>(
                d_poffsets.get(), d_ps.get(), d_pe.get(),
                d_productions.get(), m);
            gpuAssert(cudaGetLastError());
            gpuAssert(cudaDeviceSynchronize());
        }
        std::vector<production_t> h_prods(num_prods);
        gpuAssert(cudaMemcpy(h_prods.data(), d_productions.get(),
                             num_prods * sizeof(production_t),
                             cudaMemcpyDeviceToHost));
        out_prods.resize(num_prods);
        for (I i = 0; i < num_prods; i++) out_prods[i] = (uint64_t)h_prods[i];
    }

    return true;
}

