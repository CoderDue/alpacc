// parser.cu — fused single-kernel LLP parser, appended by the code generator
// after common.cu, scan.cu, pse.cu, and the grammar constants.
//
// The whole pipeline runs in ONE cooperative kernel (parserFusedKernel),
// launched via cudaLaunchCooperativeKernel with grid.sync() between phases:
//
//   Phase A+B: per-position hash-table key lookup (FNV-1a + linear probe)
//            fused with the per-tile scans of both length arrays: lengths
//            go straight from the lookup into shared memory, are scanned
//            there, and only the tile-local scan values + tile aggregates
//            reach global memory (a second pass scans the aggregates).
//            Phase C recovers each length as the difference of adjacent
//            tile-local scan values, so no lengths array exists at all.
//   Phase C: segmented copies of STACKS spans (brackets) and
//            PRODUCTIONS spans into their compacted output arrays.
//   Phase D: cooperative inclusive scan of ±1 bracket deltas (computed
//            on the fly from d_brackets) → depths; validity
//            check (no negative prefix, last must be 0).
//   Phase E: PSE(<=) over depths via apsepDeviceSPT (pse.cu), which is
//            itself structured as grid.sync() phases.
//   Phase F: bracket symbol check of each right bracket against its match.
//
// Combined (lexer+parser) mode additionally computes the compact parse
// tree (mirrors `parents` / `parse_int` in parser.fut), still inside the
// same cooperative kernel:
//   Phase G: exclusive scan of (arity - 1) over the productions.
//   Phase H: PSE(<=) over that scan → parent vector (parents[0] = 0).
//   Phase I: scan of is-terminal flags → compaction: terminal slots are
//            dropped, parents remapped into the compacted numbering, and
//            each terminal slot records its parent as token_parents[lexeme].
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
// Pass 1: per-tile inclusive block scan of f(d_in[i]) into d_out (tile-local
//         values only) and the tile aggregate into d_agg[tile].
//         The functor maps the (possibly narrower) source element to the
//         scanned value, so ±1 deltas / arities / flags never round-trip
//         through global memory as index_t arrays.
//         Safe with d_in == d_out (in-place).
// Pass 2: block 0 turns d_agg into its inclusive scan, chunk by chunk with a
//         sequential carry (num_tiles is small: n / FUSED_BS).
// The caller grid.sync()s between and after the passes and then combines
// tile-local values with d_agg[tile-1] element-wise.
// ---------------------------------------------------------------------------

template<typename T, typename S, typename F>
__device__ void
coopScanPass1(const S* __restrict__ d_in, T* d_out, T* d_agg, index_t n, F f)
{
    __shared__ volatile T sh[FUSED_BS];
    const index_t num_tiles = (n + (index_t)FUSED_BS - (index_t)1) / (index_t)FUSED_BS;
    for (index_t tile = (index_t)blockIdx.x; tile < num_tiles; tile += (index_t)gridDim.x) {
        index_t gid = tile * (index_t)FUSED_BS + (index_t)threadIdx.x;
        sh[threadIdx.x] = (gid < n) ? f(d_in[gid]) : (T)0;
        __syncthreads();
        scanBlock<T, uint32_t, Add<T>>(sh, Add<T>());
        if (gid < n) d_out[gid] = sh[threadIdx.x];
        if (threadIdx.x == FUSED_BS - 1) d_agg[tile] = sh[threadIdx.x];
        __syncthreads();
    }
}

// Pass-1 input functors (map source element → scanned index_t value).
struct BracketDelta {
    __device__ __forceinline__ index_t operator()(bracket_t b) const {
        return bracket_is_left(b) ? (index_t)1 : (index_t)-1;
    }
};
struct ArityMinusOne {
    __device__ __forceinline__ index_t operator()(production_t p) const {
        return (index_t)PRODUCTION_TO_ARITY[p] - (index_t)1;
    }
};
struct IsTermFlag {
    __device__ __forceinline__ index_t operator()(production_t p) const {
        return PRODUCTION_TO_TERMINAL_IS_VALID[p] ? (index_t)1 : (index_t)0;
    }
};

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

    int32_t* d_ss;                  // [m] packed (span_start << 4 | slen)
    int32_t* d_ps;                  // [m] packed (span_start << 4 | plen)
    index_t* d_agg_s;               // [ceil(m/FUSED_BS)] inter-tile bracket-length prefix sums
    index_t* d_agg_p;               // [ceil(m/FUSED_BS)] inter-tile production-length prefix sums
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
    unsigned* d_p3_next;

#ifdef HAS_LEXER
    // Combined mode (parse_int): parents + compact-tree assembly.
    // d_scan/d_agg_d/d_match are reused for the parents scan and PSE;
    // d_match holds the (old-numbering) parent vector after Phase H.
    bool           with_parents;    // runtime: skip G/H/I on token-only paths
    index_t        num_lexemes;
    production_t*  d_tree_prods;    // [max_pr] compacted tree production ids
    index_t*       d_tree_parents;  // [max_pr] compacted tree parent indices
    index_t*       d_token_parents; // [max_pr] per lexeme: parent tree index
#endif
};

// ---------------------------------------------------------------------------
// Fused Phase A+B+C in two grid-striding passes:
//
// Pass 1 (lookupScanPass1):
//   Each tile: hash-table lookup → lengths + span starts in shared/registers
//   → block scan of lengths in shared → write tile aggregate to d_agg_s/d_agg_p
//   and pack (ss, slen) / (ps, plen) into d_ss / d_ps (int32: span start in
//   upper 28 bits, length in lower 4 bits — safe since MAX_*_PER_POSITION ≤ 15).
//   No separate d_soffsets / d_poffsets arrays written.
//
// [grid.sync() + coopScanPass2: d_agg_s/d_agg_p become inter-tile prefix sums]
//
// Pass 2 (scatterPass2):
//   Re-scan lengths in shared (unpacked from d_ss/d_ps, no hash lookup),
//   compute exclusive intra-tile offset from shared scan, add L2-resident
//   inter-tile prefix, scatter directly to d_brackets / d_productions.
// ---------------------------------------------------------------------------

__device__ void
lookupScanPass1(const FusedBufs& b)
{
    __shared__ volatile index_t sh_s[FUSED_BS];
    __shared__ volatile index_t sh_p[FUSED_BS];
    const index_t m = b.m;
    const index_t num_tiles = (m + (index_t)FUSED_BS - (index_t)1) / (index_t)FUSED_BS;
    for (index_t tile = (index_t)blockIdx.x; tile < num_tiles; tile += (index_t)gridDim.x) {
        index_t gid = tile * (index_t)FUSED_BS + (index_t)threadIdx.x;
        index_t slen = (index_t)0;
        index_t plen = (index_t)0;
        if (gid < m) {
            int32_t ss, se, ps, pe;
            bool valid = lookupSpans(b.d_arr, m, gid, ss, se, ps, pe);
            if (!valid) atomicAnd(b.d_valid, 0);
            slen = valid ? (index_t)(se - ss) : (index_t)0;
            plen = valid ? (index_t)(pe - ps) : (index_t)0;
            // Pack span start (upper bits) + length (lower 4 bits) into one int32.
            b.d_ss[gid] = (ss << 4) | (int32_t)slen;
            b.d_ps[gid] = (ps << 4) | (int32_t)plen;
        }
        sh_s[threadIdx.x] = slen;
        sh_p[threadIdx.x] = plen;
        __syncthreads();
        scanBlock<index_t, uint32_t, Add<index_t>>(sh_s, Add<index_t>());
        scanBlock<index_t, uint32_t, Add<index_t>>(sh_p, Add<index_t>());
        if (threadIdx.x == FUSED_BS - 1) {
            b.d_agg_s[tile] = sh_s[threadIdx.x];
            b.d_agg_p[tile] = sh_p[threadIdx.x];
        }
        __syncthreads();
    }
}

__device__ void
scatterPass2(const FusedBufs& b)
{
    __shared__ volatile index_t sh_s[FUSED_BS];
    __shared__ volatile index_t sh_p[FUSED_BS];
    const index_t m = b.m;
    const index_t num_tiles = (m + (index_t)FUSED_BS - (index_t)1) / (index_t)FUSED_BS;
    for (index_t tile = (index_t)blockIdx.x; tile < num_tiles; tile += (index_t)gridDim.x) {
        index_t gid = tile * (index_t)FUSED_BS + (index_t)threadIdx.x;
        index_t slen = (index_t)0, plen = (index_t)0;
        int32_t ss = 0, ps = 0;
        if (gid < m) {
            int32_t packed_s = b.d_ss[gid];
            int32_t packed_p = b.d_ps[gid];
            slen = (index_t)(packed_s & 0xf);
            plen = (index_t)(packed_p & 0xf);
            ss   = packed_s >> 4;
            ps   = packed_p >> 4;
        }
        sh_s[threadIdx.x] = slen;
        sh_p[threadIdx.x] = plen;
        __syncthreads();
        scanBlock<index_t, uint32_t, Add<index_t>>(sh_s, Add<index_t>());
        scanBlock<index_t, uint32_t, Add<index_t>>(sh_p, Add<index_t>());
        index_t s_excl = sh_s[threadIdx.x] - slen;
        index_t p_excl = sh_p[threadIdx.x] - plen;
        __syncthreads();
        index_t s_tile_pfx = (tile > 0) ? b.d_agg_s[tile - 1] : (index_t)0;
        index_t p_tile_pfx = (tile > 0) ? b.d_agg_p[tile - 1] : (index_t)0;
        if (gid < m) {
            index_t soff = s_tile_pfx + s_excl;
            index_t poff = p_tile_pfx + p_excl;
            for (index_t j = 0; j < slen; j++)
                b.d_brackets[soff + j] = STACKS[ss + (int32_t)j];
            for (index_t j = 0; j < plen; j++)
                b.d_productions[poff + j] = PRODUCTIONS[ps + (int32_t)j];
        }
        __syncthreads();
    }
}

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

    // ---- Phase A+B: per-tile lookup + shared-memory scan → tile aggregates ----
    const index_t num_tiles_m = (m + (index_t)FUSED_BS - (index_t)1) / (index_t)FUSED_BS;
    lookupScanPass1(b);
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

    // ---- Phase C: scatter using packed d_ss/d_ps, offsets via shared scan ----
    // No d_soffsets/d_poffsets: lengths unpacked from d_ss/d_ps into shared,
    // scanned there, combined with L2-resident tile prefix to get global offset.
    scatterPass2(b);
    grid.sync();

    if (num_brackets != (index_t)0) {   // grid-uniform
        // ---- Phase D: inclusive delta scan → depths + validity ----
        // Deltas (±1) are computed on the fly from d_brackets.
        const index_t num_tiles_b = (num_brackets + (index_t)FUSED_BS - (index_t)1) / (index_t)FUSED_BS;
        coopScanPass1(b.d_brackets, b.d_scan, b.d_agg_d, num_brackets, BracketDelta());
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
                b.d_tree, b.d_prefix_min, b.d_p3_next);
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
    // (arity - 1) is computed on the fly from d_productions.
    const index_t num_tiles_p = (num_prods + (index_t)FUSED_BS - (index_t)1) / (index_t)FUSED_BS;
    coopScanPass1(b.d_productions, b.d_scan, b.d_agg_d, num_prods, ArityMinusOne());
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
            b.d_tree, b.d_prefix_min, b.d_p3_next);
    }
    grid.sync();

    // ---- Phase I: parents fixup + compact-tree assembly ----
    // Terminal-production slots are dropped from the tree; indices are
    // remapped by subtracting the inclusive is-term scan (t_incl).  Parents
    // always point at production nodes, so their remap is the same
    // subtraction.  Terminal slot k (in order) belongs to lexeme
    // t_incl[i] - 1 and contributes token_parents[lexeme] instead.
    // The parents fixup and the flag scan are independent: no grid.sync
    // between them.
    for (index_t i = grank; i < num_prods; i += gsz) {
        if (i == (index_t)0 || b.d_match[i] < (index_t)0) b.d_match[i] = (index_t)0;
    }
    coopScanPass1(b.d_productions, b.d_scan, b.d_agg_d, num_prods, IsTermFlag());
    grid.sync();
    coopScanPass2<index_t>(b.d_agg_d, num_tiles_p);
    grid.sync();
    // safe_zip: #terminal productions must equal #lexemes
    if (grank == 0 && b.d_agg_d[num_tiles_p - 1] != b.num_lexemes)
        atomicAnd(b.d_valid, 0);
    for (index_t i = grank; i < num_prods; i += gsz) {
        production_t prod = b.d_productions[i];
        index_t tile   = i / (index_t)FUSED_BS;
        index_t t_incl = b.d_scan[i] + ((tile > 0) ? b.d_agg_d[tile - 1] : (index_t)0);
        index_t p_old  = b.d_match[i];
        index_t p_tile = p_old / (index_t)FUSED_BS;
        index_t p_new  = p_old - (b.d_scan[p_old]
                                  + ((p_tile > 0) ? b.d_agg_d[p_tile - 1] : (index_t)0));
        if (PRODUCTION_TO_TERMINAL_IS_VALID[prod]) {
            index_t lex = t_incl - (index_t)1;
            if (lex < b.num_lexemes) b.d_token_parents[lex] = p_new;
            else                     atomicAnd(b.d_valid, 0);
        } else {
            index_t j = i - t_incl;
            b.d_tree_prods[j]   = prod;
            b.d_tree_parents[j] = p_new;
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
    gpuAssert(cudaMalloc(&b.d_ss,         (size_t)max_m * sizeof(int32_t)));
    gpuAssert(cudaMalloc(&b.d_ps,         (size_t)max_m * sizeof(int32_t)));
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
    gpuAssert(cudaMalloc(&b.d_p3_next,    sizeof(unsigned)));
    b.d_arr = p.d_arr;

#ifdef HAS_LEXER
    gpuAssert(cudaMalloc(&b.d_tree_prods,    (size_t)p.max_pr * sizeof(production_t)));
    gpuAssert(cudaMalloc(&b.d_tree_parents,  (size_t)p.max_pr * sizeof(index_t)));
    gpuAssert(cudaMalloc(&b.d_token_parents, (size_t)p.max_pr * sizeof(index_t)));
    b.with_parents = false;
    b.num_lexemes  = 0;
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
    cudaFree(b.d_ss);        cudaFree(b.d_ps);
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
    cudaFree(b.d_p3_next);
#ifdef HAS_LEXER
    cudaFree(b.d_tree_prods);
    cudaFree(b.d_tree_parents);
    cudaFree(b.d_token_parents);
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
                           std::vector<production_t>& out_prods)
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
        out_prods.resize((size_t)num_prods);
        gpuAssert(cudaMemcpy(out_prods.data(), p.bufs.d_productions,
                             (size_t)num_prods * sizeof(production_t), cudaMemcpyDeviceToHost));
    }
    return true;
}

// ---------------------------------------------------------------------------
// One-shot pipeline used by the CLI
// ---------------------------------------------------------------------------

bool runParserPipeline(const terminal_t* h_tokens, uint64_t n,
                       std::vector<production_t>& out_prods)
{
    if (n > (uint64_t)(std::numeric_limits<index_t>::max() - 2)) return false;
    index_t m = (index_t)(n + 2);

    // PSE indices (and SPT sizes) are ints: reject inputs whose bracket
    // capacity bound cannot be guaranteed to fit.
    if ((uint64_t)m * (uint64_t)MAX_BRACKETS_PER_POSITION > (uint64_t)INT_MAX)
        return false;

    std::vector<terminal_t> h_arr((size_t)m);
    h_arr[0] = START_TERMINAL;
    for (index_t i = 0; i < (index_t)n; i++) h_arr[(size_t)(i + 1)] = h_tokens[(size_t)i];
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

struct BothTree {
    std::vector<production_t> prods;         // [num_tree_nodes] production ids
    std::vector<index_t>      parents;       // [num_tree_nodes] parent tree indices
    std::vector<index_t>      token_parents; // [num_lexemes] parent tree index per token
};

static bool runBothFused(ParserFused& p,
                         const terminal_t* d_tokens, index_t n,
                         BothTree& out)
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

    launchParserFused(p, m);
    gpuAssert(cudaDeviceSynchronize());

    int h_valid = 0;
    gpuAssert(cudaMemcpy(&h_valid, p.bufs.d_valid, sizeof(int), cudaMemcpyDeviceToHost));
    if (!h_valid) return false;

    index_t totals[2];
    gpuAssert(cudaMemcpy(totals, p.bufs.d_totals, 2 * sizeof(index_t), cudaMemcpyDeviceToHost));
    // Validity guarantees #terminal slots == n, so the compacted tree has
    // np - n nodes.
    const index_t nt = totals[1] - n;

    out.prods.resize((size_t)nt);
    out.parents.resize((size_t)nt);
    out.token_parents.resize((size_t)n);
    if (nt > (index_t)0) {
        gpuAssert(cudaMemcpy(out.prods.data(), p.bufs.d_tree_prods,
                             (size_t)nt * sizeof(production_t), cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(out.parents.data(), p.bufs.d_tree_parents,
                             (size_t)nt * sizeof(index_t), cudaMemcpyDeviceToHost));
    }
    if (n > (index_t)0)
        gpuAssert(cudaMemcpy(out.token_parents.data(), p.bufs.d_token_parents,
                             (size_t)n * sizeof(index_t), cudaMemcpyDeviceToHost));
    return true;
}
#endif // HAS_LEXER
