// cli.cu — generated program entry point.
//
// Appended last (after common.cu, scan.cu, [lexer.cu], pse.cu, [parser.cu]
// and the grammar constants).  Provides a unified CLI for all three modes:
//   Lex-only:    framed test batches → token/span records
//   Parse-only:  framed token-ID batches → production IDs
//   Both:        framed raw-byte batches → lexer → fused parser with
//                parents/parse_int phases → CST nodes
//
// Wire format (native little-endian batch protocol, see
// docs/wire-protocols.md):
//   batch input : u64 num_tests, then per test a frame:
//                 u64 frame_len, u64 n, payload
//                 (payload: n raw bytes for lexer/both;
//                  n × sizeof(terminal_t) token ids for parser)
//   batch output: u64 num_tests, then per test a response record:
//                 u8 valid; if valid: u64 count + native-width fields
//                 using terminal_t / production_t / index_t as generated
//                 for the grammar; `--layout` prints their sizes.
//   --server    : a loop of such counted batches until EOF, flushed
//                 after each response record.
//
// Flags (all optional):
//   -i FILE              input file (default: stdin)
//   -o FILE              output file (default: stdout)
//   --block-size  N      128 or 256 (default: 256)
//   --items-per-thread N 2, 4, or 8 (default: auto from shared memory)
//   --shared-memory N    shared memory budget in bytes (default: device query)
//   --timeit             print kernel elapsed time to stderr
//   --server             counted-batch loop until EOF (flush after each
//                        response)
//   --layout             print native type sizes (key=value lines) and exit
//   --benchmark N        time N runs (GPU-only, pre-alloc, no I/O in loop)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <vector>

// Hosts are little-endian; decode by memcpy.
static inline uint64_t decode_le64(const uint8_t* p) {
    uint64_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------

// Native type sizes for the server protocol; consumed by test harnesses
// to build/parse native frames.
static void print_layout(void) {
    printf("terminal_t=%zu\n", sizeof(terminal_t));
#ifdef HAS_PARSER
    printf("production_t=%zu\n", sizeof(production_t));
#endif
    printf("index_t=%zu\n", sizeof(index_t));
}

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  -i FILE              input file (default: stdin)\n"
        "  -o FILE              output file (default: stdout)\n"
        "  --block-size N       128 or 256 (default: 256)\n"
        "  --items-per-thread N 2, 4, or 8 (default: auto)\n"
        "  --shared-memory N    shared memory budget bytes (default: device)\n"
        "  --timeit             print kernel time to stderr\n"
        "  --server             counted-batch loop until EOF (flush per response)\n"
        "  --layout             print native type sizes and exit\n"
        "  --benchmark N        time N runs (GPU-only, pre-alloc, no I/O in loop)\n"
        "  --warmup N           warmup runs before timing (default: 3)\n"
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
        } else if (strcmp(argv[i], "--layout") == 0) {
            print_layout(); exit(0);
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

// Host-byte-order (little-endian) helpers for the native protocol.
static __attribute__((unused)) uint64_t read_u64_le(FILE* f) {
    uint64_t v;
    if (fread(&v, sizeof v, 1, f) != 1) return (uint64_t)-1;
    return v;
}

static __attribute__((unused)) void write_u64_le(FILE* f, uint64_t v) {
    fwrite(&v, sizeof v, 1, f);
}

template <typename T>
static inline void write_val(FILE* f, T v) { fwrite(&v, sizeof(T), 1, f); }

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
// Lexer-only mode (framed test batches)
// ---------------------------------------------------------------------------

#ifdef HAS_LEXER

// Run the lexer on one framed test; on success fills toks/starts/ends and
// returns true, otherwise returns false (rejected input).
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
static bool lex_one(const uint8_t* bytes, uint64_t n,
                    std::vector<terminal_t>& toks,
                    std::vector<index_t>& starts,
                    std::vector<index_t>& ends) {
    toks.clear(); starts.clear(); ends.clear();
    if (n == 0) return true;

    uint8_t*    d_str = nullptr;
    terminal_t* d_tok = nullptr;
    index_t*    d_s   = nullptr;
    index_t*    d_e   = nullptr;
    gpuAssert(cudaMalloc(&d_str, n));
    gpuAssert(cudaMalloc(&d_tok, n * sizeof(terminal_t)));
    gpuAssert(cudaMalloc(&d_s,   n * sizeof(index_t)));
    gpuAssert(cudaMalloc(&d_e,   n * sizeof(index_t)));
    gpuAssert(cudaMemcpy(d_str, bytes, n, cudaMemcpyHostToDevice));

    LexerCtx<uint32_t, index_t> ctx((uint32_t)n, BLOCK_SIZE, ITEMS_PER_THREAD);
    const uint32_t nblocks = numBlocks((uint32_t)n, BLOCK_SIZE, ITEMS_PER_THREAD);
    lexer<uint32_t, index_t, BLOCK_SIZE, ITEMS_PER_THREAD>
        <<<nblocks, BLOCK_SIZE>>>(ctx, d_str, d_tok, d_s, d_e, (uint32_t)n, true);
    gpuAssert(cudaDeviceSynchronize());
    gpuAssert(cudaPeekAtLastError());
    const bool     valid   = ctx.isAccept();
    const uint32_t num_lex = ctx.terminalsSize();
    ctx.cleanUp();

    if (valid && num_lex > 0) {
        toks.resize(num_lex);
        starts.resize(num_lex);
        ends.resize(num_lex);
        gpuAssert(cudaMemcpy(toks.data(),   d_tok, num_lex * sizeof(terminal_t), cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(starts.data(), d_s,   num_lex * sizeof(index_t),    cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(ends.data(),   d_e,   num_lex * sizeof(index_t),    cudaMemcpyDeviceToHost));
    }
    cudaFree(d_str); cudaFree(d_tok); cudaFree(d_s); cudaFree(d_e);
    return valid;
}

// Response record (host byte order):
//   u8 valid; if valid: u64 num_lexemes; per lexeme:
//   terminal_t terminal, index_t span start, index_t span end
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
static void run_one_lexer_test(const uint8_t* bytes, uint64_t n, FILE* out) {
    std::vector<terminal_t> toks;
    std::vector<index_t>    starts, ends;
    if (!lex_one<BLOCK_SIZE, ITEMS_PER_THREAD>(bytes, n, toks, starts, ends)) {
        fputc(0, out);
        return;
    }
    fputc(1, out);
    write_u64_le(out, (uint64_t)toks.size());
    for (size_t i = 0; i < toks.size(); i++) {
        write_val(out, toks[i]);
        write_val(out, starts[i]);
        write_val(out, ends[i]);
    }
}

// Batch mode: u64 num_tests, then per test a frame (u64 frame_len,
// u64 n, n raw bytes).  Writes u64 num_tests followed by per-test
// response records (see run_one_lexer_test).
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
static int run_lexer_batch_impl(FILE* in, FILE* out, bool timeit) {
    size_t buf_len = 0;
    uint8_t* buf = slurp(in, &buf_len);
    if (!buf || buf_len < 8) { free(buf); fprintf(stderr, "error: truncated input\n"); return 1; }

    cudaEvent_t t0, t1;
    if (timeit) {
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
    }

    const uint8_t* p   = buf;
    const uint8_t* end = buf + buf_len;
    uint64_t num_tests = decode_le64(p); p += 8;
    write_u64_le(out, num_tests);
    int ret = 0;

    for (uint64_t t = 0; t < num_tests; t++) {
        if (p + 16 > end) { fprintf(stderr, "error: truncated input\n"); ret = 1; break; }
        uint64_t frame_len = decode_le64(p); p += 8;
        uint64_t n = decode_le64(p); p += 8;
        if (frame_len != 8 + n) { fprintf(stderr, "error: frame length mismatch\n"); ret = 1; break; }
        if (n > (uint64_t)(end - p)) { fprintf(stderr, "error: truncated input\n"); ret = 1; break; }
        if (timeit) cudaEventRecord(t0);
        run_one_lexer_test<BLOCK_SIZE, ITEMS_PER_THREAD>(p, n, out);
        if (timeit) {
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms = 0;
            cudaEventElapsedTime(&ms, t0, t1);
            fprintf(stderr, "input %zu (%zu bytes): %.2fms\n", (size_t)t, (size_t)n, ms);
        }
        p += n;
    }
    free(buf);
    fflush(out);

    if (timeit) {
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
    return ret;
}

// Server mode: counted batches (u64 num_tests, then that many frames) in
// a loop until EOF, flush after each response.
// Frame: u64 frame_len; content: u64 n + n raw bytes.
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
static int lexer_server_impl(FILE* in, FILE* out) {
    std::vector<uint8_t> frame;
    for (;;) {
        uint64_t num_tests = read_u64_le(in);
        if (feof(in)) break;
        if (num_tests == (uint64_t)-1) {
            fprintf(stderr, "error: truncated input\n"); return 1;
        }
        write_u64_le(out, num_tests);
        fflush(out);
        for (uint64_t t = 0; t < num_tests; t++) {
            uint64_t frame_len = read_u64_le(in);
            if (feof(in) || frame_len == (uint64_t)-1 || frame_len < 8) {
                fprintf(stderr, "error: bad frame length\n"); return 1;
            }
            frame.resize(frame_len);
            if (fread(frame.data(), 1, frame_len, in) != frame_len) {
                fprintf(stderr, "error: truncated frame\n"); return 1;
            }
            uint64_t n;
            memcpy(&n, frame.data(), sizeof n);
            if (frame_len != 8 + n) {
                fprintf(stderr, "error: frame length mismatch\n"); return 1;
            }
            run_one_lexer_test<BLOCK_SIZE, ITEMS_PER_THREAD>(frame.data() + 8, n, out);
            fflush(out);
        }
    }
    return 0;
}

template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
static int lexer_benchmark(FILE* in, uint32_t warmup_runs, uint32_t n_runs) {
    size_t buf_len = 0;
    uint8_t* buf = slurp(in, &buf_len);
    if (!buf || buf_len < 16) {
        free(buf); fprintf(stderr, "error: benchmark input too short\n"); return 1;
    }
    const uint8_t* p   = buf;
    const uint8_t* end = buf + buf_len;
    uint64_t num_tests = decode_le64(p); p += 8;

    cudaEvent_t ev0, ev1;
    gpuAssert(cudaEventCreate(&ev0));
    gpuAssert(cudaEventCreate(&ev1));

    for (uint64_t t = 0; t < num_tests; t++) {
        if (p + 16 > end) { free(buf); fprintf(stderr, "error: truncated input\n"); return 1; }
        uint64_t frame_len = decode_le64(p); p += 8;
        uint64_t n         = decode_le64(p); p += 8;
        if (frame_len != 8 + n || (uint64_t)(end - p) < n) {
            free(buf); fprintf(stderr, "error: input truncated\n"); return 1;
        }
        const uint8_t* data = p;
        p += n;

        uint8_t*    d_str = nullptr;
        terminal_t* d_tok = nullptr;
        index_t*    d_s   = nullptr;
        index_t*    d_e   = nullptr;
        gpuAssert(cudaMalloc(&d_str, n ? n : 1));
        gpuAssert(cudaMalloc(&d_tok, n * sizeof(terminal_t)));
        gpuAssert(cudaMalloc(&d_s,   n * sizeof(index_t)));
        gpuAssert(cudaMalloc(&d_e,   n * sizeof(index_t)));
        gpuAssert(cudaMemcpy(d_str, data, n, cudaMemcpyHostToDevice));

        std::vector<terminal_t> h_tok(n);

        const uint32_t nblocks = numBlocks((uint32_t)n, BLOCK_SIZE, ITEMS_PER_THREAD);
        LexerCtx<uint32_t, index_t> ctx((uint32_t)n, BLOCK_SIZE, ITEMS_PER_THREAD);

        for (uint32_t i = 0; i < warmup_runs; i++) {
            ctx.reset();
            lexer<uint32_t, index_t, BLOCK_SIZE, ITEMS_PER_THREAD>
                <<<nblocks, BLOCK_SIZE>>>(ctx, d_str, d_tok, d_s, d_e, (uint32_t)n, false);
            gpuAssert(cudaDeviceSynchronize());
        }

        // --- kernel-only timing ---
        std::vector<float> times_ms(n_runs);
        for (uint32_t i = 0; i < n_runs; i++) {
            ctx.reset();
            gpuAssert(cudaEventRecord(ev0));
            lexer<uint32_t, index_t, BLOCK_SIZE, ITEMS_PER_THREAD>
                <<<nblocks, BLOCK_SIZE>>>(ctx, d_str, d_tok, d_s, d_e, (uint32_t)n, false);
            gpuAssert(cudaEventRecord(ev1));
            gpuAssert(cudaEventSynchronize(ev1));
            gpuAssert(cudaEventElapsedTime(&times_ms[i], ev0, ev1));
        }

        // --- IO-bound timing (H->D + kernel + D->H) ---
        std::vector<float> times_io_ms(n_runs);
        for (uint32_t i = 0; i < n_runs; i++) {
            ctx.reset();
            gpuAssert(cudaEventRecord(ev0));
            gpuAssert(cudaMemcpyAsync(d_str, data, n, cudaMemcpyHostToDevice));
            lexer<uint32_t, index_t, BLOCK_SIZE, ITEMS_PER_THREAD>
                <<<nblocks, BLOCK_SIZE>>>(ctx, d_str, d_tok, d_s, d_e, (uint32_t)n, false);
            gpuAssert(cudaMemcpyAsync(h_tok.data(), d_tok, n * sizeof(terminal_t), cudaMemcpyDeviceToHost));
            gpuAssert(cudaEventRecord(ev1));
            gpuAssert(cudaEventSynchronize(ev1));
            gpuAssert(cudaEventElapsedTime(&times_io_ms[i], ev0, ev1));
        }

        ctx.cleanUp();
        cudaFree(d_str); cudaFree(d_tok); cudaFree(d_s); cudaFree(d_e);

        auto print_stats = [&](const char* label, std::vector<float>& tms) {
            double mean = 0, variance = 0, gbps = 0;
            double factor = (double)n / (1000.0 * n_runs);
            for (uint32_t i = 0; i < n_runs; i++) {
                double diff = fmax(1e3 * (double)tms[i], 0.5);
                mean     += diff / n_runs;
                variance += (diff * diff) / n_runs;
                gbps     += factor / diff;
            }
            double bound = (0.95 * sqrt(variance)) / sqrt((double)n_runs);
            fprintf(stderr, "%s:\n", label);
            fprintf(stderr, "        %.0fμs (95%% CI: [%.1fμs, %.1fμs]); %.0fGB/s\n",
                    mean, mean - bound, mean + bound, gbps);
        };

        char label[64];
        snprintf(label, sizeof(label), "lex_int (cuda, %zu bytes)", (size_t)n);
        print_stats(label, times_ms);
        snprintf(label, sizeof(label), "lex_int (cuda+io, %zu bytes)", (size_t)n);
        print_stats(label, times_io_ms);
    }

    gpuAssert(cudaEventDestroy(ev0));
    gpuAssert(cudaEventDestroy(ev1));
    free(buf);
    return 0;
}

#endif // HAS_LEXER

// ---------------------------------------------------------------------------
// Parser-only mode (framed token-ID batches)
//
// Frame payload: n × terminal_t token ids (host byte order).
// Response record: u8 valid; if valid: u64 np + np×production_t prod ids.
// ---------------------------------------------------------------------------

#if defined(HAS_PARSER) && !defined(HAS_LEXER)

static void run_one_parser_test(const terminal_t* tokens, uint64_t n,
                                FILE* out) {
    std::vector<production_t> prods;
    bool ok = runParserPipeline(tokens, n, prods);
    if (ok) {
        fputc(1, out);
        write_u64_le(out, (uint64_t)prods.size());
        if (!prods.empty())
            fwrite(prods.data(), sizeof(production_t), prods.size(), out);
    } else {
        fputc(0, out);
    }
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

    uint64_t num_tests = decode_le64(p); p += 8;
    write_u64_le(out, num_tests);

    std::vector<terminal_t> tokens;
    for (uint64_t t = 0; t < num_tests; t++) {
        if (p + 16 > end) { free(buf); fprintf(stderr, "truncated\n"); return 1; }
        uint64_t frame_len = decode_le64(p); p += 8;
        uint64_t n = decode_le64(p); p += 8;
        if (frame_len != 8 + n * sizeof(terminal_t)) {
            free(buf); fprintf(stderr, "error: frame length mismatch\n"); return 1;
        }
        if (p + n * sizeof(terminal_t) > end) { free(buf); fprintf(stderr, "truncated\n"); return 1; }
        tokens.resize(n);
        if (n > 0)
            memcpy(tokens.data(), p, n * sizeof(terminal_t));
        p += n * sizeof(terminal_t);
        run_one_parser_test(tokens.data(), n, out);
    }
    free(buf);
    fflush(out);
    return 0;
}

// Server mode: counted batches (u64 num_tests, then that many frames) in
// a loop until EOF, flush after each response.  Each frame: u64
// frame_byte_length, then u64 n + n×terminal_t token ids.
static int parser_server(FILE* in, FILE* out) {
    std::vector<terminal_t> tokens;
    std::vector<uint8_t>    frame;
    for (;;) {
        uint64_t num_tests = read_u64_le(in);
        if (feof(in)) break;
        if (num_tests == (uint64_t)-1) {
            fprintf(stderr, "error: truncated input\n"); return 1;
        }
        write_u64_le(out, num_tests);
        fflush(out);
        for (uint64_t t = 0; t < num_tests; t++) {
            uint64_t frame_len = read_u64_le(in);
            if (feof(in) || frame_len == (uint64_t)-1 || frame_len < 8) {
                fprintf(stderr, "error: bad frame length\n"); return 1;
            }
            frame.resize(frame_len);
            if (fread(frame.data(), 1, frame_len, in) != frame_len) {
                fprintf(stderr, "error: truncated frame\n"); return 1;
            }
            uint64_t n;
            memcpy(&n, frame.data(), sizeof n);
            if (frame_len != 8 + n * sizeof(terminal_t)) {
                fprintf(stderr, "error: frame length mismatch\n"); return 1;
            }
            tokens.resize(n);
            if (n > 0)
                memcpy(tokens.data(), frame.data() + 8, n * sizeof(terminal_t));
            run_one_parser_test(tokens.data(), n, out);
            fflush(out);
        }
    }
    return 0;
}

#endif /* HAS_PARSER && !HAS_LEXER */

#ifdef HAS_PARSER
// Benchmark mode: read all tests from `in`, pre-allocate GPU buffers once,
// run warmup passes, then time `n_runs` passes with CUDA events.
static int parser_benchmark(FILE* in, uint32_t warmup_runs, uint32_t n_runs) {
    size_t buf_len = 0;
    uint8_t* buf = slurp(in, &buf_len);
    if (!buf || buf_len < 16) {
        free(buf);
        fprintf(stderr, "error: benchmark input too short\n");
        return 1;
    }
    const uint8_t* p   = buf;
    const uint8_t* end = buf + buf_len;
    uint64_t num_tests = decode_le64(p); p += 8;

    cudaEvent_t ev0, ev1;
    gpuAssert(cudaEventCreate(&ev0));
    gpuAssert(cudaEventCreate(&ev1));

    for (uint64_t t = 0; t < num_tests; t++) {
        if (p + 16 > end) { free(buf); fprintf(stderr, "error: truncated input\n"); return 1; }
        uint64_t frame_len = decode_le64(p); p += 8;
        uint64_t n         = decode_le64(p); p += 8;
        if (frame_len != 8 + n * sizeof(terminal_t) ||
            (uint64_t)(end - p) < n * sizeof(terminal_t)) {
            free(buf); fprintf(stderr, "error: input truncated\n"); return 1;
        }

        index_t ni = (index_t)n;
        index_t m  = ni + (index_t)2;

        std::vector<terminal_t> h_arr((size_t)m);
        h_arr[0] = START_TERMINAL;
        if (ni > 0)
            memcpy(h_arr.data() + 1, p, (size_t)ni * sizeof(terminal_t));
        h_arr[(size_t)m - 1] = END_TERMINAL;
        p += n * sizeof(terminal_t);

        if ((uint64_t)m * (uint64_t)MAX_BRACKETS_PER_POSITION > (uint64_t)INT_MAX) {
            fprintf(stderr, "error: bracket capacity bound exceeds INT_MAX\n");
            free(buf); return 1;
        }

        ParserFused pre = allocParserFused(m);
        gpuAssert(cudaMemcpy(pre.d_arr, h_arr.data(),
                             (size_t)m * sizeof(terminal_t), cudaMemcpyHostToDevice));

        std::vector<production_t> dummy_prods;
        for (uint32_t i = 0; i < warmup_runs; i++)
            runParserFused(pre, h_arr.data(), m, dummy_prods);

        // --- kernel-only timing ---
        std::vector<float> times_ms(n_runs);
        for (uint32_t i = 0; i < n_runs; i++) {
            int one = 1;
            gpuAssert(cudaMemcpy(pre.bufs.d_valid, &one, sizeof(int), cudaMemcpyHostToDevice));
            gpuAssert(cudaEventRecord(ev0));
            launchParserFused(pre, m);
            gpuAssert(cudaEventRecord(ev1));
            gpuAssert(cudaEventSynchronize(ev1));
            gpuAssert(cudaEventElapsedTime(&times_ms[i], ev0, ev1));
        }

        // --- IO-bound timing (H->D + kernel + D->H) ---
        index_t totals[2];
        std::vector<production_t> h_prods((size_t)pre.max_pr);
        std::vector<float> times_io_ms(n_runs);
        for (uint32_t i = 0; i < n_runs; i++) {
            int one = 1;
            gpuAssert(cudaMemcpy(pre.bufs.d_valid, &one, sizeof(int), cudaMemcpyHostToDevice));
            gpuAssert(cudaEventRecord(ev0));
            gpuAssert(cudaMemcpyAsync(pre.d_arr, h_arr.data(),
                                      (size_t)m * sizeof(terminal_t), cudaMemcpyHostToDevice));
            launchParserFused(pre, m);
            gpuAssert(cudaMemcpyAsync(totals, pre.bufs.d_totals,
                                      2 * sizeof(index_t), cudaMemcpyDeviceToHost));
            gpuAssert(cudaMemcpyAsync(h_prods.data(), pre.bufs.d_productions,
                                      (size_t)pre.max_pr * sizeof(production_t), cudaMemcpyDeviceToHost));
            gpuAssert(cudaEventRecord(ev1));
            gpuAssert(cudaEventSynchronize(ev1));
            gpuAssert(cudaEventElapsedTime(&times_io_ms[i], ev0, ev1));
        }

        freeParserFused(pre);

        size_t bytes = (size_t)ni * sizeof(terminal_t);
        auto print_stats = [&](const char* label, std::vector<float>& tms) {
            double mean = 0, variance = 0, gbps = 0;
            double factor = (double)bytes / (1000.0 * n_runs);
            for (uint32_t i = 0; i < n_runs; i++) {
                double diff = fmax(1e3 * (double)tms[i], 0.5);
                mean     += diff / n_runs;
                variance += (diff * diff) / n_runs;
                gbps     += factor / diff;
            }
            double bound = (0.95 * sqrt(variance)) / sqrt((double)n_runs);
            fprintf(stderr, "%s:\n", label);
            fprintf(stderr, "        %.0fμs (95%% CI: [%.1fμs, %.1fμs]); %.0fGB/s\n",
                    mean, mean - bound, mean + bound, gbps);
        };

        char label[64];
        snprintf(label, sizeof(label), "parse_int (cuda, %zu tokens)", (size_t)ni);
        print_stats(label, times_ms);
        snprintf(label, sizeof(label), "parse_int (cuda+io, %zu tokens)", (size_t)ni);
        print_stats(label, times_io_ms);
    }

    gpuAssert(cudaEventDestroy(ev0));
    gpuAssert(cudaEventDestroy(ev1));
    free(buf);
    return 0;
}

#endif // HAS_PARSER

// ---------------------------------------------------------------------------
// Both mode: full pipeline (raw bytes → lexer → fused parser with parents)
//
// Frame payload: n raw bytes.  Response record: u8 valid; if valid:
// u64 num_nodes; per node: u8 is_terminal, index_t parent,
// production_t id (terminal ids fit in production_t; see d_node_ids in
// parser.cu), index_t start, index_t end (0, 0 for nonterminal nodes).
// Same format as the generated C backend's combined mode.
// ---------------------------------------------------------------------------

#if defined(HAS_LEXER) && defined(HAS_PARSER)

// Run the full pipeline on one test; fills `nodes` and returns validity.
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
static bool both_one(const uint8_t* bytes, uint64_t n, BothNodes& nodes) {
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
    return valid;
}

template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
static void run_one_both_test(const uint8_t* bytes, uint64_t n, FILE* out) {
    BothNodes nodes;
    if (!both_one<BLOCK_SIZE, ITEMS_PER_THREAD>(bytes, n, nodes)) {
        fputc(0, out);
        return;
    }
    fputc(1, out);
    write_u64_le(out, (uint64_t)nodes.ids.size());
    for (size_t i = 0; i < nodes.ids.size(); i++) {
        fputc(nodes.is_term[i] ? 1 : 0, out);
        write_val(out, nodes.parents[i]);
        write_val(out, nodes.ids[i]);
        write_val(out, nodes.starts[i]);
        write_val(out, nodes.ends[i]);
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

    uint64_t num_tests = decode_le64(p); p += 8;
    write_u64_le(out, num_tests);

    for (uint64_t t = 0; t < num_tests; t++) {
        if (p + 16 > end) { free(buf); fprintf(stderr, "truncated\n"); return 1; }
        uint64_t frame_len = decode_le64(p); p += 8;
        uint64_t n = decode_le64(p); p += 8;
        if (frame_len != 8 + n) {
            free(buf); fprintf(stderr, "error: frame length mismatch\n"); return 1;
        }
        if (n > (uint64_t)(end - p)) { free(buf); fprintf(stderr, "truncated\n"); return 1; }
        run_one_both_test<BLOCK_SIZE, ITEMS_PER_THREAD>(p, n, out);
        p += n;
    }
    free(buf);
    fflush(out);
    return 0;
}

// Server mode for the full pipeline: counted batches (u64 num_tests, then
// that many raw-byte frames) in a loop until EOF, flush after each response.
// Frame: u64 frame_len; content: u64 n + n raw bytes.
template<uint32_t BLOCK_SIZE, uint32_t ITEMS_PER_THREAD>
static int both_server_impl(FILE* in, FILE* out) {
    std::vector<uint8_t> frame;
    for (;;) {
        uint64_t num_tests = read_u64_le(in);
        if (feof(in)) break;
        if (num_tests == (uint64_t)-1) {
            fprintf(stderr, "error: truncated input\n"); return 1;
        }
        write_u64_le(out, num_tests);
        fflush(out);
        for (uint64_t t = 0; t < num_tests; t++) {
            uint64_t frame_len = read_u64_le(in);
            if (feof(in) || frame_len == (uint64_t)-1 || frame_len < 8) {
                fprintf(stderr, "error: bad frame length\n"); return 1;
            }
            frame.resize(frame_len);
            if (fread(frame.data(), 1, frame_len, in) != frame_len) {
                fprintf(stderr, "error: truncated frame\n"); return 1;
            }
            uint64_t n;
            memcpy(&n, frame.data(), sizeof n);
            if (frame_len != 8 + n) {
                fprintf(stderr, "error: frame length mismatch\n"); return 1;
            }
            run_one_both_test<BLOCK_SIZE, ITEMS_PER_THREAD>(frame.data() + 8, n, out);
            fflush(out);
        }
    }
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
    int ret;
    if (a.benchmark > 0)
        ret = DISPATCH_BS_IPT(bs, ipt, lexer_benchmark, in, a.warmup, a.benchmark);
    else if (a.server)
        ret = DISPATCH_BS_IPT(bs, ipt, lexer_server_impl, in, out);
    else
        ret = DISPATCH_BS_IPT(bs, ipt, run_lexer_batch_impl, in, out, a.timeit);
    if (in  != stdin)  fclose(in);
    if (out != stdout) fclose(out);
    return ret;

#elif defined(HAS_PARSER) && !defined(HAS_LEXER)
    // ---- Parse-only mode ----
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
    // --server loops counted raw-byte batches; --benchmark uses the
    // token-ID protocol of parse-only mode.
    int ret;
    if (a.benchmark > 0)
        ret = parser_benchmark(in, a.warmup, a.benchmark);
    else if (a.server)
        ret = DISPATCH_BS_IPT(bs, ipt, both_server_impl, in, out);
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
