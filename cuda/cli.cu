// cli.cu — generated program entry point.
//
// Appended last (after common.cu, scan.cu, [lexer.cu], pse.cu, [parser.cu]
// and the grammar constants).  Provides a unified CLI for all three modes:
//   Lex-only:    reads raw bytes from stdin/file, emits token spans
//   Parse-only:  reads binary token-ID frames, emits production IDs
//   Both:        --raw-input → raw bytes → lexer → parser → productions
//                default    → binary token-ID frames → parser → productions
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
// template<typename I> static bool runParserPipeline(const uint64_t*, uint64_t, std::vector<uint64_t>&);

static void run_one_parser_test(const uint64_t* tokens, uint64_t n,
                                FILE* out) {
    std::vector<uint64_t> prods;
    bool ok;
    if (n < (uint64_t)0x7FFFFFFF)
        ok = runParserPipeline<uint32_t>(tokens, n, prods);
    else
        ok = runParserPipeline<size_t>(tokens, n, prods);

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

    if (n >= (uint64_t)0x7FFFFFFF) {
        free(buf);
        fprintf(stderr, "error: --benchmark only supports inputs fitting uint32_t index\n");
        return 1;
    }
    uint32_t ni = (uint32_t)n;
    uint32_t m  = ni + 2;

    // Build host-side extended token array (terminal_t)
    std::vector<terminal_t> h_arr(m);
    h_arr[0] = START_TERMINAL;
    for (uint32_t i = 0; i < ni; i++) h_arr[i + 1] = (terminal_t)decode_be64(p + i * 8);
    h_arr[m - 1] = END_TERMINAL;
    free(buf);

    fprintf(stderr, "Benchmark: %u tokens, m=%u\n", ni, m);
    fprintf(stderr, "  MAX_BRACKETS_PER_POSITION=%zu  MAX_PRODS_PER_POSITION=%zu\n",
            (size_t)MAX_BRACKETS_PER_POSITION, (size_t)MAX_PRODS_PER_POSITION);

    if ((uint64_t)m * (uint64_t)MAX_BRACKETS_PER_POSITION > (uint64_t)INT_MAX) {
        fprintf(stderr, "error: bracket capacity bound exceeds INT_MAX\n");
        return 1;
    }

    // Pre-allocate GPU buffers and upload the input once
    ParserFused<uint32_t> pre = allocParserFused<uint32_t>(m);
    gpuAssert(cudaMemcpy(pre.d_arr, h_arr.data(),
                         (size_t)m * sizeof(terminal_t), cudaMemcpyHostToDevice));
    fprintf(stderr, "  Cooperative grid: %u blocks x %u threads\n",
            pre.P, (uint32_t)FUSED_BS);

    // Warmup (full runs, checks the parse succeeds)
    fprintf(stderr, "  Warmup (%u runs)...\n", warmup_runs);
    std::vector<uint64_t> dummy_prods;
    for (uint32_t i = 0; i < warmup_runs; i++) {
        bool ok = runParserFused<uint32_t>(pre, h_arr.data(), m, dummy_prods);
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
        launchParserFused<uint32_t>(pre, m);
        gpuAssert(cudaEventRecord(ev1));
        gpuAssert(cudaEventSynchronize(ev1));
        gpuAssert(cudaEventElapsedTime(&times_ms[i], ev0, ev1));
    }

    gpuAssert(cudaEventDestroy(ev0));
    gpuAssert(cudaEventDestroy(ev1));
    freeParserFused<uint32_t>(pre);

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
    fprintf(stderr, " Tokens: %u   m: %u   Warmup: %u   Runs: %u\n",
            ni, m, warmup_runs, n_runs);
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
    // ---- Both mode ----
    if (a.raw_input) {
        // Full pipeline: raw bytes -> lexer -> tokens -> parser -> productions
        // Not yet implemented; placeholder.
        fprintf(stderr, "error: --raw-input (full pipeline) not yet implemented\n");
        return 1;
    } else {
        // Binary token-ID protocol (same as parse-only)
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
    }
#else
    fprintf(stderr, "error: no lexer or parser compiled in\n");
    return 1;
#endif
}
