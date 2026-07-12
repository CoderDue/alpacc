// cli.c — generated program entry point (C backend).
//
// Appended last by the code generator, after the grammar constants,
// lexer.c (if HAS_LEXER), and parser.c (if HAS_PARSER).
//
// The generated code defines:
//   void run_test_case(uint64_t n, const uint8_t *in, FILE *out);
//     — one batch test (u64-BE test protocol), result to `out`
//   void run_server_case(uint64_t n, const uint8_t *in, FILE *out);
//     — one server frame (native types, host byte order), result to `out`
//   void print_layout(FILE *out);
//     — native type sizes as key=value lines (for --layout)
// plus the INPUT_BYTES(n) / SERVER_INPUT_BYTES(n) payload-size macros.
//
// Input format (batch mode, u64 BE regardless of native types):
//   u64 BE  num_tests
//   per test:
//     u64 BE  n          (bytes for lexer/both; token count for parser)
//     n bytes / n*8 bytes payload
//
// Server mode uses native types in host byte order (alpacc targets
// little-endian hosts): u64 frame_len; content: u64 n + payload
// (n raw bytes for lexer/both; n × sizeof(terminal_t) token ids for parser).
//
// Flags:
//   -i FILE    input file  (default: stdin)
//   -o FILE    output file (default: stdout)
//   --timeit   print wall-clock time to stderr
//   --server   length-prefixed frame loop (one test per frame, no num_tests header)
//   --layout   print native type sizes and exit
//   -h/--help  show usage

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

static uint64_t read_u64be(FILE *f) {
    uint8_t p[8];
    if (fread(p, 1, 8, f) != 8) return (uint64_t)-1;
    return ((uint64_t)p[0]<<56)|((uint64_t)p[1]<<48)|((uint64_t)p[2]<<40)|
           ((uint64_t)p[3]<<32)|((uint64_t)p[4]<<24)|((uint64_t)p[5]<<16)|
           ((uint64_t)p[6]<< 8)| (uint64_t)p[7];
}

static void write_u64be(FILE *f, uint64_t v) {
    uint8_t p[8];
    p[0]=(uint8_t)(v>>56); p[1]=(uint8_t)(v>>48);
    p[2]=(uint8_t)(v>>40); p[3]=(uint8_t)(v>>32);
    p[4]=(uint8_t)(v>>24); p[5]=(uint8_t)(v>>16);
    p[6]=(uint8_t)(v>> 8); p[7]=(uint8_t)(v);
    fwrite(p, 1, 8, f);
}

// decode_u64 / write_u64: aliases for use by the generated run_test_case().
static uint64_t decode_u64(const uint8_t *p) {
    return ((uint64_t)p[0]<<56)|((uint64_t)p[1]<<48)|((uint64_t)p[2]<<40)|
           ((uint64_t)p[3]<<32)|((uint64_t)p[4]<<24)|((uint64_t)p[5]<<16)|
           ((uint64_t)p[6]<< 8)| (uint64_t)p[7];
}

static void write_u64(FILE *f, uint64_t v) { write_u64be(f, v); }

// Host-byte-order (little-endian) helpers for the native server protocol.
static uint64_t read_u64le(FILE *f) {
    uint64_t v;
    if (fread(&v, sizeof v, 1, f) != 1) return (uint64_t)-1;
    return v;
}

static void write_u64le(FILE *f, uint64_t v) { fwrite(&v, sizeof v, 1, f); }

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  -i FILE    input file  (default: stdin)\n"
        "  -o FILE    output file (default: stdout)\n"
        "  --timeit   print wall-clock elapsed time to stderr\n"
        "  --server   length-prefixed frame mode (one test per frame, native types)\n"
        "  --layout   print native type sizes and exit\n"
        "  -h/--help  show this message\n",
        prog);
}

typedef struct {
    const char *input_file;
    const char *output_file;
    bool timeit;
    bool server;
} CliArgs;

static CliArgs parse_args(int argc, char *argv[]) {
    CliArgs a = { NULL, NULL, false, false };
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); exit(0);
        } else if (strcmp(argv[i], "--timeit") == 0) {
            a.timeit = true;
        } else if (strcmp(argv[i], "--server") == 0) {
            a.server = true;
        } else if (strcmp(argv[i], "--layout") == 0) {
            print_layout(stdout); exit(0);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            a.input_file = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            a.output_file = argv[++i];
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
            usage(argv[0]); exit(1);
        }
    }
    return a;
}

// ---------------------------------------------------------------------------
// Batch mode
//
// Reads num_tests-prefixed stream, calls run_test_case() for each test.
// In parser mode the payload per test is n*8 bytes (token IDs);
// in lexer/both mode it is n bytes (raw input).
// run_test_case() handles the payload format internally.
// ---------------------------------------------------------------------------

static int batch_mode(FILE *in, FILE *out) {
    uint64_t num_tests = read_u64be(in);
    if (num_tests == (uint64_t)-1) {
        fprintf(stderr, "error: truncated input\n"); return 1;
    }
    write_u64be(out, num_tests);

    for (uint64_t t = 0; t < num_tests; t++) {
        uint64_t n = read_u64be(in);
        if (n == (uint64_t)-1) {
            fprintf(stderr, "error: truncated input\n"); return 1;
        }
        uint64_t byte_n = INPUT_BYTES(n);
        uint8_t *buf = (uint8_t *) malloc(byte_n);
        if (fread(buf, 1, byte_n, in) != byte_n) {
            fprintf(stderr, "error: truncated input\n"); free(buf); return 1;
        }
        run_test_case(n, buf, out);
        free(buf);
    }
    fflush(out);
    return 0;
}

// ---------------------------------------------------------------------------
// Server mode (native protocol, host byte order)
//
// Loop: read u64 frame_length, read that many bytes as the frame,
// pass to run_server_case(), flush, repeat until EOF.
// Frame format: u64 n + native payload (raw bytes for lexer/both;
// n × sizeof(terminal_t) token ids for parser).
// ---------------------------------------------------------------------------

static int server_mode(FILE *in, FILE *out) {
    for (;;) {
        uint64_t frame_len = read_u64le(in);
        if (feof(in)) break;
        if (frame_len == (uint64_t)-1 || frame_len < 8) {
            fprintf(stderr, "error: bad frame length\n"); return 1;
        }
        uint8_t *frame = (uint8_t *) malloc(frame_len);
        if (fread(frame, 1, frame_len, in) != frame_len) {
            fprintf(stderr, "error: truncated frame\n"); free(frame); return 1;
        }
        // First 8 bytes of frame are n (host byte order).
        uint64_t n;
        memcpy(&n, frame, sizeof n);
        uint64_t byte_n = SERVER_INPUT_BYTES(n);
        if (frame_len != 8 + byte_n) {
            fprintf(stderr, "error: frame length mismatch\n"); free(frame); return 1;
        }
        run_server_case(n, frame + 8, out);
        fflush(out);
        free(frame);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    CliArgs a = parse_args(argc, argv);

    FILE *in  = stdin;
    FILE *out = stdout;
    if (a.input_file)  { in  = fopen(a.input_file,  "rb"); if (!in)  { perror(a.input_file);  return 1; } }
    if (a.output_file) { out = fopen(a.output_file, "wb"); if (!out) { perror(a.output_file); return 1; } }

    struct timespec t0, t1;
    if (a.timeit) clock_gettime(CLOCK_MONOTONIC, &t0);

    int ret = a.server ? server_mode(in, out) : batch_mode(in, out);

    if (a.timeit) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;
        fprintf(stderr, "Time: %.2fms\n", ms);
    }

    if (in  != stdin)  fclose(in);
    if (out != stdout) fclose(out);
    return ret;
}
