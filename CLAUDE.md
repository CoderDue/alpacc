# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Alpacc (Array Language Parallelism-Accelerated Compiler Compiler) is a parallel parser generator written in Haskell. It takes a grammar specification (`.alp` file) and generates parallel lexers and parsers targeting Futhark (for GPU/CPU parallel execution) and CUDA (for NVIDIA GPUs). The core parsing algorithm is LLP (LL-parallel), which is a subclass of LL grammars amenable to data-parallel execution.

## Build Commands

```bash
cabal build all              # Build library and executable
cabal build alpacc           # Build just the executable
cabal build lib:alpacc       # Build just the library
cabal install --installdir=$HOME/.local/bin --overwrite-policy=always exe:alpacc  # Install binary
```

**Never install to `~/bin`** — always use `~/.local/bin`. A stale `~/bin/alpacc` shadows the correct copy on PATH and causes benchmarks/tests to silently run against outdated codegen.

Run the tool directly during development:
```bash
cabal run alpacc -- <args>
```

## Testing

```bash
cabal test all               # Run all Haskell unit/property tests

# Integration tests — random grammars, all modes
bash tests/testcuda.sh                        # 10 random grammars, CUDA backend
bash tests/testfuthark.sh [q] [k] [target] [parallel_jobs] [--lexer|--parser]
bash tests/testc.sh                           # C backend

# Single-grammar long-input test (lexer + parser + combined)
bash tests/test-long-input.sh <grammar.alp> [backend] [length]
# backend: c | cuda | futhark-<target>  (default: c)
# Example:
bash tests/test-long-input.sh grammars/json.alp cuda
bash tests/test-long-input.sh grammars/numfmt.alp cuda

# CUDA PSE primitive test (requires nvcc and an NVIDIA GPU; local only)
bash tests/testpse.sh [arch]       # arch: nvcc -arch value, default native
```

The Haskell test suite uses Tasty with QuickCheck properties and HUnit assertions. Test modules live in `haskell-tests/`.

Note: `cabal test all` only exercises the Haskell code (grammar analysis, CFG parsing, LLP tables, etc.). It does **not** exercise the embedded backend templates (`backends/cuda/*.cu`, `backends/futhark/*.fut`, `backends/c/*.c`), which are only compiled into a runnable binary by the integration tests above. If a change touches only an embedded backend template, running `cabal test all` is wasted time — go straight to `test-long-input.sh` / `testcuda.sh` / `testfuthark.sh` / `testc.sh` for the affected backend.

## Tool Usage

Always run `alpacc` via `cabal exec` during development — the installed binary in `~/bin` may be stale:

```bash
cabal exec -- alpacc <args>
```

```bash
cabal exec -- alpacc futhark grammar.alp        # Generate Futhark lexer+parser
cabal exec -- alpacc cuda grammar.alp           # Generate CUDA lexer+parser
cabal exec -- alpacc random                     # Generate a random grammar
cabal exec -- alpacc test generate grammar.alp --single-long --length 1000000 --no-outputs -o data
cabal exec -- alpacc test compare grammar.alp inputs.bin outputs.bin results.bin
```

Key flags:
- `-q <INT>` / `--lookback <INT>` — lookback depth (default 0)
- `-k <INT>` / `--lookahead <INT>` — lookahead depth (default 1)
- `-l` / `--lexer` — generate lexer only
- `-p` / `--parser` — generate parser only
- `-o <FILE>` — output file
- `--index32` — use 32-bit indices (smaller, faster for inputs < 2 GB)
- `--sm-arch <INT>` — target SM architecture (e.g. 75 for Turing); auto-probed by default

Lookahead/lookback can also be set in the grammar file:
```
params {
  lookback = 1.
  lookahead = 1.
}
```

### Grammar file syntax notes

- Terminals are lowercase (regex patterns), nonterminals are uppercase.
- No comment syntax — comments are not supported in `.alp` files.
- Inside character classes `[...]`, use `\xHH` hex escapes (e.g. `\x20` for space, `\x09` for tab). `\t`, `\n` etc. are not valid there.
- A bare `.` in a regex is a literal dot (not wildcard). There is no `\.` escape.
- Terminal names must not collide with C++ keywords (`float`, `int`, etc.) — codegen will fail at nvcc time.
- LL(1) conflicts will cause a codegen error; restructure the grammar with an explicit tail nonterminal (e.g. `T -> . T -> sep N T.`) to avoid ambiguity.

## Benchmarks

Benchmark infrastructure lives in `benchmarks/`. Each grammar has its own subdirectory (e.g. `benchmarks/json/`, `benchmarks/numfmt/`) with a `Makefile`.

### Per-grammar Makefile targets

```bash
# From the repo root:
make -C benchmarks/json build-cuda        # Generate + compile CUDA lexer and parser
make -C benchmarks/json bench-lexer-cuda  # Run CUDA lexer benchmark (10 reps, 3 warmup)
make -C benchmarks/json bench-parser-cuda # Run CUDA parser benchmark
make -C benchmarks/json bench             # All benchmarks (CUDA + Futhark)
make -C benchmarks/json reset             # Delete compiled binaries and generated .cu/.fut
make -C benchmarks/json clean             # Also delete datasets

# Override dataset size (default 52428800 tokens):
make -C benchmarks/json bench-lexer-cuda INPUT_SIZE=10485760
```

The Makefile uses `alpacc` from PATH. Run `make` inside `nix-shell` or ensure the installed binary is current. The dataset is auto-generated on first build if missing.

### CUDA sweep scripts

Find the optimal `BLOCK_SIZE` × `ITEMS_PER_THREAD` combination for a grammar:

```bash
# Lexer sweep — sweep BS and IPT, report fastest config
bash benchmarks/sweep-cuda-lexer.sh <grammar.alp> [dataset.inputs] \
    [BS_LIST] [IPT_LIST] [STATE_LIST] [INDEX_LIST]

# Parser sweep — same interface but targets the fused parser kernel
bash benchmarks/sweep-cuda-parser.sh <grammar.alp> [dataset.inputs] \
    [BS_LIST] [IPT_LIST] [STATE_LIST] [INDEX_LIST]
```

Lists are space-separated strings in a single argument:

```bash
bash benchmarks/sweep-cuda-lexer.sh grammars/json.alp \
    benchmarks/json/data-10485760.inputs \
    "128 256" "1 2 4 5 6 8 12 16" "uint8_t" "int32_t"
```

The sweep uses `alpacc` from PATH — point PATH at the in-tree binary when the installed one may be stale:

```bash
# Create a wrapper pointing at the dist-newstyle binary:
ALPACC_BIN=$(find dist-newstyle -name alpacc -type f | head -1)
mkdir -p .claude-artifacts/bin
printf '#!/bin/sh\nexec %s "$@"\n' "$ALPACC_BIN" > .claude-artifacts/bin/alpacc
chmod +x .claude-artifacts/bin/alpacc

PATH=.claude-artifacts/bin:$PATH bash benchmarks/sweep-cuda-lexer.sh ...
```

Tuning env vars: `BENCH_RUNS` (default 5), `BENCH_WARMUP` (default 2), `KEEP_LOGS=1` to retain per-config logs.

SLURM variants (`*.slurm`) target the A100 cluster; `benchmark.sh` / `profile-gpu.sh` are full profiling scripts for cluster use.

### Compiling a generated .cu manually

```bash
nvcc -O3 -std=c++17 -arch=native output.cu -o output

# Override tuning at compile time:
nvcc -O3 -std=c++17 -arch=native \
    -DALPACC_BLOCK_SIZE=256 -DALPACC_ITEMS_PER_THREAD=5 \
    output.cu -o output

# Run benchmark:
./output --benchmark 10 --warmup 3 < data.inputs
```

## Grammar File Format (`.alp`)

Terminals are lowercase (regex patterns), nonterminals are uppercase. See `grammars/` for examples. The `grammars/arithmetic.alp` and `grammars/json.alp` files are the primary reference grammars.

## Project Architecture

```
src/Alpacc/
├── Grammar.hs              # Core grammar types and operations
├── CFG.hs                  # Parser for .alp grammar files (megaparsec)
├── LL.hs                   # LL(k) algorithms: first/follow/last/before sets, LL table
├── LLP.hs                  # LLP parser tables and parallel parsing
├── Types.hs                # Integer type abstractions
├── Encode.hs               # Encoding/compression
├── Util.hs                 # Shared utilities
├── Debug.hs                # Debugging helpers
├── HashTable.hs            # Hash table implementation
├── Random.hs               # Random grammar generation (for fuzzing)
├── Lexer/
│   ├── RegularExpression.hs  # Regex AST and parsing
│   ├── NFA.hs                # NFA construction
│   ├── DFA.hs                # DFA construction and minimisation
│   ├── FSA.hs                # Finite state automaton abstraction
│   ├── ParallelLexing.hs     # Parallel lexing algorithm
│   ├── DFAParallelLexer.hs   # DFA-based parallel lexer
│   └── Encode.hs             # Lexer state encoding
├── Generator/
│   ├── Analyzer.hs           # Grammar analysis shared by all backends
│   ├── Futhark/
│   │   ├── Generator.hs      # Top-level Futhark code generator
│   │   ├── Lexer.hs          # Futhark lexer emission
│   │   ├── Parser.hs         # Futhark parser emission
│   │   └── Futharkify.hs     # Futhark pretty-printing helpers
│   └── Cuda/
│       ├── Generator.hs      # Top-level CUDA code generator
│       ├── Lexer.hs          # CUDA lexer emission
│       ├── Parser.hs         # CUDA parser emission
│       └── Cudafy.hs         # CUDA pretty-printing helpers
└── Test/
    ├── Parser.hs             # Parser test infrastructure
    ├── Lexer.hs              # Lexer test infrastructure
    └── LexerParser.hs        # Combined lexer+parser tests
app/Main.hs                   # CLI entry point (optparse-applicative)
futhark/                      # Futhark runtime templates (embedded via file-embed)
cuda/                         # CUDA runtime templates (embedded via file-embed)
grammars/                     # Example grammar files
haskell-tests/                # Tasty test suite
benchmarks/                   # Per-grammar benchmark dirs (json/, numfmt/, lisp/), sweep scripts
```

The pipeline is: `.alp` file → `CFG` → `Grammar` → `LL`/`LLP` tables → `Generator.Analyzer` → backend code generator → `.fut` or `.cu` output.

## Code Style

- Standard Haskell conventions: `PascalCase` for types and modules, `camelCase` for values
- Qualified imports for non-Prelude modules (e.g., `Data.Map qualified as Map`)
- GHC language extensions enabled project-wide: `DeriveGeneric`, `FlexibleContexts`, `OverloadedStrings`, `TupleSections`, `QuasiQuotes`, `TemplateHaskell`, `BlockArguments`
- GHC warnings enabled (`-Wall`); keep the build warning-free
- No external formatter configured; follow the style of surrounding code
- Use `GHC2021` as the base language standard

## Dependencies

Key packages: `megaparsec` (grammar parsing), `file-embed` (embed Futhark/CUDA templates at compile time), `interpolate` (QuasiQuote string interpolation), `tasty` + `tasty-quickcheck` (testing), `optparse-applicative` (CLI), `parallel` (parallel evaluation), `containers`, `mtl`, `text`.

A Nix development shell is available via `shell.nix` — it provides GHC, cabal, ghcid, HLS, Futhark, and other tools.

## CI

GitHub Actions (`.github/workflows/main.yml`) runs on push to main and on PRs labelled "test":
1. Builds with GHC 9.14.1 / Cabal 3.16.0.0
2. Runs `cabal test all`
3. Runs `testfuthark.sh` with multiple (q, k) configurations
4. Runs `testjson.sh` with the multicore backend
