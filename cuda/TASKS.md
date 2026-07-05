# CUDA parser backend — task plan

Goal: implement the LLP parser from `futhark/parser.fut` as CUDA kernels in
`cuda/parser.cu`, using the vendored PSE primitive (`cuda/pse.cu`) for
bracket matching and parent vectors. Bring-up is staged kernels first, then
fused into a single cooperative kernel. Detailed design: `cuda/PLAN.md`.

## Done

- [x] Generalize the APSEP SPT kernel to PSE(<=) via a compile-time `INCL`
      comparator (APSEP repo, `src/apsep.cuh`; costs nothing on random input,
      ~1.5x faster than strict on tie-heavy input).
- [x] Vendor the PSE primitive into `cuda/pse.cu`, split the scan machinery
      out of `common.cu` into `cuda/scan.cu`, embed both in the generators
      (`Generator.hs`, `Parser.hs`). Verified: 84/84 correctness checks vs a
      CPU stack reference (both semantics, incl. a ±1 depths pattern);
      generated `json.alp` output compiles with `nvcc -std=c++17`.
- [x] Checked-in PSE differential test: `bash tests/testpse.sh [arch]`
      (GPU vs CPU stack reference, both semantics; local runs only — needs
      a GPU). Grammar-level `alpacc test generate`/`compare` wiring for the
      CUDA parser lands with Stage 1.
- [x] **C backend** (`alpacc c grammar.alp --parser`): sequential LLP
      reference parser (`c/parser.c` + `src/Alpacc/Generator/C/`), mirrors
      `pre_productions_int` from `parser.fut` (stack-based bracket matching
      instead of depths + PSE). Reads the binary test-input format on stdin,
      writes the output format on stdout, so it plugs straight into
      `alpacc test generate`/`compare`; runs in GPU-less hosted CI and is
      the CPU reference for the CUDA kernels. Parser-only for now (other
      modes error out). Also emits START_TERMINAL/END_TERMINAL, which the
      CUDA constants block still lacks. Differential tests: json.alp
      (exhaustive length 4, single-long 20000) and arithmetic.alp
      (exhaustive length 6) all pass; random grammars via
      `bash tests/testc.sh [q] [k] [target] [jobs]` (needs GNU parallel).

## Done (continued)

- [x] **Generator fixes** (`Cuda/Parser.hs`): emit `START_TERMINAL` and
      `END_TERMINAL`; fix `PRODUCTIONS` declaration to use `production_t` and
      `PRODUCTIONS_SIZE`; change `HASH_TABLE_STACKS_SPAN` and
      `HASH_TABLE_PRODUCTIONS_SPAN` from `size_t` to `int64_t` to handle -1
      sentinels correctly; add `__device__` to all constant array declarations
      so they are accessible from CUDA kernels.
- [x] **Stage 1 — key lookup + span offsets** (`cuda/parser.cu`):
      `parserKeysSpans<I>` kernel builds the Q+K window (FNV-1a + linear
      probe), writes stack/prod lengths and raw int64_t spans. Two separate
      inclusive scans (one per length array) using `inclusiveScanIKernel`
      (wraps scan.cu's decoupled-lookback machinery) give per-position
      offsets and totals. Index type templated on `I` (uint32_t / size_t).
- [x] **Stage 2 — bracket matching**: `parserStacks<I>` copies STACKS spans;
      `bracketsToDeltas<I>` maps ±1; inclusive int32_t scan via
      `inclusiveScanI32Kernel`; `depthsAndValidate<I>` checks no prefix < 0
      and last == 0; `runSPT<int32_t, 128, 4, true>` (PSE(<=)) over depths;
      `checkBracketMatches<I>` verifies symbol equality via bracket_unpack.
- [x] **Stage 3 — productions**: `parserProductions<I>` copies PRODUCTIONS
      spans into the output array, completing `pre_productions_int`.
- [x] **Binary I/O test harness** in `cuda/parser.cu` `main()`: reads the
      same binary format as `c/parser.c`, routes to `uint32_t` or `size_t`
      instantiation based on input length, writes results compatible with
      `alpacc test compare --parser`.
- [x] **`tests/testcuda.sh`**: differential test script mirroring
      `testc.sh`; uses `--length 4` to keep GPU memory bounded; 10/10 random
      grammars pass for q∈{0,1}, k∈{1,2}.

## Pending

- [ ] **json.alp end-to-end test**: `alpacc test compare` on json grammar
      with `--length 3` (55 KB) — verify pass; `--single-long` for throughput.
- [ ] **Stage 3 — parents** (for `parse_int` / lexer+parser mode): arity−1
      map over productions, exclusive scan, PSE(<=) for the parent vector;
      optional terminal-node scatter. Not needed until lexer+parser mode.
- [ ] **Fuse** the stages into one cooperative kernel; benchmark vs the
      staged version and vs the Futhark backend.
