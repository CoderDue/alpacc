# CUDA combined-mode `--benchmark` measured the wrong thing

## The bug

In the generated CUDA CLI (`backends/cuda/cli.cu`), combined lexer+parser
binaries dispatched `--benchmark` to `parser_benchmark`, the benchmark for
*parse-only* binaries. That function speaks the token-ID protocol: it
reinterprets the framed payload as an array of terminal ids. When the
combined-mode benchmark is fed raw text (as the benchmark Makefiles do), this
meant:

- Raw bytes were treated as terminal ids — garbage input, and ~2.1× more
  elements than the real token stream (every byte becomes a "token").
- The lexer was never run inside the timed region.
- `d_valid` was forced to 1 and never checked, so the invalid parse was
  silently timed anyway.
- `with_parents` stayed false, so the compact-tree phases (parents,
  production/parent compaction) of the fused kernel were skipped.

## How it manifested

After the SoA compact-tree wire-format change (01d10f1) and a benchmark
Makefile fix that made the Futhark datasets real (`dd bs=24 skip=1` to strip
the frame header), the Futhark `parse_int` number at 10M tokens became an
apparently-honest 10.6 ms (later found to be a failed lex — see
`futhark-chunked-lexer-bug.md`), while the CUDA "parser benchmark" reported
~30 ms. This looked
like a CUDA regression caused by the format change.

## How it was identified

Incremental comparison against a pre-format-change worktree (4b120ee), one
variable at a time, all on a GTX 1660 Ti with the same 22.4 MB json input:

- Benchmark-mode kernel times old vs new: 29.08 ms vs 29.15 ms — no change.
- Real batch pipeline (nsys, `with_parents=true`): 35.189 ms vs 35.189 ms —
  identical.
- `git log -L` on the dispatch site showed combined-mode `--benchmark` had
  called `parser_benchmark` since long before the format change.

So no kernel regressed; the benchmark had simply never measured the real
pipeline, and the newly-correct Futhark number exposed it.

## The fix

A new `both_benchmark` in `backends/cuda/cli.cu`, used by combined-mode
binaries. Each timed run is the real pipeline, matching what the Futhark
`parse_int` entry point measures: lexer kernel → device-to-device token
staging → fused parser kernel with `with_parents=true` (full compact tree).
Details:

- All allocations (lexer context, `ParserFused` buffers, sentinel staging)
  happen once per test, outside the timed region — the fair equivalent of
  Futhark's free-list allocator, which caches device allocations across runs.
- Two variants are reported: kernel-only (`parse_int (cuda, N bytes)`) and
  `+io`, which additionally times the input H2D copy and the result D2H
  copies (tokens, spans, compact tree, token parents).
- `d_valid` is checked after the timed loops; an input that does not lex or
  parse fails the benchmark instead of being timed silently.
- `parser_benchmark` is now compiled only for parse-only binaries
  (`HAS_PARSER && !HAS_LEXER`), where the token-ID protocol is the correct
  input format.

First honest numbers (json, 22.4 MB, GTX 1660 Ti): CUDA 35.0 ms kernel-only
vs Futhark `parse_int` 10.6 ms — which looked like a 3.3× gap. **Update:** the
Futhark side of that comparison was itself bogus: the Futhark chunked lexer
failed on inputs > 2^24 bytes (see `futhark-chunked-lexer-bug.md`), so the
10.6 ms timed a failed lex with zero tokens and no parsing. With the Futhark
lexer fixed, the honest numbers (json, 22.4 MB, GTX 1660 Ti) are: CUDA
36.1 ms vs Futhark `parse_int` 75.4 ms kernel-only — the CUDA fused pipeline
is ~2.1× *faster*, not 3.3× slower; lexer-only is CUDA 8.4 ms vs Futhark
9.9 ms.

## Trade-offs of the fix

- The discovery lexer run assumes token count and lexability are fixed for a
  fixed input (true: the pipeline is deterministic), so result-buffer sizes
  are computed once.
- The kernel-only variant includes a D2D copy of the token stream between the
  lexer and parser kernels. A fused lexer+parser launch could avoid it, but
  the copy is what the runtime (`runBothFused`) actually does, so timing it is
  honest.
- Timing per run with `cudaEventElapsedTime` slightly perturbs short runs;
  runs are clamped to 0.5 μs in the stats, same as the other benchmark modes.

## Alternatives considered

- **Keep the token-ID benchmark but feed it real token dumps**: still skips
  the lexer and tree phases, so it measures a subset of the pipeline nobody
  runs; also needs a second dataset format.
- **Time `both_batch_impl` (batch mode) directly**: includes host-side
  framing, encode/decode, and per-batch cudaMalloc/free, which the Futhark
  benchmark excludes — numbers would not be comparable.
- **Benchmark via nsys/ncu externally**: accurate but not reproducible with a
  single `make bench` invocation, and kernel sums ignore gaps between
  kernels.
