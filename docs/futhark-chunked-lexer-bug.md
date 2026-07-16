# Futhark chunked-lexer boundary bugs

## What it is

The generated Futhark lexer (`backends/futhark/lexer.fut`) processes input in
chunks of `chunk_size` bytes (default 2^24 = 16,777,216) with a one-byte
overlap: each chunk after the first starts at the last byte of the previous
chunk, and the scan over the chunk is seeded with the state carried out of the
previous chunk (`prev_state`). Three independent bugs affected every input
that needed more than one chunk:

1. **Overlap byte double-composed.** `trans_to_state` composed `prev_state`
   with the transition of the chunk's byte 0 for *every* chunk, not just the
   first one. For chunks ≥ 2, byte 0 is the overlap byte, whose transition is
   already folded into `prev_state` — so its transition was applied twice.
2. **Scatter sentinel clobber.** In `lex_step`, positions that emit no token
   carry a `-1` sentinel in `offsets`; the scatter indices were computed as
   `map (+ prev_size) offsets`, turning every sentinel into `prev_size - 1` —
   a *valid* index for chunks ≥ 2. All non-token positions then scattered
   garbage onto the previous chunk's last token (scatter keeps the last
   duplicate, so the token was replaced by the state/span of the chunk's
   final byte).
3. **Span double-offset.** The globalisation of spans
   (`map (\(t,(s,e)) -> (t, (s + offset, ...)))`) was applied to the whole
   result array *after* the scatter, so tokens emitted by earlier chunks —
   whose spans were already global — got the current chunk's offset added
   again on every subsequent `lex_step` call.

## How it manifests

- Bug 1: double-applying a transition is harmless when it is idempotent in
  the composition monoid (digit runs, string-interior characters), but
  keyword mid-states (`null`, `true`, `false` in the JSON grammar) are not:
  the composed state goes dead and the whole lex returns `#none`. Any JSON
  input > 2^24 bytes with a keyword character at a chunk boundary failed to
  lex at all.
- Bug 2: one wrong terminal id per interior chunk boundary. The lex still
  reports success and the token *count* is correct, but the parse fails (or
  worse, could succeed on a wrong token stream).
- Bug 3: all tokens of chunks 1..j−1 have spans shifted by the accumulated
  offsets of later chunks (observed: first-chunk spans shifted by exactly
  2^24 on a two-chunk input). Terminal ids are unaffected, so this corrupts
  only the reported source locations.

Inputs ≤ 2^24 bytes (single chunk) were completely unaffected, which is why
the test suites and the 1MB benchmarks never caught any of this.

Downstream consequence: the Futhark `parse_int` benchmark at 10M scale
(22.4MB input) reported ~10.6 ms — but due to bug 1 that timed a *failed*
lex producing an empty token stream and no parsing at all. That number was
used as the baseline in `docs/cuda-benchmark-protocol-bug.md` and made the
CUDA parser look 3.3× slower than Futhark; the comparison was meaningless.

## How it was identified

1. A probe entry `parse_some` (returns `(lex ok, parse ok, #tokens)`) on the
   22.4MB benchmark input returned `false false 0` — the "fast" Futhark run
   was lexing nothing.
2. Bisection on input size: a 16.78MB token-aligned prefix (1 chunk) lexed
   fine; a 17.5MB prefix (2 chunks) failed. Minimal reproducer with
   `chunk_size = 4`: `[null,9]` fails, `[9999,9]` / `["abcd",9]` pass →
   non-idempotent keyword transitions at the boundary → bug 1.
3. After fixing bug 1, lexing succeeded but parsing still failed on the
   full input. A synthetic-input bisection first suggested a parser limit at
   2^24 *tokens*, but the synthetic inputs had one byte per token, so the
   apparent threshold was really the lexer's single→multi-chunk transition
   in *bytes* — the parser was exonerated by forcing a single chunk with
   `chunk_size = 2^30` (the `i32` parameter allows this), which parsed the
   full 22.4MB input fine.
4. Differential probe: lex the same input with `chunk_size = 2^24` vs
   `2^30` and report the first differing token. Full-record diff → index 0
   (spans shifted by 2^24 → bug 3). Terminal-only diff → exactly the token
   before the chunk boundary, replaced by a spurious copy of the chunk-final
   state (`]` where `,` should be) → bug 2.

A methodological trap worth recording: an earlier small-chunk-size
differential test "passed" because the hand-written test input failed to lex
under *every* chunk size, so all outputs were identical empty arrays. Probes
must assert non-emptiness before comparing.

## How it was solved

All three fixes are in `lex_step`/`trans_to_state` in
`backends/futhark/lexer.fut`:

1. `trans_to_state` and `traverse` take a `first: bool` flag (true only for
   the chunk at offset 0). For later chunks, position 0 yields `prev_state`
   itself — the overlap byte's transition is not re-applied.
2. Scatter indices keep the sentinel negative:
   `map (\o -> if o < 0 then -1 else o + prev_size) offsets`, so non-token
   positions stay out of bounds and are ignored by `scatter`.
3. Spans are globalised on `vs` *before* the scatter, so already-written
   entries are never touched again.

Verified by:
- `chunk_size = 4` minimal reproducers (`[null,9]` etc.).
- Differential probe on the real 22.4MB input: token streams (terminals and
  spans) byte-identical between `chunk_size = 2^24` and `2^30`.
- Differential probe with `chunk_size` 2–11 on a 39-byte input covering
  keywords, numbers, strings, and nesting (non-vacuous: 22 tokens).
- `parse_int` on the full 22.4MB input now succeeds (10,535,757 tokens).
- `cabal test all` (93 tests) and
  `tests/test-long-input.sh grammars/json.alp futhark-multicore` pass.

## Trade-offs of the chosen solution

- Adds a `bool` parameter threaded through `traverse`/`trans_to_state` and a
  branch in the scatter-index map; both are cheap and data-parallel.
- Keeps the one-byte-overlap chunk layout unchanged, so no changes to
  `lex_step` slicing or the driver loop.
- The chunked path is still only exercised by inputs > `chunk_size`; the
  generated `test` entry hardcodes 2^24, so routine test runs do not cover
  chunking (see below).

## Downsides of alternative approaches

- **Drop the overlap entirely** (chunks of exactly `chunk_size`, seed the
  scan with the carried state): cleaner, but changes the slicing arithmetic
  in `lex_int_flag` and the produce/start-flag logic in `lex_step`, which
  peeks at `states[i + 1]` across the boundary — a larger, riskier rewrite.
- **Seed the scan with `prev_state` as initial accumulator**: Futhark's
  `scan` takes a neutral element, not an initial accumulator; a non-neutral
  seed breaks the scan's semantics.
- **Filter out non-token positions before the scatter** (instead of the
  sentinel branch): an extra `filter` pass costs more than mapping the
  sentinel through.

## Follow-up

The generated `test`/`parse_int`/`lex` entries hardcode
`chunk_size = 16777216`, so no existing test exercises multi-chunk lexing on
reasonably-sized inputs, even though the underlying `lex_int` takes the chunk
size as a runtime argument (which is how the bug was isolated with 39-byte
inputs). Making the chunk size reachable from the test harness would turn
this whole class of bug into a routine regression test.
