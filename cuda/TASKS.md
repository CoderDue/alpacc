# CUDA parser backend — task plan

Goal: implement the LLP parser from `futhark/parser.fut` as CUDA kernels in
`cuda/parser.cu`, using the vendored PSE primitive (`cuda/pse.cu`) for
bracket matching and parent vectors. Bring-up is staged kernels first, then
fused into a single cooperative kernel.

## Done

- [x] Generalize the APSEP SPT kernel to PSE(<=) via a compile-time `INCL`
      comparator (APSEP repo, `src/apsep.cuh`; costs nothing on random input,
      ~1.5x faster than strict on tie-heavy input).
- [x] Vendor the PSE primitive into `cuda/pse.cu`, split the scan machinery
      out of `common.cu` into `cuda/scan.cu`, embed both in the generators
      (`Generator.hs`, `Parser.hs`). Verified: 84/84 correctness checks vs a
      CPU stack reference (both semantics, incl. a ±1 depths pattern);
      generated `json.alp` output compiles with `nvcc -std=c++17`.

## Pending

- [ ] **Stage 1 — key lookup + span offsets** (`parser.cu`): for each
      position, hash the q+k terminal window (FNV-1a, as in `parser.fut`
      `hash`/`lookup`) and linear-probe `HASH_TABLE_KEYS`; produce validity
      flag plus (stack_len, prod_len) pairs; exclusive scan of the pairs for
      segmented-copy offsets. Validate against the Futhark parser.
- [ ] **Stage 2 — bracket matching**: segmented copy of `STACKS` spans, ±1
      depth scan with validity, PSE(<=) over depths (`runSPT<..., true>`),
      bracket symbol comparison, global validity reduce (ballot + atomic).
- [ ] **Stage 3 — parents**: segmented copy of `PRODUCTIONS` spans, arity−1
      exclusive scan, PSE(<=) for the parent vector; optional terminal-node
      scatter for `parse_int`-style semantics.
- [ ] **Fuse** the stages into one cooperative kernel; benchmark vs the
      staged version and vs the Futhark backend.
