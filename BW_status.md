# BW implementation status — 2026-10-24 (first full session)

Branch `block-wiedemann` in this worktree (base msieve 0f25603 + lpbound patch).

## Done

- **M0**: worktree; pinned CADO snapshot b5ca9c4 vendored under common/bw/
  (lingen core, bblas, gf2x LGPL variant, fmt; provenance in
  common/bw/README.licensing). spmv_engine: per-engine cache-config +
  stream-aware `spmv_engine_run_on`. bw_kernel.cu (oprod64/combine64/
  dense64/anyset64) with lane-private shared accumulation.
- **M1**: full C skeleton (bw.c/bw_matrix/bw_gpu/bw_krylov/bw_mksol/
  bw_checkpoint/bw_seq). Selftest vs CPU references on synthetic
  matrices (dense rows 0/1/63/64/65, excess edges) passes on 1/2/3 GPUs,
  including a complete Krylov->lingen->mksol->nilpotent->A*x=0 loop
  (BW_TEST_FULL=1).
- **M2**: lingen ported single-rank (bw-local trims: no MPI bigmatpoly,
  no checkpoints/tuning/FFT plugins; companions synthesized on demand,
  recurse = L >= threshold with a 64-coefficient leaf floor). Random-
  recurrence identity test passes on basecase and recursive paths.
- **M3**: end-to-end on a 200k-dim SNFS-146 orphan matrix: 2 GPUs,
  64 CPU-verified dependencies, BWTime 43 s (31 s single-GPU; the
  26 s lingen dominates at this size). gf2.c integration + BW=1
  build system; BW=0 clean build verified (no BW symbols).

## Key bugs found by the tests (and fixed)

1. oprod/dense shared-memory accumulation raced across lanes ->
   per-lane accumulators, warp-0-only flush.
2. mksol combine received a host pointer (device deref) -> staged
   d_fcoef.
3. MKSOL_STEP allgather missed the copy-stream wait on ev_comp ->
   staged D2H raced SpMV/combine (only visible multi-GPU).
4. F file is written transposed (CADO convention) and must be
   consumed trailing-first for z = sum A^i Y Ft[d-c].
5. The padded tail [nrows,N) carries real dependency bits; it was
   wrongly zeroed. Now collected via worker-0 h_tail.
6. bw_gpus=1,2 parsed as "1" (comma = option separator) -> dedicated
   list parsing; tokenizer keeps digit-comma runs together.
7. block_nnz default 100k fragmented slabs into ~172 launches
   (~120 ms/step on the 200k matrix). L2-calibrated default
   (bw_l2_bytes; ~514M on L20) -> ~0.2 ms/step, Krylov 4736 steps
   in ~1 s. Small matrices are lingen-bound; multi-GPU payoff needs
   C254/C263 step times (13-27 ms).
8. Makefile: .bwo objects now depend on the BW headers (a stale
   bw_krylov.o with an older bw_worker_t ABI caused cross-TU struct
   layout mismatches).

## Open (next session)

- RSA-100 full closed loop (poly -> cuda-sieve -> filter -> BW -> sqrt
  vs known factors); multi-GPU scaling table on that matrix.
- -ncr resume path exercised only via checkpoint unit tests.
- C254/C264 benchmarks (M4/M5), (m,n) sweeps (M6), hierarchical
  allgather, lingen OpenMP/GPL-Toom levers.
