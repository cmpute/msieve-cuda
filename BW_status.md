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

## RSA-100 closed loop (2026-10-24, complete)

poly 6 min (GPU1) -> cuda-sieve 20M relations in ~3.5 min ->
msieve filter (matrix 399k x 399k + excess, 91 dense rows) ->
BW on GPUs 1,2,3: Krylov 6 s, lingen 69 s, mksol 3 s, 64
CPU-verified dependencies, BWTime 102 s -> sqrt dependency #2:

  p50 factor: 37975227936943673922808872755445627854565536638199
  p50 factor: 40094690950920881030683735292761468389214899724061

Both match the published RSA-100 factors. (Dependency #1 gave
GCD 1 -- the expected coin flip per dependency; msieve's sqrt
tries one dep by default, `-nc3 "dep_first=N dep_last=N"` selects
another. Note dep_first/dep_last must be space-separated: a comma
triggers sqrt.c's legacy 'X,Y' override parser.)

The Lanczos differential on the same matrix: BL 85 s -> same two
factors from its dep #1, confirming the matrix/cycle/fb pipeline;
BW's LA GPU time (Krylov+mksol, ~9 s on 3 GPUs) vs BL 85 s.

## Quad lingen (self-written quadratic GPU basecase) — complete

2026-10-24, later session. bw_lingen_impl=quad now passes the whole
ladder: planted identity test (m=64 and m=128 geometry), QUAD
BW_TEST_FULL closed loop, and the RSA-100 e2e end to end
(Krylov 12 s -> quad lingen 68 s vs cado 69 s -> mksol 4 s ->
47 CPU-verified deps -> sqrt dep 40 gives the two published p50
factors; deps live in bit-planes 18-64 of the dep bitmap, planes
1-17 come out zero, so use dep_first>=18).

The five root causes behind the earlier non-minimal generator /
failing identity, for the record:

1. P-arena sizing: a row's P content spans [0,L) plus one
   coefficient per pivot shift, so ncoeff must cover L + L*m/b
   plus the degree spread (observed +9*b over the mean at
   L=9469); sized L + L*m/b + L/4 + 4b + 1024 now.
2. Aug contribution folding must be XOR, not OR: W[j] =
   g_id[j] + sum_rp x^{t0-kk} g_aug[rp] lands several
   contributions on the same coordinate; OR-folding destroyed
   the cancellations (corrupted every W, let W=0 rows through
   selection).
3. t0 = max kk + 1, not max kk: the identity window must start
   strictly above every pick (CADO's kA <= t0-1). With
   t0 == max kk the top pick duplicates an identity column, and
   the trivial self-relations map to W = 0, costing solution
   dimensions (62/64 on the planted test).
4. The file is trailing-first: block c carries W degree Dlen-1-c
   (lingen_output_to_singlefile convention; the test pairs
   A[k-d+c] with block c, mksol's Horner feeds block ncoef-1-c
   at step c). The emit loop builds ascending; blocks are
   reversed before writing.
5. Exact degree trim: Dlen = max deg(W) + 1 so the leading
   block is nonzero (consumers rely on it). Verified junk stays
   below it (the low-side boundary is bounded by t0 + the id
   degree, and aug-topped candidates sit lower still).

Also: pivot discipline restored to CADO's (already-pivoted rows
are inert for the rest of the coefficient — with our batched
end-of-coefficient shifts that means skipping them as XOR
targets too); without it real sequences never stabilize (0 rows
reached luck_mini=3 on the RSA-100 matrix even 3000 coefficients
past the theoretical completion point).

F degree 6236 on the real matrix (theory 2N/m = 6239); quad BM
~5-10 ms/step at m=128/b=192, GPU resident.

## Independent recursive lingen (bw_lingen_impl=own) — working

2026-10-24, evening. Fully independent lingen: no CADO/gf2x code
whatsoever (user decision; the GPU-leaf fusion into the vendored
recursion was built and then reverted for the same reason).

Pieces (all public domain):
- bw_gfpoly.c/.h: GF(2)[x] Karatsuba over bit-packed words,
  word-aligned splits, chunked handling of unbalanced operands.
  Validated against a reference schoolbook (4000 random <=500x500
  under ASAN, exhaustive 10..120, 3000 each at several large/unbalanced
  sizes). Three real bugs found on the way, worth remembering:
  mul_word must not early-return on the k=0 bit; the Karatsuba cross
  term needs z2 in its RAW (unshifted) frame; unbalanced splits must
  accumulate the high product into a zeroed tail (gpmul is
  write-only).
- bw_lingen_own.cpp: recursive block BM, own invariants (basecase
  = the quad GPU engine via bwq_basecase; pi = piL * piR with the
  middle-product window [half, L) of E*piL). The scalar-vector
  products decompose per BIT plane (a word-level "plane" loses the
  bit independence of scalar x vector multiplication).
- Delta threading through the recursion is REQUIRED, not an
  optimization: without it every column churns to max degree in every
  sub-window, and the composed W junk bound (t0 + degree) crosses the
  declared file degree. With it, early-frozen columns stay small
  (test F degree 7 vs quad's 8 vs cado's 6).
- The quad engine got exported glue: bwq_select / bwq_basecase
  (gamma_in/out) / bwq_emit, all inside extern "C" in
  bw_lingen_quad.h.

Status: identity tests (m=64/128), BW_TEST_FULL closed loop, and the
RSA-100 e2e all pass. Timings on the RSA-100 matrix (3 GPUs):
cado 69 s / quad 68 s / own 93 s (lingen only; Krylov 6 s, mksol 3 s).
F degree 6354 (theory 2N/m = 6239). RSA-100 factors recovered from
its single CPU-verified dependency (dep bit-plane 43).

Known weakness: the own route currently yields only 1 nilpotent-verified
dependency (cado 64, quad 47) — the smallest-degree-first selection
picks W-independent but kernel-correlated rows. Fine for factoring
(needs 1), worth improving before production use.

Build-mode trap (bit us twice): plain `make` (BW=0) recompiles
gf2.o without HAVE_BW; a later `make BW=1 ...` only relinks, so msieve
silently loses the BW dispatch and runs Block Lanczos. Touch
gnfs/gf2.c (the only HAVE_BW-dependent TU) when switching back.
Also `make msieve` is a no-op (msieve is not a real target; the
default goal builds it).

## CADO removed; own+gf2x is the default lingen (2026-10-24, night)

User decision: keep gf2x, drop CADO entirely. The vendored
common/bw/cado/ tree and shims/ are deleted (history in git);
bw_gfpoly.c (our Karatsuba) went with it -- bw_lingen_own.cpp now
multiplies through gf2x (gf2x_mul_r with per-thread pools). The
bw_lingen_impl values collapsed to 0 = own (default) / 1 = quad;
tests default to own on device 1.

Effect on the RSA-100 matrix (3 GPUs): lingen 93 s (our Karatsuba)
-> 22 s (gf2x) -- 3.1x faster than the removed cado path (69 s) and
the fastest of the three. Whole -nc2 now 37 s wall (Krylov 6 +
lingen 22 + mksol 3 + verify); 47 CPU-verified dependencies; sqrt
reproduces the published factors. Identity tests (m=64/128), both
closed-loop selftests (own default, quad) green after the removal.

Remaining LGPL-2.1+ surface: gf2x only (LGPL build by default, the
GPL Toom file swapped for the placeholder; BW_GPL_GF2X=1 opts in).

## RSA-120: end-to-end + the multi-GPU timing table (2026-10-24, night)

First full pipeline at c120 scale, all on our own code paths: msieve
GPU polyfind (4m54s) -> cuda-sieve 3 GPUs (52% of three 1M-wide q
bands, 163M unique relations) -> msieve filter (matrix 2,099,776 x
2,099,776 + 89 dense rows) -> BW (own+gf2x lingen) -> sqrt. RSA-120
factors out of dependency plane 23: two p60, product verified.

Pipeline bug found and fixed on the way:
- the own recursion's delta bookkeeping double-counted (deltaL +
  deltaR at each combine), inflating column degrees ~2x per level
  until basecase arenas reached 26.7 GB and cudaMalloc failed. The
  correct semantics (CADO's) is a single shared array mutated in
  place: left updates it, right inherits and updates, no addition at
  the combine. Symptom before the fix: slow 69-153 s leaf calls
  (10-GB-class arenas); after: whole -nc2 fits and the ladder stays
  green. Also: bwq_alloc now reports size + CUDA error, and the
  engine destroys its stream per call.

Multi-GPU timing table (RSA-120 matrix N=2.1M, L=49408, phases from
the log; PCIe, no NVLink; flat allgather unless noted):

  gpus | krylov | lingen | mksol | deps
  -----+--------+--------+-------+-----
    1  |  76 s  | 205 s  |  85 s | 44
    2  | 113 s  | 247 s  |  80 s | 44
    3  | 107 s  | 183 s  |  64 s | 44
  3,hier | 127 s | 174 s | 65 s | 44

FINDING: multi-GPU Krylov is NET NEGATIVE at this size -- 2 GPUs is
49% slower than 1. Per-step: 1.54 ms (1 GPU) vs 2.17-2.57 ms (2-3
GPUs): the per-step cross-GPU gather (host-mediated, barrier per
step) costs more than the slab-parallel spmv saves at N=2.1M on
PCIe. lingen stays single-GPU+CPU (its variance across rows is BL-job
CPU contention, load avg ~30). mksol improves mildly (85->64 s).

So for THIS hardware the Krylov allgather path is the blocker, not
the compute. Candidates (next): overlap gather with spmv (pipelined
per-slab), device-mediated gather (peer copies, no host bounce),
vector-window xfer_min batching across steps, or run one BW instance
per GPU with independent Y blocks (embarrassingly parallel, no
gather) and merge at lingen input. The 4-GPU row waits for GPU0
(the C263 BL job holds it).

## Open (next session)

- multi-GPU scaling table on this matrix (1 vs 2 vs 3 GPUs).
- -ncr resume path exercised only via checkpoint unit tests.
- C254/C264 benchmarks (M4/M5), (m,n) sweeps (M6), hierarchical
  allgather, lingen OpenMP/GPL-Toom levers.
