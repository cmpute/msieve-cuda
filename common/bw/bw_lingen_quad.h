/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

Fully self-contained quadratic lingen for the BW fallback track: reads
the Krylov A file, runs a GPU block Berlekamp-Massey on the augmented
sequence, writes the generator F file. No CADO or gf2x code.

Math (all own convention):
  - augmentation: scan A's columns (coefficient-major, earliest first)
    with an m-bit Gaussian basis until m independent (k_r, j_r) pairs
    are found; F0 = [x^-t0 I_n | x^-k_1 e_{j_1} ... x^-k_m e_{j_m}],
    t0 = max k_r; E' = A*F0 (m x b, b = m+n).
  - BM: b candidate rows over m-bit discrepancies (the quad engine);
    a lucky row r satisfies E'(x)*g_r(x) ~= 0 on the high window,
    g_r = row r of the generator arena.
  - extraction: W_r = x^t0 * F0 * g_r satisfies A*W_r ~= 0 on the
    shifted window; the F file stacks n lucky W_r as columns
    (file coefficient c, word i = W_{r_i} coefficient c), which is
    exactly the transposed convention mksol consumes.
--------------------------------------------------------------------*/

#ifndef _COMMON_BW_BW_LINGEN_QUAD_H_
#define _COMMON_BW_BW_LINGEN_QUAD_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Full pipeline. Returns 0 on success (F file written), -1 on failure
 * (message on stderr). luck_mini is the required zero-discrepancy
 * streak per solution row. */
int bw_lingen_quad_full(const char *afile, const char *ffile,
			uint32_t m, uint32_t n, uint64_t seq_len,
			uint32_t nrows, uint32_t ncols, int dev,
			uint32_t luck_mini, uint32_t nthreads);

#ifdef __cplusplus
}
#endif

#endif /* !_COMMON_BW_BW_LINGEN_QUAD_H_ */
