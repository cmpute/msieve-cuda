/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

Quadratic GPU basecase for the block Berlekamp-Massey (lingen),
self-written fallback track per BW.md. Owns none of the CADO
machinery: it consumes the raw coefficient buffers of the augmented
sequence E (m x b, b = m+n, coefficient-major matpoly<true> layout)
and produces the generator pi the same way lingen_qcode does, with
bm.delta / bm.lucky / bm.t maintained.

Algorithm (Coppersmith's block BM, explicit row-operation form):
  maintain pi (b x b) as b row-polynomials, each stored relative to
  its shift gamma_r (so "multiply row by X" is a counter increment),
  and P = pi * E^T maintained incrementally: row XORs on pi apply to
  P too, and shifts are free. Each step t: Delta = P rows at absolute
  coefficient t (b x m bits); Gauss-eliminate Delta's m columns
  (pivot preference: smallest gamma); pivot rows get gamma++ ; the
  same row ops are applied to pi and P on the GPU as batched,
  gamma-aligned XORs. Termination: n rows spontaneously zero for
  luck_mini consecutive steps.
--------------------------------------------------------------------*/

#ifndef _COMMON_BW_BW_LINGEN_QUAD_H_
#define _COMMON_BW_BW_LINGEN_QUAD_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* runs the basecase on the GPU. E and pi_out are plain coefficient
 * buffers: E in matpoly<true> layout (m*ceil(b/64) words per
 * coefficient, row-major rows, ascending coefficients); pi_out must
 * hold ceil(b/64) words per row per coefficient, (E_len + b + 64)
 * coefficients. gamma[] carries delta in/out (set to t0 by the
 * caller), lucky[] (b ints) and *t_out are updated. Returns the pi
 * degree bound (highest gamma), or -1 on failure. */
long long bw_lingen_quad_core(const uint64_t *E, uint32_t m, uint32_t b,
			uint32_t E_len, uint32_t *gamma, int *lucky,
			uint32_t *t_out, uint64_t *pi_out, int dev);

#ifdef __cplusplus
}
#endif

#endif /* !_COMMON_BW_BW_LINGEN_QUAD_H_ */
