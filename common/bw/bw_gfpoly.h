/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

GF(2)[x] bit-packed polynomial arithmetic for the independent lingen
(bw_lingen_own). Polynomials are little-endian word arrays, bit k of
word k/64 is the coefficient of x^k.
--------------------------------------------------------------------*/

#ifndef _COMMON_BW_BW_GFPOLY_H_
#define _COMMON_BW_BW_GFPOLY_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* r = a * b. r must have na+nb words and is fully written (no need to
 * zero it first). a and b must not alias r. Small operands use
 * shift-XOR schoolbook; larger ones Karatsuba. Returns r. */
uint64_t *bw_gfpoly_mul(uint64_t *r, const uint64_t *a, uint32_t na,
			const uint64_t *b, uint32_t nb);

/* Number of scratch words bw_gfpoly_mul may want (for callers that
 * keep per-thread buffers). */
uint32_t bw_gfpoly_scratch_words(uint32_t na, uint32_t nb);

#ifdef __cplusplus
}
#endif

#endif /* !_COMMON_BW_BW_GFPOLY_H_ */
