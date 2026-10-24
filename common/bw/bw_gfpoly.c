/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

GF(2)[x] multiplication: schoolbook (shift-XOR) below the cutoff,
Karatsuba above. Splits happen at whole words, so the recombination
shifts are word-aligned (plain XOR at an offset, no bit shifting).
Unbalanced operands fall back to chunked two-way multiplication
instead of padding. Plain C, thread-safe, no globals.
--------------------------------------------------------------------*/

#include <stdlib.h>
#include <string.h>
#include "bw_gfpoly.h"

#define BWGP_CUTOFF 12	/* words; below this schoolbook wins */

/* r[0..na] ^= a * w (r must have na+1 valid words) */
static void mul_word(uint64_t *r, const uint64_t *a, uint32_t na,
			uint64_t w)
{
	while (w) {
		uint32_t k = __builtin_ctzll(w);
		w &= w - 1;
		if (k == 0) {
			uint32_t i;
			for (i = 0; i < na; i++)
				r[i] ^= a[i];
		} else {
			uint64_t carry = 0;
			uint32_t i;
			for (i = 0; i < na; i++) {
				uint64_t t = a[i];
				r[i] ^= (t << k) | carry;
				carry = t >> (64 - k);
			}
			r[na] ^= carry;
		}
	}
}

static void schoolbook(uint64_t *r, const uint64_t *a, uint32_t na,
			const uint64_t *b, uint32_t nb)
{
	uint32_t j;
	memset(r, 0, (size_t)(na + nb) * 8);
	for (j = 0; j < nb; j++)
		if (b[j])
			mul_word(r + j, a, na, b[j]);
}

static void gpmul(uint64_t *r, const uint64_t *a, uint32_t na,
		const uint64_t *b, uint32_t nb, uint64_t *sc);

/* scratch bound: a Karatsuba level parks as/bs/z1 (<= 3*(2h+4)
 * words) or an unbalanced temp (na+nb-h) plus the recursion's own
 * need; summed over the depth this stays inside 8*(na+nb)+1024. */
uint32_t bw_gfpoly_scratch_words(uint32_t na, uint32_t nb)
{
	return 8 * (na + nb) + 1024;
}

static void gpmul(uint64_t *r, const uint64_t *a, uint32_t na,
		const uint64_t *b, uint32_t nb, uint64_t *sc)
{
	if (na > nb) {
		const uint64_t *tp = a;
		uint32_t tn;
		a = b; b = tp;
		tn = na; na = nb; nb = tn;
	}
	/* now na <= nb */
	if (na <= BWGP_CUTOFF) {
		schoolbook(r, a, na, b, nb);
		return;
	}

	uint32_t i;
	uint32_t h = nb / 2;
	if (na <= h) {
		/* unbalanced: b = b0 + x^{64h}*b1. The low product only
		 * defines r[0, na+h); zero the tail so the high product
		 * can be XOR-accumulated into it. */
		gpmul(r, a, na, b, h, sc);
		memset(r + na + h, 0, (size_t)(nb - h) * 8);
		uint32_t nt = na + (nb - h);
		uint64_t *tmp = sc;
		gpmul(tmp, a, na, b + h, nb - h, sc + nt);
		for (i = 0; i < nt; i++)
			r[i + h] ^= tmp[i];
		return;
	}

	/* a = a0 + x^{64h}*a1, b = b0 + x^{64h}*b1 */
	const uint64_t *a0 = a, *a1 = a + h;
	const uint64_t *b0 = b, *b1 = b + h;
	uint32_t na1 = na - h, nb1 = nb - h;

	/* z0 -> r[0, 2h), z2 -> r[2h, na+nb) */
	gpmul(r, a0, h, b0, h, sc);
	gpmul(r + 2 * h, a1, na1, b1, nb1, sc);

	uint32_t nsa = (h > na1 ? h : na1) + 1;
	uint32_t nsb = (h > nb1 ? h : nb1) + 1;
	uint64_t *as = sc, *bs = sc + nsa, *z1 = sc + nsa + nsb;
	uint64_t *sc2 = z1 + nsa + nsb;

	memset(as, 0, (size_t)nsa * 8);
	memcpy(as, a0, (size_t)h * 8);
	for (i = 0; i < na1; i++)
		as[i] ^= a1[i];
	memset(bs, 0, (size_t)nsb * 8);
	memcpy(bs, b0, (size_t)h * 8);
	for (i = 0; i < nb1; i++)
		bs[i] ^= b1[i];

	gpmul(z1, as, nsa, bs, nsb, sc2);

	/* cross = z1 ^ z0 ^ z2raw: all three in the compressed degree
	 * frame. z2raw is z2's coefficient array as stored (at offset
	 * 2h), NOT its shifted position. Then r += x^{64h}*cross. */
	uint64_t *t = sc;
	uint32_t nz1 = nsa + nsb;
	memcpy(t, z1, (size_t)nz1 * 8);
	for (i = 0; i < 2 * h && i < nz1; i++)
		t[i] ^= r[i];			/* z0 */
	for (i = 0; i + 2 * h < na + nb && i < nz1; i++)
		t[i] ^= r[i + 2 * h];		/* z2, raw frame */
	for (i = 0; i + h < na + nb && i < nz1; i++)
		r[i + h] ^= t[i];
}

uint64_t *bw_gfpoly_mul(uint64_t *r, const uint64_t *a, uint32_t na,
			const uint64_t *b, uint32_t nb)
{
	uint32_t scw = bw_gfpoly_scratch_words(na, nb);
	uint64_t *sc = malloc((size_t)scw * 8);
	if (!sc)
		return NULL;
	gpmul(r, a, na, b, nb, sc);
	free(sc);
	return r;
}
