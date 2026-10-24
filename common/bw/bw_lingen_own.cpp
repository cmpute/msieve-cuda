/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

Independent recursive lingen (bw_lingen_impl=own). No CADO or gf2x
code: the basecase is the self-contained GPU engine from
bw_lingen_quad.cpp, the polynomial arithmetic is the Karatsuba
multiplier in bw_gfpoly.c, and the divide-and-conquer below follows
the standard recursive block-Berlekamp-Massey structure (Thome's
approach), re-derived here from our own invariants:

  basecase(E, L): returns pi with (E[0:L) * pi)_k = 0 for k in [0, L)
    (this is exactly the GPU engine's guarantee, transposed);

  recursion: with half = L - L/2,
    piL = lingen(E[0:half))
    R'_j = (sum_k piL[k,j] * E_k)[half : L)      (the "middle product"
          window; coefficients below half are zero by piL's guarantee)
    piR = lingen(R')
    pi  = piL * piR (per output row j: sum_k piR[k,j] * piL_k)

  Correctness of the composition for k < half needs only that
  negative-degree coefficients are zero; for half <= k < L it is the
  basecase guarantee applied to R'. Row orientation: "row j" of an
  arena is candidate j's polynomial vector (bits across the b
  candidates for pi, across the m matrix rows for E).

Everything stays in row-major arenas: row r, coefficient c, word w at
buf[r*stride + c*wpc + w].
--------------------------------------------------------------------*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <vector>
#include <algorithm>

#include "bw_lingen_quad.h"
#include "gf2x.h"
#include "bw_seq.h"

static double own_now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/*--------------------------------------------------------------------
out[j] = sum_k s_{j,k}(x) * in_k(x), restricted to the coefficient
window [w0, w1) of the full products. s_{j,k} is the scalar polynomial
whose coefficient c is bit k of s's row j at coefficient c. in rows
have in_len coefficients of wpc_in words; s rows have s_len
coefficients of s_wpc words. out gets b compact rows of (w1-w0)
coefficients, wpc_in words each. OpenMP over the output rows.
--------------------------------------------------------------------*/
static void own_vecmul(std::vector<uint64_t> &out, uint64_t w0,
		uint64_t w1, const std::vector<uint64_t> &in,
		uint32_t wpc_in, uint64_t in_len,
		const std::vector<uint64_t> &s, uint32_t s_wpc,
		uint64_t s_len, uint32_t b)
{
	const uint64_t in_w = (in_len + 63) / 64;
	const uint64_t s_w = (s_len + 63) / 64;
	const uint64_t full_w = in_w + s_w;
	const uint64_t out_len = w1 - w0;
	const uint32_t bits = wpc_in * 64;   /* vector width per row */

	/* pre-extract the input rows' bit planes: plane (k, i) is the
	 * packed scalar polynomial of bit i across row k's coefficients.
	 * A scalar times a vector polynomial multiplies every bit line
	 * independently, so the products are per-bit scalar products --
	 * exactly the m*b*b (or b*b*b) entry products of the matrix
	 * polynomial product. */
	std::vector<uint64_t> planes_buf((size_t)b * bits * in_w, 0);
	for (uint32_t k = 0; k < b; k++) {
		const uint64_t *row = &in[(size_t)k * in_len * wpc_in];
		for (uint64_t c = 0; c < in_len; c++) {
			for (uint32_t w = 0; w < wpc_in; w++) {
				uint64_t v = row[c * wpc_in + w];
				while (v) {
					uint32_t i = w * 64 +
						__builtin_ctzll(v);
					planes_buf[((size_t)k * bits + i) *
						in_w + (c >> 6)] |=
						1ull << (c & 63);
					v &= v - 1;
				}
			}
		}
	}

	out.assign((size_t)b * out_len * wpc_in, 0);

#pragma omp parallel
	{
		/* per-thread gf2x scratch pool (reentrant variant) */
		gf2x_mul_pool_t pool;
		gf2x_mul_pool_init(pool);
#pragma omp for schedule(dynamic)
		for (int64_t j1 = 0; j1 < (int64_t)b; j1++) {
			uint32_t j = (uint32_t)j1;
			const uint64_t *srow = &s[(size_t)j * s_len * s_wpc];
			std::vector<uint64_t> acc((size_t)bits * full_w, 0);
			std::vector<uint64_t> scalar((size_t)s_w),
					prod((size_t)full_w);
			for (uint32_t k = 0; k < b; k++) {
				uint32_t kw = k >> 6, kb = k & 63;
				std::fill(scalar.begin(), scalar.end(), 0);
				int nz = 0;
				for (uint64_t c = 0; c < s_len; c++) {
					if ((srow[c * s_wpc + kw] >> kb) & 1) {
						scalar[c >> 6] |= 1ull << (c & 63);
						nz = 1;
					}
				}
				if (!nz)
					continue;
				for (uint32_t i = 0; i < bits; i++) {
					gf2x_mul_r(prod.data(), scalar.data(),
						(unsigned long)s_w,
						&planes_buf[((size_t)k * bits + i)
							* in_w],
						(unsigned long)in_w, pool);
					uint64_t *a = &acc[(size_t)i * full_w];
					for (uint64_t i2 = 0; i2 < full_w; i2++)
						a[i2] ^= prod[i2];
				}
			}
			uint64_t *orow = &out[(size_t)j * out_len * wpc_in];
			for (uint32_t i = 0; i < bits; i++) {
				const uint64_t *a = &acc[(size_t)i * full_w];
				for (uint64_t c = 0; c < out_len; c++) {
					uint64_t gc = c + w0;
					if ((a[gc >> 6] >> (gc & 63)) & 1)
						orow[c * wpc_in + (i >> 6)] |=
							1ull << (i & 63);
				}
			}
		}
		gf2x_mul_pool_clear(pool);
	}
}

/* highest nonzero coefficient across all rows + 1 */
static uint64_t own_tops(const std::vector<uint64_t> &rows,
		uint32_t b, uint32_t wpc, uint64_t len)
{
	for (uint64_t c = len; c-- > 0; ) {
		for (uint32_t r = 0; r < b; r++) {
			const uint64_t *row = &rows[(size_t)r * len * wpc];
			uint64_t orv = 0;
			for (uint32_t w = 0; w < wpc; w++)
				orv |= row[c * wpc + w];
			if (orv)
				return c + 1;
		}
	}
	return 1;
}

/* compact copy of the first ncoeff coefficients of each row */
static void own_copy_rows(std::vector<uint64_t> &dst,
		const std::vector<uint64_t> &src, uint32_t b,
		uint32_t wpc, uint64_t src_len, uint64_t ncoeff)
{
	dst.assign((size_t)b * ncoeff * wpc, 0);
	for (uint32_t r = 0; r < b; r++)
		memcpy(&dst[(size_t)r * ncoeff * wpc],
			&src[(size_t)r * src_len * wpc],
			(size_t)ncoeff * wpc * 8);
}

/*--------------------------------------------------------------------
The recursion. E: b compact rows (wpc = mb), L coefficients. pi: b
compact rows (wpc = bb), *pi_len coefficients. Basecase capacity
sizing follows bw_lingen_quad.cpp (window + shift share + spread).
--------------------------------------------------------------------*/
static int own_rec(const std::vector<uint64_t> &E, uint32_t b,
		uint32_t m, uint64_t L, uint32_t thresh, int dev,
		std::vector<uint32_t> &delta,
		std::vector<uint64_t> &pi, uint64_t &pi_len)
{
	const uint32_t mb = (m + 63) / 64;
	const uint32_t bb = (b + 63) / 64;

	if (L <= thresh) {
		uint32_t maxinit = 0;
		for (uint32_t r = 0; r < b; r++)
			maxinit = std::max(maxinit, delta[r]);
		uint64_t ncoeff = L + L * m / b + maxinit + L / 4 +
					4 * b + 1024;
		std::vector<uint64_t> F((size_t)b * ncoeff * bb, 0);
		double t0 = own_now();
		if (bwq_basecase(E.data(), L * mb, b, m, (uint32_t)L,
				(uint32_t)ncoeff, dev, delta.data(),
				delta.data(), F.data())) {
			fprintf(stderr, "BW own: basecase failed (L=%llu)\n",
				(unsigned long long)L);
			return -1;
		}
		uint64_t len = own_tops(F, b, bb, ncoeff);
		own_copy_rows(pi, F, b, bb, ncoeff, len);
		fprintf(stderr, "BW own: basecase L=%llu -> deg %llu "
			"(%.2fs)\n", (unsigned long long)L,
			(unsigned long long)(len - 1), own_now() - t0);
		pi_len = len;
		return 0;
	}

	uint64_t half = L - L / 2;
	double t0 = own_now();

	/* left half */
	std::vector<uint64_t> El((size_t)b * half * mb, 0);
	for (uint32_t r = 0; r < b; r++)
		memcpy(&El[(size_t)r * half * mb],
			&E[(size_t)r * L * mb], (size_t)half * mb * 8);
	std::vector<uint64_t> piL;
	uint64_t lenL;
	if (own_rec(El, b, m, half, thresh, dev, delta, piL, lenL))
		return -1;
	fprintf(stderr, "BW own: split L=%llu left %llu done (%.2fs)\n",
		(unsigned long long)L, (unsigned long long)half,
		own_now() - t0);

	/* middle-product residual: R'_j = (E * piL)[half : L) */
	std::vector<uint64_t> R;
	own_vecmul(R, half, L, E, mb, L, piL, bb, lenL, b);
	fprintf(stderr, "BW own: split L=%llu MP done (%.2fs)\n",
		(unsigned long long)L, own_now() - t0);

	std::vector<uint64_t> piR;
	uint64_t lenR;
	if (own_rec(R, b, m, L - half, thresh, dev, delta, piR, lenR))
		return -1;

	/* combine: pi_j = sum_k piR[k,j] * piL_k */
	std::vector<uint64_t> piC;
	own_vecmul(piC, 0, lenL + lenR, piL, bb, lenL, piR, bb, lenR, b);
	uint64_t len = own_tops(piC, b, bb, lenL + lenR);
	own_copy_rows(pi, piC, b, bb, lenL + lenR, len);
	pi_len = len;
	fprintf(stderr, "BW own: split L=%llu combine -> deg %llu "
		"(%.2fs)\n", (unsigned long long)L,
		(unsigned long long)(len - 1), own_now() - t0);
	return 0;
}

/*--------------------------------------------------------------------
vecmul self-check (BWOWN_CHK=1): small random inputs against a naive
bit-level reference, so a consistent-but-wrong product law cannot
validate itself through the recursion oracle.
--------------------------------------------------------------------*/
static void own_vecmul_selftest(void)
{
	const uint32_t B = 8, WPC = 1, IN_LEN = 100, S_LEN = 37;
	uint64_t rs = 0x9e3779b97f4a7c15ull;
	auto rnd = [&]() {
		rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
		return rs;
	};
	std::vector<uint64_t> in((size_t)B * IN_LEN * WPC),
			s((size_t)B * S_LEN * 2);
	for (auto &x : in) x = rnd();
	for (auto &x : s) x = rnd();
	std::vector<uint64_t> out;
	own_vecmul(out, 10, 90, in, WPC, IN_LEN, s, 2, S_LEN, B);
	/* naive: out_j[c] = sum_k sum_i s_{j,k}_i * in_k[c+10-i] */
	for (uint32_t j = 0; j < B; j++) {
		for (uint64_t c = 0; c < 80; c++) {
			uint64_t want = 0;
			for (uint32_t k = 0; k < B; k++)
				for (uint64_t i = 0; i < S_LEN; i++) {
					if (!((s[(size_t)j * S_LEN * 2 +
						i * 2 + (k >> 6)] >>
						(k & 63)) & 1))
						continue;
					int64_t src = (int64_t)(c + 10) -
							(int64_t)i;
					if (src < 0 || (uint64_t)src >= IN_LEN)
						continue;
					want ^= in[(size_t)k * IN_LEN * WPC +
						(uint64_t)src * WPC];
				}
			uint64_t got = out[(size_t)j * 80 * WPC +
						c * WPC];
			if (got != want) {
				fprintf(stderr, "BW own: vecmul selftest "
					"FAIL j=%u c=%llu want=%llx "
					"got=%llx\n", j,
					(unsigned long long)c,
					(unsigned long long)want,
					(unsigned long long)got);
				return;
			}
		}
	}
	fprintf(stderr, "BW own: vecmul selftest ok\n");
}

/*--------------------------------------------------------------------
Full pipeline: A file -> F file (same file formats as the quad route).
--------------------------------------------------------------------*/
int bw_lingen_own_full(const char *afile, const char *ffile,
			uint32_t m, uint32_t n, uint64_t seq_len,
			uint32_t nrows, uint32_t ncols, int dev,
			uint32_t nthreads, uint32_t threshold)
{
	if (getenv("BWOWN_CHK"))
		own_vecmul_selftest();

	if (nthreads > 0) {
		char nt[16];
		snprintf(nt, sizeof(nt), "%u", nthreads);
		setenv("OMP_NUM_THREADS", nt, 0);
	}

	/* ingest the A file */
	FILE *f = fopen(afile, "rb");
	if (!f) {
		fprintf(stderr, "BW own: cannot open %s\n", afile);
		return -1;
	}
	bw_filehdr_t hdr;
	if (fread(&hdr, 1, 64, f) != 64 || hdr.magic != BW_A_MAGIC ||
			hdr.m != m || hdr.n != n) {
		fprintf(stderr, "BW own: bad A header\n");
		fclose(f);
		return -1;
	}
	const uint64_t L = hdr.seq_len ? hdr.seq_len : seq_len;
	std::vector<uint64_t> A((size_t)L * m);
	if (fread(A.data(), 8, (size_t)L * m, f) != (size_t)L * m) {
		fprintf(stderr, "BW own: A truncated\n");
		fclose(f);
		return -1;
	}
	fclose(f);

	/* augmentation selection and E window (same as the quad route) */
	const uint32_t maug = m;
	std::vector<uint32_t> kk(maug), jj(maug);
	int t0 = bwq_select(A.data(), m, maug, n, L, kk.data(), jj.data());
	if (t0 < 0) {
		fprintf(stderr, "BW own: selection failed\n");
		return -1;
	}
	fprintf(stderr, "BW own: t0=%d\n", t0);
	const uint32_t b = n + maug;
	const uint32_t mb = (m + 63) / 64;
	const uint64_t LE = L - (uint64_t)t0;

	/* build E: b compact rows (wpc = mb), LE coefficients */
	std::vector<uint64_t> E((size_t)b * LE * mb, 0);
	for (uint64_t c = 0; c < LE; c++) {
		for (uint32_t r = 0; r < n; r++) {
			const uint64_t *Ak = A.data() + (c + (uint64_t)t0) * m;
			uint64_t *dst = &E[(size_t)r * LE * mb +
						(uint64_t)c * mb];
			for (uint32_t i = 0; i < m; i++)
				if ((Ak[i] >> r) & 1)
					dst[i >> 6] |= 1ull << (i & 63);
		}
		for (uint32_t r = 0; r < maug; r++) {
			const uint64_t *Ak = A.data() +
					(c + (uint64_t)kk[r]) * m;
			uint64_t *dst = &E[(size_t)(n + r) * LE * mb +
						(uint64_t)c * mb];
			for (uint32_t i = 0; i < m; i++)
				if ((Ak[i] >> jj[r]) & 1)
					dst[i >> 6] |= 1ull << (i & 63);
		}
	}

	/* the recursion */
	std::vector<uint64_t> pi;
	uint64_t pi_len;
	std::vector<uint32_t> delta0(b, 0);
	if (own_rec(E, b, m, LE, threshold, dev, delta0, pi,
			pi_len)) {
		fprintf(stderr, "BW own: recursion failed\n");
		return -1;
	}

	if (getenv("BWOWN_CHK")) {
		/* oracle: (E * pi) must vanish on [0, LE) */
		std::vector<uint64_t> chk;
		own_vecmul(chk, 0, LE, E, mb, LE, pi,
				(b + 63) / 64, pi_len, b);
		uint64_t nbad = 0, first = ~0ull;
		for (uint64_t c = 0; c < LE; c++)
			for (uint32_t r = 0; r < b; r++) {
				uint64_t orv = 0;
				for (uint32_t w = 0; w < mb; w++)
					orv |= chk[(size_t)r * LE * mb +
						c * mb + w];
				if (orv) {
					nbad++;
					if (first == ~0ull)
						first = c;
				}
			}
		fprintf(stderr, "BW own: oracle %llu/%llu bad coeffs "
			"(first %llu)\n", (unsigned long long)nbad,
			(unsigned long long)(LE * b),
			first == ~0ull ? ~0ull : first);
		/* second oracle: A*W for every row via the emit formula */
		const uint32_t bb2 = (b + 63) / 64;
		uint64_t nbad2 = 0, first2 = ~0ull;
		for (uint32_t r = 0; r < b; r++) {
			const uint64_t *pr = &pi[(size_t)r * pi_len * bb2];
			for (uint64_t d = (uint64_t)t0; d < L; d++) {
				uint64_t acc = 0;   /* m=64 assumption for the
						   * probe; fine for tests */
				for (uint64_t c = 0; c <= d; c++) {
					uint64_t Wv = 0;
					if (c < pi_len) {
						uint64_t fw = pr[c * bb2];
						while (fw) {
							uint32_t j2 =
							 __builtin_ctzll(fw);
							if (j2 < n)
								Wv |= 1ull << j2;
							fw &= fw - 1;
						}
						for (uint32_t rp = 0; rp < maug; rp++) {
							int64_t cc = (int64_t)c -
								t0 + kk[rp];
							if (cc < 0 || (uint64_t)cc >= pi_len)
								continue;
							if ((pr[(uint64_t)cc * bb2 +
								((n + rp) >> 6)] >>
								((n + rp) & 63)) & 1)
								Wv ^= 1ull << jj[rp];
						}
					}
					if (!Wv || d - c >= L)
						continue;
					const uint64_t *Ak = &A.data()[(d - c) * m];
					while (Wv) {
						uint32_t j2 = __builtin_ctzll(Wv);
						uint64_t col = 0;
						for (uint32_t i2 = 0; i2 < m; i2++)
							if ((Ak[i2] >> j2) & 1)
								col |= 1ull << i2;
						acc ^= col;
						Wv &= Wv - 1;
					}
				}
				if (acc) {
					nbad2++;
					if (first2 == ~0ull)
						first2 = d;
				}
			}
		}
		fprintf(stderr, "BW own: A*W oracle %llu bad (first %llu)\n",
			(unsigned long long)nbad2,
			first2 == ~0ull ? ~0ull : first2);
	}

	/* every pi row annihilates the window: select by actual row
	 * degree (ascending) plus W independence; rows that froze early
	 * through the recursion carry the small degrees, which keeps the
	 * emitted W's low-side junk below the declared file degree */
	const uint32_t bb2 = (b + 63) / 64;
	std::vector<int> lucky(b, 1);
	for (uint32_t r = 0; r < b; r++) {
		const uint64_t *pr = &pi[(size_t)r * pi_len * bb2];
		uint32_t top = 0;
		for (uint64_t c = pi_len; c-- > 0; ) {
			uint64_t orv = 0;
			for (uint32_t w = 0; w < bb2; w++)
				orv |= pr[c * bb2 + w];
			if (orv) {
				top = (uint32_t)c + 1;
				break;
			}
		}
		lucky[r] = 0x40000000 - (int)top;
	}
	return bwq_emit(pi.data(), pi_len * bb2, b, m, n, maug,
			t0, kk.data(), jj.data(), lucky.data(), pi_len, 1,
			ffile, nrows, ncols);
}
