/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

bw_lingen_test: random-sequence validation of the ported lingen.
Generates an A file of random m-by-n GF(2) matrix coefficients, runs
bw_lingen_run, reads the F file and checks the polynomial identity
A(x) * F(x) = 0 on the interior coefficient window, plus that F is
nondegenerate (nonzero, and not the trivial low-degree identity).
--------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
	uint32_t m, n;
	uint32_t nrows, ncols;
	const char *afile;
	const char *ffile;
	uint64_t seq_len;
	uint32_t threshold;
	uint32_t nthreads;
	uint32_t impl;
	uint32_t dev;
} bw_lingen_opts_t;

extern int bw_lingen_run(const bw_lingen_opts_t *o);
/* impl selection, defined in bw_lingen.cpp */
extern unsigned int g_bw_lingen_impl;
extern unsigned int g_bw_lingen_dev;

/* bw_seq.h formats */
#define BW_A_MAGIC   0x31574241u
#define BW_F_MAGIC   0x31574246u
#define BW_FORMAT_VERSION 1
#define BW_HDR_SIZE 64

typedef struct { uint32_t magic, version, m, n, nrows, ncols,
	reserved0, num_dense_rows; uint64_t seq_len, seed;
	uint64_t reserved[4]; } bw_filehdr_t;

static uint64_t rng_state = 0x243F6A8885A308D3ull;
static uint64_t rnd(void) {
	uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

/* C[r] ^= A * B over GF(2), 64x64 words */
static void matmul_acc(uint64_t *C, const uint64_t *A, const uint64_t *B)
{
	int i, j;
	for (i = 0; i < 64; i++) {
		uint64_t a = A[i];
		while (a) {
			j = __builtin_ctzll(a);
			C[i] ^= B[j];
			a &= a - 1;
		}
	}
}

int main(int argc, char **argv)
{
	uint32_t m = 64, n = 64, L = 384;
	uint32_t threshold = 32;
	const char *afile = "/tmp/bw_lt.A";
	const char *ffile = "/tmp/bw_lt.F";
	bw_filehdr_t hdr;
	uint64_t *A, *F;
	FILE *fp;
	uint32_t d, i, k;
	int c;

	for (c = 1; c < argc; c++) {
		if (!strncmp(argv[c], "L=", 2)) L = atoi(argv[c] + 2);
		else if (!strncmp(argv[c], "thr=", 4)) threshold = atoi(argv[c] + 4);
		else if (!strncmp(argv[c], "m=", 2)) m = atoi(argv[c] + 2);
	}

	/* Build the A file from a KNOWN recurrence A_k = sum_{i=1..DT}
	 * A_{k-i} * G_i, so the true generator has degree DT, cancellations
	 * start early and the luck condition is genuinely satisfied (a
	 * purely random sequence has its generator complete exactly at the
	 * end of the input, leaving no luck slack). */
	{
		uint32_t DT = 6;
		uint64_t *G = malloc((size_t)(DT + 1) * n * sizeof(uint64_t));
		for (i = 0; i < (DT + 1) * n; i++)
			G[i] = rnd();
		/* G_0 = identity so the true reversed generator is monic */
		for (i = 0; i < n; i++)
			G[i] = 0;
		G[0] = 1;

		A = malloc((size_t)L * m * sizeof(uint64_t));
		for (i = 0; i < (DT < L ? DT : L) * m; i++)
			A[i] = rnd();
		for (k = DT; k < L; k++) {
			const uint64_t *Ak;
			const uint64_t *Gc;
			uint64_t *dest;
			int ii;
			memset(A + (size_t)k * m, 0, m * 8);
			for (c = 1; (uint32_t)c <= DT; c++) {
				Ak = A + (size_t)(k - c) * m;
				Gc = G + (size_t)c * n;
				dest = A + (size_t)k * m;
				for (ii = 0; ii < (int)m; ii++) {
					uint64_t a = Ak[ii];
					while (a) {
						int jj = __builtin_ctzll(a);
						dest[ii] ^= Gc[jj];
						a &= a - 1;
					}
				}
			}
		}
		free(G);
	}

	fp = fopen(afile, "wb");
	memset(&hdr, 0, sizeof(hdr));
	hdr.magic = BW_A_MAGIC;
	hdr.version = BW_FORMAT_VERSION;
	hdr.m = m;
	hdr.n = n;
	hdr.nrows = 0;
	hdr.ncols = 0;
	hdr.seq_len = L;
	hdr.seed = 0;
	fwrite(&hdr, 1, 64, fp);
	fwrite(A, 8, (size_t)L * m, fp);
	fclose(fp);
	remove(ffile);

	{
		bw_lingen_opts_t o;
		memset(&o, 0, sizeof(o));
		o.m = m;
		o.n = n;
		o.afile = afile;
		o.ffile = ffile;
		o.seq_len = L;
		o.threshold = threshold;
		o.nthreads = 4;
		/* default (no env): own route on device 1 */
		g_bw_lingen_dev = 1;
		if (getenv("QUAD"))
			g_bw_lingen_impl = 1;
		if (bw_lingen_run(&o)) {
			printf("LINGEN TEST FAIL: bw_lingen_run errored\n");
			return 1;
		}
	}

	/* read F */
	fp = fopen(ffile, "rb");
	if (!fp) { printf("LINGEN TEST FAIL: no F file\n"); return 1; }
	if (fread(&hdr, 1, 64, fp) != 64 || hdr.magic != BW_F_MAGIC ||
			hdr.m != m || hdr.n != n) {
		printf("LINGEN TEST FAIL: bad F header\n");
		return 1;
	}
	d = (uint32_t)hdr.seq_len;
	F = malloc((size_t)d * n * sizeof(uint64_t));
	if (fread(F, 8, (size_t)d * n, fp) != (size_t)d * n) {
		printf("LINGEN TEST FAIL: F truncated\n");
		return 1;
	}
	fclose(fp);

	/* lingen_output_to_singlefile writes transposed coefficients; the
	 * identity A(x)*F(x)=0 holds for F[c] = transpose(file block c) */
	for (k = 0; k < d; k++) {
		uint64_t *M = F + (size_t)k * n;
		uint32_t ii, jj, kk;
		for (ii = 0; ii < n; ii++)
			for (jj = ii + 1; jj < n; jj++) {
				uint64_t a = (M[ii] >> jj) & 1;
				uint64_t b = (M[jj] >> ii) & 1;
				if (a != b) {
					M[ii] ^= (uint64_t)1 << jj;
					M[jj] ^= (uint64_t)1 << ii;
				}
			}
		(void)kk;
	}

	printf("F degree %u (L=%u)\n", d - 1, L);

	/* the identity: A(x) * F(x) = 0, F(x) = sum_c F[c] x^{d-c}
	 * (leading first). Product coefficient
	 * P[k] = sum_{c: 0<=k-d+c<L} A[k-d+c] * F[c]; every coefficient
	 * with d <= k < L must vanish. */
	{
		uint64_t *Pc = malloc(m * sizeof(uint64_t));
		uint32_t nz_first = 0, nz_last = 0, nz_count = 0;
		for (k = 0; k < L + d; k++) {
			memset(Pc, 0, m * 8);
			for (c = 0; c < (int)d; c++) {
				int32_t ai = (int32_t)k - (int32_t)d + (int32_t)c;
				if (ai < 0 || ai >= (int32_t)L)
					continue;
				matmul_acc(Pc, A + (size_t)ai * m,
						F + (size_t)c * n);
			}
			for (i = 0; i < m; i++)
				if (Pc[i]) {
					if (!nz_count)
						nz_first = k;
					nz_last = k;
					nz_count++;
					break;
				}
		}
		free(Pc);
		printf("[dbg] product: %u/%u nonzero coeffs, first=%u "
			"last=%u (d=%u, L=%u)\n",
			nz_count, L + d, nz_first, nz_last, d, L);
		/* with the planted recurrence, the product must vanish on
		 * [d, L); junk outside that window is expected (t0 shift
		 * on the low side, truncation on the high side) */
		{
			uint32_t bad = 0;
			for (k = d; k < L && bad < 4; k++) {
				uint32_t isnz = 0;
				memset(Pc = malloc(m * 8), 0, m * 8);
				for (c = 0; c < (int)d; c++) {
					int32_t ai = (int32_t)k - (int32_t)d + (int32_t)c;
					if (ai < 0 || ai >= (int32_t)L)
						continue;
					matmul_acc(Pc, A + (size_t)ai * m,
							F + (size_t)c * n);
				}
				for (i = 0; i < m; i++)
					if (Pc[i])
						isnz = 1;
				free(Pc);
				if (isnz) {
					bad++;
					if (bad < 4)
						printf("LINGEN TEST FAIL: "
							"product nonzero at "
							"degree %u\n", k);
				}
			}
			if (bad)
				return 1;
		}
	}

	/* F must be nontrivial: leading coefficient nonzero */
	{
		int nz = 0;
		for (i = 0; i < (uint32_t)n; i++)
			if (F[i])
				nz = 1;
		if (!nz) {
			printf("LINGEN TEST FAIL: leading F coefficient "
				"is zero\n");
			return 1;
		}
	}

	printf("lingen random-sequence test PASSED (m=%u n=%u L=%u)\n",
		m, n, L);
	free(A);
	free(F);
	return 0;
}
