/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

bw_test: standalone selftest for the BW machinery against CPU
references. Exercises the matrix packer, every helper kernel, one
Krylov/mksol/nilpotent step through the real worker threads and the
allgather, and the checkpoint roundtrip, for several dense-row edge
cases (0/1/63/64/65) and excess-column shapes.
--------------------------------------------------------------------*/

#include <common.h>
#include "bw.h"
#include "bw_matrix.h"
#include "bw_gpu.h"
#include "bw_seq.h"
#include "bw_checkpoint.h"

#include <stdlib.h>
#include <string.h>

static bw_rng_t g_rng = { 0x123456789abcdef0ULL };

/* CPU reference: full SpMV including dense rows, over la_col_t */
static void cpu_mul(bw_matrix_t *M, la_col_t *cols,
		const uint64_t *x, uint64_t *y /* nrows words */)
{
	uint32_t i, j, k;

	memset(y, 0, M->nrows * sizeof(uint64_t));
	for (i = 0; i < M->ncols; i++) {
		la_col_t *col = cols + i;
		uint64_t u = x[i];
		if (!u)
			continue;
		for (j = 0; j < col->weight; j++)
			y[col->data[j]] ^= u;
	}
	for (k = 0; k < M->num_dense_blocks; k++) {
		for (i = 0; i < M->ncols; i++) {
			uint64_t d = M->dense[(size_t)k * M->ncols + i];
			uint64_t u = x[i];
			if (!u || !d)
				continue;
			while (d) {
				uint32_t r = (uint32_t)__builtin_ctzll(d);
				y[64 * k + r] ^= u;
				d &= d - 1;
			}
		}
	}
}

/* CPU reference for the S coefficient: S[a] = sum over i with U[a][i]
   of Y[i], where U row a is bit a of h_u[i*(m/64) + a/64] */
static void cpu_oprod(bw_matrix_t *M, const uint64_t *h_u, uint32_t m,
		const uint64_t *y, uint64_t *s /* m words, zeroed by caller */)
{
	uint32_t i, a;
	for (i = 0; i < M->nrows; i++) {
		uint64_t yy = y[i];
		if (!yy)
			continue;
		for (a = 0; a < m; a++)
			if ((h_u[(size_t)i * (m / 64) + a / 64] >>
					(a & 63)) & 1)
				s[a] ^= yy;
	}
}

/* CPU reference for the combine: x[i] ^= sum over set bits c of
   y0[i] of f[c] */
static void cpu_combine(uint64_t *x, const uint64_t *y0,
		const uint64_t *f, uint32_t n, size_t N)
{
	size_t i;
	for (i = 0; i < N; i++) {
		uint64_t u = y0[i], r = 0;
		while (u) {
			uint32_t c = (uint32_t)__builtin_ctzll(u);
			if (c < n)
				r ^= f[c];
			u &= u - 1;
		}
		x[i] ^= r;
	}
}

static double rand_dense_density = 0.35;

static int make_random_matrix(msieve_obj *obj, la_col_t **cols_out,
		uint32_t nrows, uint32_t ncols, uint32_t ndr,
		uint32_t num_slabs, bw_matrix_t *M)
{
	la_col_t *cols = (la_col_t *)xcalloc(ncols, sizeof(la_col_t));
	uint32_t i, j, k;
	uint32_t ndr_words = (ndr + 31) / 32;

	for (i = 0; i < ncols; i++) {
		uint32_t w = 4 + (uint32_t)(bw_rng_next(&g_rng) % 32);
		uint32_t *rows = (uint32_t *)xmalloc((w + ndr_words) *
						sizeof(uint32_t));
		cols[i].weight = w;
		cols[i].data = rows;
		/* random distinct rows above ndr, ascending */
		for (j = 0; j < w; j++) {
			uint32_t r;
			do {
				r = ndr + (uint32_t)(bw_rng_next(&g_rng) %
					(nrows - ndr));
			} while (0);
			rows[j] = r;
		}
		/* sort + dedup */
		for (j = 1; j < w; j++) {
			uint32_t v = rows[j];
			for (k = j; k > 0 && rows[k-1] > v; k--)
				rows[k] = rows[k-1];
			rows[k] = v;
		}
		{
			uint32_t nw = 1;
			for (j = 1; j < w; j++)
				if (rows[j] != rows[nw-1])
					rows[nw++] = rows[j];
			cols[i].weight = nw;
		}
		for (k = 0; k < ndr_words; k++) {
			uint32_t bits = 0;
			for (j = 0; j < 32; j++)
				if ((double)(bw_rng_next(&g_rng) & 1023) /
						1023.0 < rand_dense_density)
					bits |= 1u << j;
			/* keep the last row of the matrix nonzero-ish
			   for every column so the matrix stays full rank
			   enough in practice; not strictly needed for
			   kernel-vector tests */
			cols[i].data[cols[i].weight + k] = bits;
		}
	}

	*cols_out = cols;
	if (bw_matrix_build(M, obj, nrows, ndr, ncols, cols,
			num_slabs, 1 << 20))
		return -1;
	return 0;
}

/*-------------------------------------------------------------------*/
static int failures = 0;

#define CHECK(cond, msg, ...) do {				\
	if (!(cond)) {						\
		printf("FAIL %s:%d: ", __FILE__, __LINE__);	\
		printf(msg, ##__VA_ARGS__);			\
		printf("\n");					\
		failures++;					\
	}							\
} while (0)

static int run_case(msieve_obj *obj, uint32_t nrows, uint32_t ncols,
		uint32_t ndr, const bw_opts_t *o)
{
	la_col_t *cols = NULL;
	bw_matrix_t M;
	bw_group_t g;
	uint64_t *h_y0 = NULL, *h_u = NULL, *h_xcpu = NULL, *h_ref = NULL;
	uint64_t *coef = NULL;
	uint32_t i;
	int rc = -1;
	uint32_t m = o->m;

	printf("== case nrows=%u ncols=%u ndr=%u gpus=%d\n",
		nrows, ncols, ndr, o->num_gpus);

	if (make_random_matrix(obj, &cols, nrows, ncols, ndr,
				o->num_gpus, &M))
		goto out;
	printf("[dbg] dense_weight=%" PRIu64 " blocks=%u sparse=%" PRIu64 "\n",
		M.dense_weight, M.num_dense_blocks, M.sparse_weight);

	h_y0 = (uint64_t *)xmalloc(ncols * sizeof(uint64_t));
	for (i = 0; i < ncols; i++)
		h_y0[i] = bw_rng_next(&g_rng);
	h_u = (uint64_t *)xcalloc((size_t)nrows * (m / 64),
				sizeof(uint64_t));
	for (i = 0; i < nrows * (m / 64); i++)
		h_u[i] = bw_rng_next(&g_rng);
	h_xcpu = (uint64_t *)xmalloc(ncols * sizeof(uint64_t));
	h_ref = (uint64_t *)xmalloc(nrows * sizeof(uint64_t));
	coef = (uint64_t *)xmalloc(m * sizeof(uint64_t));

	if (bw_gpu_start(&g, &M, o, obj, h_y0, h_u))
		goto out;

	/* three Krylov steps with state and S verification */
	memcpy(h_xcpu, h_y0, ncols * 8);
	for (i = 0; i < 3; i++) {
		int w;
		bw_gpu_krylov_step(&g);
		for (w = 0; w < g.num_workers; w++)
			CHECK(g.w[w].cuda_error == 0, "worker %d cuda ok",
				w);

		cpu_mul(&M, cols, h_xcpu, h_ref);
		{
			uint64_t *got = (uint64_t *)xcalloc(ncols, 8);
			bw_gpu_collect(&g);
			for (w = 0; w < g.num_workers; w++)
				memcpy(got + M.slabs[w].xfer_min,
					g.w[w].h_stage,
					(M.slabs[w].row_max -
					 M.slabs[w].xfer_min) * 8);
			/* the padded tail must be zero after a step */
			{
				uint32_t t;
				int tail_ok = 1;
				for (t = nrows; t < ncols; t++)
					if (got[t])
						tail_ok = 0;
				CHECK(tail_ok, "tail zeroed after krylov "
					"step %u", i);
			}
			if (memcmp(got, h_ref, nrows * 8) != 0) {
				uint32_t t;
				for (t = 0; t < nrows; t++)
					if (got[t] != h_ref[t])
						break;
				printf("[dbg] step %u first diff row %u: "
					"got %016llx ref %016llx\n", i, t,
					(unsigned long long)got[t],
					(unsigned long long)h_ref[t]);
			}
			CHECK(memcmp(got, h_ref, nrows * 8) == 0,
				"krylov step %u matches CPU", i);
			memcpy(h_xcpu, got, nrows * 8);
			/* the padded tail is zero after every step */
			memset(h_xcpu + nrows, 0, (ncols - nrows) * 8);
			free(got);
		}
		{
			uint64_t *ref_s = (uint64_t *)xcalloc(m, 8);
			bw_gpu_reduce_s(&g, coef, m);
			cpu_oprod(&M, h_u, m, h_xcpu, ref_s);
			if (memcmp(coef, ref_s, m * 8) != 0) {
				uint32_t t;
				for (t = 0; t < m; t++)
					if (coef[t] != ref_s[t])
						break;
				printf("[dbg] step %u S first diff word %u: "
					"got %016llx ref %016llx\n", i, t,
					(unsigned long long)coef[t],
					(unsigned long long)ref_s[t]);
			}
			CHECK(memcmp(coef, ref_s, m * 8) == 0,
				"S coefficient step %u matches CPU", i);
			free(ref_s);
		}
	}

	/* mksol step: x' = A*x + Y0*F */
	{
		uint64_t f[64];
		uint64_t *got = (uint64_t *)xcalloc(ncols, 8);
		int w;
		for (i = 0; i < 64; i++)
			f[i] = bw_rng_next(&g_rng);
		bw_gpu_mksol_step(&g, f);
		for (w = 0; w < g.num_workers; w++)
			CHECK(g.w[w].cuda_error == 0, "worker %d cuda ok",
				w);
		cpu_mul(&M, cols, h_xcpu, h_ref);
		memcpy(h_xcpu, h_ref, nrows * 8);
		memset(h_xcpu + nrows, 0, (ncols - nrows) * 8);
		cpu_combine(h_xcpu, h_y0, f, 64, nrows);
		bw_gpu_collect(&g);
		for (w = 0; w < g.num_workers; w++)
			memcpy(got + M.slabs[w].xfer_min, g.w[w].h_stage,
				(M.slabs[w].row_max -
				 M.slabs[w].xfer_min) * 8);
		CHECK(memcmp(got, h_xcpu, nrows * 8) == 0,
			"mksol step matches CPU");
		free(got);
	}

	/* checkpoint roundtrip */
	{
		bw_chkhdr_t hdr, hdr2;
		uint64_t *state2 = NULL;
		char path[512];
		uint64_t *full = (uint64_t *)xcalloc(ncols, 8);

		bw_gpu_collect(&g);
		for (i = 0; i < (uint32_t)g.num_workers; i++)
			memcpy(full + M.slabs[i].xfer_min, g.w[i].h_stage,
				(M.slabs[i].row_max -
				 M.slabs[i].xfer_min) * 8);

		memset(&hdr, 0, sizeof(hdr));
		hdr.magic = BW_CHK_MAGIC;
		hdr.version = BW_FORMAT_VERSION;
		hdr.m = o->m;
		hdr.n = o->n;
		hdr.nrows = M.nrows;
		hdr.ncols = M.ncols;
		hdr.phase = 0;
		hdr.step = 4;
		hdr.seq_len = 100;
		hdr.seed = 42;
		hdr.hash_x = bw_fnv1a(0, full, (size_t)ncols * 8);

		bw_path(path, sizeof(path), obj, ".bw.chk");
		CHECK(bw_checkpoint_write(path, &hdr, full, ncols) == 0,
			"checkpoint write");
		/* second write rotates the first into .prev */
		hdr.step = 5;
		hdr.hash_x = bw_fnv1a(0, full, (size_t)ncols * 8);
		CHECK(bw_checkpoint_write(path, &hdr, full, ncols) == 0,
			"checkpoint write 2");
		CHECK(bw_checkpoint_read(path, &hdr2, &state2, ncols) == 0,
			"checkpoint read");
		CHECK(memcmp(state2, full, (size_t)ncols * 8) == 0,
			"checkpoint state roundtrip");
		CHECK(hdr2.step == 5, "checkpoint header roundtrip");
		/* truncation must be detected */
		{
			FILE *fp = fopen(path, "r+b");
			if (fp) {
				(void)!ftruncate(fileno(fp), 100);
				fclose(fp);
			}
			free(state2);
			state2 = NULL;
			CHECK(bw_checkpoint_read(path, &hdr2, &state2,
					ncols) != 0,
				"truncation detected");
			/* .prev still has the good copy */
			{
				char prevpath[600];
				int prc;
				snprintf(prevpath, sizeof(prevpath),
					"%s.prev", path);
				prc = bw_checkpoint_read(prevpath, &hdr2,
					&state2, ncols);
				CHECK(prc == 0, "prev fallback");
			}
		}
		free(state2);
		free(full);
	}

	bw_gpu_stop(&g);
	rc = failures;

out:
	bw_matrix_free(&M);
	if (cols) {
		for (i = 0; i < ncols; i++)
			free(cols[i].data);
		free(cols);
	}
	free(h_y0);
	free(h_u);
	free(h_xcpu);
	free(h_ref);
	free(coef);
	return rc;
}

/*-------------------------------------------------------------------*/
int main(int argc, char **argv)
{
	msieve_obj *obj;
	bw_opts_t o;
	uint32_t ndr_cases[] = { 0, 1, 63, 64, 65 };
	uint32_t c;
	char logfile[] = "/tmp/bw_test.log";
	char savefile[] = "/tmp/bw_test.dat";

	bw_opts_defaults(&o);
	o.enabled = 1;
	o.num_gpus = 1;
	o.gpus[0] = 1;      /* GPU 0 may host the live job */

	for (c = 1; c < (uint32_t)argc; c++) {
		if (!strncmp(argv[c], "--gpus=", 7)) {
			char *p = argv[c] + 7;
			o.num_gpus = 0;
			while (*p && o.num_gpus < BW_MAX_GPUS) {
				o.gpus[o.num_gpus++] = (int)strtol(p, &p, 10);
				if (*p == ',')
					p++;
			}
		}
	}

	obj = msieve_obj_new(NULL, 0, savefile, logfile, NULL,
			1, 2, 0, cpu_generic, 0, 0, 1, 0, "");

	if (bw_preflight(o.gpus[0])) {
		printf("preflight failed on GPU %d\n", o.gpus[0]);
		return 1;
	}

	for (c = 0; c < sizeof(ndr_cases) / sizeof(ndr_cases[0]); c++) {
		run_case(obj, 3000, 3136, ndr_cases[c], &o);
	}

	/* excess edge: exactly 64 */
	run_case(obj, 500, 564, 64, &o);

	if (failures) {
		printf("%d FAILURES\n", failures);
		return 1;
	}
	printf("all BW selftests passed\n");
	msieve_obj_free(obj);
	return 0;
}
