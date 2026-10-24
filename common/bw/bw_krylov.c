/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

Krylov sequence generation: run L steps of x <- A*x, reducing the
m-word S coefficient per step into the A file, with checkpoints and
interrupt-safe restart.
--------------------------------------------------------------------*/

#include <common.h>
#include "bw_gpu.h"
#include "bw_matrix.h"
#include "bw_seq.h"
#include "bw_checkpoint.h"
#include "bw.h"

#include <time.h>

/* entry from bw.c */
int bw_run_krylov(bw_group_t *g, msieve_obj *obj, const bw_opts_t *o,
		uint64_t seed, uint64_t hash_y0, uint64_t hash_u,
		uint32_t L, uint32_t interval, int have_resume,
		const bw_chkhdr_t *resume, uint32_t *stop_requested);

/* local */
static int krylov_checkpoint(bw_group_t *g, msieve_obj *obj,
			const bw_opts_t *o, uint64_t seed,
			uint64_t hash_y0, uint64_t hash_u,
			uint64_t hash_s, uint64_t step, uint64_t a_offset,
			uint32_t L, const char *chk_path, FILE *a_fp,
			uint64_t *scratch);

int bw_run_krylov(bw_group_t *g, msieve_obj *obj, const bw_opts_t *o,
		uint64_t seed, uint64_t hash_y0, uint64_t hash_u,
		uint32_t L, uint32_t interval, int have_resume,
		const bw_chkhdr_t *resume, uint32_t *stop_requested)
{
	char apath[512], chk_path[512];
	bw_filehdr_t ahdr;
	FILE *a_fp = NULL;
	uint64_t *coef = NULL, *scratch = NULL;
	uint64_t step0 = 0, hash_s = 0;
	uint32_t j;
	int rc = -1;
	time_t last_chk = time(NULL);
	time_t t0 = time(NULL);
	uint64_t steps_done = 0;

	bw_path(apath, sizeof(apath), obj, ".bw.A");
	bw_path(chk_path, sizeof(chk_path), obj, ".bw.chk");

	memset(&ahdr, 0, sizeof(ahdr));
	ahdr.magic = BW_A_MAGIC;
	ahdr.version = BW_FORMAT_VERSION;
	ahdr.m = o->m;
	ahdr.n = o->n;
	ahdr.nrows = g->M->nrows;
	ahdr.ncols = g->M->ncols;
	ahdr.num_dense_rows = g->M->num_dense_rows;
	ahdr.seq_len = L;
	ahdr.seed = seed;

	if (have_resume && resume->phase == 0) {
		a_fp = fopen(apath, "r+b");
		if (a_fp == NULL) {
			logprintf(obj, "error: BW checkpoint refers to a "
					"missing A file\n");
			goto out;
		}
		/* the state upload happened in bw.c */
		step0 = resume->step;
		hash_s = resume->hash_s;
		if (fseek(a_fp, (long)resume->a_offset, SEEK_SET)) {
			logprintf(obj, "error: BW A file shorter than the "
					"checkpoint offset\n");
			goto out;
		}
	}
	else {
		a_fp = fopen(apath, "wb");
		if (a_fp == NULL) {
			logprintf(obj, "error: cannot open %s\n", apath);
			goto out;
		}
		bw_put_hdr(a_fp, &ahdr);
	}

	coef = (uint64_t *)xmalloc(o->m * sizeof(uint64_t));
	scratch = (uint64_t *)xmalloc((size_t)g->N * sizeof(uint64_t));

	logprintf(obj, "BW krylov: L=%u steps from %llu%s\n",
			L, (unsigned long long)step0,
			step0 ? " (resumed)" : "");

	for (j = (uint32_t)step0; j < L; j++) {
		bw_gpu_krylov_step(g);

		{
			int w;
			memset(coef, 0, o->m * sizeof(uint64_t));
			for (w = 0; w < g->num_workers; w++)
				if (g->w[w].cuda_error) {
					logprintf(obj, "error: BW worker %d "
						"CUDA failure\n", w);
					goto out;
				}
			for (w = 0; w < g->num_workers; w++) {
				uint32_t i;
				for (i = 0; i < o->m; i++)
					coef[i] ^= g->w[w].h_s[i];
			}
		}

		if (fwrite(coef, sizeof(uint64_t), o->m, a_fp) != o->m) {
			logprintf(obj, "error: BW A file write failed\n");
			goto out;
		}
		hash_s = bw_fnv1a(hash_s, coef, o->m * sizeof(uint64_t));
		steps_done++;

		if (time(NULL) - last_chk >= (time_t)o->checkpoint_s ||
				obj->flags & MSIEVE_FLAG_STOP_SIEVING ||
				j + 1 == L) {
			uint64_t off = (uint64_t)BW_HDR_SIZE +
					((uint64_t)j + 1) * o->m * 8;
			fflush(a_fp);
			if (krylov_checkpoint(g, obj, o, seed, hash_y0,
					hash_u, hash_s, j + 1, off, L,
					chk_path, a_fp, scratch))
				goto out;
			last_chk = time(NULL);
		}

		if (obj->flags & MSIEVE_FLAG_STOP_SIEVING) {
			*stop_requested = 1;
			rc = 1;    /* clean stop; resume later */
			goto out;
		}

		if (steps_done % (uint64_t)interval == 0 ||
				(steps_done < 20 && steps_done % 1 == 0)) {
			double el = difftime(time(NULL), t0);
			double rate = steps_done / (el > 0 ? el : 1);
			double eta = (L - (step0 + steps_done)) /
					(rate > 0 ? rate : 1);
			logprintf(obj, "BW krylov: step %u/%u "
					"(%0.1f steps/s, ETA %uh%02um)\n",
					j + 1, L, rate,
					(unsigned)(eta / 3600),
					(unsigned)((eta / 60)) % 60);
		}
	}

	if (fflush(a_fp) || fsync(fileno(a_fp))) {
		logprintf(obj, "error: cannot finalize BW A file\n");
		goto out;
	}
	rc = 0;

	bw_gpu_time_report(g, obj, "krylov");
	bw_gpu_time_reset(g);

out:
	if (a_fp)
		fclose(a_fp);
	free(coef);
	free(scratch);
	return rc;
}

/*-------------------------------------------------------------------*/
static int krylov_checkpoint(bw_group_t *g, msieve_obj *obj,
			const bw_opts_t *o, uint64_t seed,
			uint64_t hash_y0, uint64_t hash_u,
			uint64_t hash_s, uint64_t step, uint64_t a_offset,
			uint32_t L, const char *chk_path, FILE *a_fp,
			uint64_t *scratch)
{
	bw_chkhdr_t chk;
	int w;
	uint64_t h = 0;

	/* collect the current state slab by slab */
	bw_gpu_collect(g);
	for (w = 0; w < g->num_workers; w++)
		if (g->w[w].cuda_error) {
			logprintf(obj, "error: BW worker %d CUDA failure "
					"during checkpoint\n", w);
			return -1;
		}
	bw_gpu_merge_state(g, scratch);

	h = bw_fnv1a(0, scratch, (size_t)g->N * 8);

	memset(&chk, 0, sizeof(chk));
	chk.magic = BW_CHK_MAGIC;
	chk.version = BW_FORMAT_VERSION;
	chk.m = o->m;
	chk.n = o->n;
	chk.nrows = g->M->nrows;
	chk.ncols = g->M->ncols;
	chk.num_dense_rows = g->M->num_dense_rows;
	chk.phase = 0;
	chk.step = step;
	chk.seq_len = L;
	chk.seed = seed;
	chk.a_offset = a_offset;
	chk.hash_y0 = hash_y0;
	chk.hash_u = hash_u;
	chk.hash_s = hash_s;
	chk.hash_x = h;

	if (bw_checkpoint_write(chk_path, &chk, scratch, g->M->ncols)) {
		logprintf(obj, "error: BW checkpoint write failed\n");
		return -1;
	}
	(void)a_fp;
	return 0;
}
