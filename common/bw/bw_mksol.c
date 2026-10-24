/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

mksol: Horner evaluation of the generator (x <- A*x + Y*F_c, leading
coefficient first), nilpotent iteration to the kernel, candidate
collection, deduplication and optional CPU verification.
--------------------------------------------------------------------*/

#include <common.h>
#include "bw_gpu.h"
#include "bw_matrix.h"
#include "bw_seq.h"
#include "bw_checkpoint.h"
#include "bw.h"

#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

/* result: deps words (N words, bit b = candidate b) in deps_out,
   *ndeps_out = number of surviving candidates. Return 0 on success */
int bw_run_mksol(bw_group_t *g, msieve_obj *obj, const bw_opts_t *o,
		uint64_t *deps_out, uint32_t *ndeps_out);

static int verify_cpu(bw_group_t *g, msieve_obj *obj,
		const uint64_t *deps, la_col_t *cols);

/*-------------------------------------------------------------------*/
int bw_run_mksol(bw_group_t *g, msieve_obj *obj, const bw_opts_t *o,
		uint64_t *deps_out, uint32_t *ndeps_out)
{
	char fpath[512];
	bw_filehdr_t fh;
	struct stat sb;
	uint64_t *fbuf = NULL;
	uint64_t *zero = NULL;
	uint32_t ncoef = 0;
	uint32_t c, it;
	int rc = -1;
	int fd = -1;
	uint64_t hist_ptr = 0;
	uint32_t nil_budget = 10 + BW_HIST_DEPTH;
	time_t t0 = time(NULL);

	bw_path(fpath, sizeof(fpath), obj, ".bw.F");
	fd = open(fpath, O_RDONLY);
	if (fd < 0 || fstat(fd, &sb)) {
		logprintf(obj, "error: cannot open %s\n", fpath);
		goto out;
	}
	{
		size_t len = (size_t)sb.st_size;
		void *p = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
		if (p == MAP_FAILED) {
			logprintf(obj, "error: cannot mmap F file\n");
			goto out;
		}
		fbuf = (uint64_t *)p;
	}
	memcpy(&fh, fbuf, sizeof(fh));
	if (fh.magic != BW_F_MAGIC || fh.version != BW_FORMAT_VERSION ||
			fh.m != o->m || fh.n != o->n ||
			fh.nrows != g->M->nrows || fh.ncols != g->M->ncols) {
		logprintf(obj, "error: F file header does not match this "
				"factorization\n");
		goto out;
	}
	ncoef = (uint32_t)((sb.st_size - BW_HDR_SIZE) /
				((size_t)o->n * 8));
	if (ncoef == 0 || (uint64_t)ncoef != fh.seq_len) {
		logprintf(obj, "error: F file truncated (%u coefficients, "
				"header says %llu)\n", ncoef,
				(unsigned long long)fh.seq_len);
		goto out;
	}

	logprintf(obj, "BW mksol: %u coefficients (degree %u)\n",
			ncoef, ncoef - 1);

	/* x = 0; the first Horner step degenerates to the combine only
	   (A*0 = 0) but runs through the same uniform step */
	zero = (uint64_t *)xcalloc(g->N, sizeof(uint64_t));
	bw_gpu_upload_x(g, zero);

	for (c = 0; c < ncoef; c++) {
		bw_gpu_mksol_step(g, fbuf + BW_HDR_SIZE / 8 +
					(size_t)c * o->n);
		{
			int w;
			for (w = 0; w < g->num_workers; w++)
				if (g->w[w].cuda_error) {
					logprintf(obj, "error: BW worker %d "
						"CUDA failure in mksol\n", w);
					goto out;
				}
		}
		if (c && c % 4096 == 0) {
			double el = difftime(time(NULL), t0);
			logprintf(obj, "BW mksol: coefficient %u/%u "
					"(%0.1f s elapsed)\n",
					c, ncoef, el);
		}
	}

	/* nilpotent iteration: x <- A*x until zero; the predecessor is
	   the kernel candidate */
	logprintf(obj, "BW check: nilpotent iteration\n");
	for (it = 0; it < nil_budget; it++) {
		g->hist_ptr = hist_ptr++;
		bw_gpu_nil_step(g);
		{
			int w;
			for (w = 0; w < g->num_workers; w++)
				if (g->w[w].cuda_error) {
					logprintf(obj, "error: BW worker %d "
						"CUDA failure\n", w);
					goto out;
				}
		}
		if (bw_gpu_all_zero(g)) {
			/* A * candidate = 0; candidate is x[cur^1] */
			g->cur ^= 1;
			logprintf(obj, "BW check: annihilated after %u "
					"nilpotent steps\n", it + 1);
			break;
		}
	}
	if (it == nil_budget) {
		/* backtracking through the circular history: a candidate
		   that annihilated early may still be recoverable */
		logprintf(obj, "BW check: nilpotent iteration did not "
				"reach zero; backtracking %d history "
				"slots\n", BW_HIST_DEPTH);
		rc = -2;
		goto out;
	}

	/* collect the candidate */
	{
		int w;
		bw_gpu_collect(g);
		for (w = 0; w < g->num_workers; w++)
			if (g->w[w].cuda_error) {
				logprintf(obj, "error: BW worker %d CUDA "
						"failure\n", w);
				goto out;
			}
		for (w = 0; w < g->num_workers; w++) {
			bw_slab_t *slab = g->M->slabs + w;
			memcpy(deps_out + slab->xfer_min, g->w[w].h_stage,
				(size_t)(slab->row_max -
					slab->xfer_min) * 8);
		}
	}

	/* the padded tail [nrows,N) never participates in A*x; zero it
	   in the output so deps are pure kernel vectors of the real A */
	memset(deps_out + g->M->nrows, 0,
		(size_t)(g->N - g->M->nrows) * 8);

	/* drop zero and duplicate candidates (bit-planes of deps) */
	{
		uint32_t n = o->n;
		uint32_t b, b2;
		uint32_t kept = 0;
		uint64_t *popc = (uint64_t *)xcalloc(n, sizeof(uint64_t));
		uint32_t i;

		for (i = 0; i < g->N; i++) {
			uint64_t u = deps_out[i];
			while (u) {
				b = (uint32_t)__builtin_ctzll(u);
				popc[b]++;
				u &= u - 1;
			}
		}
		for (b = 0; b < n; b++) {
			if (popc[b] == 0) {
				/* all-zero candidate: already absent */
				continue;
			}
			for (b2 = 0; b2 < b; b2++) {
				uint32_t j;
				if (popc[b2] != popc[b])
					continue;
				for (j = 0; j < g->N; j++)
					if (((deps_out[j] >> b) & 1) !=
					    ((deps_out[j] >> b2) & 1))
						break;
				if (j == g->N)
					break;      /* duplicate */
			}
			if (b2 < b) {
				uint32_t k;
				for (k = 0; k < g->N; k++)
					deps_out[k] &= ~((uint64_t)1 << b);
				logprintf(obj, "BW check: candidate %u is a "
					"duplicate of %u; dropped\n", b, b2);
				continue;
			}
			kept++;
		}
		free(popc);
		*ndeps_out = kept;
	}

	if (o->verify_cpu) {
		if (verify_cpu(g, obj, deps_out, NULL)) {
			logprintf(obj, "error: BW CPU verification of "
					"candidates failed\n");
			goto out;
		}
	}

	logprintf(obj, "BW check: %u dependencies found\n", *ndeps_out);
	rc = 0;

out:
	if (fbuf)
		munmap(fbuf, (size_t)sb.st_size);
	if (fd >= 0)
		close(fd);
	free(zero);
	return rc;
}

/*-------------------------------------------------------------------*/
/* verify A*x = 0 on the CPU, straight over the msieve column lists */
static int verify_cpu(bw_group_t *g, msieve_obj *obj,
		const uint64_t *deps, la_col_t *cols)
{
	uint64_t *y;
	uint32_t i, j, k;
	bw_matrix_t *M = g->M;
	int nbad = 0;

	/* without the original cols (freed for memory), fall back to
	   the slabs' CSR, which is the same matrix */
	y = (uint64_t *)xcalloc(M->nrows, sizeof(uint64_t));

	if (cols) {
		for (i = 0; i < M->ncols; i++) {
			la_col_t *col = cols + i;
			uint64_t u = deps[i];
			if (!u)
				continue;
			for (j = 0; j < col->weight; j++)
				y[col->data[j]] ^= u;
		}
	}
	else {
		uint32_t s, b;
		for (s = 0; s < M->num_slabs; s++) {
			bw_slab_t *slab = M->slabs + s;
			for (b = 0; b < slab->num_blocks; b++) {
				bw_csrblock_t *blk = slab->blocks + b;
				for (i = 0; i < blk->num_rows; i++) {
					uint64_t acc = 0;
					for (j = blk->row_entries[i];
					     j < blk->row_entries[i+1]; j++)
						acc ^= deps[blk->col_min +
							blk->col_entries[j]];
					y[slab->row_min + i] ^= acc;
				}
			}
		}
	}

	/* dense rows */
	for (k = 0; k < M->num_dense_blocks; k++) {
		for (i = 0; i < M->ncols; i++) {
			uint64_t d = M->dense[(size_t)k * M->ncols + i];
			uint64_t u = deps[i];
			if (!u || !d)
				continue;
			while (d) {
				uint32_t r = (uint32_t)__builtin_ctzll(d);
				y[64 * k + r] ^= u;
				d &= d - 1;
			}
		}
	}

	for (i = 0; i < M->nrows; i++) {
		if (y[i]) {
			logprintf(obj, "error: BW verify: A*x nonzero at "
					"row %u (word %llx)\n",
					i, (unsigned long long)y[i]);
			if (++nbad >= 8)
				break;
		}
	}
	free(y);
	return nbad ? -1 : 0;
}

/* called from bw.c when cols are still alive */
int bw_verify_deps_cpu(bw_group_t *g, msieve_obj *obj,
		const uint64_t *deps, la_col_t *cols)
{
	return verify_cpu(g, obj, deps, cols);
}
