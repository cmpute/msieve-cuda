/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

Option parsing and phase sequencing for the Block Wiedemann solver.
--------------------------------------------------------------------*/

#include <common.h>
#include "bw.h"
#include "bw_matrix.h"
#include "bw_gpu.h"
#include "bw_seq.h"
#include "bw_checkpoint.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

/* lingen impl selection, read by bw_lingen_run; defined there so
 * the lingen test link needs no bw.c objects */
extern uint32_t g_bw_lingen_impl;
extern uint32_t g_bw_lingen_dev;

/* lingen C ABI (bw_lingen.cpp) */
typedef struct {
	uint32_t m, n;
	uint32_t nrows, ncols;   /* matrix dims for the F header */
	const char *afile;
	const char *ffile;
	uint64_t seq_len;
	uint32_t threshold;
	uint32_t nthreads;
	uint32_t impl;           /* 0 = cado subquadratic, 1 = quad GPU */
	uint32_t dev;            /* device for the quad basecase */
} bw_lingen_opts_t;

extern int bw_lingen_run(const bw_lingen_opts_t *o);

/* drivers */
extern int bw_run_krylov(bw_group_t *g, msieve_obj *obj, const bw_opts_t *o,
		uint64_t seed, uint64_t hash_y0, uint64_t hash_u,
		uint32_t L, uint32_t interval, int have_resume,
		const bw_chkhdr_t *resume, uint32_t *stop_requested);
extern int bw_run_mksol(bw_group_t *g, msieve_obj *obj, const bw_opts_t *o,
		uint64_t *deps_out, uint32_t *ndeps_out);
extern int bw_verify_deps_cpu(bw_group_t *g, msieve_obj *obj,
		const uint64_t *deps, la_col_t *cols);

/*-------------------------------------------------------------------*/
void bw_opts_defaults(bw_opts_t *o) {
	memset(o, 0, sizeof(bw_opts_t));
	o->m = 128;
	o->n = 64;
	o->num_gpus = 1;
	o->gpus[0] = 0;
	o->block_nnz = 0;          /* 0 = auto (L2-calibrated) */
	o->peer = -1;              /* auto */
	o->allgather_hier = 0;
	o->lingen_threshold = 2048;
	o->checkpoint_s = 1800;
	o->seed = 0;
	o->retry = 0;
	o->verify_cpu = 0;
	o->keep = 0;
}

/*-------------------------------------------------------------------*/
static int parse_uint(const char *s, uint64_t *out) {
	char *end;
	unsigned long long v;
	if (*s == '\0')
		return -1;
	v = strtoull(s, &end, 10);
	if (*end != '\0')
		return -1;
	*out = v;
	return 0;
}

/* apply one key=value pair; returns 0 ok, -1 unknown/hard error */
static int bw_opt_one(bw_opts_t *o, const char *key, const char *val,
			msieve_obj *obj)
{
	uint64_t v;

	if (!strcmp(key, "bw") || !strcmp(key, "bw_enabled"))
		o->enabled = (val[0] == '1');
	else if (!strcmp(key, "bw_m")) {
		if (parse_uint(val, &v) || v % 64 || v < 64 || v > 512) {
			logprintf(obj, "error: bw_m must be a multiple of "
					"64 in [64,512]\n");
			return -1;
		}
		o->m = (uint32)v;
	}
	else if (!strcmp(key, "bw_n")) {
		if (parse_uint(val, &v) || v != 64) {
			logprintf(obj, "error: bw_n must be 64 (VBITS)\n");
			return -1;
		}
		o->n = (uint32)v;
	}
	else if (!strcmp(key, "bw_gpus")) {
		/* the value is a comma list, but ',' also separates nfs_args
		 * options: consume digits and commas only, so
		 * bw_gpus=1,2,bw_keep=1 parses as two options */
		o->num_gpus = 0;
		while (*val && o->num_gpus < BW_MAX_GPUS) {
			char *end;
			unsigned long d = strtoul(val, &end, 10);
			if (end == val)
				return -1;
			o->gpus[o->num_gpus++] = (int)d;
			val = end;
			if (*val != ',')
				break;      /* next option or end */
			val++;
			if (!isdigit((unsigned char)*val))
				break;      /* ",bw_..." style separator */
		}
		if (o->num_gpus == 0) {
			logprintf(obj, "error: bw_gpus needs at least one "
					"device ordinal\n");
			return -1;
		}
		{
			int i, j;
			for (i = 0; i < o->num_gpus; i++)
				for (j = i + 1; j < o->num_gpus; j++)
					if (o->gpus[i] == o->gpus[j]) {
						logprintf(obj, "error: "
							"bw_gpus repeats "
							"device %d\n",
							o->gpus[i]);
						return -1;
					}
		}
	}
	else if (!strcmp(key, "bw_block_nnz")) {
		if (parse_uint(val, &v) || v < 100000 || v > 4000000000ull) {
			logprintf(obj, "error: bw_block_nnz out of range "
					"[100000,4e9]\n");
			return -1;
		}
		o->block_nnz = v;
	}
	else if (!strcmp(key, "bw_peer")) {
		if (!strcmp(val, "auto")) o->peer = -1;
		else if (!strcmp(val, "0")) o->peer = 0;
		else if (!strcmp(val, "1")) o->peer = 1;
		else {
			logprintf(obj, "error: bw_peer must be auto|0|1\n");
			return -1;
		}
	}
	else if (!strcmp(key, "bw_allgather")) {
		if (!strcmp(val, "flat")) o->allgather_hier = 0;
		else if (!strcmp(val, "hier")) o->allgather_hier = 1;
		else {
			logprintf(obj, "error: bw_allgather must be "
					"flat|hier\n");
			return -1;
		}
	}
	else if (!strcmp(key, "bw_lingen_threshold")) {
		if (parse_uint(val, &v) || v < 16 || v > 65536) {
			logprintf(obj, "error: bw_lingen_threshold out of "
					"range [16,65536]\n");
			return -1;
		}
		o->lingen_threshold = (uint32)v;
	}
	else if (!strcmp(key, "bw_checkpoint")) {
		if (parse_uint(val, &v) || v < 60 || v > 86400) {
			logprintf(obj, "error: bw_checkpoint out of range "
					"[60,86400] s\n");
			return -1;
		}
		o->checkpoint_s = (uint32)v;
	}
	else if (!strcmp(key, "bw_seed")) {
		if (parse_uint(val, &v))
			return -1;
		o->seed = v;
	}
	else if (!strcmp(key, "bw_retry")) {
		o->retry = (val[0] == '1');
	}
	else if (!strcmp(key, "bw_verify_cpu")) {
		o->verify_cpu = (val[0] == '1');
	}
	else if (!strcmp(key, "bw_keep")) {
		o->keep = (val[0] == '1');
	}
	else if (!strcmp(key, "bw_lingen_impl")) {
		/* stored globally; bw_lingen_run picks it up:
		 * 0 = own (default, independent recursive), 1 = quad */
		extern uint32_t g_bw_lingen_impl;
		g_bw_lingen_impl = !strcmp(val, "quad");
	}
	else {
		logprintf(obj, "error: unknown BW option '%s'\n", key);
		return -1;
	}
	return 0;
}

/* scan a substring of the form key=val (terminated by ',' or NUL) */
static int bw_scan(bw_opts_t *o, const char *p, msieve_obj *obj) {
	char key[64], val[256];
	const char *eq = strchr(p, '=');
	size_t klen, vlen;
	if (eq == NULL)
		return -1;
	klen = (size_t)(eq - p);
	vlen = strcspn(eq + 1, ",");
	/* bw_gpus is itself a comma list; take everything up to the next
	 * ",bw" (option separator) or end of string */
	if (klen == 7 && !strncmp(p, "bw_gpus", 7)) {
		const char *q = eq + 1;
		const char *c;
		vlen = 0;
		while ((c = strchr(q + vlen, ',')) != NULL) {
			if (!strncmp(c, ",bw_", 4))
				break;
			vlen = (size_t)(c - q) + 1;
		}
		if (!c)
			vlen = strlen(q);
	}
	if (klen >= sizeof(key) || vlen >= sizeof(val))
		return -1;
	memcpy(key, p, klen); key[klen] = '\0';
	memcpy(val, eq + 1, vlen); val[vlen] = '\0';
	return bw_opt_one(o, key, val, obj);
}

int bw_opts_parse(bw_opts_t *o, msieve_obj *obj, const char *args) {
	const char *p = args;

	while (p && *p) {
		const char *comma = strchr(p, ',');
		size_t len = comma ? (size_t)(comma - p) : strlen(p);
		char tok[512];
		/* keep "bw_gpus=1,2,3" as one token: extend past commas
		 * that are followed by a digit */
		while (comma && isdigit((unsigned char)comma[1])) {
			const char *next = strchr(comma + 1, ',');
			size_t nlen = next ? (size_t)(next - p)
					: strlen(p);
			if (nlen >= sizeof(tok))
				break;
			len = nlen;
			comma = next;
			if (!comma)
				break;
		}
		if (len >= sizeof(tok))
			return -1;
		memcpy(tok, p, len);
		tok[len] = '\0';
		if (strncmp(tok, "bw", 2) == 0) {
			if (bw_scan(o, tok, obj))
				return -1;
		}
		p = comma ? comma + 1 : NULL;
	}

	/* environment: MSIEVE_BW_GPU_LIST etc., applied after the
	   command line so operators can override without editing the
	   job script */
	{
		const char *e = getenv("MSIEVE_BW_GPU_LIST");
		if (e && *e) {
			char tok[256];
			snprintf(tok, sizeof(tok), "bw_gpus=%s", e);
			if (bw_scan(o, tok, obj))
				return -1;
		}
		e = getenv("MSIEVE_BW_M");
		if (e && *e) {
			char tok[64];
			snprintf(tok, sizeof(tok), "bw_m=%s", e);
			if (bw_scan(o, tok, obj))
				return -1;
		}
	}

	if (o->enabled && o->num_gpus < 1)
		return -1;
	return 0;
}

/*-------------------------------------------------------------------*/
static uint32_t next_pow2(uint32_t x) {
	uint32_t p = 1;
	while (p < x)
		p <<= 1;
	return p;
}

static uint32_t isqrt64(uint64_t x) {
	uint64_t r = (uint64_t)sqrt((double)x);
	while (r && r * r > x) r--;
	while ((r + 1) * (r + 1) <= x) r++;
	return (uint32_t)r;
}

/* Krylov length per BW.md (CADO-derived) */
uint32_t bw_krylov_length(uint64_t N, uint32_t m, uint32_t n,
				uint32_t *interval_out)
{
	uint64_t L = (N + m - 1) / m + (N + n - 1) / n +
			2ull * ((m + n - 1) / n) + 2ull * ((n + m - 1) / m) + 10;
	uint32_t interval = next_pow2(isqrt64(L));
	if (interval < 64)
		interval = 64;
	L = ((L + interval - 1) / interval) * interval;
	*interval_out = interval;
	return (uint32_t)L;
}

/*-------------------------------------------------------------------*/
uint64_t *bw_solve(msieve_obj *obj,
		uint32_t nrows, uint32_t max_nrows, uint32_t start_row,
		uint32_t num_dense_rows,
		uint32_t ncols, uint32_t max_ncols, uint32_t start_col,
		la_col_t *cols, uint32_t *deps_found)
{
	bw_opts_t o;
	bw_matrix_t M;
	bw_group_t g;
	uint64_t *deps = NULL;
	uint64_t *h_y0 = NULL, *h_u = NULL;
	uint64_t seed, hash_y0 = 0, hash_u = 0;
	uint32_t L, interval;
	uint32_t N = ncols;
	uint32_t stop_requested = 0;
	int i, rc;
	int have_resume = 0;
	bw_chkhdr_t resume;
	uint64_t *resume_state = NULL;
	char apath[512], fpath[512], chk_path[512];
	time_t t_krylov, t_lingen, t_mksol;
	FILE *fp;

#if VBITS != 64
	logprintf(obj, "error: BW requires VBITS=64 builds\n");
	return NULL;
#endif
#ifdef HAVE_MPI
	logprintf(obj, "error: BW is single-process; build without MPI "
			"or use the Lanczos path\n");
	return NULL;
#endif
	if (start_row != 0 || start_col != 0 ||
			nrows != max_nrows || ncols != max_ncols) {
		logprintf(obj, "error: BW needs the full matrix "
				"(no MPI submatrix)\n");
		return NULL;
	}
	if (ncols <= nrows || ncols - nrows < 64) {
		logprintf(obj, "error: BW needs excess columns >= 64 "
				"(have %u)\n", ncols - nrows);
		return NULL;
	}
	bw_opts_defaults(&o);
	o.num_gpus = 0;   /* bw_gpus must be given explicitly */
	o.gpus[0] = 0;
	if (bw_opts_parse(&o, obj, obj->nfs_args)) {
		logprintf(obj, "error: bad BW options\n");
		return NULL;
	}
	if (!o.enabled) {
		logprintf(obj, "error: bw=1 not set\n");
		return NULL;
	}
	if (o.num_gpus == 0) {
		o.num_gpus = 1;
		o.gpus[0] = 0;
		logprintf(obj, "BW: no bw_gpus given; defaulting to GPU 0\n");
	}
	if ((uint64)nrows - num_dense_rows < (uint64)o.num_gpus) {
		logprintf(obj, "error: BW needs at least one sparse row "
				"per GPU\n");
		return NULL;
	}

	/* seed material: mix the msieve seeds with the explicit BW seed */
	seed = ((uint64)obj->seed1 << 32) ^ obj->seed2 ^ o.seed ^
		0x9E3779B97F4A7C15ull;

	L = bw_krylov_length(N, o.m, o.n, &interval);

	bw_path(apath, sizeof(apath), obj, ".bw.A");
	bw_path(fpath, sizeof(fpath), obj, ".bw.F");
	bw_path(chk_path, sizeof(chk_path), obj, ".bw.chk");

	logprintf(obj, "BW: m=%u n=%u gpus=", o.m, o.n);
	for (i = 0; i < o.num_gpus; i++)
		logprintf(obj, "%s%d", i ? "," : "", o.gpus[i]);
	logprintf(obj, ", N=%u, L=%u (interval %u)\n", N, L, interval);

	/* Y0: N words random (nonzero tail included) */
	h_y0 = (uint64_t *)xmalloc((size_t)N * sizeof(uint64_t));
	{
		bw_rng_t r; r.x = seed;
		for (i = 0; i < (int)N; i++)
			h_y0[i] = bw_rng_next(&r);
	}
	hash_y0 = bw_fnv1a(0, h_y0, (size_t)N * 8);

	/* U: m/64 words per row, nrows rows (zero on the padded tail) */
	h_u = (uint64_t *)xcalloc((size_t)nrows * (o.m / 64),
				sizeof(uint64_t));
	{
		bw_rng_t r; r.x = seed ^ 0xabcdef0123456789ull;
		uint32_t k;
		for (k = 0; k < nrows * (o.m / 64); k++)
			h_u[k] = bw_rng_next(&r);
	}
	hash_u = bw_fnv1a(0, h_u, (size_t)nrows * (o.m / 64) * 8);

	/* resume? */
	if (obj->flags & MSIEVE_FLAG_NFS_LA_RESTART) {
		if (bw_checkpoint_read(chk_path, &resume, &resume_state,
					ncols) == 0) {
			if (resume.m != o.m || resume.n != o.n ||
					resume.nrows != nrows ||
					resume.ncols != ncols ||
					resume.seed != seed) {
				logprintf(obj, "error: BW checkpoint "
					"dimensions or seed mismatch; "
					"remove %s to start fresh\n",
					chk_path);
				goto fail;
			}
			if (resume.hash_y0 != hash_y0 ||
					resume.hash_u != hash_u) {
				logprintf(obj, "error: BW checkpoint hash "
					"mismatch (Y0/U changed)\n");
				goto fail;
			}
			if (resume.seq_len != L) {
				logprintf(obj, "error: BW checkpoint length "
					"%llu != %u\n",
					(unsigned long long)resume.seq_len,
					L);
				goto fail;
			}
			have_resume = 1;
			logprintf(obj, "BW: resuming from checkpoint "
					"(phase %u, step %llu)\n",
					resume.phase,
					(unsigned long long)resume.step);
		}
		else {
			logprintf(obj, "BW: -ncr given but no usable "
					"checkpoint; starting fresh\n");
		}
	}
	if (!have_resume) {
		/* stale files from an abandoned run would corrupt the
		   lingen input */
		remove(apath);
		remove(fpath);
		remove(chk_path);
	}

	/* matrix packaging (slabs in GPU list order) */
	if (bw_matrix_build(&M, obj, nrows, num_dense_rows, ncols, cols,
			o.num_gpus, o.block_nnz, o.gpus[0])) {
		logprintf(obj, "error: BW matrix packaging failed\n");
		goto fail;
	}

	memset(&g, 0, sizeof(bw_group_t));
	if (bw_gpu_start(&g, &M, &o, obj, h_y0, h_u))
		goto fail_after_matrix;

	if (have_resume && resume.phase == 0)
		bw_gpu_upload_x(&g, resume_state);

	/* phase 1: Krylov sequence */
	t_krylov = time(NULL);
	rc = bw_run_krylov(&g, obj, &o, seed, hash_y0, hash_u,
			L, interval, have_resume,
			have_resume ? &resume : NULL, &stop_requested);
	if (rc < 0)
		goto fail_after_gpu;
	if (stop_requested) {
		logprintf(obj, "BW: interrupted; checkpoint saved, "
				"rerun with -ncr\n");
		deps = NULL;
		goto clean_exit;
	}
	logprintf(obj, "BW krylov done in %u s\n",
			(uint32)(time(NULL) - t_krylov));

	/* the GPU packaging no longer needs the msieve column lists;
	   give the memory back before lingen runs (~12 GB on C263).
	   CPU verification needs them, so keep when asked */
	if (!o.verify_cpu) {
		uint32_t c2;
		for (c2 = 0; c2 < ncols; c2++) {
			free(cols[c2].data);
			cols[c2].data = NULL;
		}
	}

	/* phase 2: lingen */
	t_lingen = time(NULL);
	if (have_resume && resume.phase == 1) {
		/* F already exists */
		struct stat sb;
		if (stat(fpath, &sb) != 0) {
			logprintf(obj, "error: checkpoint says mksol but "
					"F file is missing\n");
			goto fail_after_gpu;
		}
		logprintf(obj, "BW lingen: reusing existing F file\n");
	}
	else {
		bw_lingen_opts_t lo;
		memset(&lo, 0, sizeof(lo));
		lo.m = o.m;
		lo.n = o.n;
		lo.nrows = nrows;
		lo.ncols = ncols;
		g_bw_lingen_dev = o.gpus[0];
		lo.afile = apath;
		lo.ffile = fpath;
		lo.seq_len = L;
		lo.threshold = o.lingen_threshold;
		lo.nthreads = 0;   /* 0 = keep omp default (machine threads) */
		if (bw_lingen_run(&lo)) {
			logprintf(obj, "error: BW lingen failed\n");
			goto fail_after_gpu;
		}
		logprintf(obj, "BW lingen done in %u s\n",
				(uint32)(time(NULL) - t_lingen));
	}

	/* phase 3: mksol */
	t_mksol = time(NULL);
	deps = (uint64_t *)xcalloc(ncols, sizeof(uint64_t));
	if (have_resume && resume.phase == 1) {
		bw_gpu_upload_x(&g, resume_state);
		/* BW TODO: fast-forward the Horner loop to resume.step */
		logprintf(obj, "error: mksol resume not implemented; "
				"restart mksol from F (state replay "
				"TODO)\n");
		goto fail_after_gpu;
	}
	{
		uint32_t nd = 0;
		if (bw_run_mksol(&g, obj, &o, deps, &nd)) {
			logprintf(obj, "error: BW mksol failed\n");
			goto fail_after_gpu;
		}
		*deps_found = nd;
		if (nd == 0) {
			logprintf(obj, "error: BW found no dependencies%s\n",
				o.retry ? "; retry with a new bw_seed" : "");
			goto fail_after_gpu;
		}
	}
	logprintf(obj, "BW mksol done in %u s\n",
			(uint32)(time(NULL) - t_mksol));

	/* golden hash of the final deps */
	{
		char hpath[512];
		bw_path(hpath, sizeof(hpath), obj, ".bw.deps.hash");
		fp = fopen(hpath, "w");
		if (fp) {
			uint64_t h = bw_fnv1a(0, deps, (size_t)ncols * 8);
			fprintf(fp, "%llu\n", (unsigned long long)h);
			fclose(fp);
		}
	}

clean_exit:
	bw_gpu_stop(&g);
	bw_matrix_free(&M);
	if (!o.keep && deps && *deps_found) {
		char prevpath[600];
		remove(apath);
		remove(fpath);
		remove(chk_path);
		snprintf(prevpath, sizeof(prevpath), "%s.prev", chk_path);
		remove(prevpath);
	}
	else if (!o.keep && !deps) {
		/* keep checkpoint files for the -ncr restart */
	}
	free(h_y0);
	free(h_u);
	free(resume_state);
	return deps;

fail_after_gpu:
	bw_gpu_stop(&g);
fail_after_matrix:
	bw_matrix_free(&M);
fail:
	free(h_y0);
	free(h_u);
	free(resume_state);
	free(deps);
	return NULL;
}
