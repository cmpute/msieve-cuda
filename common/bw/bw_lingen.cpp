/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its authors.
You may use it for any purpose, free of charge, without having to notify
anyone. We disclaim any responsibility for any errors.

Thin dispatcher for the BW lingen implementations. The CADO-vendored
subquadratic path was removed (2026-10-24): the default is the
independent recursive lingen (bw_lingen_own, GPU basecase + gf2x
polynomial arithmetic); bw_lingen_impl=quad selects the standalone
quadratic GPU route. No CADO code remains in the tree.
--------------------------------------------------------------------*/

#include <cstdio>
#include <cstdint>
#include "bw_lingen_quad.h"

extern "C" {

typedef struct {
	uint32_t m, n;
	uint32_t nrows, ncols;   /* matrix dims for the F header */
	const char *afile;
	const char *ffile;
	uint64_t seq_len;
	uint32_t threshold;
	uint32_t nthreads;
	uint32_t impl;           /* 0 = own (default), 1 = quad GPU */
	uint32_t dev;            /* device for the GPU basecase */
} bw_lingen_opts_t;

/* set from the bw=1 option string (bw.c, C linkage) */
uint32_t g_bw_lingen_impl = 0;
uint32_t g_bw_lingen_dev = 0;

int bw_lingen_run(const bw_lingen_opts_t *o)
{
	if (g_bw_lingen_impl == 1) {
		/* the quad track: fully self-contained quadratic GPU BM */
		if (bw_lingen_quad_full(o->afile, o->ffile, o->m, o->n,
					o->seq_len, o->nrows, o->ncols,
					(int)g_bw_lingen_dev, 3,
					o->nthreads)) {
			fprintf(stderr, "BW quad: full lingen failed\n");
			return -1;
		}
		fprintf(stderr, "BW lingen: F written to %s\n", o->ffile);
		return 0;
	}

	/* default: the independent recursive lingen */
	if (bw_lingen_own_full(o->afile, o->ffile, o->m, o->n,
				o->seq_len, o->nrows, o->ncols,
				(int)g_bw_lingen_dev, o->nthreads,
				o->threshold)) {
		fprintf(stderr, "BW own: full lingen failed\n");
		return -1;
	}
	fprintf(stderr, "BW lingen: F written to %s\n", o->ffile);
	return 0;
}

} /* extern "C" */
