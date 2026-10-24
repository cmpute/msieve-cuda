/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

STUB: the CADO lingen port lands in this file next (M2). This stub
keeps BW=1 builds linkable and fails with a clear message so nobody
burns GPU hours on a solve that cannot finish.
--------------------------------------------------------------------*/

#include <stdio.h>
#include <stdint.h>

extern "C" {

typedef struct {
	uint32_t m, n;
	const char *afile;
	const char *ffile;
	uint64_t seq_len;
	uint32_t threshold;
	uint32_t nthreads;
} bw_lingen_opts_t;

int bw_lingen_run(const bw_lingen_opts_t *o)
{
	fprintf(stderr,
	 "BW error: the lingen port is not built into this binary yet\n"
	 "(bw_lingen.cpp is the stub; see BW.md 'Lingen port').\n");
	(void)o;
	return -1;
}

} /* extern C */
