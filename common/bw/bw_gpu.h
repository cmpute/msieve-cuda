/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

One process, one worker thread per GPU. Workers own their device, two
streams and an spmv_engine instance; all CUDA calls happen on worker
threads. The main thread coordinates steps through two reusable
barriers: it publishes a command, releases the workers at b_start, and
collects results at b_end. Workers synchronize the allgather midpoint
at b_mid (workers only). Host barriers are ~10us against 13+ ms steps.
--------------------------------------------------------------------*/

#ifndef _COMMON_BW_BW_GPU_H_
#define _COMMON_BW_BW_GPU_H_

#include <pthread.h>
#include <cuda_runtime.h>
#include "bw.h"
#include "bw_matrix.h"
#include "../cub/spmv_engine.h"

typedef struct bw_worker bw_worker_t;
typedef struct bw_group bw_group_t;

typedef enum {
	BW_CMD_NONE = 0,
	BW_CMD_UPLOAD_INIT,     /* upload Y0/U/dense; x[0] = Y0 */
	BW_CMD_UPLOAD_X,        /* x[cur] = group->h_x (resume) */
	BW_CMD_KRYLOV_STEP,     /* x[next] = A*x[cur]; partial S to h_s */
	BW_CMD_MKSOL_STEP,      /* x[next] = A*x[cur] + Y0*F; no S */
	BW_CMD_NIL_STEP,        /* x[next] = A*x[cur]; history + zero flag */
	BW_CMD_COLLECT,         /* D2H x[cur] slab into h_stage */
	BW_CMD_STOP
} bw_cmd_t;

#define BW_HIST_DEPTH 16

struct bw_worker {
	int idx;
	int dev;
	bw_group_t *g;

	pthread_t thread;
	void *spmv;              /* spmv_engine instance */
	cudaStream_t compute, copy;
	cudaEvent_t ev_comp, ev_comb;

	uint64_t *d_x[2];        /* full N words, replicated per GPU */
	uint64_t *d_y0;          /* Y0 replica, full N words */
	uint64_t *d_u;           /* U masks for this slab:
				    (m/64) words per coordinate row */
	uint64_t *d_u_dense;    /* U masks for slab 0's dense-row prefix
				    [0,row_min), same layout */
	uint64_t *d_s;           /* partial S, m words */
	uint64_t *d_fcoef;       /* mksol coefficient, n words (the host
				    pointer cannot be dereferenced on the
				    device, so it is staged here first) */
	uint64_t *d_hist[BW_HIST_DEPTH + 1];  /* [0,BW_HIST_DEPTH) are the
				    circular-history slabs, [BW_HIST_DEPTH]
				    is a one-word zero flag */
	uint64_t *d_dense;       /* dense bitmap replica (all GPUs) */

	/* host copies of the CSR block descriptors */
	spmv_data_t *sd;
	uint32_t num_sd;

	/* pinned staging */
	uint64_t *h_s;           /* m words */
	uint64_t *h_stage;       /* slab words */
	uint64_t h_flag;         /* slab-zero flag after NIL_STEP */
	int cuda_error;          /* first error code, sticky */
};

struct bw_group {
	bw_matrix_t *M;
	const bw_opts_t *o;
	uint32_t N;              /* ncols words == padded dimension words */
	uint32_t nrows_words;    /* ceil(nrows/64)... always nrows since
				    rows are words: nrows words exactly */
	int num_workers;

	bw_cmd_t cmd;
	int cur;                 /* which d_x is current */
	const uint64_t *fcoef;   /* MKSOL_STEP: n words of the coefficient */
	uint64_t hist_ptr;       /* NIL_STEP history slot to fill */
	int need_stage;          /* allgather mode for this step */

	bw_worker_t *w;

	pthread_barrier_t b_start;       /* main + workers */
	pthread_barrier_t b_mid;         /* workers only */
	pthread_barrier_t b_end;         /* main + workers */

	/* peer-access matrix and allgather mode */
	int can_peer[BW_MAX_GPUS][BW_MAX_GPUS];
	int use_peer;            /* 1 if every ordered pair has access */

	/* host-side shared inputs/outputs */
	const uint64_t *h_y0;    /* N words */
	const uint64_t *h_u;     /* (m/64) words per row, nrows*... words */
	const uint64_t *h_x;     /* N words (upload target / collect src */
	uint64_t *h_xbuf;        /* N words owned by the group */

	/* accounting filled in by workers */
	uint64_t slab_words[BW_MAX_GPUS];
	uint64_t slab_off[BW_MAX_GPUS];  /* word offset into the state */
};

/* spawn P workers bound to o->gpus; each thread sets its device,
   allocates its buffers, uploads its slab of the matrix, runs the
   occupancy preflights, and parks on b_start. Returns 0 on success */
int bw_gpu_start(bw_group_t *g, bw_matrix_t *M, const bw_opts_t *o,
		msieve_obj *obj, const uint64_t *h_y0, const uint64_t *h_u);

/* run one command and wait for completion (barriers around it).
   bw_gpu_step() wrappers below fill g->cmd then call this */
void bw_gpu_run(bw_group_t *g, bw_cmd_t cmd);

static inline void bw_gpu_krylov_step(bw_group_t *g) {
	bw_gpu_run(g, BW_CMD_KRYLOV_STEP);
	g->cur ^= 1;
}
static inline void bw_gpu_mksol_step(bw_group_t *g, const uint64_t *f) {
	g->fcoef = f;
	bw_gpu_run(g, BW_CMD_MKSOL_STEP);
	g->cur ^= 1;
}
static inline void bw_gpu_nil_step(bw_group_t *g) {
	bw_gpu_run(g, BW_CMD_NIL_STEP);
	g->cur ^= 1;
}
static inline void bw_gpu_collect(bw_group_t *g) {
	bw_gpu_run(g, BW_CMD_COLLECT);   /* into w[i].h_stage */
}
static inline void bw_gpu_upload_x(bw_group_t *g, const uint64_t *x) {
	g->h_x = x;
	bw_gpu_run(g, BW_CMD_UPLOAD_X);
}

void bw_gpu_stop(bw_group_t *g);   /* BW_CMD_STOP + join + free */

/* XOR all worker partials into out (m words); also return the OR of
   worker zero-flags (NIL_STEP) */
static inline void bw_gpu_reduce_s(bw_group_t *g, uint64_t *out, uint32_t m) {
	uint32_t i;
	int w;
	memset(out, 0, m * sizeof(uint64_t));
	for (w = 0; w < g->num_workers; w++)
		for (i = 0; i < m; i++)
			out[i] ^= g->w[w].h_s[i];
}
static inline int bw_gpu_all_zero(bw_group_t *g) {
	int w;
	uint64_t f = 0;
	for (w = 0; w < g->num_workers; w++)
		f |= g->w[w].h_flag;
	return f == 0;
}

/* CUDA kernels (bw_kernel.cu); runtime-API launches, VBITS=64 */
#pragma pack(push, 8)
#ifdef __cplusplus
extern "C" {
#endif

void bw_oprod64(uint64_t *s, const uint64_t *u, const uint64_t *y,
		uint32_t nmasks, size_t nrows, cudaStream_t stream);
	/* s[m=64*nmasks words] = sum over i of (bits of u[i]) selected
	   rows of y: s[64k+a] ^= bit a of u_k[i] ? y[i] */

void bw_combine64(uint64_t *x, const uint64_t *y0, const uint64_t *fw,
		size_t n, cudaStream_t stream);
	/* x[i] ^= sum over set bits c of y0[i] of fw[c] */

void bw_dense64(uint64_t *y, const uint64_t *dense, const uint64_t *x,
		uint32_t ndense_blocks, size_t n, cudaStream_t stream);
	/* y[r] ^= sum over columns c with bit r of dense[k][c] of x[c],
	   for every dense row 64k+r */

void bw_anyset64(const uint64_t *x, uint64_t *flag, size_t n,
		cudaStream_t stream);
	/* flag[0] |= OR of x[0..n); flag is zeroed first */

int bw_preflight(int dev);  /* occupancy checks; 0 ok, -1 fail */

#ifdef __cplusplus
}
#endif
#pragma pack(pop)

#endif /* !_COMMON_BW_BW_GPU_H_ */
