/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

Worker threads: device ownership, upload, one Krylov/mksol/nilpotent
step, and the flat allgather (peer push or pinned-host staged).
Everything CUDA happens on worker threads; the main thread only touches
host memory and barriers.
--------------------------------------------------------------------*/

#include <common.h>
#include "bw_gpu.h"
#include "bw_matrix.h"
#include "bw.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

/* BW_TIME wall clock, milliseconds */
static double bw_now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* single-binary CUDA build: the engine objects are linked in, same
   declaration style as lanczos_matmul_gpu.c */
extern void *spmv_engine_init(int *vbits);
extern void spmv_engine_free(void *engine);
extern void spmv_engine_run_on(void *engine, spmv_data_t *spmv_data,
			CUstream stream);

#define CUDA_CHK(w, call) do {					\
	cudaError_t _e = (call);					\
	if (_e != cudaSuccess && (w)->cuda_error == 0) {		\
		(w)->cuda_error = (int)_e;				\
		printf("BW worker %d (GPU %d) CUDA error: %s at %s:%d\n",\
			(w)->idx, (w)->dev,				\
			cudaGetErrorString(_e), __FILE__, __LINE__);	\
	}								\
} while (0)

/*-------------------------------------------------------------------*/
static void worker_step_spmv(bw_worker_t *w)
{
	/* x[next] slab = A * x[cur], sparse + dense. The engine
	   XOR-accumulates, so the slab is zeroed first; the padded tail
	   [nrows,N) is zeroed every step (zero rows of A_pad). */

	bw_group_t *g = w->g;
	bw_matrix_t *M = g->M;
	bw_slab_t *slab = M->slabs + w->idx;
	uint64_t *xin = w->d_x[g->cur];
	uint64_t *xout = w->d_x[g->cur ^ 1];
	uint32_t i;

	if (g->timing)
		cudaEventRecord(w->ev_t0, w->compute);

	/* slab 0's sparse CSR starts at num_dense_rows, but its output
	   range still begins at row 0: the dense kernel accumulates
	   there, so the whole slab range must be zeroed. Plane split:
	   every worker owns the full-range slab, so every worker takes
	   the slab-0 role */
	{
		int lead = (w->idx == 0 || g->plane_split);
		CUDA_CHK(w, cudaMemsetAsync(xout + (lead ? 0 :
							slab->row_min), 0,
				(size_t)(slab->row_max - (lead ? 0 :
							slab->row_min)) * 8,
				w->compute));
	}
	CUDA_CHK(w, cudaMemsetAsync(xout + M->nrows, 0,
			(size_t)(g->N - M->nrows) * 8,
			w->compute));

	for (i = 0; i < slab->num_blocks; i++) {
		spmv_data_t *sd = w->sd + i;
		sd->vector_in = (CUdeviceptr)(xin + slab->blocks[i].col_min);
		sd->vector_out = (CUdeviceptr)(xout + slab->row_min);
		spmv_engine_run_on(w->spmv, sd, w->compute);
	}

	if ((w->idx == 0 || g->plane_split) && M->num_dense_blocks)
		bw_dense64(xout, w->d_dense, xin,
				M->num_dense_blocks, g->N, w->compute);
}

/*-------------------------------------------------------------------*/
static void worker_allgather(bw_worker_t *w, int need_mid_barrier)
{
	/* flat v1: every worker pushes its slab of x[next] to everyone.
	   Peer pushes need only the local compute event; the staged path
	   needs one extra barrier between the D2H and H2D halves. The
	   mksol combine touches the full x[next] range, so its exchange
	   always goes through the mid barrier to order compute vs copies
	   across workers. */

	bw_group_t *g = w->g;
	bw_matrix_t *M = g->M;
	bw_slab_t *slab = M->slabs + w->idx;
	uint64_t *xout = w->d_x[g->cur ^ 1];
	int j;

	/* nothing to exchange: plane split keeps every plane resident on
	   one GPU, and a single worker owns everything. Only the S/flag
	   copies on the copy stream need completing. Skipping here also
	   saves the staged path's pointless full-vector D2H that a lone
	   worker used to do every mksol step */
	if (g->plane_split || g->num_workers == 1) {
		if (g->timing) {
			float ms;
			double t0 = bw_now_ms();
			CUDA_CHK(w, cudaStreamSynchronize(w->copy));
			w->t_copy_sync += bw_now_ms() - t0;
			w->n_steps++;
			cudaEventElapsedTime(&ms, w->ev_t0, w->ev_comp);
			w->t_gpu_compute += ms;
		}
		else
			CUDA_CHK(w, cudaStreamSynchronize(w->copy));
		return;
	}

	if (g->use_peer && !need_mid_barrier) {
		/* pushes ordered by the compute event only */
		for (j = 0; j < g->num_workers; j++) {
			if (j == w->idx)
				continue;
			CUDA_CHK(w, cudaMemcpyPeerAsync(
				w->g->w[j].d_x[g->cur ^ 1] + slab->xfer_min,
				g->w[j].dev,
				xout + slab->xfer_min, w->dev,
				(size_t)(slab->row_max -
					slab->xfer_min) * 8,
				w->copy));
		}
		if (g->timing) {
			float ms;
			double t0 = bw_now_ms();
			CUDA_CHK(w, cudaStreamSynchronize(w->copy));
			w->t_copy_sync += bw_now_ms() - t0;
			w->n_steps++;
			cudaEventElapsedTime(&ms, w->ev_t0, w->ev_comp);
			w->t_gpu_compute += ms;
		}
		else
			CUDA_CHK(w, cudaStreamSynchronize(w->copy));
		return;
	}

	CUDA_CHK(w, cudaMemcpyAsync(w->h_stage, xout + slab->xfer_min,
			(size_t)(slab->row_max - slab->xfer_min) * 8,
			cudaMemcpyDeviceToHost, w->copy));
	{
		double t0 = g->timing ? bw_now_ms() : 0;
		CUDA_CHK(w, cudaStreamSynchronize(w->copy));
		if (g->timing) {
			float ms;
			w->t_copy_sync += bw_now_ms() - t0;
			w->n_steps++;
			cudaEventElapsedTime(&ms, w->ev_t0, w->ev_comp);
			w->t_gpu_compute += ms;
		}
	}

	{
		double t0 = g->timing ? bw_now_ms() : 0;
		pthread_barrier_wait(&g->b_mid);
		if (g->timing)
			w->t_bar_mid += bw_now_ms() - t0;
	}

	for (j = 0; j < g->num_workers; j++) {
		bw_worker_t *o = g->w + j;
		if (j == w->idx)
			continue;
		CUDA_CHK(w, cudaMemcpyAsync(
			xout + M->slabs[j].xfer_min, o->h_stage,
			(size_t)(M->slabs[j].row_max -
				M->slabs[j].xfer_min) * 8,
			cudaMemcpyHostToDevice, w->copy));
	}
	{
		double t0 = g->timing ? bw_now_ms() : 0;
		CUDA_CHK(w, cudaStreamSynchronize(w->copy));
		if (g->timing)
			w->t_copy_sync += bw_now_ms() - t0;
	}
}

/*-------------------------------------------------------------------*/
static void worker_do_step(bw_worker_t *w)
{
	bw_group_t *g = w->g;
	bw_matrix_t *M = g->M;
	bw_slab_t *slab = M->slabs + w->idx;
	uint64_t *xout = w->d_x[g->cur ^ 1];
	double t0 = g->timing ? bw_now_ms() : 0;

	worker_step_spmv(w);

	if (g->cmd == BW_CMD_KRYLOV_STEP) {
		if (getenv("BW_DBG_HS")) {
			static const uint64_t *last[8] = {0};
			if (last[w->idx] && last[w->idx] != w->h_s)
				fprintf(stderr, "[w%d] h_s CHANGED %p -> %p\n",
					w->idx, (void *)last[w->idx],
					(void *)w->h_s);
			last[w->idx] = w->h_s;
		}
		/* the oprod kernel XOR-accumulates into s */
		CUDA_CHK(w, cudaMemsetAsync(w->d_s, 0,
				g->o->m * 8, w->compute));
		bw_oprod64(w->d_s, w->d_u, xout + slab->row_min,
				g->o->m / 64, slab->row_max - slab->row_min,
				w->compute);
		/* rows [0,num_dense_rows) have no U coverage: the U block
		   is m x nrows but the dense rows are part of nrows, so
		   the oprod above misses them. The dense rows are genuine
		   state and must feed S too: run the oprod over the dense
		   prefix with the corresponding U words. */
		if ((w->idx == 0 || g->plane_split) && slab->row_min > 0)
			bw_oprod64(w->d_s, w->d_u_dense,
					xout,
					g->o->m / 64, slab->row_min,
					w->compute);
		CUDA_CHK(w, cudaEventRecord(w->ev_comp, w->compute));
		CUDA_CHK(w, cudaStreamWaitEvent(w->copy, w->ev_comp, 0));
		CUDA_CHK(w, cudaMemcpyAsync(w->h_s, w->d_s,
				g->o->m * 8,
				cudaMemcpyDeviceToHost, w->copy));
	}
	else if (g->cmd == BW_CMD_MKSOL_STEP) {
		CUDA_CHK(w, cudaMemcpyAsync(w->d_fcoef, g->fcoef,
				g->o->n * 8, cudaMemcpyHostToDevice,
				w->compute));
		/* plane split: only this worker's planes of the Horner
		   state are its business; mask the coefficient rows so
		   the combine never writes foreign planes */
		bw_combine64(xout, w->d_y0, w->d_fcoef,
				g->plane_split ? w->plane_mask : ~0ull,
				g->N, w->compute);
		CUDA_CHK(w, cudaEventRecord(w->ev_comp, w->compute));
		/* the allgather reads xout on the copy stream: it must
		 * wait for the compute stream's SpMV+combine (this was
		 * missing, and the staged D2H raced the kernels) */
		CUDA_CHK(w, cudaStreamWaitEvent(w->copy, w->ev_comp, 0));
	}
	else {	/* BW_CMD_NIL_STEP */
		bw_anyset64(xout + slab->xfer_min,
				w->d_hist[BW_HIST_DEPTH],
				slab->row_max - slab->xfer_min, w->compute);
		/* remember the predecessor while it is still x[cur] */
		CUDA_CHK(w, cudaMemcpyAsync(
			w->d_hist[g->hist_ptr % BW_HIST_DEPTH],
			w->d_x[g->cur] + slab->xfer_min,
			(size_t)(slab->row_max - slab->xfer_min) * 8,
			cudaMemcpyDeviceToDevice, w->compute));
		CUDA_CHK(w, cudaEventRecord(w->ev_comp, w->compute));
		CUDA_CHK(w, cudaStreamWaitEvent(w->copy, w->ev_comp, 0));
		CUDA_CHK(w, cudaMemcpyAsync(&w->h_flag,
			w->d_hist[BW_HIST_DEPTH], 8,
			cudaMemcpyDeviceToHost, w->copy));
		if (getenv("BW_DEBUG_NIL")) {
			uint64_t dbg = 0xfeed;
			cudaStreamSynchronize(w->copy);
			cudaMemcpy(&dbg, w->d_hist[BW_HIST_DEPTH], 8,
					cudaMemcpyDeviceToHost);
			printf("[dbg] w%d h_flag=%llx direct=%llx "
				"hist_ptr=%llu\n", w->idx,
				(unsigned long long)w->h_flag,
				(unsigned long long)dbg,
				(unsigned long long)g->hist_ptr);
			fflush(stdout);
		}
	}

	if (g->timing)
		w->t_submit += bw_now_ms() - t0;
	worker_allgather(w, g->cmd == BW_CMD_MKSOL_STEP ||
				!g->use_peer || g->need_stage);
}

/*-------------------------------------------------------------------*/
static int worker_alloc_and_upload(bw_worker_t *w)
{
	bw_group_t *g = w->g;
	bw_matrix_t *M = g->M;
	bw_slab_t *slab = M->slabs + w->idx;
	uint32_t words_per_row = g->o->m / 64;
	uint32_t i;
	size_t slab_words = slab->row_max - slab->xfer_min;

	/* CSR blocks */
	w->sd = (spmv_data_t *)xcalloc(slab->num_blocks,
					sizeof(spmv_data_t));
	w->num_sd = slab->num_blocks;
	for (i = 0; i < slab->num_blocks; i++) {
		bw_csrblock_t *b = slab->blocks + i;
		void *rp = NULL, *cp = NULL;
		w->sd[i].num_rows = (int)b->num_rows;
		w->sd[i].num_col_entries = (uint32_t)b->nnz;
		CUDA_CHK(w, cudaMalloc(&rp,
				(b->num_rows + 1) * sizeof(uint32_t)));
		CUDA_CHK(w, cudaMemcpy(rp, b->row_entries,
				(b->num_rows + 1) * sizeof(uint32_t),
				cudaMemcpyHostToDevice));
		if (b->nnz) {
			CUDA_CHK(w, cudaMalloc(&cp,
					b->nnz * sizeof(uint32_t)));
			CUDA_CHK(w, cudaMemcpy(cp, b->col_entries,
					b->nnz * sizeof(uint32_t),
					cudaMemcpyHostToDevice));
		}
		w->sd[i].row_entries = (CUdeviceptr)rp;
		w->sd[i].col_entries = (CUdeviceptr)cp;
	}

	/* state and replicas */
	CUDA_CHK(w, cudaMalloc((void **)&w->d_x[0],
			(size_t)g->N * 8));
	CUDA_CHK(w, cudaMalloc((void **)&w->d_x[1],
			(size_t)g->N * 8));
	CUDA_CHK(w, cudaMalloc((void **)&w->d_y0,
			(size_t)g->N * 8));
	CUDA_CHK(w, cudaMalloc((void **)&w->d_u,
			(size_t)slab_words * words_per_row * 8));
	CUDA_CHK(w, cudaMalloc((void **)&w->d_s, g->o->m * 8));
	CUDA_CHK(w, cudaMalloc((void **)&w->d_fcoef, g->o->n * 8));
	CUDA_CHK(w, cudaMalloc((void **)&w->d_hist[BW_HIST_DEPTH], 8));
	for (i = 0; i < BW_HIST_DEPTH; i++)
		CUDA_CHK(w, cudaMalloc((void **)&w->d_hist[i],
				slab_words * 8));
	if (M->num_dense_blocks) {
		CUDA_CHK(w, cudaMalloc((void **)&w->d_dense,
				(size_t)g->N * M->num_dense_blocks * 8));
		CUDA_CHK(w, cudaMemcpy(w->d_dense, M->dense,
				(size_t)g->N * M->num_dense_blocks * 8,
				cudaMemcpyHostToDevice));
	}

	CUDA_CHK(w, cudaHostAlloc((void **)&w->h_s, g->o->m * 8,
				cudaHostAllocDefault));
	if (getenv("BW_DBG_HS"))
		fprintf(stderr, "[w%d] h_s=%p after alloc\n", w->idx,
			(void *)w->h_s);
	CUDA_CHK(w, cudaHostAlloc((void **)&w->h_stage,
				slab_words * 8, cudaHostAllocDefault));
	if ((w->idx == 0 || g->plane_split) && g->N > M->nrows)
		CUDA_CHK(w, cudaHostAlloc((void **)&w->h_tail,
				(size_t)(g->N - M->nrows) * 8,
				cudaHostAllocDefault));

	/* Y0 and the initial state. Y0 stays full on every GPU (the
	   mksol combine reads all its planes); the running state carries
	   only this worker's planes in split mode */
	CUDA_CHK(w, cudaMemcpy(w->d_y0, g->h_y0,
			(size_t)g->N * 8, cudaMemcpyHostToDevice));
	if (g->plane_split && w->plane_mask != ~0ull) {
		uint64_t *tmp = (uint64_t *)xmalloc((size_t)g->N * 8);
		uint64_t i2;
		for (i2 = 0; i2 < g->N; i2++)
			tmp[i2] = g->h_y0[i2] & w->plane_mask;
		CUDA_CHK(w, cudaMemcpy(w->d_x[g->cur], tmp,
				(size_t)g->N * 8, cudaMemcpyHostToDevice));
		free(tmp);
	}
	else {
		CUDA_CHK(w, cudaMemcpy(w->d_x[g->cur], g->h_y0,
				(size_t)g->N * 8, cudaMemcpyHostToDevice));
	}
	CUDA_CHK(w, cudaMemcpy(w->d_u,
			g->h_u + (size_t)slab->row_min * words_per_row,
			(size_t)(slab->row_max - slab->row_min) *
				words_per_row * 8,
			cudaMemcpyHostToDevice));
	/* U words for the dense-row prefix of slab 0 (rows
	   [0,row_min)) sit BEFORE the main slab upload; give them
	   their own device buffer so the prefix oprod can read them */
	if (slab->row_min > 0) {
		CUDA_CHK(w, cudaMalloc((void **)&w->d_u_dense,
				(size_t)slab->row_min * words_per_row * 8));
		CUDA_CHK(w, cudaMemcpy(w->d_u_dense, g->h_u,
				(size_t)slab->row_min * words_per_row * 8,
				cudaMemcpyHostToDevice));
	}

	if (getenv("BW_DBG_HS"))
		fprintf(stderr, "[w%d] h_s=%p leaving alloc (err=%d)\n",
			w->idx, (void *)w->h_s, w->cuda_error);
	return w->cuda_error ? -1 : 0;
}

/*-------------------------------------------------------------------*/
static void worker_free(bw_worker_t *w)
{
	uint32_t i;
	if (w->spmv)
		spmv_engine_free(w->spmv);
	if (w->compute)
		cudaStreamDestroy(w->compute);
	if (w->copy)
		cudaStreamDestroy(w->copy);
	if (w->ev_comp)
		cudaEventDestroy(w->ev_comp);
	if (w->ev_comb)
		cudaEventDestroy(w->ev_comb);
	if (w->ev_t0)
		cudaEventDestroy(w->ev_t0);
	for (i = 0; i < w->num_sd; i++) {
		if (w->sd[i].row_entries)
			cudaFree((void *)(uintptr_t)w->sd[i].row_entries);
		if (w->sd[i].col_entries)
			cudaFree((void *)(uintptr_t)w->sd[i].col_entries);
	}
	free(w->sd);
	if (w->d_x[0]) cudaFree(w->d_x[0]);
	if (w->d_x[1]) cudaFree(w->d_x[1]);
	if (w->d_y0) cudaFree(w->d_y0);
	if (w->d_u) cudaFree(w->d_u);
	if (w->d_u_dense) cudaFree(w->d_u_dense);
	if (w->d_s) cudaFree(w->d_s);
	if (w->d_fcoef) cudaFree(w->d_fcoef);
	if (w->d_dense) cudaFree(w->d_dense);
	for (i = 0; i <= BW_HIST_DEPTH; i++)  /* includes the flag slot */
		if (w->d_hist[i]) cudaFree(w->d_hist[i]);
	if (w->h_s) cudaFreeHost(w->h_s);
	if (w->h_stage) cudaFreeHost(w->h_stage);
	if (w->h_tail) cudaFreeHost(w->h_tail);
	memset(&w->spmv, 0, sizeof(*w) - offsetof(bw_worker_t, spmv));
}

/*-------------------------------------------------------------------*/
static void *worker_main(void *arg)
{
	bw_worker_t *w = (bw_worker_t *)arg;
	bw_group_t *g = w->g;

	CUDA_CHK(w, cudaSetDevice(w->dev));
	CUDA_CHK(w, cudaStreamCreate(&w->compute));
	CUDA_CHK(w, cudaStreamCreate(&w->copy));
	CUDA_CHK(w, cudaEventCreate(&w->ev_comp));
	CUDA_CHK(w, cudaEventCreate(&w->ev_comb));
	if (g->timing)
		CUDA_CHK(w, cudaEventCreate(&w->ev_t0));
	w->spmv = spmv_engine_init(NULL);

	/* peer probe and (best-effort) enable */
	{
		int j, ok;
		for (j = 0; j < g->num_workers; j++) {
			if (j == w->idx)
				continue;
			if (cudaDeviceCanAccessPeer(&ok, w->dev, g->w[j].dev)
					== cudaSuccess)
				g->can_peer[w->idx][j] = ok;
			if (cudaDeviceEnablePeerAccess(g->w[j].dev, 0)
					!= cudaSuccess) {
				/* not fatal: already enabled or unsupported */
				cudaGetLastError();
			}
		}
	}

	while (1) {
		int stop;
		double tb;

		tb = g->timing ? bw_now_ms() : 0;
		pthread_barrier_wait(&g->b_start);
		if (g->timing)
			w->t_bar_start += bw_now_ms() - tb;

		stop = (g->cmd == BW_CMD_STOP);

		if (!stop && w->cuda_error)  /* keep barrier count balanced */
			goto end_of_step;

		switch (g->cmd) {
		case BW_CMD_UPLOAD_INIT:
			pthread_barrier_wait(&g->b_mid);
			if (w->idx == 0) {
				int i, j, all = 1;
				for (i = 0; i < g->num_workers; i++)
					for (j = 0; j < g->num_workers; j++)
						if (i != j &&
						    !g->can_peer[i][j])
							all = 0;
				g->use_peer = (g->o->peer != 0) && all;
				if (g->o->peer == 1 && !all)
					printf("BW: bw_peer=1 requested "
						"but not all GPU pairs "
						"have peer access\n");
			}
			pthread_barrier_wait(&g->b_mid);
			if (worker_alloc_and_upload(w)) {
				/* main sees cuda_error at b_end */
			}
			break;

		case BW_CMD_UPLOAD_X:
			if (g->plane_split && w->plane_mask != ~0ull) {
				uint64_t *tmp = (uint64_t *)xmalloc(
							(size_t)g->N * 8);
				uint64_t i2;
				for (i2 = 0; i2 < g->N; i2++)
					tmp[i2] = g->h_x[i2] & w->plane_mask;
				CUDA_CHK(w, cudaMemcpy(w->d_x[g->cur],
						tmp, (size_t)g->N * 8,
						cudaMemcpyHostToDevice));
				free(tmp);
			}
			else {
				CUDA_CHK(w, cudaMemcpy(w->d_x[g->cur],
						g->h_x, (size_t)g->N * 8,
						cudaMemcpyHostToDevice));
			}
			break;

		case BW_CMD_KRYLOV_STEP:
		case BW_CMD_MKSOL_STEP:
		case BW_CMD_NIL_STEP:
			worker_do_step(w);
			break;

		case BW_CMD_COLLECT:
			{
				bw_slab_t *slab = g->M->slabs + w->idx;
				CUDA_CHK(w, cudaMemcpy(w->h_stage,
					w->d_x[g->cur] + slab->xfer_min,
					(size_t)(slab->row_max -
						slab->xfer_min) * 8,
					cudaMemcpyDeviceToHost));
				if (w->h_tail)
					CUDA_CHK(w, cudaMemcpy(w->h_tail,
						w->d_x[g->cur] +
							g->M->nrows,
						(size_t)(g->N -
							g->M->nrows) * 8,
						cudaMemcpyDeviceToHost));
			}
			break;

		default:
			break;
		}

end_of_step:
		tb = g->timing ? bw_now_ms() : 0;
		pthread_barrier_wait(&g->b_end);
		if (g->timing)
			w->t_bar_end += bw_now_ms() - tb;
		if (stop)
			break;
	}

	worker_free(w);
	return NULL;
}

/*-------------------------------------------------------------------*/
int bw_gpu_start(bw_group_t *g, bw_matrix_t *M, const bw_opts_t *o,
		msieve_obj *obj, const uint64_t *h_y0, const uint64_t *h_u)
{
	int i;
	bw_matrix_t *Mx = M;

	memset(g, 0, sizeof(bw_group_t));
	g->timing = (getenv("BW_TIME") != NULL);
	g->plane_split = (o->split != 0 && o->num_gpus > 1);
	g->M = Mx;
	g->o = o;
	g->N = Mx->ncols;
	g->num_workers = o->num_gpus;
	g->h_y0 = h_y0;
	g->h_u = h_u;
	g->cur = 0;

	/* worker 0 needs the slabs in order before startup; the matrix
	   packer already ordered slabs to match the gpu list */
	for (i = 0; i < g->num_workers; i++) {
		g->slab_words[i] = Mx->slabs[i].row_max -
					Mx->slabs[i].xfer_min;
		g->slab_off[i] = Mx->slabs[i].xfer_min;
	}

	g->w = (bw_worker_t *)xcalloc(g->num_workers, sizeof(bw_worker_t));
	for (i = 0; i < g->num_workers; i++) {
		g->w[i].idx = i;
		g->w[i].dev = o->gpus[i];
		g->w[i].g = g;
		if (g->plane_split) {
			g->w[i].plane_lo = (64 * i) / g->num_workers;
			g->w[i].plane_hi = (64 * (i + 1)) /
						g->num_workers;
		}
		else {
			g->w[i].plane_lo = 0;
			g->w[i].plane_hi = 64;
		}
		g->w[i].plane_mask = (g->w[i].plane_hi > g->w[i].plane_lo)
			? ((g->w[i].plane_hi - g->w[i].plane_lo == 64)
				? ~0ull
				: (((1ull << (g->w[i].plane_hi -
					g->w[i].plane_lo)) - 1) <<
					g->w[i].plane_lo))
			: 0ull;
	}

	pthread_barrier_init(&g->b_start, NULL, g->num_workers + 1);
	pthread_barrier_init(&g->b_mid, NULL, g->num_workers);
	pthread_barrier_init(&g->b_end, NULL, g->num_workers + 1);

	for (i = 0; i < g->num_workers; i++) {
		if (pthread_create(&g->w[i].thread, NULL,
					worker_main, g->w + i)) {
			logprintf(obj, "error: BW cannot spawn worker "
					"threads\n");
			return -1;
		}
	}

	/* UPLOAD_INIT round: workers set devices, probe peers, then
	   allocate and upload */
	bw_gpu_run(g, BW_CMD_UPLOAD_INIT);

	for (i = 0; i < g->num_workers; i++)
		if (g->w[i].cuda_error) {
			logprintf(obj, "error: BW worker %d failed to "
					"initialize GPU %d\n", i, o->gpus[i]);
			return -1;
		}

	logprintf(obj, "BW gpu group: %d workers, allgather %s\n",
			g->num_workers, g->use_peer ? "peer" : "staged");
	(void)obj;
	return 0;
}

/*-------------------------------------------------------------------*/
void bw_gpu_run(bw_group_t *g, bw_cmd_t cmd) {
	g->cmd = cmd;
	pthread_barrier_wait(&g->b_start);
	pthread_barrier_wait(&g->b_end);
}

/*-------------------------------------------------------------------*/
void bw_gpu_stop(bw_group_t *g) {
	int i;

	bw_gpu_run(g, BW_CMD_STOP);
	for (i = 0; i < g->num_workers; i++)
		pthread_join(g->w[i].thread, NULL);

	pthread_barrier_destroy(&g->b_start);
	pthread_barrier_destroy(&g->b_mid);
	pthread_barrier_destroy(&g->b_end);
	free(g->w);
	g->w = NULL;
}

/*-------------------------------------------------------------------*/
/* BW_TIME summary. Call with the workers parked at b_start (between
   driver phases); prints per-worker phase averages and the per-step
   exchange volumes implied by the slab geometry. */
void bw_gpu_time_report(bw_group_t *g, msieve_obj *obj, const char *phase)
{
	int i, j;

	if (!g->timing)
		return;
	logprintf(obj, "BW time [%s]: %d workers\n", phase,
			g->num_workers);
	for (i = 0; i < g->num_workers; i++) {
		bw_worker_t *w = g->w + i;
		uint64_t n = w->n_steps ? w->n_steps : 1;
		uint64_t out_b = 0, in_b = 0;
		if (!g->plane_split && g->num_workers > 1) {
			for (j = 0; j < g->num_workers; j++) {
				bw_slab_t *s = g->M->slabs + j;
				uint64_t bytes = (uint64_t)(s->row_max -
						s->xfer_min) * 8;
				if (j == i)
					out_b += bytes;
				else
					in_b += bytes;
			}
			out_b *= (g->num_workers - 1);
		}
		logprintf(obj, "  w%d(gpu%d): %llu steps | "
			"submit %.3f sync %.3f barmid %.3f barend %.3f "
			"barstart %.3f | gpu_compute %.3f ms/step | "
			"xfer %.1f out + %.1f in MB/step\n",
			i, w->dev, (unsigned long long)w->n_steps,
			w->t_submit / n, w->t_copy_sync / n,
			w->t_bar_mid / n, w->t_bar_end / n,
			w->t_bar_start / n, w->t_gpu_compute / n,
			(double)out_b / 1e6,
			(double)in_b / 1e6);
	}
}

void bw_gpu_time_reset(bw_group_t *g)
{
	int i;

	if (!g->timing)
		return;
	for (i = 0; i < g->num_workers; i++) {
		bw_worker_t *w = g->w + i;
		w->t_submit = w->t_copy_sync = w->t_bar_mid = 0;
		w->t_bar_end = w->t_bar_start = w->t_gpu_compute = 0;
		w->n_steps = 0;
	}
}
