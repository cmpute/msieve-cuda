/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

Block Wiedemann linear algebra for msieve's NFS postprocessing, as an
opt-in replacement for Block Lanczos. One process owns every local GPU
(one worker thread each); only forward SpMV is ever needed. See BW.md
in the repository root for the design document.

This is the only header gnfs/gf2.c includes.
--------------------------------------------------------------------*/

#ifndef _COMMON_BW_BW_H_
#define _COMMON_BW_BW_H_

/* included from gf2.c, which pulls common.h first */

#define BW_MAX_GPUS 8

typedef struct {
	int enabled;
	uint32 m;               /* generator block height, multiple of 64 */
	uint32 n;               /* generator block width; == 64 (== VBITS) */
	int gpus[BW_MAX_GPUS];  /* CUDA device ordinals */
	int num_gpus;
	uint64 block_nnz;       /* column-window CSR block target (nnz of the
				   full matrix per window, so the gather
				   window width is P-independent) */
	int peer;               /* -1 auto, 0 disable, 1 require */
	int allgather_hier;     /* 0 flat, 1 hierarchical (PIX pairs) */
	uint32 lingen_threshold;/* matrix-multiply recursion threshold */
	uint32 checkpoint_s;    /* seconds between checkpoints */
	uint64 seed;            /* extra seed material; 0 = derive */
	int retry;              /* retry Krylov with a new seed on failure */
	int verify_cpu;         /* verify candidates against cols[] on CPU */
	int keep;               /* keep .bw.* files on success */
} bw_opts_t;

void bw_opts_defaults(bw_opts_t *o);

/* parse bw_* keys from an nfs_args-style option string plus MSIEVE_BW_*
   environment variables. Returns 0 on success, -1 on a hard error
   (nothing is silently clamped); on error a reason is logged */
int bw_opts_parse(bw_opts_t *o, msieve_obj *obj, const char *args);

/* the BW counterpart of block_lanczos(). Takes ownership of cols[i].data
   (frees them after device upload); cols[] itself is left alone. On
   return *deps_found is the number of usable dependencies and the return
   value is the dependencies array (max_ncols words, dump_dependencies
   format), or NULL on failure */
uint64 *bw_solve(msieve_obj *obj,
		uint32 nrows, uint32 max_nrows, uint32 start_row,
		uint32 num_dense_rows,
		uint32 ncols, uint32 max_ncols, uint32 start_col,
		la_col_t *cols, uint32 *deps_found);

#endif /* !_COMMON_BW_BW_H_ */
