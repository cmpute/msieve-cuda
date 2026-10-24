/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

Host-side ingestion of the msieve matrix and the per-GPU row-slab
column-window CSR packaging.
--------------------------------------------------------------------*/

#ifndef _COMMON_BW_BW_MATRIX_H_
#define _COMMON_BW_BW_MATRIX_H_

/* one column-window CSR block of one row slab. The layout matches what
   cub/spmv_engine consumes: row_entries[num_rows+1] CSR pointers and
   col_entries[nnz] column indices, columns relative to col_min so the
   SpMV gather reads x[col_min .. col_min+num_cols) only */
typedef struct {
	uint32_t num_rows;      /* rows in the block (== slab height) */
	uint32_t num_cols;      /* column window width */
	uint32_t col_min;       /* window start, absolute column */
	uint64_t nnz;           /* must fit uint32 (spmv ABI); enforced */
	uint32_t *row_entries;  /* num_rows+1 */
	uint32_t *col_entries;  /* nnz */
} bw_csrblock_t;

typedef struct {
	uint32_t row_min;       /* absolute row range of the slab */
	uint32_t row_max;
	uint32_t xfer_min;      /* state exchange window start: 0 for
				   slab 0 (its dense rows [0,num_dense_rows)
				   are real state), row_min otherwise */
	bw_csrblock_t *blocks;
	uint32_t num_blocks;
	uint64_t total_nnz;
} bw_slab_t;

typedef struct {
	uint32_t nrows;         /* unpadded rows */
	uint32_t ncols;         /* N = ncols, padded dimension */
	uint32_t num_dense_rows;
	uint64_t sparse_weight;
	uint64_t dense_weight;

	/* dense bitmap converted to one word per column per 64 dense
	   rows: bit r of dense[64k*ncolsw + j] is dense row 64k+r,
	   column j. ncolum words per block, ceil(ndr/64) blocks */
	uint64_t *dense;
	uint32_t num_dense_blocks;

	bw_slab_t *slabs;
	uint32_t num_slabs;
} bw_matrix_t;

/* build the packaging. Splits rows [0,nrows) into num_slabs contiguous
   slabs balanced by nonzero count (band histogram + greedy boundaries),
   then packs each slab into column-window blocks targeting block_nnz
   nonzeros of the *full-matrix* column window (so the gather window
   width stays P-independent), reusing msieve's extract_block /
   pack_matrix_block / radix_sort logic. Does not free cols[] data. */
int bw_matrix_build(bw_matrix_t *M, msieve_obj *obj,
		uint32_t nrows, uint32_t num_dense_rows, uint32_t ncols,
		la_col_t *cols, uint32_t num_slabs, uint64_t block_nnz,
		int l2_dev);

void bw_matrix_free(bw_matrix_t *M);   /* host packaging only */

#endif /* !_COMMON_BW_BW_MATRIX_H_ */
