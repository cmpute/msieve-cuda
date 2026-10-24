/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

Host-side packaging of the msieve matrix for the BW GPU workers. The
block extraction, radix sort and COO->CSR conversion are lifted from
common/lanczos/gpu/lanczos_matmul_gpu.c (same public-domain tree) and
adapted: rows are split into contiguous per-GPU slabs first, then each
slab is packed into column-window blocks whose *column window width*
targets block_nnz nonzeros of the full matrix, so the L2 gather window
tuning of the single-GPU Lanczos path carries over unchanged.
--------------------------------------------------------------------*/

#include <common.h>
#include "bw_matrix.h"
#include "bw.h"

/* Largest block_nnz accepted: one block's nonzero count must fit the
   uint32 CSR ABI of spmv_engine (same ceiling as lanczos_matmul_gpu) */
#define BW_MAX_BLOCK_NNZ 4000000000ull

typedef struct {
	uint32_t row_off;
	uint32_t col_off;
} entry_idx_t;

/*-------------------------------------------------------------------*/
static uint64_t bw_extract_block(la_col_t *cols,
			uint32_t row_min, uint32_t row_max,
			uint32_t col_min, uint32_t col_max,
			uint64_t nnz, uint32_t *blocksize,
			entry_idx_t **entries_in,
			uint64_t *max_entries_in)
{
	uint32_t i, j;
	uint64_t num_entries = 0;
	entry_idx_t *entries = *entries_in;
	uint64_t max_entries = *max_entries_in;

	if (entries == NULL)      /* first block of the slab */
		entries = (entry_idx_t *)xmalloc(
				max_entries * sizeof(entry_idx_t));

	for (i = col_min; (i < col_max) && (num_entries < nnz); i++) {

		la_col_t *col = cols + i;

		for (j = 0; j < col->weight; j++) {
			uint32_t idx = col->data[j];

			if (idx >= row_max)
				break;

			if (idx >= row_min) {

				entry_idx_t *e;

				if (num_entries == max_entries) {
					max_entries *= 2;
					entries = (entry_idx_t *)xrealloc(
							entries,
							max_entries *
							sizeof(entry_idx_t));
				}

				e = entries + num_entries++;
				e->row_off = idx;
				e->col_off = i;
			}
		}
	}

	*blocksize = i - col_min;
	*entries_in = entries;
	*max_entries_in = max_entries;
	return num_entries;
}

/*-------------------------------------------------------------------*/
static void bw_radix_sort(entry_idx_t *arr, uint64_t n) {

	/* 8-byte keys (row<<32|col), 8 radix passes; same as
	   lanczos_matmul_gpu.c radix_sort() */

	uint64_t i;
	uint32_t pass, skip;
	uint64_t *a, *b, *from, *to, *temp;

	a = (uint64_t *) xmalloc(n * sizeof(uint64_t));
	b = (uint64_t *) xmalloc(n * sizeof(uint64_t));

	for (i = 0; i < n; i++) {
		entry_idx_t *e = arr + i;
		a[i] = ((uint64_t)(e->row_off) << 32) | (uint64_t)(e->col_off);
	}

	from = a;
	to = b;
	skip = 0;
	for (pass = 0; pass < 8; pass++)  {
		uint64_t box[256] = { 0 };

		for (i = 0; i < n; i++) box[ (from[i] >> (8*pass)) & 255]++;
		if (box[0] == n) { /* this word is all 0's, don't need to sort */
			skip++;
			continue;
		}
		for (i = 1; i < 256; i++) box[i] += box[i-1];
		for (i = n - 1; i != (uint64_t)(-1); i--) to[--box[(from[i] >> (8*pass)) & 255]] = from[i];

		temp = from;
		from = to;
		to = temp;
	}

	if (skip & 1) to = b;
	else to = a;
	for (i = 0; i < n; i++) {
		entry_idx_t *e = arr + i;
		e->row_off = (uint32_t)(to[i] >> 32);
		e->col_off = (uint32_t)(to[i]);
	}

	free(a);
	free(b);
}

/*-------------------------------------------------------------------*/
static int bw_pack_block(bw_csrblock_t *b,
			entry_idx_t *entries, uint64_t num_entries,
			uint32_t row_min, uint32_t col_min,
			uint32_t num_rows, uint32_t blocksize)
{
	uint64_t i, j;
	uint32_t *col_entries;
	uint32_t *row_entries;

	if (num_entries > BW_MAX_BLOCK_NNZ) {
		printf("error: BW matrix block holds %" PRIu64 " nonzeros, "
			"above what a CSR block can address; lower "
			"bw_block_nnz\n", num_entries);
		return -1;
	}

	if (num_entries == 0) {
		/* slab rows with no nonzeros in this window: an empty CSR
		   keeps num_rows so the descriptor stays well-formed */
		memset(b, 0, sizeof(*b));
		b->num_rows = num_rows;
		b->num_cols = blocksize;
		b->col_min = col_min;
		b->nnz = 0;
		b->row_entries = (uint32_t *)xcalloc(num_rows + 1,
							sizeof(uint32_t));
		b->col_entries = NULL;
		return 0;
	}

	col_entries = (uint32_t *)xmalloc(num_entries * sizeof(uint32_t));
	row_entries = (uint32_t *)xcalloc(num_rows + 1, sizeof(uint32_t));

	bw_radix_sort(entries, num_entries);

	for (i = j = 0; i < num_entries; i++, j++) {

		entry_idx_t *e = entries + i;

		col_entries[i] = e[0].col_off - col_min;

		if (i > 0 && e[0].row_off != e[-1].row_off) {
			row_entries[e[-1].row_off - row_min] = (uint32_t)j;
			j = 0;
		}
	}
	row_entries[entries[i-1].row_off - row_min] = (uint32_t)j;

	for (i = j = 0; i < num_rows; i++) {
		uint32_t t = row_entries[i];
		row_entries[i] = (uint32_t)j;
		j += t;
	}
	row_entries[num_rows] = (uint32_t)num_entries;

	b->num_rows = num_rows;
	b->num_cols = blocksize;
	b->col_min = col_min;
	b->nnz = num_entries;
	b->row_entries = row_entries;
	b->col_entries = col_entries;
	return 0;
}

/*-------------------------------------------------------------------*/
#define BW_BAND_SHIFT 12   /* band granularity cap: 4096 rows */

static int bw_choose_slabs(bw_matrix_t *M, la_col_t *cols,
			uint32_t num_slabs, uint32_t *bounds)
{
	/* balance the P row slabs by nonzero count: one pass over cols
	   builds a per-band histogram, then greedy boundaries pick the
	   cut points with equal nnz above num_dense_rows (dense rows go
	   to slab 0's sparse CSR start, per the design). The band size
	   adapts down for small matrices so P distinct cuts always
	   exist; the fallback balances raw row counts. */

	uint32_t shift = BW_BAND_SHIFT;
	uint32_t nbands, b;
	uint64_t *band_nnz;
	uint32_t i, j;
	uint64_t total = 0, accum = 0, target;
	int s;

	while ((M->nrows >> shift) < 16u * num_slabs && shift > 3)
		shift--;

	nbands = (M->nrows + ((uint32_t)1 << shift) - 1) >> shift;
	band_nnz = (uint64_t *)xcalloc(nbands, sizeof(uint64_t));

	for (i = 0; i < M->ncols; i++) {
		la_col_t *col = cols + i;
		for (j = 0; j < col->weight; j++) {
			uint32_t idx = col->data[j];
			if (idx < M->num_dense_rows)
				continue;      /* dense rows not in slabs */
			band_nnz[idx >> shift]++;
			total++;
		}
	}

	M->sparse_weight = total;

	bounds[0] = M->num_dense_rows;
	bounds[num_slabs] = M->nrows;

	s = 1;
	accum = 0;
	target = total / num_slabs;
	for (b = 0; b < nbands && s < (int)num_slabs; b++) {
		accum += band_nnz[b];
		if (accum >= (uint64_t)s * target &&
		    ((b + 1) << shift) > M->num_dense_rows) {
			uint32_t cut = (b + 1) << shift;
			if (cut > M->nrows)
				cut = M->nrows;
			if (cut > bounds[s-1])
				bounds[s++] = cut;
		}
	}

	/* fallback: fill any remaining cuts by equal row counts */
	while (s < (int)num_slabs) {
		uint64_t span = (uint64_t)M->nrows - bounds[s-1];
		uint32_t need = num_slabs - s;
		uint32_t cut = bounds[s-1] + (uint32_t)((span + need - 1) / need);
		if (cut >= M->nrows)
			cut = M->nrows - 1;
		if (cut <= bounds[s-1])
			cut = bounds[s-1] + 1;
		bounds[s++] = cut;
	}

	for (s = 1; s <= (int)num_slabs; s++)
		if (bounds[s] <= bounds[s-1])
			return -1;

	free(band_nnz);
	return 0;
}

/*-------------------------------------------------------------------*/
static int bw_build_dense(bw_matrix_t *M, la_col_t *cols)
{
	/* convert msieve's per-column dense bitfields (after weight
	   entries: ceil(num_dense_rows/32) uint32 words, row-major
	   within the column) into per-64-row bit planes: dense[k*ncols+j]
	   bit r = dense row 64k+r, column j */

	uint32_t ndr = M->num_dense_rows;
	uint32_t dense_row_words = (ndr + 31) / 32;
	uint32_t ndb = (ndr + 63) / 64;
	uint64_t *d;
	uint32_t i, k;
	uint64_t w = 0;

	if (ndr == 0) {
		M->dense = NULL;
		M->num_dense_blocks = 0;
		M->dense_weight = 0;
		return 0;
	}

	d = (uint64_t *)xcalloc((size_t)ndb * M->ncols, sizeof(uint64_t));

	for (i = 0; i < M->ncols; i++) {
		la_col_t *col = cols + i;
		uint32_t *src = col->data + col->weight;

		for (k = 0; k < dense_row_words; k++) {
			uint32_t t = src[k];
			uint32_t row0 = 32 * k;
			uint32_t kb = row0 >> 6;
			uint32_t shift = row0 & 63;
			d[(uint64_t)kb * M->ncols + i] |=
				(uint64_t)t << shift;
			if (shift && kb + 1 < ndb)
				d[(uint64_t)(kb+1) * M->ncols + i] |=
					(uint64_t)t >> (64 - shift);
			w += __builtin_popcount(t);
		}
	}

	M->dense = d;
	M->num_dense_blocks = ndb;
	M->dense_weight = w;
	if (ndr > 512)
		return 1;    /* caller warns */
	return 0;
}

/*-------------------------------------------------------------------*/
int bw_matrix_build(bw_matrix_t *M, msieve_obj *obj,
		uint32_t nrows, uint32_t num_dense_rows, uint32_t ncols,
		la_col_t *cols, uint32_t num_slabs, uint64_t block_nnz)
{
	uint32_t bounds[BW_MAX_GPUS + 1];
	uint32_t s, col_min;
	uint64_t sparse_total = 0;
	uint64_t avg_col_weight;
	int dense_warn;

	memset(M, 0, sizeof(bw_matrix_t));
	M->nrows = nrows;
	M->ncols = ncols;
	M->num_dense_rows = num_dense_rows;
	M->num_slabs = num_slabs;

	if (num_slabs < 1 || num_slabs > BW_MAX_GPUS)
		return -1;
	if (num_dense_rows >= nrows)
		return -1;

	dense_warn = bw_build_dense(M, cols);

	if (bw_choose_slabs(M, cols, num_slabs, bounds))
		return -1;

	avg_col_weight = M->sparse_weight ? (M->sparse_weight + ncols - 1) / ncols : 1;
	if (avg_col_weight == 0)
		avg_col_weight = 1;

	/* clamp the requested block_nnz the same way the Lanczos packer
	   does, but as a *window width* target: the per-slab extraction
	   below asks for block_nnz * (slab nnz fraction) nonzeros */
	if (block_nnz < 100000)
		block_nnz = 100000;
	if (block_nnz > BW_MAX_BLOCK_NNZ)
		block_nnz = BW_MAX_BLOCK_NNZ;

	M->slabs = (bw_slab_t *)xcalloc(num_slabs, sizeof(bw_slab_t));

	logprintf(obj, "BW matrix pack: %u rows x %u cols, %u dense rows, "
			"%" PRIu64 " sparse + %" PRIu64 " dense nnz, "
			"%u slabs\n",
			nrows, ncols, num_dense_rows,
			M->sparse_weight, M->dense_weight, num_slabs);

	for (s = 0; s < num_slabs; s++) {
		bw_slab_t *slab = M->slabs + s;
		uint32_t row_min = bounds[s];
		uint32_t row_max = bounds[s+1];
		uint64_t slab_frac_nnz;
		uint32_t max_blocks = 0;
		entry_idx_t *entries = NULL;
		uint64_t max_entries = 1 << 20;

		slab->row_min = row_min;
		slab->row_max = row_max;
		slab->xfer_min = (s == 0) ? 0 : row_min;

		/* count slab nnz to size the extraction target */
		{
			uint32_t i, j;
			slab->total_nnz = 0;
			for (i = 0; i < ncols; i++) {
				la_col_t *col = cols + i;
				for (j = 0; j < col->weight; j++) {
					uint32_t idx = col->data[j];
					if (idx >= row_max)
						break;
					if (idx >= row_min)
						slab->total_nnz++;
				}
			}
		}
		sparse_total += slab->total_nnz;

		/* slab fraction of the full matrix: extraction target that
		   keeps the full-matrix column window width at block_nnz */
		slab_frac_nnz = block_nnz * slab->total_nnz /
					(M->sparse_weight ? M->sparse_weight : 1);
		if (slab_frac_nnz == 0)
			slab_frac_nnz = 1;

		max_blocks = 64;
		slab->blocks = (bw_csrblock_t *)xmalloc(max_blocks *
						sizeof(bw_csrblock_t));

		col_min = 0;
		while (col_min < ncols) {
			uint32_t blocksize;
			uint64_t n = bw_extract_block(cols, row_min, row_max,
						col_min, ncols,
						slab_frac_nnz, &blocksize,
						&entries, &max_entries);
			bw_csrblock_t *b;

			if (slab->num_blocks == max_blocks) {
				max_blocks *= 2;
				slab->blocks = (bw_csrblock_t *)xrealloc(
						slab->blocks,
						max_blocks *
						sizeof(bw_csrblock_t));
			}
			b = slab->blocks + slab->num_blocks;

			if (bw_pack_block(b, entries, n, row_min, col_min,
					row_max - row_min, blocksize))
				return -1;

			slab->num_blocks++;
			col_min += blocksize;
			if (blocksize == 0) {   /* cannot happen: nnz>=1 */
				return -1;
			}
		}

		free(entries);
		logprintf(obj, "BW slab %u: rows [%u,%u), %" PRIu64 " nnz, "
				"%u blocks\n",
				s, row_min, row_max, slab->total_nnz,
				slab->num_blocks);
	}

	if (dense_warn)
		logprintf(obj, "warning: %u dense rows is above the "
				"expected 64-72 range\n", num_dense_rows);

	if (sparse_total != M->sparse_weight) {
		logprintf(obj, "error: BW slab nnz total %" PRIu64
				" != histogram %" PRIu64 "\n",
				sparse_total, M->sparse_weight);
		return -1;
	}

	return 0;
}

/*-------------------------------------------------------------------*/
void bw_matrix_free(bw_matrix_t *M) {
	uint32_t s, b;
	for (s = 0; s < M->num_slabs; s++) {
		bw_slab_t *slab = M->slabs + s;
		for (b = 0; b < slab->num_blocks; b++) {
			free(slab->blocks[b].row_entries);
			free(slab->blocks[b].col_entries);
		}
		free(slab->blocks);
	}
	free(M->slabs);
	free(M->dense);
	memset(M, 0, sizeof(bw_matrix_t));
}
