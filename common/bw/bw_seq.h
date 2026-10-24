/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

On-disk formats shared by the Krylov driver, the lingen port and the
checkpoint code. Everything is host-endian little; these files are
internal to one factorization run.

A file  {savefile}.bw.A : the sequence S_0..S_{L-1}, coefficient j at
       byte offset HDR + j*m*8. Coefficient j is m words; bit b of word
       a is S_j[a][b] (row a of the m-by-n coefficient, n <= 64 bits).
       Matches CADO lingen_io_matpoly<true> with transpose=0.

F file  {savefile}.bw.F : the generator F in CADO's convention,
       leading coefficient first, degree d = count-1; coefficient is
       n words, bit b of word a is F_c[a][b].

chk     {savefile}.bw.chk : see bw_checkpoint.h. State vector is
       N = ncols words (the full replicated bit-sliced block).
--------------------------------------------------------------------*/

#ifndef _COMMON_BW_BW_SEQ_H_
#define _COMMON_BW_BW_SEQ_H_

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BW_FORMAT_VERSION 1

#define BW_A_MAGIC   0x31574241u   /* "ABW1" */
#define BW_F_MAGIC   0x31574246u   /* "FBW1" */
#define BW_CHK_MAGIC 0x31574243u   /* "CBW1" */
#define BW_EOF_MAGIC 0x21464F45u   /* "EOF!" completeness sentinel */

/* fixed 64-byte header for A and F files */
typedef struct {
	uint32_t magic;         /* BW_A_MAGIC or BW_F_MAGIC */
	uint32_t version;       /* BW_FORMAT_VERSION */
	uint32_t m, n;          /* block dimensions */
	uint32_t nrows;         /* matrix rows (unpadded) */
	uint32_t ncols;         /* N, matrix columns (padded dim) */
	uint32_t reserved0;     /* padding to keep 8-alignment */
	uint32_t num_dense_rows;
	uint64_t seq_len;       /* A: L. F: number of coefficients */
	uint64_t seed;          /* the seed material used for Y0, U */
	uint64_t reserved[4];
} bw_filehdr_t;

/* checkpoint header, fixed size */
typedef struct {
	uint32_t magic;         /* BW_CHK_MAGIC */
	uint32_t version;
	uint32_t m, n;
	uint32_t nrows;
	uint32_t ncols;
	uint32_t num_dense_rows;
	uint32_t phase;         /* 0 = krylov, 1 = mksol */
	uint32_t reserved0;
	uint64_t step;          /* next step to execute in that phase */
	uint64_t seq_len;       /* L for the A file this belongs to */
	uint64_t seed;
	uint64_t a_offset;      /* valid bytes of the A file (hdr included) */
	uint64_t hash_y0;       /* FNV-1a of the Y0 block (ncols words) */
	uint64_t hash_u;        /* FNV-1a of the U block (m*nrows bits) */
	uint64_t hash_s;        /* FNV-1a of the S prefix written so far */
	uint64_t hash_x;        /* FNV-1a of the checkpointed state */
	uint64_t reserved[3];
} bw_chkhdr_t;

/* follows: ncols words of state (the x[cur] block), then the 8-byte
   BW_EOF_MAGIC sentinel located by seeking from EOF */

#define BW_HDR_SIZE 64

static inline void bw_put_hdr(FILE *f, const bw_filehdr_t *h) {
	unsigned char pad[BW_HDR_SIZE];
	size_t used = 0;
	memset(pad, 0, sizeof(pad));
	fwrite(h, 1, sizeof(bw_filehdr_t), f);
	used = sizeof(bw_filehdr_t);
	if (used < BW_HDR_SIZE)
		fwrite(pad, 1, BW_HDR_SIZE - used, f);
}

static inline int bw_get_hdr(FILE *f, bw_filehdr_t *h) {
	unsigned char pad[BW_HDR_SIZE];
	if (fread(h, 1, sizeof(bw_filehdr_t), f) != sizeof(bw_filehdr_t))
		return -1;
	if (sizeof(bw_filehdr_t) < BW_HDR_SIZE &&
	    fread(pad, 1, BW_HDR_SIZE - sizeof(bw_filehdr_t), f) !=
			BW_HDR_SIZE - sizeof(bw_filehdr_t))
		return -1;
	return 0;
}

/* FNV-1a over raw bytes; the golden-hash verification cascade */
static inline uint64_t bw_fnv1a(uint64_t h, const void *data, size_t n) {
	const unsigned char *p = (const unsigned char *)data;
	size_t i;
	if (h == 0)
		h = 1469598103934665603ULL;
	for (i = 0; i < n; i++) {
		h ^= p[i];
		h *= 1099511628211ULL;
	}
	return h;
}

/* deterministic PRNG for Y0/U (splitmix64) */
typedef struct { uint64_t x; } bw_rng_t;

static inline uint64_t bw_rng_next(bw_rng_t *r) {
	uint64_t z = (r->x += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

#endif /* !_COMMON_BW_BW_SEQ_H_ */
