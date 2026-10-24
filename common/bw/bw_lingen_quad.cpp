/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

Host driver for the quadratic GPU lingen basecase (algorithm in
bw_lingen_quad.h). Rows are ABSOLUTE-coefficient arenas: an "X
multiply" is a block move the GPU does once per step for all pivot
rows, and row XORs need no alignment. P = pi * E^T is maintained
incrementally because it is linear in the rows of pi.
--------------------------------------------------------------------*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <vector>
#include <algorithm>

#include "bw_lingen_quad.h"

struct bwq_op_t { uint32_t dst, src, cext, cextP; };
struct bwq_arenas_t { unsigned long long *F, *P;
		uint64_t rowcapF, rowcapP; uint32_t wpcF, wpcP; };

extern "C" {
int bwq_xor_pivot_launch(const bwq_arenas_t *, const bwq_arenas_t *,
		const bwq_op_t *, uint32_t, bwq_op_t *, void *);
int bwq_shift_rows_launch(const bwq_arenas_t *,
		const uint32_t *, const uint32_t *, const uint32_t *,
		uint32_t, uint32_t *, uint32_t *, uint32_t *, void *);
int bwq_gather_delta_launch(const unsigned long long *, uint64_t,
		uint32_t, uint32_t, uint32_t, unsigned long long *, void *);
int bwq_upload(const void *, void *, uint64_t, void *);
int bwq_download(void *, const void *, uint64_t, void *);
int bwq_alloc(void **, uint64_t);
void bwq_free(void *);
int bwq_sync(void *);
int bwq_set_device(int, void **);
}

static double bwq_now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

extern "C" long long bw_lingen_quad_core(const uint64_t *E, uint32_t m,
			uint32_t b, uint32_t E_len, uint32_t *gamma,
			int *lucky, uint32_t *t_out, uint64_t *pi_out,
			int dev)
{
	const uint32_t t0_in = gamma[0];
	const uint32_t bb = (b + 63) / 64;
	const uint32_t mb = (m + 63) / 64;
	const uint64_t ncoeff = (uint64_t)E_len + b + 64;
	const uint64_t rowcapF = ncoeff * bb;
	const uint64_t rowcapP = ncoeff * mb;

	void *stream = NULL;
	unsigned long long *dF = NULL, *dP = NULL;
	unsigned long long *d_delta = NULL;
	bwq_op_t *d_ops = NULL;
	uint32_t *d_rows = NULL, *d_ext = NULL, *d_ext2 = NULL;
	bwq_arenas_t hA, *dA = NULL;

	if (bwq_set_device(dev, &stream))
		return -1;
	if (bwq_alloc((void **)&dF, (uint64_t)b * rowcapF * 8) ||
	    bwq_alloc((void **)&dP, (uint64_t)b * rowcapP * 8) ||
	    bwq_alloc((void **)&d_delta, (uint64_t)b * mb * 8) ||
	    bwq_alloc((void **)&d_ops, (uint64_t)b * sizeof(bwq_op_t)) ||
	    bwq_alloc((void **)&d_rows, (uint64_t)b * 4) ||
	    bwq_alloc((void **)&d_ext, (uint64_t)b * 4) ||
	    bwq_alloc((void **)&d_ext2, (uint64_t)b * 4) ||
	    bwq_alloc((void **)&dA, sizeof(bwq_arenas_t))) {
		fprintf(stderr, "BW quad: device allocation failed\n");
		return -1;
	}
	hA.F = dF; hA.P = dP; hA.rowcapF = rowcapF; hA.rowcapP = rowcapP;
	hA.wpcF = bb; hA.wpcP = mb;
	if (bwq_upload(&hA, dA, sizeof(hA), stream) || bwq_sync(stream)) {
		fprintf(stderr, "BW quad: arena setup failed\n");
		return -1;
	}

	/* initial P = E^T (b rows x m bits per coefficient); F = I.
	 * matpoly<true> layout is ENTRY-major: entry (row i, col j) is a
	 * bit-packed polynomial: word (i*b + j)*W + (k/64), bit k%64 */
	const uint64_t W = (E_len + 63) / 64;
	std::vector<uint64_t> P0((size_t)b * rowcapP, 0);
	for (uint32_t i = 0; i < m; i++) {
		for (uint32_t j = 0; j < b; j++) {
			const uint64_t *ent = E + ((size_t)i * b + j) * W;
			for (uint32_t k = 0; k < E_len; k++) {
				if ((ent[k >> 6] >> (k & 63)) & 1) {
					P0[(size_t)j * rowcapP +
						(uint64_t)k * mb +
						(i >> 6)] |=
						1ull << (i & 63);
				}
			}
		}
	}
	std::vector<uint64_t> F0((size_t)b * rowcapF, 0);
	for (uint32_t r = 0; r < b; r++)
		F0[(size_t)r * rowcapF + (r >> 6)] = 1ull << (r & 63);

	if (bwq_upload(F0.data(), dF, (uint64_t)b * rowcapF * 8, stream) ||
	    bwq_upload(P0.data(), dP, (uint64_t)b * rowcapP * 8, stream)) {
		fprintf(stderr, "BW quad: upload failed\n");
		return -1;
	}

	std::vector<uint64_t> h_delta((size_t)b * mb);
	std::vector<uint32_t> h_gamma(b, 0);
	std::vector<uint32_t> hi(b, 1);
	std::vector<uint32_t> hiP(b, E_len);
	std::vector<uint8_t> used(b);
	std::vector<uint32_t> order, shift_rows, shift_extF,
					shift_extP;
	std::vector<bwq_op_t> ops;

	uint32_t t = 0;
	double t_start = bwq_now();
	uint32_t maxg = 0;


	for (t = 0; t < E_len; t++) {
		if (bwq_gather_delta_launch(dP, rowcapP, mb, t, b,
					d_delta, stream) ||
		    bwq_download(h_delta.data(), d_delta,
					(uint64_t)b * mb * 8, stream)) {
			fprintf(stderr, "BW quad: delta failed\n");
			return -1;
		}

		for (uint32_t r = 0; r < b; r++) {
			int z = 1;
			for (uint32_t w = 0; w < mb; w++)
				if (h_delta[(size_t)r * mb + w])
					z = 0;
			if (z) lucky[r]++;
			else lucky[r] = 0;
		}

		/* cross-check: recompute Delta from F directly */
		if (getenv("BW_QUAD_CHECK") && t < 40) {
			/* step-level printout */
			if (t < 4) {
				uint32_t nz = 0;
				for (uint32_t r = 0; r < b; r++) {
					int z = 1;
					for (uint32_t w = 0; w < mb; w++)
						if (h_delta[(size_t)r*mb+w]) z = 0;
					if (!z) nz++;
				}
				fprintf(stderr, "[quad-dbg] t=%u: nonzero delta "
					"rows %u\n", t, nz);
			}
			static std::vector<uint64_t> Fh;
			Fh.resize((size_t)b * rowcapF);
			bwq_download(Fh.data(), dF,
				(uint64_t)b * rowcapF * 8, stream);
			for (uint32_t r = 0; r < b; r++) {
				uint64_t acc[8] = {0};
				for (uint32_t k = 0; k <= t; k++) {
					/* F_r coefficient k (bb words) times
					 * E^T coefficient t-k (P0 rows) */
					for (uint32_t j = 0; j < bb; j++) {
						uint64_t fw =
							Fh[(size_t)r * rowcapF
								+ (uint64_t)k * bb + j];
						while (fw) {
							uint32_t c =
							 j*64 + __builtin_ctzll(fw);
							if (c < b) {
								uint32_t t2 = t - k;
								if (t2 < E_len)
									for (uint32_t w = 0; w < mb; w++)
										acc[w] ^= P0[(size_t)c * rowcapP + (uint64_t)t2 * mb + w];
							}
							fw &= fw - 1;
						}
					}
				}
				for (uint32_t w = 0; w < mb; w++)
					if (acc[w] != h_delta[(size_t)r * mb + w]) {
						fprintf(stderr, "[quad-dbg] t=%u row %u "
							"word %u: maintained %llx "
							"true %llx\n", t, r, w,
							(unsigned long long)h_delta[(size_t)r*mb+w],
							(unsigned long long)acc[w]);
						break;
					}
							}
		}

		order.resize(b);
		for (uint32_t r = 0; r < b; r++) order[r] = r;
		/* pivot preference: CADO's PLE takes the first row by
		 * index; gamma preference breaks the extraction's
		 * column-degree invariant (see BW_status notes) */
		std::stable_sort(order.begin(), order.end(),
			[&](uint32_t x, uint32_t y) {
				if (h_gamma[x] != h_gamma[y])
					return h_gamma[x] < h_gamma[y];
				return x < y;
			});
		std::fill(used.begin(), used.end(), 0);
		shift_rows.clear();
		shift_extF.clear();
		shift_extP.clear();

		for (uint32_t col = 0; col < m; col++) {
			uint32_t cw = col >> 6, cb = col & 63;
			int pivot = -1;
			for (uint32_t idx = 0; idx < b; idx++) {
				uint32_t r = order[idx];
				if (!used[r] &&
				    (h_delta[(size_t)r * mb + cw] >>
							cb) & 1) {
					pivot = (int)r;
					break;
				}
			}
			if (pivot < 0)
				continue;
			used[pivot] = 1;

			ops.clear();
			for (uint32_t r = 0; r < b; r++) {
				if (r == (uint32_t)pivot)
					continue;
				if ((h_delta[(size_t)r * mb + cw] >>
							cb) & 1) {
					bwq_op_t op;
					op.dst = r;
					op.src = (uint32_t)pivot;
					op.cext = std::max(hi[r],
							hi[pivot]);
					op.cextP = std::max(hiP[r],
							hiP[pivot]);
					ops.push_back(op);
					for (uint32_t w = 0; w < mb; w++)
						h_delta[(size_t)r * mb + w] ^=
						    h_delta[(size_t)pivot * mb + w];
					hi[r] = op.cext;
					hiP[r] = op.cextP;
				}
			}
			if (!ops.empty() &&
			    bwq_xor_pivot_launch(&hA, dA, ops.data(),
					(uint32_t)ops.size(), d_ops,
					stream)) {
				fprintf(stderr, "BW quad: xor failed\n");
				return -1;
			}
			/* pivot row multiplied by X (deferred batch) */
			h_gamma[pivot]++;
			shift_rows.push_back((uint32_t)pivot);
			/* extents are taken at step end: the row may still
			 * grow as a destination of later pivots this step,
			 * and the X-multiply must cover the whole final row
			 * (Shift*Tau*F order) */
			shift_extF.push_back(0);
			shift_extP.push_back(0);
			hi[pivot]++;
			hiP[pivot]++;
			maxg = std::max(maxg, h_gamma[pivot]);
			if ((uint64_t)hi[pivot] + 2 >= ncoeff) {
				fprintf(stderr, "BW quad: row overflow\n");
				return -1;
			}
		}

		/* fill the shift extents from the final watermarks */
		for (uint32_t i2 = 0; i2 < shift_rows.size(); i2++) {
			uint32_t r = shift_rows[i2];
			shift_extF[i2] = hi[r];
			shift_extP[i2] = hiP[r];
		}

		/* apply all X-shifts of this step in one batch */
		if (!shift_rows.empty() &&
		    bwq_shift_rows_launch(dA, shift_rows.data(),
				shift_extF.data(), shift_extP.data(),
				(uint32_t)shift_rows.size(), d_rows,
				d_ext, d_ext2, stream)) {
			fprintf(stderr, "BW quad: shift failed\n");
			return -1;
		}

		if ((t & 2047) == 0) {
			double el = bwq_now() - t_start;
			fprintf(stderr, "BW quad: t=%u/%u (%.2fs, "
				"%.4f s/step)\n", t, E_len, el,
				el / (t ? t : 1));
		}
	}

	/* pi = F directly (absolute storage), written in matpoly<true>
	 * entry-major layout */
	{
		std::vector<uint64_t> Fh((size_t)b * rowcapF);
		if (bwq_download(Fh.data(), dF,
					(uint64_t)b * rowcapF * 8, stream)) {
			fprintf(stderr, "BW quad: F download failed\n");
			return -1;
		}
		const uint64_t Wout = (ncoeff + 63) / 64;
		memset(pi_out, 0, (size_t)b * b * Wout * 8);
		/* convention probe: transpose switch via env */
		int trans = getenv("BW_QUAD_TRANS") != NULL;
		for (uint32_t r = 0; r < b; r++) {
			for (uint32_t c = 0; c < b; c++) {
				uint32_t rr = trans ? c : r;
				uint32_t cc = trans ? r : c;
				uint64_t *ent = pi_out +
					((size_t)rr * b + cc) * Wout;
				for (uint32_t k = 0; k < hi[r]; k++) {
					if ((Fh[(size_t)r * rowcapF +
							(uint64_t)k * bb +
							(c >> 6)] >>
							(c & 63)) & 1)
						ent[k >> 6] |=
							1ull << (k & 63);
				}
			}
		}
		fprintf(stderr, "BW quad: done, max gamma %u\n", maxg);
	}

	for (uint32_t r = 0; r < b; r++)
		gamma[r] = h_gamma[r] + t0_in;
	*t_out = t;

	bwq_free(dF); bwq_free(dP); bwq_free(d_delta);
	bwq_free(d_ops); bwq_free(d_rows); bwq_free(d_ext);
	bwq_free(d_ext2); bwq_free(dA);
	return (long long)ncoeff;
}
