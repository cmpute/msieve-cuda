/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

Host driver for the self-contained quadratic GPU lingen (the math is
documented in bw_lingen_quad.h). Stages: A-file ingest, Gaussian
augmentation-column selection, the GPU BM core (row XORs and X-shifts
batched per pivot), lucky-row extraction into the F file.
--------------------------------------------------------------------*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <vector>
#include <algorithm>

#include "bw_lingen_quad.h"
#include "bw_seq.h"

/* ---- GPU bridge (bw_lingen_quad.cu) ---- */
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

/*--------------------------------------------------------------------
Selection: m independent columns of A, earliest coefficients first.
Returns t0 = max k_r, or -1 (as int) on failure.
--------------------------------------------------------------------*/
static int quad_select(const uint64_t *A, uint32_t m, uint32_t n,
			uint64_t L, std::vector<uint32_t> &kk,
			std::vector<uint32_t> &jj)
{
	const uint32_t nw = (m + 63) / 64;
	std::vector<uint64_t> basis((size_t)m * nw, 0);
	std::vector<int32_t> piv(m, -1);
	uint32_t found = 0;
	const uint64_t kmax = std::min<uint64_t>(L, 8192);

	for (uint64_t k = 0; k < kmax; k++) {
		for (uint32_t j = 0; j < n; j++) {
			if (found == m)
				break;
			std::vector<uint64_t> v(nw, 0);
			const uint64_t *Ak = A + k * m;
			for (uint32_t i = 0; i < m; i++)
				if ((Ak[i] >> j) & 1)
					v[i >> 6] |= 1ull << (i & 63);
			for (uint32_t p = 0; p < found; p++) {
				if ((v[(uint32_t)piv[p] >> 6] >>
						(piv[p] & 63)) & 1) {
					const uint64_t *b =
						&basis[(size_t)p * nw];
					for (uint32_t w = 0; w < nw; w++)
						v[w] ^= b[w];
				}
			}
			int32_t lead = -1;
			for (uint32_t w = 0; w < nw; w++)
				if (v[w]) {
					lead = (int32_t)(w * 64 +
						__builtin_ctzll(v[w]));
					break;
				}
			if (lead < 0)
				continue;
			memcpy(&basis[(size_t)found * nw], v.data(),
				nw * 8);
			piv[found] = lead;
			kk[found] = (uint32_t)k;
			jj[found] = j;
			found++;
		}
		if (found == m)
			break;
	}
	if (found < m)
		return -1;
	uint32_t t0 = 0;
	for (uint32_t r = 0; r < m; r++)
		t0 = std::max(t0, kk[r]);
	return (int)t0;
}

/*--------------------------------------------------------------------
The GPU BM core: P0 is the augmented transposed sequence (b rows,
mb words per coefficient, L coefficients, arena layout). On return
Fh holds the generator rows, lucky[] the streaks, hi[] the row lengths.
--------------------------------------------------------------------*/
static int quad_bm(const std::vector<uint64_t> &P0, uint32_t b,
		uint32_t m, uint32_t L, uint32_t ncoeff, int dev,
		std::vector<uint64_t> &Fh, std::vector<int> &lucky,
		std::vector<uint32_t> &hi, std::vector<uint32_t> &gamma)
{
	const uint32_t bb = (b + 63) / 64;
	const uint32_t mb = (m + 63) / 64;
	const uint64_t rowcapF = (uint64_t)ncoeff * bb;
	const uint64_t rowcapP = (uint64_t)ncoeff * mb;

	void *stream = NULL;
	unsigned long long *dF = NULL, *dP = NULL, *d_delta = NULL;
	bwq_op_t *d_ops = NULL;
	uint32_t *d_rows = NULL, *d_extF = NULL, *d_extP = NULL;
	bwq_arenas_t hA, *dA = NULL;

	if (bwq_set_device(dev, &stream))
		return -1;
	if (bwq_alloc((void **)&dF, (uint64_t)b * rowcapF * 8) ||
	    bwq_alloc((void **)&dP, (uint64_t)b * rowcapP * 8) ||
	    bwq_alloc((void **)&d_delta, (uint64_t)b * mb * 8) ||
	    bwq_alloc((void **)&d_ops, (uint64_t)b * sizeof(bwq_op_t)) ||
	    bwq_alloc((void **)&d_rows, (uint64_t)b * 4) ||
	    bwq_alloc((void **)&d_extF, (uint64_t)b * 4) ||
	    bwq_alloc((void **)&d_extP, (uint64_t)b * 4) ||
	    bwq_alloc((void **)&dA, sizeof(bwq_arenas_t))) {
		fprintf(stderr, "BW quad: device allocation failed\n");
		return -1;
	}
	hA.F = dF; hA.P = dP; hA.rowcapF = rowcapF; hA.rowcapP = rowcapP;
	hA.wpcF = bb; hA.wpcP = mb;
	if (bwq_upload(&hA, dA, sizeof(hA), stream) || bwq_sync(stream))
		return -1;

	/* F starts as the identity at coefficient 0 */
	std::vector<uint64_t> F0((size_t)b * rowcapF, 0);
	for (uint32_t r = 0; r < b; r++)
		F0[(size_t)r * rowcapF + (r >> 6)] = 1ull << (r & 63);
	if (bwq_upload(F0.data(), dF, (uint64_t)b * rowcapF * 8, stream) ||
	    bwq_upload(P0.data(), dP, (uint64_t)b * rowcapP * 8, stream))
		return -1;

	lucky.assign(b, 0);
	gamma.assign(b, 0);
	hi.assign(b, 1);
	std::vector<uint64_t> h_delta((size_t)b * mb);
	std::vector<uint8_t> used(b);
	std::vector<uint32_t> order, srows, sextF, sextP;
	std::vector<bwq_op_t> ops;
	double t_start = bwq_now();

	for (uint32_t t = 0; t < L; t++) {
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



		order.resize(b);
		for (uint32_t r = 0; r < b; r++) order[r] = r;
		std::stable_sort(order.begin(), order.end(),
			[&](uint32_t x, uint32_t y) {
				return gamma[x] < gamma[y];
			});
		std::fill(used.begin(), used.end(), 0);
		srows.clear();
		sextF.clear();
		sextP.clear();

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
					op.cext = t + 2;   /* safe bound:
					 * any row's F content
					 * lives in [0, t+2) after t
					 * steps; watermarks proved
					 * unreliable */
					op.cextP = std::max(
						(uint64_t)hi[pivot] + 1,
						hi[r] ?
						  std::max((uint64_t)hi[r],
						    (uint64_t)L) : (uint64_t)L);
					/* P rows always span the whole
					 * sequence: extent = L */
					op.cextP = L;
					ops.push_back(op);
					for (uint32_t w = 0; w < mb; w++)
						h_delta[(size_t)r * mb + w] ^=
						    h_delta[(size_t)pivot * mb + w];
					hi[r] = op.cext;
				}
			}
			if (!ops.empty() &&
			    bwq_xor_pivot_launch(&hA, dA, ops.data(),
					(uint32_t)ops.size(), d_ops,
					stream)) {
				fprintf(stderr, "BW quad: xor failed\n");
				return -1;
			}

			gamma[pivot]++;
			srows.push_back((uint32_t)pivot);
			sextF.push_back(0);
			sextP.push_back(0);
			hi[pivot]++;
			if ((uint64_t)hi[pivot] + 2 >= (uint64_t)ncoeff) {
				fprintf(stderr, "BW quad: row overflow\n");
				return -1;
			}
		}

		for (uint32_t i2 = 0; i2 < srows.size(); i2++) {
			sextF[i2] = t + 2;   /* full live range, see above */
			sextP[i2] = L + 1;
		}

		if (!srows.empty() &&
		    bwq_shift_rows_launch(dA, srows.data(), sextF.data(),
				sextP.data(), (uint32_t)srows.size(),
				d_rows, d_extF, d_extP, stream)) {
			fprintf(stderr, "BW quad: shift failed\n");
			return -1;
		}


		if ((t & 2047) == 0) {
			double el = bwq_now() - t_start;
			fprintf(stderr, "BW quad: t=%u/%u (%.2fs, "
				"%.4f s/step)\n", t, L, el,
				el / (t ? t : 1));
		}
	}

	Fh.resize((size_t)b * rowcapF);
	if (bwq_download(Fh.data(), dF, (uint64_t)b * rowcapF * 8, stream))
		return -1;

	bwq_free(dF); bwq_free(dP); bwq_free(d_delta);
	bwq_free(d_ops); bwq_free(d_rows); bwq_free(d_extF);
	bwq_free(d_extP); bwq_free(dA);
	return 0;
}

/*--------------------------------------------------------------------
Full pipeline: A file -> F file.
--------------------------------------------------------------------*/
int bw_lingen_quad_full(const char *afile, const char *ffile,
			uint32_t m, uint32_t n, uint64_t seq_len,
			uint32_t nrows, uint32_t ncols, int dev,
			uint32_t luck_mini, uint32_t nthreads)
{
	(void)nthreads;
	/* ingest the A file */
	FILE *f = fopen(afile, "rb");
	if (!f) {
		fprintf(stderr, "BW quad: cannot open %s\n", afile);
		return -1;
	}
	bw_filehdr_t hdr;
	if (fread(&hdr, 1, 64, f) != 64 || hdr.magic != BW_A_MAGIC ||
			hdr.m != m || hdr.n != n) {
		fprintf(stderr, "BW quad: bad A header\n");
		fclose(f);
		return -1;
	}
	const uint64_t L = hdr.seq_len ? hdr.seq_len : seq_len;
	std::vector<uint64_t> A((size_t)L * m);
	if (fread(A.data(), 8, (size_t)L * m, f) != (size_t)L * m) {
		fprintf(stderr, "BW quad: A truncated\n");
		fclose(f);
		return -1;
	}
	fclose(f);

	/* augmentation selection */
	std::vector<uint32_t> kk(m), jj(m);
	int t0 = quad_select(A.data(), m, n, L, kk, jj);
	if (t0 < 0) {
		fprintf(stderr, "BW quad: selection failed\n");
		return -1;
	}
	fprintf(stderr, "BW quad: t0=%d\n", t0);
	const uint32_t b = m + n;
	const uint32_t bb = (b + 63) / 64;
	const uint32_t nw = (n + 63) / 64;
	const uint32_t mb = (m + 63) / 64;
	/* E' in coefficient-window form: identity columns shifted by
	 * t0, augmentation columns by kk; BM annihilates this window */
	const uint64_t LE = L - (uint64_t)t0;
	const uint64_t ncoeff = LE + b + 64;
	const uint64_t rowcapP = ncoeff * mb;
	const uint64_t rowcapF = ncoeff * bb;

	/* build P0 = E''^T (see the layout note above) */
	std::vector<uint64_t> P0((size_t)b * rowcapP, 0);
	for (uint64_t c = 0; c < LE; c++) {
		for (uint32_t r = 0; r < n; r++) {
			const uint64_t *Ak = A.data() + (c + (uint64_t)t0) * m;
			uint64_t *dst = &P0[(size_t)r * rowcapP +
						(uint64_t)c * mb];
			for (uint32_t i = 0; i < m; i++)
				if ((Ak[i] >> r) & 1)
					dst[i >> 6] |= 1ull << (i & 63);
		}
		for (uint32_t r = 0; r < m; r++) {
			const uint64_t *Ak = A.data() +
					(c + (uint64_t)kk[r]) * m;
			uint64_t *dst = &P0[(size_t)(n + r) * rowcapP +
						(uint64_t)c * mb];
			for (uint32_t i = 0; i < m; i++)
				if ((Ak[i] >> jj[r]) & 1)
					dst[i >> 6] |= 1ull << (i & 63);
		}
	}



	/* run the BM */
	std::vector<uint64_t> Fh;
	std::vector<int> lucky;
	std::vector<uint32_t> hi, gamma;
	if (quad_bm(P0, b, m, (uint32_t)LE, (uint32_t)ncoeff, dev,
			Fh, lucky, hi, gamma)) {
		fprintf(stderr, "BW quad: BM failed\n");
		return -1;
	}




	/* pick n solution rows: streak-qualified, W-independent, W != 0.
	 * (Trivial relations between duplicated augmented columns reach the
	 * top streaks but map to W = 0 through F0, so independence over
	 * the n-bit polynomial vectors is the real selector.) */
	std::vector<uint32_t> cand;
	{
		std::vector<uint32_t> idx(b);
		for (uint32_t r = 0; r < b; r++) idx[r] = r;
		std::stable_sort(idx.begin(), idx.end(),
			[&](uint32_t x, uint32_t y) {
				return lucky[x] > lucky[y];
			});
		/* Gaussian basis over flattened W vectors (n * Dmax bits) */
		const uint32_t Dmax = (uint32_t)ncoeff;
		const uint64_t vbits = (uint64_t)n * Dmax;
		std::vector<uint64_t> basis;
		std::vector<int32_t> pivots;
		auto try_add = [&](uint32_t r) -> bool {
			std::vector<uint64_t> v((vbits + 63) / 64, 0);
			/* W_r coefficients: part1 + part2, flattened */
			for (uint64_t c = 0; c < Dmax; c++) {
				uint64_t W[8] = {0};
				const uint64_t *Fr = &Fh[(size_t)r * rowcapF];
				for (uint32_t wq = 0; wq < bb; wq++) {
					uint64_t fw = Fr[c * bb + wq];
					while (fw) {
						uint32_t rr2 = wq * 64 +
							__builtin_ctzll(fw);
						if (rr2 < n)
							W[rr2 >> 6] |=
								1ull << (rr2 & 63);
						fw &= fw - 1;
					}
				}
				for (uint32_t rp = 0; rp < m; rp++) {
					int64_t cc = (int64_t)c - t0 +
						(int64_t)kk[rp];
					if (cc < 0 || (uint64_t)cc >= ncoeff)
						continue;
					const uint64_t *Fr2 = &Fh[(size_t)r *
							rowcapF];
					uint32_t bit = n + rp;
					if ((Fr2[(uint64_t)cc * bb +
						(bit >> 6)] >> (bit & 63)) & 1)
						W[jj[rp] >> 6] |=
							1ull << (jj[rp] & 63);
				}
				for (uint32_t w2 = 0; w2 < nw; w2++) {
					uint64_t word = W[w2];
					uint64_t bitbase = (uint64_t)c * n +
						w2 * 64;
					while (word) {
						uint32_t obit = __builtin_ctzll(word);
						uint64_t absbit = bitbase + obit;
						v[absbit >> 6] |=
							1ull << (absbit & 63);
						word &= word - 1;
					}
				}
			}
			/* zero vector? */
			int nz = 0;
			for (uint64_t w2 = 0; w2 < (uint64_t)v.size(); w2++)
				if (v[w2]) { nz = 1; break; }
			if (!nz)
				return false;
			/* reduce */
			for (uint32_t p2 = 0; p2 < pivots.size(); p2++) {
				uint64_t pb = (uint64_t)pivots[p2];
				if ((v[pb >> 6] >> (pb & 63)) & 1)
					for (uint64_t w2 = 0;
						w2 < (uint64_t)v.size(); w2++)
						v[w2] ^= basis[(size_t)p2 *
							v.size() + w2];
			}
			int64_t lead = -1;
			for (uint64_t w2 = 0; w2 < (uint64_t)v.size(); w2++)
				if (v[w2]) {
					lead = (int64_t)(w2 * 64 +
						__builtin_ctzll(v[w2]));
					break;
				}
			if (lead < 0)
				return false;
			basis.insert(basis.end(), v.begin(), v.end());
			pivots.push_back((int32_t)lead);
			return true;
		};
		for (uint32_t r = 0; r < b && cand.size() < n; r++) {
			if (lucky[idx[r]] < (int)luck_mini)
				break;
			if (try_add(idx[r]))
				cand.push_back(idx[r]);
		}
		if (cand.size() < n) {
			fprintf(stderr, "BW quad: only %zu/%u independent "
				"solutions\n", cand.size(), n);
			return -1;
		}
	}

	/* extract W per candidate and emit the file words */


	/* the identity window starts at each row's detection point:
	 * A*W_r vanishes on [LE - lucky[r] + t0, L). Pad the file so
	 * d = D-1 covers the latest detection across the candidates. */
	/* A*W = E''*g vanishes from each candidate's detection point
	 * t_c = LE - lucky[r]; pad the file degree past the worst one */
	/* true row extents by scanning (watermarks unreliable) */
	std::vector<uint32_t> top(b, 0);
	for (uint32_t r = 0; r < b; r++) {
		const uint64_t *Fr = &Fh[(size_t)r * rowcapF];
		for (uint32_t c = (uint32_t)ncoeff; c-- > 0; ) {
			uint64_t orv = 0;
			for (uint32_t wq = 0; wq < bb; wq++)
				orv |= Fr[(uint64_t)c * bb + wq];
			if (orv) { top[r] = c + 1; break; }
		}
	}
	uint64_t Dreq = 0;
	for (uint32_t r : cand)
		Dreq = std::max(Dreq, (uint64_t)top[r] + 2 * t0 + 2);
	Dreq = std::min(Dreq, (uint64_t)ncoeff);
	uint64_t Dlen = 1;
	std::vector<uint64_t> words;   /* coefficient-major n words each */
	for (uint64_t c = 0; c < ncoeff; c++) {
		/* file layout: coefficient c word i = candidate i's W
		 * coefficient c (ascending, leading first; consumers
		 * transpose on read and get G's columns = the W's) */
		int any = 0;
		for (uint32_t i = 0; i < cand.size(); i++) {
			uint32_t r = cand[i];
			uint64_t Wc[8] = {0};
			{
				const uint64_t *Fr = &Fh[(size_t)r *
							rowcapF];
				for (uint32_t j = 0; j < n; j++)
					if ((Fr[c * bb + (j >> 6)] >>
							(j & 63)) & 1)
						Wc[j >> 6] |=
						1ull << (j & 63);
			}
			for (uint32_t rp = 0; rp < m; rp++) {
				int64_t cc = (int64_t)c - t0 +
						(int64_t)kk[rp];
				if (cc < 0)
					continue;
				const uint64_t *Fr = &Fh[(size_t)r *
							rowcapF];
				uint32_t bit = n + rp;
				if ((Fr[(uint64_t)cc * bb + (bit >> 6)] >>
						(bit & 63)) & 1)
					Wc[jj[rp] >> 6] |=
						1ull << (jj[rp] & 63);
			}
			for (uint32_t w = 0; w < nw; w++)
				if (Wc[w])
					any = 1;
			words.insert(words.end(), Wc, Wc + nw);
		}
		(void)any;
		Dlen = c + 1;
		/* early stop: nothing new can appear beyond the longest
		 * row; keep scanning while c < max extent */
		{
			uint64_t maxext = 0;
			for (uint32_t r : cand)
				maxext = std::max(maxext,
					(uint64_t)top[r] + 2 * t0 + 2);
			if (c + 1 >= maxext && c + 1 >= Dreq)
				break;
			if (c + 1 >= (uint64_t)ncoeff)
				break;
		}
	}
	fprintf(stderr, "BW quad: F degree %llu (t0=%d)\n",
		(unsigned long long)Dlen - 1, t0);


	/* write the F file */
	f = fopen(ffile, "wb");
	if (!f) {
		fprintf(stderr, "BW quad: cannot write %s\n", ffile);
		return -1;
	}
	memset(&hdr, 0, sizeof(hdr));
	hdr.magic = BW_F_MAGIC;
	hdr.version = BW_FORMAT_VERSION;
	hdr.m = m;
	hdr.n = n;
	hdr.nrows = nrows;
	hdr.ncols = ncols;
	hdr.num_dense_rows = 0;
	hdr.seq_len = Dlen;
	if (fwrite(&hdr, 1, 64, f) != 64 ||
	    fwrite(words.data(), 8, words.size(), f) != words.size()) {
		fprintf(stderr, "BW quad: F write failed\n");
		fclose(f);
		return -1;
	}
	fclose(f);
	return 0;
}
