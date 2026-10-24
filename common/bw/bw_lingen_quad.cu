/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

GPU kernels for the quadratic lingen basecase (see bw_lingen_quad.h).

Storage is ABSOLUTE-coefficient: row r's coefficient k lives at
arena[r*rowcap + k*wpc + w]. "Multiply row by X" is then a whole-word
block move of the row (no bit crossing), and row XORs need no
alignment. Per step: one parallel launch per pivot (its destinations
are distinct), then one batched shift launch for all pivot rows.
--------------------------------------------------------------------*/

#include <cuda_runtime.h>
#include <cstdint>

typedef unsigned long long ull_t;

struct bwq_op_t {
	uint32_t dst;
	uint32_t src;
	uint32_t cext;      /* F extent to XOR, in coefficients */
	uint32_t cextP;     /* P extent to XOR (P outgrows F) */
};

struct bwq_arenas_t {
	ull_t *F;
	ull_t *P;
	uint64_t rowcapF;   /* words per F row */
	uint64_t rowcapP;   /* words per P row */
	uint32_t wpcF;      /* words per F coefficient */
	uint32_t wpcP;      /* words per P coefficient */
};

/* dst[k] ^= src[k] for k < cext, both arenas */
__global__ void bwq_xor_pivot(const bwq_arenas_t *dA_ptr,
			const bwq_op_t *ops, uint32_t nops)
{
	uint32_t o = blockIdx.x;
	if (o >= nops) return;
	bwq_arenas_t A = *dA_ptr;
	bwq_op_t op = ops[o];

	ull_t *dF = A.F + (uint64_t)op.dst * A.rowcapF;
	const ull_t *sF = A.F + (uint64_t)op.src * A.rowcapF;
	ull_t *dP = A.P + (uint64_t)op.dst * A.rowcapP;
	const ull_t *sP = A.P + (uint64_t)op.src * A.rowcapP;

	for (uint32_t k = threadIdx.x; k < op.cext; k += blockDim.x)
		for (uint32_t w = 0; w < A.wpcF; w++) {
			uint64_t i = (uint64_t)k * A.wpcF + w;
			if (i < A.rowcapF)
				dF[i] ^= sF[i];
		}
	for (uint32_t k = threadIdx.x; k < op.cextP; k += blockDim.x)
		for (uint32_t w = 0; w < A.wpcP; w++) {
			uint64_t i = (uint64_t)k * A.wpcP + w;
			if (i < A.rowcapP)
				dP[i] ^= sP[i];
		}
}

/* rows[]: multiply each listed row by X (block move up one coeff,
 * zero coefficient 0); ext[] is the row's live length */
__global__ void bwq_shift_rows(const bwq_arenas_t *dA_ptr,
			const uint32_t *rows, const uint32_t *extF,
			const uint32_t *extP, uint32_t nrows)
{
	uint32_t r = blockIdx.x;
	if (r >= nrows) return;
	bwq_arenas_t A = *dA_ptr;
	uint32_t row = rows[r];
	uint32_t nf = extF[r];
	uint32_t np = extP[r];

	ull_t *F = A.F + (uint64_t)row * A.rowcapF;
	ull_t *P = A.P + (uint64_t)row * A.rowcapP;
	for (uint32_t k = nf; k-- > 1; )
		for (uint32_t w = threadIdx.x; w < A.wpcF; w += blockDim.x)
			F[(uint64_t)k * A.wpcF + w] =
				F[(uint64_t)(k-1) * A.wpcF + w];
	for (uint32_t w = threadIdx.x; w < A.wpcF; w += blockDim.x)
		F[w] = 0;
	for (uint32_t k = np; k-- > 1; )
		for (uint32_t w = threadIdx.x; w < A.wpcP; w += blockDim.x)
			P[(uint64_t)k * A.wpcP + w] =
				P[(uint64_t)(k-1) * A.wpcP + w];
	for (uint32_t w = threadIdx.x; w < A.wpcP; w += blockDim.x)
		P[w] = 0;
}

/* Delta_r = P_r at absolute coefficient t (b rows x wpc words) */
__global__ void bwq_gather_delta(const ull_t *P, uint64_t rowcap,
			uint32_t wpc, uint32_t t, uint32_t b, ull_t *out)
{
	uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= b) return;
	for (uint32_t w = 0; w < wpc; w++)
		out[(uint64_t)r * wpc + w] =
			P[(uint64_t)r * rowcap + (uint64_t)t * wpc + w];
}

extern "C" {

int bwq_xor_pivot_launch(const bwq_arenas_t *hA, const bwq_arenas_t *dA,
			const bwq_op_t *h_ops, uint32_t nops,
			bwq_op_t *d_ops, void *stream)
{
	cudaStream_t s = (cudaStream_t)stream;
	cudaError_t e;
	if (!nops)
		return 0;
	e = cudaMemcpyAsync(d_ops, h_ops, sizeof(bwq_op_t) * nops,
			cudaMemcpyHostToDevice, s);
	if (e != cudaSuccess) return -1;
	bwq_xor_pivot<<<nops, 256, 0, s>>>(dA,
			(const bwq_op_t *)d_ops, nops);
	e = cudaStreamSynchronize(s);
	return e == cudaSuccess ? 0 : -1;
}

int bwq_shift_rows_launch(const bwq_arenas_t *dA,
			const uint32_t *h_rows, const uint32_t *h_extF,
			const uint32_t *h_extP, uint32_t nrows,
			uint32_t *d_rows, uint32_t *d_ext,
			uint32_t *d_ext2, void *stream)
{
	cudaStream_t s = (cudaStream_t)stream;
	cudaError_t e;
	if (!nrows)
		return 0;
	e = cudaMemcpyAsync(d_rows, h_rows, 4 * nrows,
			cudaMemcpyHostToDevice, s);
	if (e != cudaSuccess) return -1;
	e = cudaMemcpyAsync(d_ext, h_extF, 4 * nrows,
			cudaMemcpyHostToDevice, s);
	if (e != cudaSuccess) return -1;
	e = cudaMemcpyAsync(d_ext2, h_extP, 4 * nrows,
			cudaMemcpyHostToDevice, s);
	if (e != cudaSuccess) return -1;
	bwq_shift_rows<<<nrows, 64, 0, s>>>(dA,
			(const uint32_t *)d_rows,
			(const uint32_t *)d_ext,
			(const uint32_t *)d_ext2, nrows);
	e = cudaStreamSynchronize(s);
	return e == cudaSuccess ? 0 : -1;
}

int bwq_gather_delta_launch(const ull_t *P, uint64_t rowcap, uint32_t wpc,
			uint32_t t, uint32_t b, ull_t *d_out, void *stream)
{
	cudaStream_t s = (cudaStream_t)stream;
	uint32_t blocks = (b + 255) / 256;
	bwq_gather_delta<<<blocks, 256, 0, s>>>(P, rowcap, wpc, t, b,
			d_out);
	cudaError_t e = cudaStreamSynchronize(s);
	return e == cudaSuccess ? 0 : -1;
}

int bwq_upload(const void *h, void *d, uint64_t bytes, void *stream)
{
	cudaError_t e = cudaMemcpyAsync(d, h, bytes,
			cudaMemcpyHostToDevice, (cudaStream_t)stream);
	return e == cudaSuccess ? 0 : -1;
}

int bwq_download(void *h, const void *d, uint64_t bytes, void *stream)
{
	cudaError_t e = cudaMemcpyAsync(h, d, bytes,
			cudaMemcpyDeviceToHost, (cudaStream_t)stream);
	if (e != cudaSuccess) return -1;
	e = cudaStreamSynchronize((cudaStream_t)stream);
	return e == cudaSuccess ? 0 : -1;
}

int bwq_alloc(void **p, uint64_t bytes)
{
	cudaError_t e = cudaMalloc(p, bytes);
	return e == cudaSuccess ? 0 : -1;
}

void bwq_free(void *p) { cudaFree(p); }

int bwq_sync(void *stream)
{
	cudaError_t e = cudaStreamSynchronize((cudaStream_t)stream);
	return e == cudaSuccess ? 0 : -1;
}

int bwq_set_device(int dev, void **stream_out)
{
	cudaError_t e = cudaSetDevice(dev);
	if (e != cudaSuccess) return -1;
	return cudaStreamCreate((cudaStream_t *)stream_out)
			== cudaSuccess ? 0 : -1;
}

} /* extern C */
