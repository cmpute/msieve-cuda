/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

BW helper kernels (VBITS=64 bit-sliced layout: one 64-bit state word
per matrix coordinate, bit b of the word = vector b at that
coordinate). Runtime-API launches on a caller-owned stream and device.
--------------------------------------------------------------------*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

/* device atomics overload on unsigned long long, not uint64_t
   (which is unsigned long on linux). The extern wrappers below keep
   this type too; it is ABI-identical to the uint64_t* declared in
   bw_gpu.h on LP64. */
typedef unsigned long long ull_t;

#define BW_THREADS 256

/*--------------------------------------------------------------------
bw_oprod64: partial S = U^T * Y for one row slab.

S[64k + a] ^= (bit a of u_k[i]) ? Y[i], summed over the slab's rows i.
u has nmasks (=m/64) words per row i; the block accumulates S in
shared memory and flushes with one atomic per word.
--------------------------------------------------------------------*/
__global__ void bw_oprod64_kernel(ull_t *s,
			const ull_t *u, const ull_t *y,
			uint32_t nmasks, uint32_t nrows)
{
	extern __shared__ ull_t sh[];    /* nmasks*64 words */
	uint32_t i;
	uint32_t tid = threadIdx.x;

	for (i = tid; i < nmasks * 64; i += blockDim.x)
		sh[i] = 0;
	__syncthreads();

	for (i = blockIdx.x * blockDim.x + tid; i < nrows;
			i += (uint32_t)gridDim.x * blockDim.x) {
		ull_t yy = y[i];
		uint32_t k;
		if (!yy)
			continue;
		for (k = 0; k < nmasks; k++) {
			ull_t uu = u[(size_t)i * nmasks + k];
			while (uu) {
				uint32_t a = (uint32_t)__ffsll((long long)uu) - 1;
				sh[64 * k + a] ^= yy;
				uu &= uu - 1;
			}
		}
	}
	__syncthreads();

	/* few blocks; atomics on nmasks*64 words are cheap */
	for (i = tid; i < nmasks * 64; i += blockDim.x)
		atomicXor(&s[i], sh[i]);
}

extern "C" void bw_oprod64(ull_t *s, const ull_t *u,
			const ull_t *y, uint32_t nmasks, size_t nrows,
			cudaStream_t stream)
{
	size_t shbytes = (size_t)nmasks * 64 * sizeof(ull_t);
	int blocks = (int)((nrows + BW_THREADS - 1) / BW_THREADS);
	if (blocks > 4096)
		blocks = 4096;
	if (!blocks)
		blocks = 1;
	bw_oprod64_kernel<<<blocks, BW_THREADS, shbytes, stream>>>
			(s, u, y, nmasks, (uint32_t)nrows);
}

/*--------------------------------------------------------------------
bw_combine64: x[i] ^= Y0[i] * F (64x64 GF(2) map, row-major words).

F sits in constant-like shared memory; per word the work is the
weight of Y0[i] (32 XORs on average).
--------------------------------------------------------------------*/
__global__ void bw_combine64_kernel(ull_t *x, const ull_t *y0,
			const ull_t *fw, size_t n)
{
	__shared__ ull_t sh[64];
	uint32_t tid = threadIdx.x;
	size_t i;

	if (tid < 64)
		sh[tid] = fw[tid];
	__syncthreads();

	for (i = (size_t)blockIdx.x * blockDim.x + tid; i < n;
			i += (size_t)gridDim.x * blockDim.x) {
		ull_t u = y0[i];
		ull_t r = 0;
		while (u) {
			uint32_t c = (uint32_t)__ffsll((long long)u) - 1;
			r ^= sh[c];
			u &= u - 1;
		}
		x[i] ^= r;
	}
}

extern "C" void bw_combine64(ull_t *x, const ull_t *y0,
			const ull_t *fw, size_t n, cudaStream_t stream)
{
	int blocks = (int)((n + BW_THREADS - 1) / BW_THREADS);
	if (blocks > 8192)
		blocks = 8192;
	bw_combine64_kernel<<<blocks, BW_THREADS, 0, stream>>>
			(x, y0, fw, n);
}

/*--------------------------------------------------------------------
bw_dense64: y[r] += sum over columns c with bit (r%64) of
dense[(r/64)*n + c] set, of x[c].

Per column word, the few set bits select rows whose accumulators
get x[c] XORed in shared memory, flushed with one atomic per word.
--------------------------------------------------------------------*/
__global__ void bw_dense64_kernel(ull_t *y, const ull_t *dense,
			const ull_t *x, uint32_t ndb, size_t n)
{
	extern __shared__ ull_t sh[];    /* ndb*64 words */
	uint32_t tid = threadIdx.x;
	size_t j;
	uint32_t k;

	for (uint32_t i = tid; i < ndb * 64; i += blockDim.x)
		sh[i] = 0;
	__syncthreads();

	for (j = (size_t)blockIdx.x * blockDim.x + tid; j < n;
			j += (size_t)gridDim.x * blockDim.x) {
		ull_t xx = x[j];
		if (!xx)
			continue;
		for (k = 0; k < ndb; k++) {
			ull_t dd = dense[(size_t)k * n + j];
			while (dd) {
				uint32_t r = (uint32_t)__ffsll((long long)dd) - 1;
				sh[64 * k + r] ^= xx;
				dd &= dd - 1;
			}
		}
	}
	__syncthreads();

	for (uint32_t i = tid; i < ndb * 64; i += blockDim.x)
		atomicXor(&y[i], sh[i]);
}

extern "C" void bw_dense64(ull_t *y, const ull_t *dense,
			const ull_t *x, uint32_t ndb, size_t n,
			cudaStream_t stream)
{
	size_t shbytes = (size_t)ndb * 64 * sizeof(ull_t);
	int blocks = (int)((n + BW_THREADS - 1) / BW_THREADS);
	if (blocks > 4096)
		blocks = 4096;
	if (!blocks)
		blocks = 1;
	bw_dense64_kernel<<<blocks, BW_THREADS, shbytes, stream>>>
			(y, dense, x, ndb, n);
}

/*--------------------------------------------------------------------
bw_anyset64: flag[0] = OR of x[0..n). flag is zeroed by the kernel.
--------------------------------------------------------------------*/
__global__ void bw_anyset64_kernel(const ull_t *x, ull_t *flag,
			size_t n)
{
	__shared__ ull_t sh;
	uint32_t tid = threadIdx.x;
	ull_t acc = 0;
	size_t i;

	if (tid == 0)
		sh = 0;
	__syncthreads();

	for (i = (size_t)blockIdx.x * blockDim.x + tid; i < n;
			i += (size_t)gridDim.x * blockDim.x)
		acc |= x[i];
	if (acc)
		atomicOr(&sh, acc);
	__syncthreads();
	if (tid == 0 && sh)
		atomicOr(&flag[0], sh);
}

extern "C" void bw_anyset64(const ull_t *x, ull_t *flag,
			size_t n, cudaStream_t stream)
{
	cudaMemsetAsync(flag, 0, 8, stream);
	int blocks = (int)((n + BW_THREADS - 1) / BW_THREADS);
	if (blocks > 4096)
		blocks = 4096;
	if (!blocks)
		blocks = 1;
	bw_anyset64_kernel<<<blocks, BW_THREADS, 0, stream>>>(x, flag, n);
}

/*--------------------------------------------------------------------
occupancy preflights: launch-bound sanity for every kernel above;
a zero here means the configuration is unusable and must be a hard
error, never a silent clamp.
--------------------------------------------------------------------*/
extern "C" int bw_preflight(int dev)
{
	cudaError_t e;
	int blocks = 0;

	e = cudaSetDevice(dev);
	if (e != cudaSuccess) {
		printf("BW preflight: cudaSetDevice(%d): %s\n", dev,
			cudaGetErrorString(e));
		return -1;
	}

	{
		int sm = 1;
		cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
		e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,
			bw_oprod64_kernel, BW_THREADS, 8 * 64 * sizeof(ull_t));
		if (e != cudaSuccess || blocks <= 0 || sm <= 0) {
			printf("BW preflight: bw_oprod64 occupancy failed\n");
			return -1;
		}
		e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,
			bw_combine64_kernel, BW_THREADS, 64 * sizeof(ull_t));
		if (e != cudaSuccess || blocks <= 0) {
			printf("BW preflight: bw_combine64 occupancy failed\n");
			return -1;
		}
		e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,
			bw_dense64_kernel, BW_THREADS, 64 * sizeof(ull_t));
		if (e != cudaSuccess || blocks <= 0) {
			printf("BW preflight: bw_dense64 occupancy failed\n");
			return -1;
		}
		e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,
			bw_anyset64_kernel, BW_THREADS, 0);
		if (e != cudaSuccess || blocks <= 0) {
			printf("BW preflight: bw_anyset64 occupancy failed\n");
			return -1;
		}
	}
	return 0;
}
