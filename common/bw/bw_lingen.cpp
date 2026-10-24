/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

C ABI bridge from the BW solver to the vendored CADO lingen. Mirrors
lingen_main.cpp's single-rank flow: file input -> E_from_A(t0/F0) ->
bw_lingen_single -> luck check -> gather_reverse -> F_from_PI ->
output_to_singlefile. No MPI, no checkpoints, no tuning: companions
are synthesized on demand by bmstatus::companion() using the fixed
recursion threshold passed in the options.
--------------------------------------------------------------------*/

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "cado_config.h"
#include "lingen_bmstatus.hpp"
#include "lingen_io_wrappers.hpp"
#include "lingen_matpoly_select.hpp"
#include "lingen_expected_pi_length.hpp"
#include <omp.h>
#include <cstdio>
#include <cstring>
#include "bw_seq.h"

/* defined in lingen.cpp (template + explicit instantiation) */
template<typename matpoly_type>
matpoly_type bw_lingen_single(bmstatus<matpoly_type::is_binary> & bm,
		matpoly_type & E);
extern template matpoly<true> bw_lingen_single<matpoly<true>>(
		bmstatus<true> &, matpoly<true> &);

extern "C" {

typedef struct {
	uint32_t m, n;
	const char *afile;
	const char *ffile;
	uint64_t seq_len;
	uint32_t threshold;
	uint32_t nthreads;
} bw_lingen_opts_t;

int bw_lingen_run(const bw_lingen_opts_t *o)
{
	cxx_mpz const p(2);
	bmstatus<true> bm(o->m, o->n, p);
	bm.lingen_threshold = o->threshold;

	if (o->nthreads)
		omp_set_num_threads((int)o->nthreads);

	typename matpoly<true>::memory_guard const guard(SIZE_MAX);

	std::unique_ptr<lingen_file_input<true>> A(
		new lingen_file_input<true>(&bm.d.ab, o->m, o->n,
				o->afile, 0, (size_t)o->seq_len));

	/* initial read: computes t0 and F0 */
	std::unique_ptr<lingen_E_from_A<true>> E_series(
		new lingen_E_from_A<true>(bm.d, *A));

	bm.set_t0(E_series->t0);

	size_t const L = E_series->guessed_length();

	matpoly<true> E(&bm.d.ab, o->m, o->m + o->n, L);
	lingen_scatter<matpoly<true>> fill_E(E);
	lingen_F0<true> const & F0 = *E_series;

	pipe<true>(*E_series, fill_E, "Read");

	auto pi = bw_lingen_single<matpoly<true>>(bm, E);

	bm.stats.final_print();
	bm.display_deltas();

	constexpr const unsigned int simd = ULONG_BITS;
	pi.zero_pad(simd * iceildiv(pi.get_size(), simd));

	/* luck check: every lucky[j] must reach the expected pi length */
	{
		unsigned int const luck_mini =
			expected_pi_length<true>(bm.d);
		unsigned int nlucky = 0;
		for (unsigned int j = 0; j < o->m + o->n; j++)
			nlucky += bm.lucky[j] >= luck_mini;
		fprintf(stderr, "BW lingen: %u lucky columns (need %u, "
			"luck_mini=%u)\n", nlucky, o->n, luck_mini);
		if (nlucky < o->n) {
			fprintf(stderr, "BW lingen: FAILED luck condition "
				"(%u < %u)\n", nlucky, o->n);
			return -1;
		}
	}

	lingen_gather_reverse<matpoly<true>> read_PI(pi);
	lingen_F_from_PI<true> Fsrc(bm, read_PI, F0);
	/* lingen_output_to_singlefile writes the bare coefficient stream;
	 * wrap it with the bw_seq.h header (magic/dims/length) */
	std::string rawfile = std::string(o->ffile) + ".cadoraw";
	/* scoped so the ofstream flushes and closes before we reread */
	{
		lingen_output_to_singlefile<true> Fdst(&bm.d.ab, o->n, o->n,
				rawfile, 0);
		pipe<true>(Fsrc, Fdst, "Written", true);
	}

	{
		FILE *in = fopen(rawfile.c_str(), "rb");
		FILE *out = fopen(o->ffile, "wb");
		bw_filehdr_t hdr;
		long rawlen;
		char buf[65536];
		size_t k;

		if (!in || !out) {
			fprintf(stderr, "BW lingen: cannot finalize F "
				"file\n");
			return -1;
		}
		fseek(in, 0, SEEK_END);
		rawlen = ftell(in);
		fseek(in, 0, SEEK_SET);

		memset(&hdr, 0, sizeof(hdr));
		hdr.magic = BW_F_MAGIC;
		hdr.version = BW_FORMAT_VERSION;
		hdr.m = o->m;
		hdr.n = o->n;
		hdr.nrows = 0;
		hdr.ncols = 0;
		hdr.seq_len = (uint64_t)(rawlen / ((long)o->n * 8));
		if (fwrite(&hdr, 1, 64, out) != 64)
			return -1;
		while ((k = fread(buf, 1, sizeof(buf), in)) > 0)
			if (fwrite(buf, 1, k, out) != k)
				return -1;
		fclose(in);
		fclose(out);
		remove(rawfile.c_str());
	}

	fprintf(stderr, "BW lingen: F written to %s\n", o->ffile);
	return 0;
}

} /* extern C */
