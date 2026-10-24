/* bw-local shim: this port forces the plain gf2x multiplication path
 * (FFT_NONE), so none of the transform types are compiled in. The
 * original (b5ca9c4 linalg/bwc/lingen_fft_select.hpp) pulls in
 * gf2x-fft / flint / ternary plugins. */
#ifndef CADO_LINGEN_FFT_SELECT_HPP
#define CADO_LINGEN_FFT_SELECT_HPP
/* inert marker types so the vendored specializations compile; the
 * FFT_NONE path never constructs them */
struct gf2x_fake_fft_info;
struct gf2x_cantor_fft_info;
struct gf2x_ternary_fft_info;
#endif
