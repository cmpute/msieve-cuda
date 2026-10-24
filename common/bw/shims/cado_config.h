/* bw-local shim: the handful of configure-detected macros the vendored
 * CADO files consume. Generated CADO builds get these from cado's own
 * configure; we fix them for x86-64 + SSE2 hosts. */
#ifndef _CADO_CONFIG_H_
#define _CADO_CONFIG_H_
#define ULONG_BITS 64
#define ULONGLONG_BITS 64
#define SIZEOF_UNSIGNED_LONG 8
#define HAVE_SSE2 1
#define HAVE_SSE3 1
#define HAVE_SSSE3 1
#define HAVE_SSE41 1
#define HAVE_PCLMUL 1
#define HAVE_AVX2 1
#define GMP_LIMB_BITS 64
#define HAVE_MINGW_SPECIAL 0
#define HAVE_STD_TO_STRING 1
#define HAVE_GETRUSAGE 1
#define HAVE_SYSCONF 1
#define HAVE_STRDUP 1
#define HAVE_ASPRINTF 1
#define HAVE_STRNDUP 1
#define HAVE_STRNLEN 1
#define HAVE_STRLCPY 1
#define HAVE_STRLCAT 1
#define HAVE_GETTID 1
#define HAVE_RUSAGE_THREAD 1
#define HAVE_GCC_STYLE_AMD64_INLINE_ASM 1
#define ULONGLONG_MAX_ 18446744073709551615ULL
#define CADO_CONFIG_H
#endif
