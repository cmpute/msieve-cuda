/* bw-local shim: lingen log lines print memory use; report zeros */
#ifndef _CADO_MEMUSAGE_H_
#define _CADO_MEMUSAGE_H_
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
size_t Memusage(void);
size_t Memusage2(void);
size_t PeakMemusage(void);
size_t memtorture(void);
#ifdef __cplusplus
}
#endif
#endif
