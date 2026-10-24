/* bw-local shim: the lingen port runs single-process. The vendored
 * code keeps its MPI call sites; they all compile to rank 0 / size 1
 * no-ops. BW_NO_MPI is always defined for this build. */
#ifndef _SELECT_MPI_H_
#define _SELECT_MPI_H_
#include <string.h>
typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef int MPI_Request;
typedef struct { int MPI_SOURCE; int MPI_TAG; } MPI_Status;
#define MPI_COMM_WORLD ((MPI_Comm)0)
#define MPI_CHAR ((MPI_Datatype)1)
#define MPI_INT ((MPI_Datatype)2)
#define MPI_UNSIGNED ((MPI_Datatype)3)
#define MPI_BYTE ((MPI_Datatype)4)
#define MPI_LONG_LONG ((MPI_Datatype)5)
#define MPI_UNSIGNED_LONG ((MPI_Datatype)6)
#define MPI_DOUBLE ((MPI_Datatype)7)
#define MPI_UNSIGNED_LONG_LONG ((MPI_Datatype)8)
#define MPI_SUM ((MPI_Op)1)
#define MPI_MAX ((MPI_Op)2)
#define MPI_LXOR ((MPI_Op)3)
#define MPI_STATUS_IGNORE ((MPI_Status *)0)
#define MPI_SUCCESS 0
#ifdef __cplusplus
extern "C" {
#endif
static inline int MPI_Comm_rank(MPI_Comm c, int *rank) { (void)c; *rank = 0; return 0; }
static inline int MPI_Comm_size(MPI_Comm c, int *size) { (void)c; *size = 1; return 0; }
static inline int MPI_Barrier(MPI_Comm c) { (void)c; return 0; }
static inline int MPI_Bcast(void *buf, int n, MPI_Datatype t, int root, MPI_Comm c)
{ (void)buf; (void)n; (void)t; (void)root; (void)c; return 0; }
static inline int MPI_Init(int *argc, char ***argv) { (void)argc; (void)argv; return 0; }
static inline int MPI_Finalize(void) { return 0; }
#ifdef __cplusplus
}
#endif
#endif
