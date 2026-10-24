/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.

Atomic checkpoint files: .tmp + fsync + rename, rotate .prev, EOF
sentinel so truncation is detected without reading the state, and
FNV-1a golden hashes of everything that must reproduce on resume.
--------------------------------------------------------------------*/

#ifndef _COMMON_BW_BW_CHECKPOINT_H_
#define _COMMON_BW_BW_CHECKPOINT_H_

#include "bw_seq.h"

/* write hdr + state (ncols words) atomically to path, rotating
   path.prev. Returns 0 on success */
int bw_checkpoint_write(const char *path, const bw_chkhdr_t *hdr,
			const uint64_t *state, uint32_t ncols);

/* validate and load. Returns 0 and fills *hdr_out, *state_out (ncols
   words, xmalloc'd) on success; -1 with a reason logged if the file
   is absent/corrupt/truncated/mismatched (dims or hashes) */
int bw_checkpoint_read(const char *path, bw_chkhdr_t *hdr_out,
			uint64_t **state_out, uint32_t ncols);

/* path helpers shared by the drivers */
void bw_path(char *buf, size_t n, msieve_obj *obj, const char *suffix);

#endif /* !_COMMON_BW_BW_CHECKPOINT_H_ */
