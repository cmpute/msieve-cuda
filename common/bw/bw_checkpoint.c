/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author.
You may use it for any purpose, free of charge, without having to notify
anyone. I disclaim any responsibility for any errors.
--------------------------------------------------------------------*/

#include <common.h>
#include "bw_checkpoint.h"
#include "bw.h"

#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

void bw_path(char *buf, size_t n, msieve_obj *obj, const char *suffix) {
	snprintf(buf, n, "%s%s", obj->savefile.name, suffix);
}

/*-------------------------------------------------------------------*/
static int write_all(int fd, const void *buf, size_t n) {
	const char *p = (const char *)buf;
	while (n) {
		ssize_t k = write(fd, p, n);
		if (k <= 0) {
			if (errno == EINTR)
				continue;
			return -1;
		}
		p += k;
		n -= (size_t)k;
	}
	return 0;
}

int bw_checkpoint_write(const char *path, const bw_chkhdr_t *hdr,
			const uint64_t *state, uint32_t ncols) {

	char tmp[512], prev[512];
	uint64_t eof = BW_EOF_MAGIC;
	int fd;
	struct stat sb;

	snprintf(tmp, sizeof(tmp), "%s.tmp", path);
	snprintf(prev, sizeof(prev), "%s.prev", path);

	fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (fd < 0)
		return -1;

	if (write_all(fd, hdr, sizeof(bw_chkhdr_t)) ||
	    write_all(fd, state, (size_t)ncols * sizeof(uint64_t)) ||
	    write_all(fd, &eof, sizeof(eof))) {
		close(fd);
		remove(tmp);
		return -1;
	}
	if (fsync(fd) || close(fd))
		return -1;

	/* rotate: existing checkpoint (if any) becomes .prev, and the
	   .prev sentinel state (never mid-write) is what a damaged .chk
	   falls back to */
	if (stat(path, &sb) == 0) {
		remove(prev);
		if (rename(path, prev))
			return -1;
	}
	if (rename(tmp, path))
		return -1;
	return 0;
}

/*-------------------------------------------------------------------*/
static int read_all(int fd, void *buf, size_t n) {
	char *p = (char *)buf;
	while (n) {
		ssize_t k = read(fd, p, n);
		if (k <= 0) {
			if (k < 0 && errno == EINTR)
				continue;
			return -1;
		}
		p += k;
		n -= (size_t)k;
	}
	return 0;
}

int bw_checkpoint_read(const char *path, bw_chkhdr_t *hdr_out,
			uint64_t **state_out, uint32_t ncols) {

	bw_chkhdr_t hdr;
	uint64_t *state, eof;
	char prev[512];
	int fd, rc = -1;

	fd = open(path, O_RDONLY);
	if (fd < 0) {
		/* fall back to the previous rotation if present */
		snprintf(prev, sizeof(prev), "%s.prev", path);
		fd = open(prev, O_RDONLY);
		if (fd < 0)
			return -1;
	}

	/* header */
	if (read_all(fd, &hdr, sizeof(hdr)))
		goto out;
	if (hdr.magic != BW_CHK_MAGIC || hdr.version != BW_FORMAT_VERSION)
		goto out;
	/* the caller re-checks dims/m/n/seed to print a precise error */

	/* EOF sentinel, located from the end: header + state + sentinel */
	{
		off_t want = (off_t)sizeof(bw_chkhdr_t) +
				(off_t)ncols * sizeof(uint64_t) + 8;
		off_t have = lseek(fd, 0, SEEK_END);
		if (have != want)
			goto out;    /* truncated or foreign */
		if (lseek(fd, -8, SEEK_END) == (off_t)-1)
			goto out;
		if (read_all(fd, &eof, 8) || eof != BW_EOF_MAGIC)
			goto out;
	}

	if (lseek(fd, sizeof(bw_chkhdr_t), SEEK_SET) == (off_t)-1)
		goto out;
	state = (uint64_t *)xmalloc((size_t)ncols * sizeof(uint64_t));
	if (read_all(fd, state, (size_t)ncols * sizeof(uint64_t)))
		goto out;

	/* golden hash of the state itself */
	{
		uint64_t h = bw_fnv1a(0, state, (size_t)ncols * sizeof(uint64_t));
		if (h != hdr.hash_x)
			goto out;
	}

	*hdr_out = hdr;
	*state_out = state;
	rc = 0;
out:
	close(fd);
	return rc;
}
