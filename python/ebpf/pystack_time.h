/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
#ifndef __PYSTACK_TIME_H
#define __PYSTACK_TIME_H

#define MAX_FRAMES 20
#define FILENAME_LEN 128
#define FUNCNAME_LEN 64

struct pyframe_event {
    __u32 pid;
    char filenames[MAX_FRAMES][FILENAME_LEN];
    char funcnames[MAX_FRAMES][FUNCNAME_LEN];
    __u32 depth;
};

#endif /* __PYSTACK_TIME_H */