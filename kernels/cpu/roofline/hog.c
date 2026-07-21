// hog.c - allocate and touch <MB> of anonymous memory, then exit.
// Purpose: force global reclaim of page cache in a container where
// /proc/sys/vm/drop_caches is read-only. Freeing page cache is required so
// that first-touch NUMA placement in the benchmarks actually lands on the
// intended node (node0 had ~1 GB free with ~100 GB of page cache; without
// this, "node0" first-touches silently fall back to node1 - verified via
// /proc/self/numa_maps: only 17.3% landed on node0).
#define _GNU_SOURCE
#include "common.h"

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <MB>\n", argv[0]); return 1; }
    size_t mb = (size_t)atol(argv[1]);
    size_t step = 1024; // touch in 1 GB steps, report progress
    size_t done = 0;
    while (done < mb) {
        size_t chunk = (mb - done < step) ? (mb - done) : step;
        uint8_t *p = alloc_anon(chunk << 20);
        memset(p, 1, chunk << 20);
        g_sink += p[12345];
        done += chunk;
        fprintf(stderr, "hog: touched %zu / %zu MB\n", done, mb);
    }
    return 0;
}
