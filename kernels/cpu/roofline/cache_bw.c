// cache_bw.c - single-core streaming read bandwidth vs working-set size
//
// Usage: cache_bw <cpu> <reps>
// Working sets: 16KB, 256KB, 1MB, 8MB, 32MB. Pinned to <cpu>.
// Per rep: enough passes over the WS to total >= ~256 MB (>=0.1 s even at L1 speed
// we cap the total per rep at 4 GB).
// CSV: cache_bw,<cpu>,<ws_bytes>,<rep>,<passes>,<secs>,<GBps>
#define _GNU_SOURCE
#include "common.h"

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <cpu> <reps>\n", argv[0]); return 1; }
    int cpu = atoi(argv[1]);
    int reps = atoi(argv[2]);
    pin_cpu(cpu);

    size_t sizes[] = { 16ul << 10, 256ul << 10, 1ul << 20, 8ul << 20, 32ul << 20 };
    int nsizes = 5;
    for (int s = 0; s < nsizes; s++) {
        size_t ws = sizes[s];
        uint8_t *buf = (uint8_t *)alloc_anon(ws);
        memset(buf, 1, ws);
        // choose passes: target ~2 GB traffic per rep for small WS, >=6 passes for big
        long passes = (long)(2.0e9 / (double)ws);
        if (passes < 6) passes = 6;
        // warmup
        uint64_t cs = avx512_read_pass(buf, ws);
        double *gbps = calloc(reps, sizeof(double));
        for (int r = 0; r < reps; r++) {
            double t0 = now_sec();
            for (long p = 0; p < passes; p++) cs += avx512_read_pass(buf, ws);
            double dt = now_sec() - t0;
            gbps[r] = (double)ws * passes / dt / 1e9;
            printf("cache_bw,%d,%zu,%d,%ld,%.6f,%.3f\n", cpu, ws, r, passes, dt, gbps[r]);
        }
        g_sink = cs;
        fprintf(stderr, "SUMMARY cache_bw cpu=%d ws=%zu median=%.3f GB/s min=%.3f max=%.3f\n",
                cpu, ws, median_of(gbps, reps), min_of(gbps, reps), max_of(gbps, reps));
        munmap(buf, ws);
        free(gbps);
    }
    return 0;
}
