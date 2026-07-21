// triad.c - STREAM-style triad a[i] = b[i] + s*c[i], 24 threads node-local
// (threads pinned to physical cores 0..23, per-thread slices first-touched
//  by the owning thread -> node-local memory)
//
// Usage: triad <MB_per_array> <nthreads> <reps>
// Reports STREAM-convention GB/s = 3*8*N/time (write-allocate/RFO traffic
// makes true bus traffic ~4/3 of this; noted in README).
// CSV: triad,<threads>,<MB_per_array>,<rep>,<secs>,<GBps_stream_convention>
#define _GNU_SOURCE
#include "common.h"

typedef struct {
    int tid, cpu, reps;
    double *a, *b, *c;
    size_t n; // elements in this thread's slice
    pthread_barrier_t *bar;
    double *rep_secs;
} targ_t;

static void *worker(void *arg) {
    targ_t *t = (targ_t *)arg;
    pin_cpu(t->cpu);
    t->a = (double *)alloc_anon(t->n * 8);
    t->b = (double *)alloc_anon(t->n * 8);
    t->c = (double *)alloc_anon(t->n * 8);
    for (size_t i = 0; i < t->n; i++) { t->a[i] = 0.0; t->b[i] = 1.5; t->c[i] = 2.5; }
    const double s = 3.0;
    pthread_barrier_wait(t->bar);
    for (int r = 0; r < t->reps; r++) {
        pthread_barrier_wait(t->bar);
        double t0 = now_sec();
        double *restrict a = t->a; const double *restrict b = t->b, *restrict c = t->c;
        for (size_t i = 0; i < t->n; i++) a[i] = b[i] + s * c[i];
        pthread_barrier_wait(t->bar);
        if (t->tid == 0) t->rep_secs[r] = now_sec() - t0;
    }
    g_sink = (uint64_t)t->a[t->n / 2];
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <MB_per_array> <nthreads> <reps>\n", argv[0]); return 1; }
    size_t mb = (size_t)atol(argv[1]);
    int nthreads = atoi(argv[2]);
    int reps = atoi(argv[3]);
    size_t n_total = (mb << 20) / 8;
    size_t n_per = (n_total / (size_t)nthreads) & ~(size_t)63;

    pthread_barrier_t bar; pthread_barrier_init(&bar, NULL, nthreads);
    double *rep_secs = calloc(reps, sizeof(double));
    targ_t *ta = calloc(nthreads, sizeof(targ_t));
    pthread_t *th = calloc(nthreads, sizeof(pthread_t));
    for (int t = 0; t < nthreads; t++) {
        ta[t] = (targ_t){ .tid = t, .cpu = t, .reps = reps, .n = n_per,
                          .bar = &bar, .rep_secs = rep_secs };
        pthread_create(&th[t], NULL, worker, &ta[t]);
    }
    for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);

    double *gbps = calloc(reps, sizeof(double));
    double bytes = 3.0 * 8.0 * (double)n_per * nthreads;
    for (int r = 0; r < reps; r++) {
        gbps[r] = bytes / rep_secs[r] / 1e9;
        printf("triad,%d,%zu,%d,%.6f,%.3f\n", nthreads, mb, r, rep_secs[r], gbps[r]);
    }
    fprintf(stderr, "SUMMARY triad threads=%d MB/array=%zu median=%.3f GB/s min=%.3f max=%.3f\n",
            nthreads, mb, median_of(gbps, reps), min_of(gbps, reps), max_of(gbps, reps));
    return 0;
}
