// mmap_read.c - read a (page-cache-warm) file via mmap with N unpinned threads.
// Quantifies the placement penalty vs malloc'd node-local memory (the
// "mmap'd GGUF pp512 plateau" hypothesis).
//
// Usage: mmap_read <file> <nthreads> <reps>
// Dumps the /proc/self/numa_maps line for the mapping (N0=/N1= page counts)
// to stderr as placement evidence.
// CSV: mmap_read,<threads>,<MB>,<rep>,<passes>,<secs>,<GBps>
#define _GNU_SOURCE
#include "common.h"
#include <fcntl.h>
#include <sys/stat.h>

typedef struct {
    int tid, reps, passes;
    const uint8_t *base;
    size_t bytes;
    pthread_barrier_t *bar;
    double *rep_secs;
} targ_t;

static void *worker(void *arg) {
    targ_t *a = (targ_t *)arg;
    // NOT pinned - this is the point (mimics default runtime thread placement)
    uint64_t cs = avx512_read_pass(a->base, a->bytes); // warm
    pthread_barrier_wait(a->bar);
    for (int r = 0; r < a->reps; r++) {
        pthread_barrier_wait(a->bar);
        double t0 = now_sec();
        for (int p = 0; p < a->passes; p++)
            cs += avx512_read_pass(a->base, a->bytes);
        pthread_barrier_wait(a->bar);
        if (a->tid == 0) a->rep_secs[r] = now_sec() - t0;
    }
    g_sink = cs;
    return NULL;
}

static void dump_numa_maps(void *addr) {
    FILE *f = fopen("/proc/self/numa_maps", "r");
    if (!f) { fprintf(stderr, "numa_maps: unavailable\n"); return; }
    char line[4096];
    unsigned long want = (unsigned long)addr;
    while (fgets(line, sizeof(line), f)) {
        unsigned long a = strtoul(line, NULL, 16);
        if (a == want) { fprintf(stderr, "numa_maps: %s", line); break; }
    }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <file> <nthreads> <reps>\n", argv[0]); return 1; }
    const char *path = argv[1];
    int nthreads = atoi(argv[2]);
    int reps = atoi(argv[3]);

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(fd, &st);
    size_t bytes = (size_t)st.st_size & ~(size_t)4095;
    uint8_t *base = mmap(NULL, bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }

    // warm the page cache fully (sequential read of whole mapping)
    uint64_t cs = avx512_read_pass(base, bytes);
    cs += avx512_read_pass(base, bytes);
    g_sink = cs;
    dump_numa_maps(base);

    size_t slice = (bytes / (size_t)nthreads) & ~(size_t)4095;
    int passes = (int)((6.0e9 / (double)(slice * nthreads)) + 0.999);
    if (passes < 1) passes = 1;

    pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, nthreads);
    double *rep_secs = calloc(reps, sizeof(double));
    targ_t *ta = calloc(nthreads, sizeof(targ_t));
    pthread_t *th = calloc(nthreads, sizeof(pthread_t));
    for (int t = 0; t < nthreads; t++) {
        ta[t] = (targ_t){ .tid = t, .reps = reps, .passes = passes,
                          .base = base + (size_t)t * slice, .bytes = slice,
                          .bar = &bar, .rep_secs = rep_secs };
        pthread_create(&th[t], NULL, worker, &ta[t]);
    }
    for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);

    double *gbps = calloc(reps, sizeof(double));
    double bytes_per_rep = (double)slice * nthreads * passes;
    for (int r = 0; r < reps; r++) {
        gbps[r] = bytes_per_rep / rep_secs[r] / 1e9;
        printf("mmap_read,%d,%zu,%d,%d,%.6f,%.3f\n",
               nthreads, bytes >> 20, r, passes, rep_secs[r], gbps[r]);
    }
    fprintf(stderr, "SUMMARY mmap_read threads=%d MB=%zu median=%.3f GB/s min=%.3f max=%.3f\n",
            nthreads, bytes >> 20, median_of(gbps, reps), min_of(gbps, reps), max_of(gbps, reps));
    return 0;
}
