// stream_read.c - multithreaded DRAM streaming READ bandwidth (AVX-512 loads)
//
// Usage: stream_read <total_MB> <nthreads> <mode a|b|c> <reps>
//   mode a: all threads pinned to node0 (physical cores 0,2,..,22 first, then
//           node0 SMT 24,26,..,46); each thread first-touches its own chunk
//           -> all memory on node0, all compute on node0. Max 24 threads.
//   mode b: threads pinned to cpus 0..47 in order (0..23 = 24 physical cores
//           alternating node0/node1; 24..47 = SMT siblings); each thread
//           first-touches its own chunk -> node-local access for everyone.
//   mode c: same pinning as b, but ALL chunks are first-touched by the main
//           thread pinned to cpu0 (node0) -> node1 threads read remote memory
//           (the mmap/page-cache-on-one-node scenario).
//
// Per rep: barrier; every thread does enough full passes over its chunk to
// take >=~150 ms; barrier. Aggregate GB/s = total bytes read / wall time.
// CSV to stdout: stream_read,<mode>,<threads>,<total_MB>,<rep>,<passes>,<secs>,<GBps>
#define _GNU_SOURCE
#include "common.h"

typedef struct {
    int tid, cpu, nthreads, reps, passes;
    uint8_t *chunk;
    size_t chunk_bytes;
    int self_touch;
    pthread_barrier_t *bar;
    double *rep_secs;   // written by tid 0
    uint64_t checksum;
} targ_t;

// Placement evidence: sum N0=/N1= page counts over all large anonymous
// mappings (our chunks are >=80 MB; everything else is tiny). Guards against
// the kernel silently falling back to the other node when the first-touch
// node is short on free memory.
static void dump_anon_numa(void) {
    FILE *f = fopen("/proc/self/numa_maps", "r");
    if (!f) { fprintf(stderr, "numa_maps unavailable\n"); return; }
    char line[8192];
    long n0 = 0, n1 = 0;
    while (fgets(line, sizeof(line), f)) {
        if (!strstr(line, "anon=")) continue;
        long anon = 0;
        char *p = strstr(line, "anon=");
        if (p) anon = atol(p + 5);
        if (anon < 1000) continue; // skip small mappings (<4 MB)
        p = strstr(line, "N0=");
        if (p) n0 += atol(p + 3);
        p = strstr(line, "N1=");
        if (p) n1 += atol(p + 3);
    }
    fclose(f);
    fprintf(stderr, "PLACEMENT large-anon pages node0=%ld node1=%ld (%.1f%% on node0)\n",
            n0, n1, 100.0 * (double)n0 / (double)(n0 + n1 ? n0 + n1 : 1));
}

static int cpu_for_thread(char mode, int t) {
    if (mode == 'a') {
        return (t < 12) ? 2 * t : 24 + 2 * (t - 12);
    }
    return t; // modes b,c: cpus 0..47 (0..23 physical alternating nodes)
}

static void *worker(void *arg) {
    targ_t *a = (targ_t *)arg;
    pin_cpu(a->cpu);
    if (a->self_touch) {
        a->chunk = (uint8_t *)alloc_anon(a->chunk_bytes);
        memset(a->chunk, (a->tid & 63) + 1, a->chunk_bytes); // first-touch here
    }
    uint64_t cs = 0;
    // warmup pass (page tables, TLB, license)
    cs += avx512_read_pass(a->chunk, a->chunk_bytes);
    pthread_barrier_wait(a->bar);
    for (int r = 0; r < a->reps; r++) {
        pthread_barrier_wait(a->bar);
        double t0 = now_sec();
        for (int p = 0; p < a->passes; p++)
            cs += avx512_read_pass(a->chunk, a->chunk_bytes);
        pthread_barrier_wait(a->bar);
        if (a->tid == 0) a->rep_secs[r] = now_sec() - t0;
    }
    a->checksum = cs;
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <total_MB> <nthreads> <a|b|c> <reps>\n", argv[0]);
        return 1;
    }
    size_t total_mb = (size_t)atol(argv[1]);
    int nthreads = atoi(argv[2]);
    char mode = argv[3][0];
    int reps = atoi(argv[4]);
    if (mode == 'a' && nthreads > 24) {
        fprintf(stderr, "mode a supports at most 24 threads (node0 only)\n");
        return 1;
    }
    size_t total_bytes = total_mb << 20;
    size_t chunk = (total_bytes / (size_t)nthreads) & ~(size_t)4095;
    total_bytes = chunk * (size_t)nthreads;

    // choose passes so a rep lasts >= ~150 ms even at high aggregate BW.
    // conservative aggregate estimate 40 GB/s -> bytes needed = 6e9
    int passes = (int)((6.0e9 / (double)total_bytes) + 0.999);
    if (passes < 1) passes = 1;

    pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, nthreads);
    double *rep_secs = calloc(reps, sizeof(double));
    targ_t *ta = calloc(nthreads, sizeof(targ_t));
    pthread_t *th = calloc(nthreads, sizeof(pthread_t));

    int self_touch = (mode != 'c');
    if (!self_touch) {
        // main thread pins to cpu0 (node0) and first-touches everything
        pin_cpu(0);
        for (int t = 0; t < nthreads; t++) {
            ta[t].chunk = (uint8_t *)alloc_anon(chunk);
            memset(ta[t].chunk, (t & 63) + 1, chunk);
        }
    }
    for (int t = 0; t < nthreads; t++) {
        ta[t].tid = t;
        ta[t].cpu = cpu_for_thread(mode, t);
        ta[t].nthreads = nthreads;
        ta[t].reps = reps;
        ta[t].passes = passes;
        ta[t].chunk_bytes = chunk;
        ta[t].self_touch = self_touch;
        ta[t].bar = &bar;
        ta[t].rep_secs = rep_secs;
        pthread_create(&th[t], NULL, worker, &ta[t]);
    }
    uint64_t cs = 0;
    for (int t = 0; t < nthreads; t++) { pthread_join(th[t], NULL); cs ^= ta[t].checksum; }
    g_sink = cs;
    dump_anon_numa();

    double *gbps = calloc(reps, sizeof(double));
    double bytes_per_rep = (double)total_bytes * passes;
    for (int r = 0; r < reps; r++) {
        gbps[r] = bytes_per_rep / rep_secs[r] / 1e9;
        printf("stream_read,%c,%d,%zu,%d,%d,%.6f,%.3f\n",
               mode, nthreads, total_mb, r, passes, rep_secs[r], gbps[r]);
    }
    fprintf(stderr, "SUMMARY stream_read mode=%c threads=%d totalMB=%zu passes=%d "
            "median=%.3f GB/s min=%.3f max=%.3f (checksum %llx)\n",
            mode, nthreads, total_mb, passes,
            median_of(gbps, reps), min_of(gbps, reps), max_of(gbps, reps),
            (unsigned long long)cs);
    return 0;
}
