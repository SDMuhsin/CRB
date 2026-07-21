// common.h - shared helpers for roofline microbenchmarks
// Build with -D_GNU_SOURCE -O3 -march=icelake-server -pthread
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <immintrin.h>
#include <sys/mman.h>

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static inline void pin_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set)) {
        perror("sched_setaffinity");
        exit(1);
    }
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_of(double *v, int n) {
    qsort(v, n, sizeof(double), cmp_double);
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double min_of(double *v, int n) {
    double m = v[0];
    for (int i = 1; i < n; i++) if (v[i] < m) m = v[i];
    return m;
}
static double max_of(double *v, int n) {
    double m = v[0];
    for (int i = 1; i < n; i++) if (v[i] > m) m = v[i];
    return m;
}

// Allocate page-aligned anonymous memory (not yet touched -> first-touch decides NUMA node)
static void *alloc_anon(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    return p;
}

// Streaming AVX-512 read of [buf, buf+bytes), 8 independent accumulators.
// Returns a checksum so the compiler cannot elide the loads.
static inline uint64_t avx512_read_pass(const uint8_t *buf, size_t bytes) {
    const __m512i *p = (const __m512i *)buf;
    size_t n = bytes / 64;
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    __m512i a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 = _mm512_add_epi64(a0, _mm512_load_si512(p + i + 0));
        a1 = _mm512_add_epi64(a1, _mm512_load_si512(p + i + 1));
        a2 = _mm512_add_epi64(a2, _mm512_load_si512(p + i + 2));
        a3 = _mm512_add_epi64(a3, _mm512_load_si512(p + i + 3));
        a4 = _mm512_add_epi64(a4, _mm512_load_si512(p + i + 4));
        a5 = _mm512_add_epi64(a5, _mm512_load_si512(p + i + 5));
        a6 = _mm512_add_epi64(a6, _mm512_load_si512(p + i + 6));
        a7 = _mm512_add_epi64(a7, _mm512_load_si512(p + i + 7));
    }
    a0 = _mm512_add_epi64(a0, a1);
    a2 = _mm512_add_epi64(a2, a3);
    a4 = _mm512_add_epi64(a4, a5);
    a6 = _mm512_add_epi64(a6, a7);
    a0 = _mm512_add_epi64(a0, a2);
    a4 = _mm512_add_epi64(a4, a6);
    a0 = _mm512_add_epi64(a0, a4);
    return (uint64_t)_mm512_reduce_add_epi64(a0);
}

// Global sink to defeat dead-code elimination
volatile uint64_t g_sink;
