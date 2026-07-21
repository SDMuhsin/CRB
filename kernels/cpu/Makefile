# DOML CPU kernel R&D — P1 (format + reference decode).
# gcc 11.4; -march=icelake-server per brief. ref/ stays scalar (no intrinsics).
CC      = gcc
CFLAGS  = -std=c11 -O3 -march=icelake-server -Wall -Wextra -Werror -D_GNU_SOURCE

BIN     = tests/dpka_test
SRCS    = tests/dpka_test.c fmt/dpka_load.c ref/ref_decode.c
HDRS    = fmt/dpka.h ref/ref_decode.h

# P2b: DOML GEMV decode microkernel (gemv/)
GEMV_TEST  = gemv/gemv_test
GEMV_BENCH = gemv/gemv_bench
GEMV_SRCS  = gemv/doml_gemv.c fmt/dpka_load.c ref/ref_decode.c
GEMV_HDRS  = $(HDRS) gemv/doml_gemv.h

# P2c: fused-slab GEMV v2 (gemv2/) — links the P2b kernel too (activation
# prep + pool v1 for the barrier before/after microbench)
GEMV2_TEST  = gemv2/gemv2_test
GEMV2_BENCH = gemv2/gemv2_bench
GEMV2_SRCS  = gemv2/doml_gemv2.c $(GEMV_SRCS)
GEMV2_HDRS  = $(GEMV_HDRS) gemv2/doml_gemv2.h

# P3: prefill GEMM (gemm/) — slab-convert + VNNI on the frozen v2 i8 slab
GEMM_TEST  = gemm/gemm_test
GEMM_BENCH = gemm/gemm_bench
GEMM_SRCS  = gemm/doml_gemm.c $(GEMV2_SRCS)
GEMM_HDRS  = $(GEMV2_HDRS) gemm/doml_gemm.h

all: $(BIN) $(GEMV_TEST) $(GEMV_BENCH) $(GEMV2_TEST) $(GEMV2_BENCH) \
     $(GEMM_TEST) $(GEMM_BENCH)

$(BIN): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) -lm

$(GEMV_TEST): gemv/gemv_test.c $(GEMV_SRCS) $(GEMV_HDRS)
	$(CC) $(CFLAGS) -o $@ gemv/gemv_test.c $(GEMV_SRCS) -lm -lpthread

$(GEMV_BENCH): gemv/gemv_bench.c $(GEMV_SRCS) $(GEMV_HDRS)
	$(CC) $(CFLAGS) -o $@ gemv/gemv_bench.c $(GEMV_SRCS) -lm -lpthread

$(GEMV2_TEST): gemv2/gemv2_test.c $(GEMV2_SRCS) $(GEMV2_HDRS)
	$(CC) $(CFLAGS) -o $@ gemv2/gemv2_test.c $(GEMV2_SRCS) -lm -lpthread

$(GEMV2_BENCH): gemv2/gemv2_bench.c $(GEMV2_SRCS) $(GEMV2_HDRS)
	$(CC) $(CFLAGS) -o $@ gemv2/gemv2_bench.c $(GEMV2_SRCS) -lm -lpthread

$(GEMM_TEST): gemm/gemm_test.c $(GEMM_SRCS) $(GEMM_HDRS)
	$(CC) $(CFLAGS) -o $@ gemm/gemm_test.c $(GEMM_SRCS) -lm -lpthread

$(GEMM_BENCH): gemm/gemm_bench.c $(GEMM_SRCS) $(GEMM_HDRS)
	$(CC) $(CFLAGS) -o $@ gemm/gemm_bench.c $(GEMM_SRCS) -lm -lpthread

clean:
	rm -f $(BIN) $(GEMV_TEST) $(GEMV_BENCH) $(GEMV2_TEST) $(GEMV2_BENCH) \
	      $(GEMM_TEST) $(GEMM_BENCH)

.PHONY: all clean
