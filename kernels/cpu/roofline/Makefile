CC      = gcc
CFLAGS  = -O3 -march=icelake-server -mprefer-vector-width=512 -D_GNU_SOURCE -Wall -Wextra -pthread
BINS    = stream_read triad cache_bw instr_tp vnni_gemm mmap_read

all: $(BINS)

%: %.c common.h
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f $(BINS)

.PHONY: all clean
