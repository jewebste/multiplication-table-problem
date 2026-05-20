# Makefile for mtable — single-processor M(n) computation.
#
# Build:   make
# Run:     ./mtable <n> [--bound <b>] [--output <file>] [--warm <file>]
#
# Optional SIMD flags (edit CXXFLAGS):
#   ARM NEON:  add -DUSE_NEON_SIMD  (optionally -DUSE_NEON_SHIFT)
#   AVX-512:   add -DUSE_AVX512_SIMD -mavx512f  (optionally -DUSE_AVX512_TABLE)

CXX      ?= c++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra

DEPS := algo4_unit_v3.cpp rollsieve.cpp

.PHONY: all clean

all: mtable

mtable: mtable.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -o $@ mtable.cpp $(DEPS)

clean:
	rm -f mtable
