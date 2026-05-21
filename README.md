# multiplication-table-problem

Single-processor C++ implementation of an efficient algorithm for computing
$M(n)$, the number of distinct entries in the $n \times n$ multiplication table.

## The Problem

$$M(n) = |\{ij : 1 \le i, j \le n\}|$$

Despite its elementary definition, $M(n)$ grows significantly slower than $n^2$.
Erdős (1955) showed $M(n) = o(n^2)$, and Ford (2008) established

$$M(n) = \Theta \left(\frac{n^2}{\Phi(n)}\right), \quad
\Phi(n) = (\log n)^c (\log\log n)^{3/2}, \quad
c = 1 - \frac{1 + \log\log 2}{\log 2} \approx 0.086$$

Equivalently, most integers less than $n^2$ are *not* products of two integers both at most $n$.

## Algorithm

The implementation is based on Algorithm 4 of:

> R. Brent, C. Pomerance, D. Purdum, J. Webster,
> *Algorithms for the Multiplication Table Problem*,
> [arXiv:1908.04251](https://arxiv.org/abs/1908.04251) (2021).

This implementation goes beyond the above and incorporates the ability to
correct shapes from unit shifts of Algorithm 4.

The key identity is $M(n) = \sum_{k=1}^{n} (k - \delta(k))$, where $\delta(k)$
counts multiples of $k$ already in the $(k-1) \times (k-1)$ table.

**Unit-shift strategy.** For a fixed *multiplier* $m$, the divisor shape for
$mk$ changes predictably as $k$ increases by 1. When $k$ is prime the shape
shifts exactly (no correction needed); when $k$ is composite a small correction
accounts for extra divisors of $mk$ not in $M \cup kM$.

**Variable-smoothness multipliers.** A multiplier $m$ is included when
$m \cdot P^+(m) \le n$, where $P^+(m)$ is the largest prime factor of $m$.
Multipliers are processed in decreasing order of $\tau(m) = |\{d : d \mid m\}|$
so that high-divisor-count multipliers claim the most composite integers early,
minimising redundant work.

## Build

Requires a C++17 compiler. No external dependencies.

```
make
```

To enable SIMD acceleration on your platform, edit `CXXFLAGS` in the Makefile:

| Platform | Flag(s) |
|----------|---------|
| ARM NEON | `-DUSE_NEON_SIMD` |
| AVX-512  | `-DUSE_AVX512_SIMD -mavx512f` |

## Usage

```
./mtable <n> [--bound <b>] [--output <file>] [--warm <file>]
```

| Argument | Description |
|----------|-------------|
| `n` | Compute $M(n)$. Must be $\le 2^{32}-1$. |
| `--bound <b>` | Include multipliers $m \le b$ (default: $\lfloor n^{2/3} \rfloor$). |
| `--output <file>` | Write `k M(k)` for $k = 1 \ldots n$, one per line, to a text file. |
| `--warm <file>` | Warm-start from a binary file of `uint32_t[n+1]` where entry $k$ holds $k - \delta(k)$. |
| `--save-warm <file>` | Write $k - \delta(k)$ for $k = 0 \ldots n$ as a binary `uint32_t[n+1]` file that can be passed to `--warm` in a later run. |

## Examples

```
./mtable 1000000
```
```
M(1000000): unit-shift algorithm (Alg. 4, var-smooth multipliers)
Multiplier bound : m <= 10000

Multipliers: 5345  [3 ms]

Exact fills    : 978668
Corrected fills: 21332
Skipped (cached): 0
Fallback (Alg.2): 0  [0 ms]
Sweep time     : 18313 ms

M(1000000) = 198878423611  [18315 ms total]
```

Write all values $M(1), M(2), \ldots, M(10000)$ to a file:

```
./mtable 10000 --output values.txt
```

Incremental computation — compute $M(10^4)$, save a warm-start file, then
extend to $M(10^6)$ without redoing the work already done:

```
./mtable 10000 --save-warm warm_10k.bin
./mtable 1000000 --warm warm_10k.bin
```

## Known values

| $n$ | $M(n)$ |
|-----|--------|
| $10^3$ | $248{,}083$ |
| $10^6$ | $198{,}878{,}423{,}611$ |
| $2^{30}-1$ | $204{,}505{,}763{,}483{,}830{,}092$ |
| $2^{32}-1$ | $3{,}215{,}709{,}724{,}700{,}470{,}901$ |

The value for $2^{30}-1$ is from Brent et al. (2021); the value for $2^{32}-1$ is new.

## Memory

The bit-vector of $n$ bits is read and written with random-access patterns
throughout the entire multiplier sweep; performance degrades once it exceeds
the L3 cache.  The `uint32_t` array of $k - \delta(k)$ values is written
with strided access (multiples of each multiplier $m$) during the sweep and
read sequentially once at the end to form the sum $M(n)$.

| $n$ | bit-vector | `uint32_t` array |
|-----|-----------|-----------------|
| $10^6$ | 125 KB | 4 MB |
| $10^8$ | 12.5 MB | 400 MB |
| $10^9$ | 125 MB | 4 GB |

## For larger $n$

`mtable` performs all steps in a single pass on a single processor, which
becomes impractical beyond roughly $n \sim 10^9$.  Scaling to $n = 2^{32}$
and beyond requires several architectural changes:

**Decouple the pipeline steps.**  The single-pass structure of `mtable`
couples multiplier generation, the unit-shift sweep, corrections, and
accumulation.  At large scale these become separate programs communicating
through files, allowing each stage to be tuned, restarted, or distributed
independently.

**Segment the bit-vector.**  The shared bit-vector of $n$ bits must fit in
RAM.  For $n = 2^{32}$ that is 512 MB — manageable on its own, but the
random-access pattern across the full vector is cache-hostile.  Segmenting
the sweep so that each pass works on a contiguous chunk of the bit-vector
that fits in L3 cache significantly improves throughput.

**Distribute multipliers across processors.**  Each multiplier is largely
independent: it owns a disjoint set of composite integers and sweeps its own
portion of the bit-vector.  Multipliers can therefore be partitioned across
nodes in a cluster, each node writing its share of the $k - \delta(k)$ values
to a memory-mapped file.  An ownership bit-vector prevents two nodes from
counting the same composite twice.

**Replace the val array with a memory-mapped file.**  At $n = 2^{32}$ the
`uint32_t` array requires 16 GB, exceeding available RAM on a single node.
A memory-mapped file spreads this across disk-backed pages and allows
multiple nodes to write to disjoint regions concurrently.

