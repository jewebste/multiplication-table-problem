// mtable.cpp
//
// Compute M(n) = |{ij : 1 <= i,j <= n}|, the number of distinct entries in
// the n x n multiplication table.
//
// Algorithm: unit-shift (Brent-Pomerance-Purdum-Webster Algorithm 4) with
// variable-smoothness multipliers.  A multiplier m is included when
// m * P+(m) <= n, where P+(m) is the largest prime factor of m.  Multipliers
// are processed in decreasing order of tau(m) = |{d : d|m}|.
//
// Reference: arXiv:1908.04251 (Algorithms for the Multiplication Table Problem)
// Reference note:  This implementation goes beyond the above and incorporates
//                  the ability to correct shapes from unit shifts of Algorithm 4.
//
// Usage:
//   mtable <n> [--bound <b>] [--output <file>] [--warm <file>]
//
//   n             compute M(n); must be <= 2^32 - 1
//   --bound <b>   include multipliers m <= b  (default: floor(n^{2/3}))
//   --output <f>  write "k M(k)" for k=1..n, one per line, to text file f
//   --warm <f>    warm-start: binary file of uint32_t[n+1], val[k] = k - delta(k)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

#include "bitvector.hpp"
#include "rollsieve.h"
#include "factorize.hpp"
#include "udl.hpp"
#include "algo4_unit_v3.hpp"

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::milliseconds;

// ---------------------------------------------------------------------------
// All divisors of m, sorted ascending, via trial-division factorization.
// ---------------------------------------------------------------------------
static std::vector<uint32_t> all_divisors_of(uint32_t m) {
    return divisors_from_factorization(factorize_u32(m));
}

// ---------------------------------------------------------------------------
// Fallback: compute delta(n) by direct Algorithm 2 (no wheel optimization).
// Used only for the small residual set of integers not covered by any
// multiplier.  `scratch` is reused across calls to amortize allocation.
// ---------------------------------------------------------------------------
static uint32_t delta_direct(uint64_t n, BitVector& scratch) {
    if (n <= 1) return 0;

    // Divisors of n that are <= sqrt(n), sorted ascending.
    uint64_t sq = static_cast<uint64_t>(sqrtl(static_cast<long double>(n)));
    while ((sq + 1) * (sq + 1) <= n) ++sq;
    std::vector<uint32_t> small;
    for (uint64_t d = 1; d <= sq; ++d)
        if (n % d == 0) small.push_back(static_cast<uint32_t>(d));

    if (small.size() < 2) return 0;  // n is 1 or prime; delta = 0

    scratch.resize(n);  // clears
    uint32_t ki = 0;
    for (uint32_t i = 1; i < small.back(); ++i) {
        if (i == small[ki]) ++ki;
        uint32_t col = static_cast<uint32_t>(n / small[ki]);
        for (uint32_t j = i; j < col; ++j)
            scratch.set(static_cast<uint64_t>(i) * j);
    }
    return static_cast<uint32_t>(scratch.popcount());
}

// ---------------------------------------------------------------------------
// Compute floor(n^{2/3}) exactly.
// Uses __uint128_t for the comparison to avoid 64-bit overflow at large n.
// ---------------------------------------------------------------------------
static uint64_t floor_n23(uint64_t n) {
    uint64_t b = static_cast<uint64_t>(cbrtl(static_cast<long double>(n) *
                                             static_cast<long double>(n)));
    while ((__uint128_t)b * b * b > (__uint128_t)n * n) --b;
    while ((__uint128_t)(b + 1) * (b + 1) * (b + 1) <= (__uint128_t)n * n) ++b;
    return b;
}

// ---------------------------------------------------------------------------
// Multiplier record: m value and its divisor count.
// ---------------------------------------------------------------------------
struct Mult { uint32_t m; uint32_t tau; };

// ---------------------------------------------------------------------------
// Generate the var-smooth multiplier set for parameter N with m < bound_excl,
// then sort by tau(m) descending (ties: smaller m first for determinism).
// ---------------------------------------------------------------------------
static std::vector<Mult> gen_multipliers(uint64_t N, uint64_t bound_excl) {
    std::vector<Mult> mults;
    mults.push_back({1u, 1u});

    Rollsieve rs(2);
    std::vector<uint64_t> distinct;

    for (uint64_t m = 2; m < bound_excl; rs.next(), ++m) {
        safe_getlist(rs, distinct);
        uint64_t p_max = distinct.empty() ? 1ULL : distinct.back();
        if (m * p_max > N) continue;  // var-smooth condition: m * P+(m) <= N

        auto f   = factorize_from_distinct(m, distinct);
        auto div = divisors_from_factorization(f);
        mults.push_back({static_cast<uint32_t>(m),
                         static_cast<uint32_t>(div.size())});
    }

    std::stable_sort(mults.begin(), mults.end(),
                     [](const Mult& a, const Mult& b) { return a.tau > b.tau; });
    return mults;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: mtable <n> [--bound <b>] [--output <file>] [--warm <file>]\n"
            "  n             compute M(n)  (n <= 2^32 - 1)\n"
            "  --bound <b>   include multipliers m <= b  "
                            "(default: floor(n^{2/3}))\n"
            "  --output <f>  write 'k M(k)' for k=1..n to text file\n"
            "  --warm <f>    warm-start binary file: uint32_t[n+1] "
                            "of (k - delta(k)) values\n");
        return 1;
    }

    uint64_t N = strtoull(argv[1], nullptr, 10);
    if (N > static_cast<uint64_t>(UINT32_MAX)) {
        fprintf(stderr, "n must be <= %llu (single-processor memory limit)\n",
                static_cast<unsigned long long>(UINT32_MAX));
        return 1;
    }
    uint32_t n = static_cast<uint32_t>(N);

    // Default bound: m <= floor(n^{2/3}), exclusive bound = floor(n^{2/3}) + 1.
    uint64_t bound_excl = floor_n23(N) + 1;

    const char* output_file = nullptr;
    const char* warm_file   = nullptr;

    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--bound") && i + 1 < argc)
            bound_excl = strtoull(argv[++i], nullptr, 10) + 1;  // inclusive -> exclusive
        else if (!strcmp(argv[i], "--output") && i + 1 < argc)
            output_file = argv[++i];
        else if (!strcmp(argv[i], "--warm") && i + 1 < argc)
            warm_file = argv[++i];
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    printf("M(%u): unit-shift algorithm (Alg. 4, var-smooth multipliers)\n", n);
    printf("Multiplier bound : m <= %llu\n\n",
           static_cast<unsigned long long>(bound_excl - 1));

    // val[k] = k - delta(k) for k = 1..n; M(n) = sum_{k=1}^{n} val[k].
    std::vector<uint32_t> val(n + 1, 0);

    // Warm-start: load pre-computed val[] from binary file.
    if (warm_file) {
        FILE* f = fopen(warm_file, "rb");
        if (!f) {
            fprintf(stderr, "Cannot open warm file: %s\n", warm_file);
            return 1;
        }
        size_t nr = fread(val.data(), sizeof(uint32_t), n + 1, f);
        fclose(f);
        if (nr < n + 1)
            fprintf(stderr, "Warning: warm file has %zu entries, expected %u\n",
                    nr, n + 1);
        else
            printf("Warm-start: loaded %zu entries from %s\n\n", nr, warm_file);
    }

    auto t0 = Clock::now();

    // Generate multiplier set sorted by tau descending.
    auto mults  = gen_multipliers(N, bound_excl);
    long long ms_gen = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
    printf("Multipliers: %zu  [%lld ms]\n\n", mults.size(), ms_gen);

    BitVector bv(n + 1);  // shared unit-shift bit vector
    BitVector aux_bv;     // reused per correction call (auto-resizes)
    BitVector scratch;    // reused for fallback delta computation

    uint64_t tot_exact = 0, tot_corrected = 0, tot_skipped = 0;

    auto t_sweep = Clock::now();

    for (const auto& mult : mults) {
        uint32_t m     = mult.m;
        uint32_t k_max = n / m;
        auto m_divs    = all_divisors_of(m);
        auto fm        = factorize_u32(m);

        // Identify which products mk are uncached, and which composite k
        // require a correction (when L_k = M union M*k is a proper subset
        // of the divisors of mk).  v3 uses parallel corr_k / mk_divs_store
        // rather than a k-indexed sparse array, since state.Lk is consumed
        // in order and the correction index must match.
        std::vector<bool>                  uncached  (k_max + 1, false);
        std::vector<bool>                  needs_corr(k_max + 1, false);
        std::vector<uint32_t>              corr_k;
        std::vector<std::vector<uint32_t>> mk_divs_store;
        uint32_t k_start = 0;

        if (!val[m]) { uncached[1] = true; k_start = 1; }

        if (k_max >= 2) {
            Rollsieve rs2(2);
            for (uint32_t k = 2; k <= k_max; rs2.next(), ++k) {
                if (val[m * k]) continue;
                uncached[k] = true;
                if (!k_start) k_start = k;

                std::vector<uint64_t> distinct;
                safe_getlist(rs2, distinct);
                bool k_prime = (distinct.size() == 1 &&
                                distinct[0] == static_cast<uint64_t>(k));
                if (k_prime) continue;  // prime k: unit-shift is exact, no correction

                auto fk      = factorize_from_distinct(k, distinct);
                auto mk_divs = divisors_from_factorization(
                                    merge_factorizations(fm, fk));
                if (unit_list_size(m_divs, k) !=
                        static_cast<uint32_t>(mk_divs.size())) {
                    needs_corr[k] = true;
                    corr_k.push_back(k);
                    mk_divs_store.push_back(std::move(mk_divs));
                }
            }
        }

        if (!k_start) { ++tot_skipped; continue; }

        // Initialise the v3 unit-shift state at k_start.  state.Lk is
        // maintained by advance() so corrections avoid rebuilding it.
        Algo4UnitV3State state =
            algo4_unit_v3_init(m, m_divs, k_start, bv);

        // corr_ci indexes into corr_k / mk_divs_store in ascending k order.
        uint32_t corr_ci = static_cast<uint32_t>(
            std::lower_bound(corr_k.begin(), corr_k.end(), k_start)
            - corr_k.begin());

        uint32_t exact_m = 0, corr_m = 0;
        for (uint32_t k = k_start; k <= k_max; ++k) {
            uint32_t mk = m * k;
            if (uncached[k]) {
                if (needs_corr[k]) {
                    // state.Lk is L_k for the current step — no internal
                    // rebuild.  aux_bv deduplicates products that appear in
                    // multiple gap rectangles of the same correction call.
                    uint32_t c = algo4_unit_v3_correct_bv(
                        state.Lk, mk_divs_store[corr_ci++], bv, aux_bv);
                    val[mk] = mk - static_cast<uint32_t>(state.weight + c);
                    ++corr_m;
                } else {
                    val[mk] = mk - static_cast<uint32_t>(state.weight);
                    ++exact_m;
                }
            }
            if (k < k_max)
                algo4_unit_v3_advance(state, m_divs, bv);
        }
        tot_exact     += exact_m;
        tot_corrected += corr_m;
    }

    long long ms_sweep =
        std::chrono::duration_cast<Ms>(Clock::now() - t_sweep).count();

    // Fallback: any k not covered by any multiplier uses Algorithm 2 directly.
    auto t_fb = Clock::now();
    uint64_t fallback = 0;
    for (uint32_t i = 1; i <= n; ++i) {
        if (!val[i]) {
            val[i] = i - delta_direct(i, scratch);
            ++fallback;
        }
    }
    long long ms_fb =
        std::chrono::duration_cast<Ms>(Clock::now() - t_fb).count();

    // Stats
    printf("Exact fills    : %llu\n", static_cast<unsigned long long>(tot_exact));
    printf("Corrected fills: %llu\n", static_cast<unsigned long long>(tot_corrected));
    printf("Skipped (cached): %llu\n", static_cast<unsigned long long>(tot_skipped));
    printf("Fallback (Alg.2): %llu  [%lld ms]\n",
           static_cast<unsigned long long>(fallback), ms_fb);
    printf("Sweep time     : %lld ms\n", ms_sweep);

    // M(n) = sum of val[1..n]
    uint64_t M = 0;
    for (uint32_t i = 1; i <= n; ++i) M += val[i];

    long long ms_total =
        std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
    printf("\nM(%u) = %llu  [%lld ms total]\n",
           n, static_cast<unsigned long long>(M), ms_total);

    // Optional: write M(k) for k = 1..n to text file.
    if (output_file) {
        FILE* f = fopen(output_file, "w");
        if (!f) {
            fprintf(stderr, "Cannot open output file: %s\n", output_file);
            return 1;
        }
        uint64_t running = 0;
        for (uint32_t k = 1; k <= n; ++k) {
            running += val[k];
            fprintf(f, "%u %llu\n", k,
                    static_cast<unsigned long long>(running));
        }
        fclose(f);
        printf("Written M(k) for k=1..%u to %s\n", n, output_file);
    }

    return 0;
}
