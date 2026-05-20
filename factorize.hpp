#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include "rollsieve.h"

// Prime-power pair: p^e in a factorization.
struct PrimePower { uint64_t p; uint32_t e; };

// Factorize a small uint32_t by trial division.  Suitable for multipliers m.
// Result is sorted ascending by p.
inline std::vector<PrimePower> factorize_u32(uint32_t n) {
    std::vector<PrimePower> f;
    for (uint32_t p = 2; (uint64_t)p * p <= n; ++p) {
        if (n % p == 0) {
            uint32_t e = 0;
            while (n % p == 0) { n /= p; ++e; }
            f.push_back({p, e});
        }
    }
    if (n > 1) f.push_back({n, 1});
    return f;
}

// getlist() has a blind spot at n = r^2 (the square of the next unrecorded prime r):
// T[pos] is empty because r is only pushed into T inside next() via the if(n==s)
// branch, which runs after the getlist opportunity.  We detect this via the public
// member rs.s == n and replace the bogus single entry {n} with the correct factor r.
inline void safe_getlist(Rollsieve& rs, std::vector<uint64_t>& distinct) {
    rs.getlist(distinct);
    uint64_t n = rs.getn();
    if (n == rs.s && distinct.size() == 1 && distinct[0] == n) {
        uint64_t r = static_cast<uint64_t>(sqrtl(static_cast<long double>(n)));
        distinct[0] = r;
    }
}

// Given the distinct prime factors returned by safe_getlist and the original n,
// recover the full factorization with exponents by trial division.
// Result is sorted ascending by p.
inline std::vector<PrimePower> factorize_from_distinct(uint64_t n,
                                                       const std::vector<uint64_t>& distinct) {
    std::vector<PrimePower> f;
    uint64_t tmp = n;
    for (uint64_t p : distinct) {
        uint32_t e = 0;
        while (tmp % p == 0) { tmp /= p; ++e; }
        f.push_back({p, e});
    }
    std::sort(f.begin(), f.end(), [](const PrimePower& a, const PrimePower& b) {
        return a.p < b.p;
    });
    return f;
}

// Merge two sorted factorizations fm and fk into the factorization of mk.
inline std::vector<PrimePower> merge_factorizations(const std::vector<PrimePower>& fm,
                                                    const std::vector<PrimePower>& fk) {
    std::vector<PrimePower> fmk;
    size_t i = 0, j = 0;
    while (i < fm.size() && j < fk.size()) {
        if (fm[i].p == fk[j].p) {
            fmk.push_back({fm[i].p, fm[i].e + fk[j].e});
            ++i; ++j;
        } else if (fm[i].p < fk[j].p) {
            fmk.push_back(fm[i++]);
        } else {
            fmk.push_back(fk[j++]);
        }
    }
    while (i < fm.size()) fmk.push_back(fm[i++]);
    while (j < fk.size()) fmk.push_back(fk[j++]);
    return fmk;
}

// Enumerate all divisors of n from its factorization, returned sorted ascending.
// mk < 2^32 for all cases in our computation, so uint32_t suffices.
inline std::vector<uint32_t> divisors_from_factorization(const std::vector<PrimePower>& f) {
    std::vector<uint32_t> divs = {1};
    for (const auto& pp : f) {
        size_t sz = divs.size();
        uint32_t pk = 1;
        for (uint32_t i = 0; i < pp.e; ++i) {
            pk *= static_cast<uint32_t>(pp.p);
            for (size_t j = 0; j < sz; ++j)
                divs.push_back(divs[j] * pk);
        }
    }
    std::sort(divs.begin(), divs.end());
    return divs;
}
