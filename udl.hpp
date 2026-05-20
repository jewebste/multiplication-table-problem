#pragma once
#include <cstdint>
#include <vector>

// Returns the number of distinct elements in {d : d|m} ∪ {k*d : d|m}.
//
// This equals tau(m*k) iff the unit-shift shape (using only m-divisors) already
// covers the exact delta shape (using mk-divisors).  When they match, no
// correction call is needed — even for some composite k.
//
// Fast paths:
//   k == 1  →  right list equals left list  →  tau(m)
//   k  > m  →  smallest right-list element is k > m = largest left-list element,
//              so the two lists are disjoint  →  2*tau(m)
// General:
//   Two-pointer merge-dedup over sorted lists m_divs and {k*d : d ∈ m_divs}.
inline uint32_t unit_list_size(const std::vector<uint32_t>& m_divs, uint64_t k) {
    uint32_t tau_m = static_cast<uint32_t>(m_divs.size());
    if (k == 1) return tau_m;
    if (k > static_cast<uint64_t>(m_divs.back())) return 2 * tau_m;

    uint32_t count = 0, lo = 0, hi = 0;
    uint64_t prev = 0;
    while (lo < tau_m || hi < tau_m) {
        uint64_t a = (lo < tau_m) ? static_cast<uint64_t>(m_divs[lo]) : UINT64_MAX;
        uint64_t b = (hi < tau_m) ? k * m_divs[hi]                    : UINT64_MAX;
        uint64_t val;
        if (a <= b) { val = a; ++lo; if (a == b) ++hi; }
        else        { val = b; ++hi; }
        if (val != prev) { ++count; prev = val; }
    }
    return count;
}
