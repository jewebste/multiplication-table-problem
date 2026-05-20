#pragma once
#include <cstdint>
#include <vector>
#include "bitvector.hpp"

// Unit-shift v3: structural improvement over v2.
//
// Changes vs v2:
//   1. col[] eliminated.  Slow path (k <= m): old_col derived via pk_prev
//      two-pointer walking L_{k-1} in parallel with L_k.  Fast path (k > m):
//      old_col = dc*(k-1) computed directly from divisor band.
//   2. rdata[] eliminated.  Fast path uses precomputed band structure instead.
//   3. pk advancement: while (Lk[pk] <= i) ++pk -> if (Lk[pk] == i) ++pk,
//      since i increments by 1; pk < lenk guard removed (unreachable).
//   4. i=1 and i=2 peeled with fill_range_and_count / fill_stride2_and_count.
//   5. Fast path band-aware adaptive loop: j-centric when band width W > dc
//      (tall rectangle), i-centric when W <= dc (wide rectangle).
//   6. Correction takes pre-built Lk (no internal rebuild).  state.Lk is kept
//      current after every init/advance call for use by the caller.
//
// NOTE: test_and_set is NOT used here.  In the non-segmented sweep the
// bitvector fills progressively; conditional test+set avoids the unconditional
// write cost that test_and_set pays as fill rate grows.  (test_and_set caused
// a ~22% regression when applied to the non-segmented sweep.)
//
// Correction: algo4_unit_v3_correct receives state.Lk directly.  For k <= m
// state.Lk = L_k (maintained by slow path).  For k > m state.Lk = M || Mk
// (rebuilt cheaply in O(tau_m) each advance).

struct FastBandV3 {
    uint32_t i_lo, i_hi;  // row range [i_lo, i_hi)
    uint64_t dc;           // column advance per k step = m_divs[tau_m-1-pk]
    bool     j_centric;   // true when band width W > dc
};

struct Algo4UnitV3State {
    uint32_t m;
    uint64_t k;        // current step
    uint64_t weight;
    // L_k for the current step — used by caller for correction.
    // Slow path: full merge result.  Fast path: M || Mk concatenation.
    std::vector<uint64_t> Lk;
    // Fast-path precomputation (k > m): bands + peel dc constants.
    std::vector<FastBandV3> bands;
    uint64_t fp_dc1;     // dc for i=1 peel
    uint64_t fp_dc2;     // dc for i=2 peel (valid only when fp_has_i2)
    bool     fp_has_i2;  // true when m > 2
};

// Correction for composite k where L_k ⊊ divs(mk).
// Lk must be L_k for the current step (use state.Lk).
// Same merge-walk as v2_correct but Lk is passed in, not rebuilt.
uint32_t algo4_unit_v3_correct(
    const std::vector<uint64_t>& Lk,
    const std::vector<uint32_t>& mk_divs,
    const BitVector& bv
);

// aux-bitvector variant: uses aux_bv for deduplication instead of vector+sort.
// Faster for multipliers with many corrections.
// aux_bv is resized/cleared internally; caller should keep it alive across calls.
uint32_t algo4_unit_v3_correct_bv(
    const std::vector<uint64_t>& Lk,
    const std::vector<uint32_t>& mk_divs,
    const BitVector& bv,
    BitVector& aux_bv
);

// Initialise at step k0.  Clears bv and populates it with all products
// i*j for rows i=1..i_lim, j=i..col_{k0}(i)-1.
// Precomputes fast-path band structure for subsequent advance() calls.
Algo4UnitV3State algo4_unit_v3_init(
    uint32_t m,
    const std::vector<uint32_t>& m_divs,
    uint64_t k0,
    BitVector& bv
);

// Advance k -> k+1.  Updates state.Lk, state.weight.  Returns weight.
uint64_t algo4_unit_v3_advance(
    Algo4UnitV3State& state,
    const std::vector<uint32_t>& m_divs,
    BitVector& bv
);
