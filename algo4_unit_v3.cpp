#include "algo4_unit_v3.hpp"
#include <algorithm>

// ---------------------------------------------------------------------------
// Build L_k = M ∪ M*k (sorted, duplicates collapsed).
// ---------------------------------------------------------------------------
static void build_merged(const std::vector<uint32_t>& m_divs, uint64_t k,
                         std::vector<uint64_t>& L)
{
    size_t n = m_divs.size();
    L.clear();
    L.reserve(2 * n);
    size_t ai = 0, bi = 0;
    while (ai < n && bi < n) {
        uint64_t a = m_divs[ai];
        uint64_t b = static_cast<uint64_t>(m_divs[bi]) * k;
        if (a < b)       { L.push_back(a); ++ai; }
        else if (a == b) { L.push_back(a); ++ai; ++bi; }
        else             { L.push_back(b); ++bi; }
    }
    while (ai < n) L.push_back(m_divs[ai++]);
    while (bi < n) L.push_back(static_cast<uint64_t>(m_divs[bi++]) * k);
}

// ---------------------------------------------------------------------------
// Correction: Lk passed in (no rebuild).
// Same merge-walk logic as algo4_unit_v2_correct.
// ---------------------------------------------------------------------------
uint32_t algo4_unit_v3_correct(
    const std::vector<uint64_t>& Lk,
    const std::vector<uint32_t>& mk_divs,
    const BitVector& bv)
{
    size_t len_lk   = Lk.size();
    size_t len_full = mk_divs.size();
    size_t s_full   = (len_full - 1) / 2;

    std::vector<uint64_t> gap_products;
    size_t   pk        = 0;
    uint64_t prev_full = 0;

    for (size_t pf = 0; pf <= s_full; ++pf) {
        uint64_t fd = mk_divs[pf];
        if (pk < len_lk && Lk[pk] == fd) {
            ++pk;
        } else {
            uint64_t exact_col = mk_divs[len_full - 1 - pf];
            uint64_t unit_col  = Lk[len_lk - 1 - pk];
            for (uint64_t row = prev_full; row < fd; ++row) {
                uint64_t j_lo = std::max(row, unit_col);
                for (uint64_t j = j_lo; j < exact_col; ++j) {
                    uint64_t prod = row * j;
                    if (!bv.test(prod)) gap_products.push_back(prod);
                }
            }
        }
        prev_full = fd;
    }

    std::sort(gap_products.begin(), gap_products.end());
    gap_products.erase(
        std::unique(gap_products.begin(), gap_products.end()),
        gap_products.end());
    return static_cast<uint32_t>(gap_products.size());
}

// ---------------------------------------------------------------------------
// algo4_unit_v3_correct_bv
//
// aux-bitvector correction.  Lk passed in (no rebuild).
// p_max computed from midpoint divisors.  p_min scanned cheaply.
// aux_bv[prod - p_min] used for deduplication — no vector+sort overhead.
// ---------------------------------------------------------------------------
uint32_t algo4_unit_v3_correct_bv(
    const std::vector<uint64_t>& Lk,
    const std::vector<uint32_t>& mk_divs,
    const BitVector& bv,
    BitVector& aux_bv)
{
    size_t len_lk   = Lk.size();
    size_t len_full = mk_divs.size();
    size_t s_full   = (len_full - 1) / 2;

    // Upper bound from midpoint divisors.
    uint64_t d_star   = mk_divs[s_full];
    uint64_t d_star_c = mk_divs[len_full - 1 - s_full];
    uint64_t p_max    = (d_star - 1) * (d_star_c - 1);

    // Lower bound: scan all gaps, no bv accesses.
    uint64_t p_min = p_max;
    {
        size_t   pk2        = 0;
        uint64_t prev_full2 = 0;
        for (size_t pf = 0; pf <= s_full; ++pf) {
            uint64_t fd = mk_divs[pf];
            if (pk2 < len_lk && Lk[pk2] == fd) {
                ++pk2;
            } else {
                uint64_t unit_col = Lk[len_lk - 1 - pk2];
                uint64_t j_lo     = std::max(prev_full2, unit_col);
                uint64_t p_gap    = prev_full2 * j_lo;
                if (p_gap < p_min) p_min = p_gap;
            }
            prev_full2 = fd;
        }
    }

    if (p_min > p_max) return 0;

    // Round p_min down to 64-bit word boundary so aux_bv is word-aligned
    // with bv: aux_bv word (w - word_offset) covers the same products as
    // bv word w, enabling stride-K mask reuse without phase adjustment.
    uint64_t p_min_aligned = (p_min >> 6) << 6;
    uint64_t word_offset   = p_min_aligned >> 6;

    uint64_t range = p_max - p_min_aligned + 1;
    aux_bv.resize(range);
    aux_bv.clear();

    // Main correction pass.
    uint32_t count     = 0;
    size_t   pk        = 0;
    uint64_t prev_full = 0;

    for (size_t pf = 0; pf <= s_full; ++pf) {
        uint64_t fd = mk_divs[pf];

        if (pk < len_lk && Lk[pk] == fd) {
            ++pk;
        } else {
            uint64_t exact_col = mk_divs[len_full - 1 - pf];
            uint64_t unit_col  = Lk[len_lk - 1 - pk];

            for (uint64_t row = prev_full; row < fd; ++row) {
                uint64_t j_lo = std::max(row, unit_col);
                if (j_lo >= exact_col) continue;
                if (row >= 1 && row <= 31) {
                    uint64_t bit_lo   = row * j_lo;
                    uint64_t bit_last = row * (exact_col - 1);
                    switch (row) {
                        case  1: count += bv.correct_strideK_filtered< 1>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case  2: count += bv.correct_strideK_filtered< 2>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case  3: count += bv.correct_strideK_filtered< 3>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case  4: count += bv.correct_strideK_filtered< 4>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case  5: count += bv.correct_strideK_filtered< 5>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case  6: count += bv.correct_strideK_filtered< 6>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case  7: count += bv.correct_strideK_filtered< 7>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case  8: count += bv.correct_strideK_filtered< 8>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case  9: count += bv.correct_strideK_filtered< 9>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 10: count += bv.correct_strideK_filtered<10>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 11: count += bv.correct_strideK_filtered<11>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 12: count += bv.correct_strideK_filtered<12>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 13: count += bv.correct_strideK_filtered<13>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 14: count += bv.correct_strideK_filtered<14>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 15: count += bv.correct_strideK_filtered<15>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 16: count += bv.correct_strideK_filtered<16>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 17: count += bv.correct_strideK_filtered<17>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 18: count += bv.correct_strideK_filtered<18>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 19: count += bv.correct_strideK_filtered<19>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 20: count += bv.correct_strideK_filtered<20>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 21: count += bv.correct_strideK_filtered<21>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 22: count += bv.correct_strideK_filtered<22>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 23: count += bv.correct_strideK_filtered<23>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 24: count += bv.correct_strideK_filtered<24>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 25: count += bv.correct_strideK_filtered<25>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 26: count += bv.correct_strideK_filtered<26>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 27: count += bv.correct_strideK_filtered<27>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 28: count += bv.correct_strideK_filtered<28>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 29: count += bv.correct_strideK_filtered<29>(bit_lo, bit_last, word_offset, aux_bv); break;
                        case 30: count += bv.correct_strideK_filtered<30>(bit_lo, bit_last, word_offset, aux_bv); break;
                        default: count += bv.correct_strideK_filtered<31>(bit_lo, bit_last, word_offset, aux_bv); break;
                    }
                } else {
                    // row == 0 produces no products; row > 31 uses scalar path.
                    for (uint64_t j = j_lo; j < exact_col; ++j) {
                        uint64_t prod = row * j;
                        if (!bv.test(prod)) {
                            uint64_t idx = prod - p_min_aligned;
                            if (!aux_bv.test(idx)) { aux_bv.set(idx); ++count; }
                        }
                    }
                }
            }
        }
        prev_full = fd;
    }

    return count;
}

// ---------------------------------------------------------------------------
// Precompute fast-path band structure and peel dc constants.
// Called once at init; bands are immutable for the lifetime of the state.
// ---------------------------------------------------------------------------
static void build_fast_bands(uint32_t m,
                              const std::vector<uint32_t>& m_divs,
                              uint64_t& fp_dc1, uint64_t& fp_dc2,
                              bool& fp_has_i2,
                              std::vector<FastBandV3>& bands)
{
    uint32_t tau_m = static_cast<uint32_t>(m_divs.size());

    // pk state after consuming i=1 (always in M since 1|m).
    size_t fp_pk1 = 1;
    fp_dc1 = (tau_m > fp_pk1) ? m_divs[tau_m - 1 - fp_pk1] : 0;

    // pk state after additionally consuming i=2 (if 2|m).
    fp_has_i2 = (m > 2);
    size_t fp_pk2 = fp_pk1;
    if (fp_has_i2 && tau_m > fp_pk2 && m_divs[fp_pk2] == 2) ++fp_pk2;
    fp_dc2 = (fp_has_i2 && tau_m > fp_pk2)
           ? m_divs[tau_m - 1 - fp_pk2] : 0;

    // Bands for i=3..m-1.  Each band is a contiguous row range sharing
    // one constant dc value.  Band ends at the next divisor of m.
    bands.clear();
    if (m <= 3) return;

    size_t   pk    = fp_pk2;
    uint32_t i_cur = 3;
    while (i_cur < m) {
        uint32_t i_end = (pk + 1 < tau_m) ? m_divs[pk] : m;
        if (i_end > m) i_end = m;
        if (i_end <= i_cur) {
            if (pk + 1 < tau_m) { ++pk; continue; } else break;
        }
        uint64_t dc_b = m_divs[tau_m - 1 - pk];
        uint32_t W    = i_end - i_cur;
        bands.push_back({i_cur, i_end, dc_b, static_cast<uint64_t>(W) > dc_b});
        i_cur = i_end;
        if (pk + 1 < tau_m) ++pk; else break;
    }
}

// ---------------------------------------------------------------------------
// algo4_unit_v3_init
//
// Clears bv, marks all products i*j for i=1..i_lim, j=i..col_{k0}(i)-1.
// Uses i=1 and i=2 word-level peels; scalar loop for i=3..i_lim-1.
// Stores L_{k0} in state.Lk for correction and sets up fast-path bands.
// ---------------------------------------------------------------------------
Algo4UnitV3State algo4_unit_v3_init(
    uint32_t m,
    const std::vector<uint32_t>& m_divs,
    uint64_t k0,
    BitVector& bv)
{
    bv.clear();
    uint64_t weight = 0;

    // Build L_{k0}.
    std::vector<uint64_t> Lk;
    build_merged(m_divs, k0, Lk);
    size_t   lenk  = Lk.size();
    uint64_t i_lim = Lk[(lenk - 1) / 2];
    size_t pk = 0;

    // i=1 peel: products 1*j for j=1..col-1 (contiguous range).
    if (i_lim > 1) {
        ++pk;
        uint64_t col = Lk[lenk - 1 - pk];
        if (col > 1)
            weight += bv.fill_range_and_count(1, col);
    }

    // i=2 peel: products 2*j for j=2..col-1 (stride-2 pattern).
    if (i_lim > 2) {
        if (Lk[pk] == 2) ++pk;
        uint64_t col = Lk[lenk - 1 - pk];
        if (col > 2)
            weight += bv.fill_stride2_and_count(4, 2 * (col - 1));
    }

    // i=3..i_lim-1: strideK for i<=31, scalar fallback beyond.
    for (uint64_t i = 3; i < i_lim; ++i) {
        if (Lk[pk] == i) ++pk;
        uint64_t col  = Lk[lenk - 1 - pk];
        uint64_t j_lo = i;
        if (j_lo >= col) continue;
        if (i <= 31) {
            uint64_t bit_lo   = i * j_lo;
            uint64_t bit_last = i * (col - 1);
            switch (i) {
                case  3: weight += bv.fill_strideK_and_count< 3>(bit_lo, bit_last); break;
                case  4: weight += bv.fill_strideK_and_count< 4>(bit_lo, bit_last); break;
                case  5: weight += bv.fill_strideK_and_count< 5>(bit_lo, bit_last); break;
                case  6: weight += bv.fill_strideK_and_count< 6>(bit_lo, bit_last); break;
                case  7: weight += bv.fill_strideK_and_count< 7>(bit_lo, bit_last); break;
                case  8: weight += bv.fill_strideK_and_count< 8>(bit_lo, bit_last); break;
                case  9: weight += bv.fill_strideK_and_count< 9>(bit_lo, bit_last); break;
                case 10: weight += bv.fill_strideK_and_count<10>(bit_lo, bit_last); break;
                case 11: weight += bv.fill_strideK_and_count<11>(bit_lo, bit_last); break;
                case 12: weight += bv.fill_strideK_and_count<12>(bit_lo, bit_last); break;
                case 13: weight += bv.fill_strideK_and_count<13>(bit_lo, bit_last); break;
                case 14: weight += bv.fill_strideK_and_count<14>(bit_lo, bit_last); break;
                case 15: weight += bv.fill_strideK_and_count<15>(bit_lo, bit_last); break;
                case 16: weight += bv.fill_strideK_and_count<16>(bit_lo, bit_last); break;
                case 17: weight += bv.fill_strideK_and_count<17>(bit_lo, bit_last); break;
                case 18: weight += bv.fill_strideK_and_count<18>(bit_lo, bit_last); break;
                case 19: weight += bv.fill_strideK_and_count<19>(bit_lo, bit_last); break;
                case 20: weight += bv.fill_strideK_and_count<20>(bit_lo, bit_last); break;
                case 21: weight += bv.fill_strideK_and_count<21>(bit_lo, bit_last); break;
                case 22: weight += bv.fill_strideK_and_count<22>(bit_lo, bit_last); break;
                case 23: weight += bv.fill_strideK_and_count<23>(bit_lo, bit_last); break;
                case 24: weight += bv.fill_strideK_and_count<24>(bit_lo, bit_last); break;
                case 25: weight += bv.fill_strideK_and_count<25>(bit_lo, bit_last); break;
                case 26: weight += bv.fill_strideK_and_count<26>(bit_lo, bit_last); break;
                case 27: weight += bv.fill_strideK_and_count<27>(bit_lo, bit_last); break;
                case 28: weight += bv.fill_strideK_and_count<28>(bit_lo, bit_last); break;
                case 29: weight += bv.fill_strideK_and_count<29>(bit_lo, bit_last); break;
                case 30: weight += bv.fill_strideK_and_count<30>(bit_lo, bit_last); break;
                default: weight += bv.fill_strideK_and_count<31>(bit_lo, bit_last); break;
            }
        } else {
            for (uint64_t j = j_lo; j < col; ++j) {
                uint64_t prod = i * j;
                if (!bv.test(prod)) { bv.set(prod); ++weight; }
            }
        }
    }

    // Precompute fast-path band structure.
    uint64_t             fp_dc1, fp_dc2;
    bool                 fp_has_i2;
    std::vector<FastBandV3> bands;
    build_fast_bands(m, m_divs, fp_dc1, fp_dc2, fp_has_i2, bands);

    return {m, k0, weight, std::move(Lk), std::move(bands),
            fp_dc1, fp_dc2, fp_has_i2};
}

// ---------------------------------------------------------------------------
// algo4_unit_v3_advance
//
// Slow path (k+1 <= m):
//   Builds L_{k+1}.  Derives old_col from state.Lk (= L_k) via pk_prev
//   two-pointer — no col[] array.  pk advances with equality check.
//   Updates state.Lk = L_{k+1}.
//
// Fast path (k+1 > m):
//   i=1, i=2 peels use precomputed fp_dc1/fp_dc2.
//   i=3..m-1: band-aware loop — j-centric for W > dc, i-centric otherwise.
//   Updates state.Lk = M || M*(k+1) in O(tau_m) for correction support.
// ---------------------------------------------------------------------------
uint64_t algo4_unit_v3_advance(
    Algo4UnitV3State& state,
    const std::vector<uint32_t>& m_divs,
    BitVector& bv)
{
    uint64_t k    = state.k + 1;
    uint32_t m    = state.m;
    bool     fast = (k > static_cast<uint64_t>(m));

    if (!fast) {
        // ------------------------------------------------------------------
        // Slow path: k <= m.
        // Build L_k, derive old_col from L_{k-1} = state.Lk via pk_prev.
        // ------------------------------------------------------------------
        std::vector<uint64_t> Lk_new;
        build_merged(m_divs, k, Lk_new);

        const auto& Lk_prev = state.Lk;
        size_t lenk     = Lk_new.size();
        size_t len_prev = Lk_prev.size();
        size_t pk       = 0;
        size_t pk_prev  = 0;

        uint64_t i_lim = Lk_new[(lenk - 1) / 2];

        // i=1 peel.
        if (i_lim > 1) {
            ++pk;
            uint64_t new_col = Lk_new[lenk - 1 - pk];
            if (len_prev > 0) ++pk_prev;
            uint64_t old_col = (pk_prev < len_prev)
                             ? Lk_prev[len_prev - 1 - pk_prev] : 0;
            uint64_t j_lo = std::max(uint64_t(1), old_col);
            if (j_lo < new_col)
                state.weight += bv.fill_range_and_count(j_lo, new_col);
        }

        // i=2 peel.
        if (i_lim > 2) {
            if (Lk_new[pk] == 2) ++pk;
            uint64_t new_col = Lk_new[lenk - 1 - pk];
            if (pk_prev < len_prev && Lk_prev[pk_prev] == 2) ++pk_prev;
            uint64_t old_col = (pk_prev < len_prev)
                             ? Lk_prev[len_prev - 1 - pk_prev] : 0;
            uint64_t j_lo = std::max(uint64_t(2), old_col);
            if (j_lo < new_col)
                state.weight += bv.fill_stride2_and_count(
                    2 * j_lo, 2 * (new_col - 1));
        }

        // i=3..i_lim-1: strideK for i<=31, scalar fallback beyond.
        // pk guard removed: pk never exhausts within i_lim range (see segk).
        // pk_prev guard kept: Lk_prev may be shorter than Lk_new.
        for (uint64_t i = 3; i < i_lim; ++i) {
            if (Lk_new[pk] == i) ++pk;
            uint64_t new_col = Lk_new[lenk - 1 - pk];

            if (pk_prev < len_prev && Lk_prev[pk_prev] == i) ++pk_prev;
            uint64_t old_col = (pk_prev < len_prev)
                             ? Lk_prev[len_prev - 1 - pk_prev] : 0;

            uint64_t j_lo = std::max(static_cast<uint64_t>(i), old_col);
            if (j_lo >= new_col) continue;
            if (i <= 31) {
                uint64_t bit_lo   = i * j_lo;
                uint64_t bit_last = i * (new_col - 1);
                switch (i) {
                    case  3: state.weight += bv.fill_strideK_and_count< 3>(bit_lo, bit_last); break;
                    case  4: state.weight += bv.fill_strideK_and_count< 4>(bit_lo, bit_last); break;
                    case  5: state.weight += bv.fill_strideK_and_count< 5>(bit_lo, bit_last); break;
                    case  6: state.weight += bv.fill_strideK_and_count< 6>(bit_lo, bit_last); break;
                    case  7: state.weight += bv.fill_strideK_and_count< 7>(bit_lo, bit_last); break;
                    case  8: state.weight += bv.fill_strideK_and_count< 8>(bit_lo, bit_last); break;
                    case  9: state.weight += bv.fill_strideK_and_count< 9>(bit_lo, bit_last); break;
                    case 10: state.weight += bv.fill_strideK_and_count<10>(bit_lo, bit_last); break;
                    case 11: state.weight += bv.fill_strideK_and_count<11>(bit_lo, bit_last); break;
                    case 12: state.weight += bv.fill_strideK_and_count<12>(bit_lo, bit_last); break;
                    case 13: state.weight += bv.fill_strideK_and_count<13>(bit_lo, bit_last); break;
                    case 14: state.weight += bv.fill_strideK_and_count<14>(bit_lo, bit_last); break;
                    case 15: state.weight += bv.fill_strideK_and_count<15>(bit_lo, bit_last); break;
                    case 16: state.weight += bv.fill_strideK_and_count<16>(bit_lo, bit_last); break;
                    case 17: state.weight += bv.fill_strideK_and_count<17>(bit_lo, bit_last); break;
                    case 18: state.weight += bv.fill_strideK_and_count<18>(bit_lo, bit_last); break;
                    case 19: state.weight += bv.fill_strideK_and_count<19>(bit_lo, bit_last); break;
                    case 20: state.weight += bv.fill_strideK_and_count<20>(bit_lo, bit_last); break;
                    case 21: state.weight += bv.fill_strideK_and_count<21>(bit_lo, bit_last); break;
                    case 22: state.weight += bv.fill_strideK_and_count<22>(bit_lo, bit_last); break;
                    case 23: state.weight += bv.fill_strideK_and_count<23>(bit_lo, bit_last); break;
                    case 24: state.weight += bv.fill_strideK_and_count<24>(bit_lo, bit_last); break;
                    case 25: state.weight += bv.fill_strideK_and_count<25>(bit_lo, bit_last); break;
                    case 26: state.weight += bv.fill_strideK_and_count<26>(bit_lo, bit_last); break;
                    case 27: state.weight += bv.fill_strideK_and_count<27>(bit_lo, bit_last); break;
                    case 28: state.weight += bv.fill_strideK_and_count<28>(bit_lo, bit_last); break;
                    case 29: state.weight += bv.fill_strideK_and_count<29>(bit_lo, bit_last); break;
                    case 30: state.weight += bv.fill_strideK_and_count<30>(bit_lo, bit_last); break;
                    default: state.weight += bv.fill_strideK_and_count<31>(bit_lo, bit_last); break;
                }
            } else {
                for (uint64_t j = j_lo; j < new_col; ++j) {
                    uint64_t prod = i * j;
                    if (!bv.test(prod)) { bv.set(prod); ++state.weight; }
                }
            }
        }

        state.Lk = std::move(Lk_new);

    } else {
        // ------------------------------------------------------------------
        // Fast path: k > m.  L_k = M || Mk (no overlap, fixed structure).
        // col(i) = dc(i)*k exactly; advance per step = dc(i).
        // ------------------------------------------------------------------

        // i=1 peel: contiguous bit range.
        {
            uint64_t new_col = state.fp_dc1 * k;
            uint64_t old_col = state.fp_dc1 * (k - 1);
            uint64_t j_lo    = std::max(uint64_t(1), old_col);
            if (j_lo < new_col)
                state.weight += bv.fill_range_and_count(j_lo, new_col);
        }

        // i=2 peel: stride-2 bit pattern.
        if (state.fp_has_i2) {
            uint64_t new_col = state.fp_dc2 * k;
            uint64_t old_col = state.fp_dc2 * (k - 1);
            uint64_t j_lo    = std::max(uint64_t(2), old_col);
            if (j_lo < new_col)
                state.weight += bv.fill_stride2_and_count(
                    2 * j_lo, 2 * (new_col - 1));
        }

        // i=3..m-1: band-aware adaptive loop.
        for (const auto& b : state.bands) {
            uint64_t dc      = b.dc;
            uint64_t old_col = dc * (k - 1);
            uint64_t new_col = dc * k;

            if (!b.j_centric) {
                // i-centric: dc >= W, wide rectangle.
                for (uint32_t i = b.i_lo; i < b.i_hi; ++i) {
                    uint64_t j_lo = std::max(static_cast<uint64_t>(i), old_col);
                    if (j_lo >= new_col) continue;
                    if (i <= 31) {
                        uint64_t bit_lo   = (uint64_t)i * j_lo;
                        uint64_t bit_last = (uint64_t)i * (new_col - 1);
                        switch (i) {
                            case  3: state.weight += bv.fill_strideK_and_count< 3>(bit_lo, bit_last); break;
                            case  4: state.weight += bv.fill_strideK_and_count< 4>(bit_lo, bit_last); break;
                            case  5: state.weight += bv.fill_strideK_and_count< 5>(bit_lo, bit_last); break;
                            case  6: state.weight += bv.fill_strideK_and_count< 6>(bit_lo, bit_last); break;
                            case  7: state.weight += bv.fill_strideK_and_count< 7>(bit_lo, bit_last); break;
                            case  8: state.weight += bv.fill_strideK_and_count< 8>(bit_lo, bit_last); break;
                            case  9: state.weight += bv.fill_strideK_and_count< 9>(bit_lo, bit_last); break;
                            case 10: state.weight += bv.fill_strideK_and_count<10>(bit_lo, bit_last); break;
                            case 11: state.weight += bv.fill_strideK_and_count<11>(bit_lo, bit_last); break;
                            case 12: state.weight += bv.fill_strideK_and_count<12>(bit_lo, bit_last); break;
                            case 13: state.weight += bv.fill_strideK_and_count<13>(bit_lo, bit_last); break;
                            case 14: state.weight += bv.fill_strideK_and_count<14>(bit_lo, bit_last); break;
                            case 15: state.weight += bv.fill_strideK_and_count<15>(bit_lo, bit_last); break;
                            case 16: state.weight += bv.fill_strideK_and_count<16>(bit_lo, bit_last); break;
                            case 17: state.weight += bv.fill_strideK_and_count<17>(bit_lo, bit_last); break;
                            case 18: state.weight += bv.fill_strideK_and_count<18>(bit_lo, bit_last); break;
                            case 19: state.weight += bv.fill_strideK_and_count<19>(bit_lo, bit_last); break;
                            case 20: state.weight += bv.fill_strideK_and_count<20>(bit_lo, bit_last); break;
                            case 21: state.weight += bv.fill_strideK_and_count<21>(bit_lo, bit_last); break;
                            case 22: state.weight += bv.fill_strideK_and_count<22>(bit_lo, bit_last); break;
                            case 23: state.weight += bv.fill_strideK_and_count<23>(bit_lo, bit_last); break;
                            case 24: state.weight += bv.fill_strideK_and_count<24>(bit_lo, bit_last); break;
                            case 25: state.weight += bv.fill_strideK_and_count<25>(bit_lo, bit_last); break;
                            case 26: state.weight += bv.fill_strideK_and_count<26>(bit_lo, bit_last); break;
                            case 27: state.weight += bv.fill_strideK_and_count<27>(bit_lo, bit_last); break;
                            case 28: state.weight += bv.fill_strideK_and_count<28>(bit_lo, bit_last); break;
                            case 29: state.weight += bv.fill_strideK_and_count<29>(bit_lo, bit_last); break;
                            case 30: state.weight += bv.fill_strideK_and_count<30>(bit_lo, bit_last); break;
                            default: state.weight += bv.fill_strideK_and_count<31>(bit_lo, bit_last); break;
                        }
                    } else {
                        for (uint64_t j = j_lo; j < new_col; ++j) {
                            uint64_t prod = static_cast<uint64_t>(i) * j;
                            if (!bv.test(prod)) { bv.set(prod); ++state.weight; }
                        }
                    }
                }
            } else {
                // j-centric: W > dc, tall rectangle.
                // For k > m: j = dc*(k-1) >= m >= i_hi for all rows in band,
                // so i < i_hi <= m <= j always — no i > j products possible.
                for (uint64_t j = old_col; j < new_col; ++j) {
                    uint64_t i_lo_j = b.i_lo;
                    uint64_t i_hi_j = b.i_hi - 1;
                    for (uint64_t i = i_lo_j; i <= i_hi_j; ++i) {
                        uint64_t prod = i * j;
                        if (!bv.test(prod)) { bv.set(prod); ++state.weight; }
                    }
                }
            }
        }

        // Rebuild state.Lk = M || Mk for correction.  O(tau_m), no merge.
        uint32_t tau_m = static_cast<uint32_t>(m_divs.size());
        state.Lk.resize(2 * tau_m);
        for (uint32_t j = 0; j < tau_m; ++j)
            state.Lk[j] = m_divs[j];
        for (uint32_t j = 0; j < tau_m; ++j)
            state.Lk[tau_m + j] = static_cast<uint64_t>(m_divs[j]) * k;
    }

    state.k = k;
    return state.weight;
}
