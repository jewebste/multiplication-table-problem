#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX512F__
#include <immintrin.h>
#endif

// Cache-line aligned bit vector with hardware popcount.
// Designed for memory efficiency: single allocation, reused across passes.
class BitVector {
public:
    BitVector() : data_(nullptr), num_words_(0) {}

    explicit BitVector(uint64_t num_bits) {
        num_words_ = (num_bits + 63) / 64;
        allocate(num_words_);
        clear();
    }

    ~BitVector() {
        free(data_);
    }

    // Non-copyable, movable
    BitVector(const BitVector&) = delete;
    BitVector& operator=(const BitVector&) = delete;

    BitVector(BitVector&& other) noexcept
        : data_(other.data_), num_words_(other.num_words_) {
        other.data_ = nullptr;
        other.num_words_ = 0;
    }

    BitVector& operator=(BitVector&& other) noexcept {
        if (this != &other) {
            free(data_);
            data_ = other.data_;
            num_words_ = other.num_words_;
            other.data_ = nullptr;
            other.num_words_ = 0;
        }
        return *this;
    }

    // Resize to hold at least num_bits. Only reallocates if capacity insufficient.
    void resize(uint64_t num_bits) {
        uint64_t needed = (num_bits + 63) / 64;
        if (needed > capacity_words_) {
            free(data_);
            allocate(needed);
        }
        num_words_ = needed;
        clear();
    }

    void clear() {
        std::memset(data_, 0, num_words_ * sizeof(uint64_t));
    }

    // Clear only the first num_bits bits (rounds up to word boundary).
    // Faster than clear() when num_bits << capacity.
    void clear_up_to(uint64_t num_bits) {
        uint64_t words = (num_bits + 63) / 64;
        if (words > num_words_) words = num_words_;
        std::memset(data_, 0, words * sizeof(uint64_t));
    }

    void set(uint64_t idx) {
        data_[idx >> 6] |= (uint64_t(1) << (idx & 63));
    }

    bool test(uint64_t idx) const {
        return (data_[idx >> 6] >> (idx & 63)) & 1;
    }

    // Set all bits in [bit_lo, bit_hi) and return the count of newly set bits.
    // Uses word-level OR for efficiency; handles partial first/last words.
    uint64_t fill_range_and_count(uint64_t bit_lo, uint64_t bit_hi) {
        if (bit_lo >= bit_hi) return 0;
        uint64_t w0  = bit_lo >> 6;
        uint64_t w1  = (bit_hi - 1) >> 6;
        int      lo_bit = static_cast<int>(bit_lo & 63);
        int      hi_bit = static_cast<int>((bit_hi - 1) & 63);
        uint64_t first_mask = ~uint64_t(0) << lo_bit;
        uint64_t last_mask  = (hi_bit == 63) ? ~uint64_t(0)
                                              : (uint64_t(1) << (hi_bit + 1)) - 1;
        uint64_t count = 0;
        if (w0 == w1) {
            uint64_t mask = first_mask & last_mask;
            uint64_t old  = data_[w0];
            count = __builtin_popcountll(mask & ~old);
            data_[w0] = old | mask;
        } else {
            uint64_t old = data_[w0];
            count += __builtin_popcountll(first_mask & ~old);
            data_[w0] = old | first_mask;
            for (uint64_t w = w0 + 1; w < w1; ++w) {
                count += __builtin_popcountll(~data_[w]);
                data_[w] = ~uint64_t(0);
            }
            old = data_[w1];
            count += __builtin_popcountll(last_mask & ~old);
            data_[w1] = old | last_mask;
        }
        return count;
    }

    // Set bits at stride-2 positions in [bit_lo, bit_last] (both inclusive, same parity).
    // base_mask = 0x5555...ULL << (bit_lo & 1) selects the correct alternating pattern.
    // Returns count of newly set bits.
    uint64_t fill_stride2_and_count(uint64_t bit_lo, uint64_t bit_last) {
        uint64_t w0        = bit_lo  >> 6;
        uint64_t w1        = bit_last >> 6;
        uint64_t base_mask = 0x5555555555555555ULL << (bit_lo & 1);
        uint64_t first_mask = base_mask & (~uint64_t(0) << (bit_lo & 63));
        int      hi_bit     = static_cast<int>(bit_last & 63);
        uint64_t last_mask  = base_mask & ((hi_bit == 63) ? ~uint64_t(0)
                                                           : (uint64_t(1) << (hi_bit + 1)) - 1);
        uint64_t count = 0;
        if (w0 == w1) {
            uint64_t mask = first_mask & last_mask;
            uint64_t old  = data_[w0];
            count = __builtin_popcountll(mask & ~old);
            data_[w0] = old | mask;
        } else {
            uint64_t old = data_[w0];
            count += __builtin_popcountll(first_mask & ~old);
            data_[w0] = old | first_mask;
            for (uint64_t w = w0 + 1; w < w1; ++w) {
                count += __builtin_popcountll(base_mask & ~data_[w]);
                data_[w] |= base_mask;
            }
            old = data_[w1];
            count += __builtin_popcountll(last_mask & ~old);
            data_[w1] = old | last_mask;
        }
        return count;
    }

    // Set bits at stride-K positions in [bit_lo, bit_last] (both inclusive,
    // same residue mod K).  Template parameter K must be in [3, 31].
    //
    // Key insight: build K masks, one per phase p in [0,K), where masks[p] has
    // bits set at positions p, p+K, p+2K, ... within [0,63].  The starting
    // phase is bit_lo % K.  Advancing one 64-bit word shifts the phase by
    // 64 % K (since 64 bits pass).  For K=4: 64%4==0, phase is constant.
    // For K=3,5,6,7 the phase cycles, requiring a different mask per word.
    //
    // Returns count of newly set bits.
    template <int K>
    uint64_t fill_strideK_and_count(uint64_t bit_lo, uint64_t bit_last) {
        static_assert(K >= 3 && K <= 31, "K must be in [3,31]");
#if defined(__AVX512F__) && defined(USE_AVX512_SIMD)
  #ifdef USE_AVX512_TABLE
        return fill_strideK_avx512_table<K>(bit_lo, bit_last);
  #else
        return fill_strideK_avx512_sllv<K>(bit_lo, bit_last);
  #endif
#elif defined(__ARM_NEON) && defined(USE_NEON_SIMD)
  #ifdef USE_NEON_SHIFT
        return fill_strideK_neon_shift<K>(bit_lo, bit_last);
  #else
        return fill_strideK_neon_table<K>(bit_lo, bit_last);
  #endif
#endif
        constexpr int PHASE_STEP = 64 % K;

        // masks[p]: bits at positions p, p+K, p+2K, ... in [0,63].
        // Computed once per template instantiation at compile time.
        static constexpr auto ms = []() constexpr {
            std::array<uint64_t, K> m = {};
            for (int bit = 0; bit < 64; ++bit)
                m[bit % K] |= (uint64_t(1) << bit);
            return m;
        }();

        uint64_t w0 = bit_lo   >> 6;
        uint64_t w1 = bit_last >> 6;

        // Phase for word w: local positions b to set satisfy
        //   (w*64 + b) ≡ bit_lo (mod K)  →  b ≡ (bit_lo - w*64) (mod K).
        // Phase decreases by (64 % K) each word.
        int r          = static_cast<int>(bit_lo % K);
        int q0         = static_cast<int>((w0 % K) * (64 % K) % K);
        int phase      = (r - q0 + K * K) % K;
        int q1         = static_cast<int>((w1 % K) * (64 % K) % K);
        int last_phase = (r - q1 + K * K) % K;

        uint64_t first_mask = ms[phase] & (~uint64_t(0) << (bit_lo & 63));
        int      hi_bit     = static_cast<int>(bit_last & 63);
        uint64_t last_mask  = ms[last_phase] &
                              ((hi_bit == 63) ? ~uint64_t(0)
                                              : (uint64_t(1) << (hi_bit + 1)) - 1);

        uint64_t count = 0;
        if (w0 == w1) {
            uint64_t mask = first_mask & last_mask;
            uint64_t old  = data_[w0];
            count = __builtin_popcountll(mask & ~old);
            data_[w0] = old | mask;
        } else {
            uint64_t old = data_[w0];
            count += __builtin_popcountll(first_mask & ~old);
            data_[w0] = old | first_mask;
            phase = (phase - PHASE_STEP + K) % K;
            for (uint64_t w = w0 + 1; w < w1; ++w) {
                uint64_t mask = ms[phase];
                count += __builtin_popcountll(mask & ~data_[w]);
                data_[w] |= mask;
                phase = (phase - PHASE_STEP + K) % K;
            }
            old = data_[w1];
            count += __builtin_popcountll(last_mask & ~old);
            data_[w1] = old | last_mask;
        }
        return count;
    }

    // -----------------------------------------------------------------------
    // Word-aligned filtered stride-K correction.
    //
    // Walks products row*j for j in [bit_lo/row, bit_last/row] (i.e. products
    // at positions bit_lo, bit_lo+row, ..., bit_last in bv).  For each such
    // product position p:
    //   unseen  = p not set in *this (not yet in the main table)
    //   new_bit = p not set in aux_bv either (first time seen in correction)
    //   if new_bit: count++, set aux_bv[p - word_offset*64]
    //
    // Precondition (enforced by caller):
    //   - p_min_aligned = word_offset * 64 is 64-aligned and <= min(bit_lo).
    //   - aux_bv.data_[w - word_offset] covers the same 64-bit product range
    //     as this->data_[w] for every word w in [bit_lo>>6, bit_last>>6].
    //     This holds because p_min_aligned is a multiple of 64.
    //
    // K must be in [1, 31].  K=1 fills every position (range correction),
    // K=2 fills every other position, K>=3 as in fill_strideK_and_count.
    // -----------------------------------------------------------------------
    template <int K>
    uint64_t correct_strideK_filtered(uint64_t bit_lo, uint64_t bit_last,
                                      uint64_t word_offset,
                                      BitVector& aux_bv) const {
        static_assert(K >= 1 && K <= 31, "K must be in [1,31]");
        constexpr int PHASE_STEP = 64 % K;

        static constexpr auto ms = []() constexpr {
            std::array<uint64_t, K> m = {};
            for (int bit = 0; bit < 64; ++bit)
                m[bit % K] |= (uint64_t(1) << bit);
            return m;
        }();

        uint64_t w0 = bit_lo   >> 6;
        uint64_t w1 = bit_last >> 6;

        int r          = static_cast<int>(bit_lo % K);
        int q0         = static_cast<int>((w0 % K) * (64 % K) % K);
        int phase      = (r - q0 + K * K) % K;
        int q1         = static_cast<int>((w1 % K) * (64 % K) % K);
        int last_phase = (r - q1 + K * K) % K;

        uint64_t first_mask = ms[phase] & (~uint64_t(0) << (bit_lo & 63));
        int      hi_bit     = static_cast<int>(bit_last & 63);
        uint64_t last_mask  = ms[last_phase] &
                              ((hi_bit == 63) ? ~uint64_t(0)
                                              : (uint64_t(1) << (hi_bit + 1)) - 1);

        uint64_t count = 0;

        if (w0 == w1) {
            uint64_t mask    = first_mask & last_mask;
            uint64_t unseen  = mask & ~data_[w0];
            uint64_t& aux_w  = aux_bv.data_[w0 - word_offset];
            uint64_t new_bits = unseen & ~aux_w;
            count += __builtin_popcountll(new_bits);
            aux_w |= unseen;
            return count;
        }

        // First word.
        {
            uint64_t unseen  = first_mask & ~data_[w0];
            uint64_t& aux_w  = aux_bv.data_[w0 - word_offset];
            uint64_t new_bits = unseen & ~aux_w;
            count += __builtin_popcountll(new_bits);
            aux_w |= unseen;
        }
        phase = (phase - PHASE_STEP + K) % K;

        // Middle words.
        for (uint64_t w = w0 + 1; w < w1; ++w) {
            uint64_t unseen  = ms[phase] & ~data_[w];
            uint64_t& aux_w  = aux_bv.data_[w - word_offset];
            uint64_t new_bits = unseen & ~aux_w;
            count += __builtin_popcountll(new_bits);
            aux_w |= unseen;
            phase = (phase - PHASE_STEP + K) % K;
        }

        // Last word.
        {
            uint64_t unseen  = last_mask & ~data_[w1];
            uint64_t& aux_w  = aux_bv.data_[w1 - word_offset];
            uint64_t new_bits = unseen & ~aux_w;
            count += __builtin_popcountll(new_bits);
            aux_w |= unseen;
        }
        return count;
    }

    // -----------------------------------------------------------------------
    // ARM NEON implementations of fill_strideK_and_count (W=2, 128-bit).
    // Enabled when __ARM_NEON is defined.
    // Compile with -DUSE_NEON_SIMD to activate from fill_strideK_and_count<K>.
    // Compile with -DUSE_NEON_SHIFT (plus USE_NEON_SIMD) for shift variant.
    // -----------------------------------------------------------------------
#ifdef __ARM_NEON

    // Popcount of a 128-bit NEON register.
    static inline uint64_t neon_popcount128(uint64x2_t v) {
        uint8x16_t bytes = vreinterpretq_u8_u64(v);
        uint8x16_t cnt   = vcntq_u8(bytes);
        uint64x2_t s64   = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(cnt)));
        return vgetq_lane_u64(s64, 0) + vgetq_lane_u64(s64, 1);
    }

    // Cross-lane right shift of 128-bit register by R bits (0 < R < 64).
    template <int R>
    static inline uint64x2_t neon_shift_right(uint64x2_t v) {
        static_assert(R > 0 && R < 64);
        return vorrq_u64(vshrq_n_u64(v, R),
                         vextq_u64(vshlq_n_u64(v, 64 - R), vdupq_n_u64(0), 1));
    }

    // Table-based NEON variant.
    // neon_groups[p] = {ms[p], ms[(p-PHASE_STEP+K)%K]} for each starting phase p.
    // Inner loop loads one 128-bit group per pair of words; phase cycles every
    // K/gcd(K, BLOCK_STEP) NEON blocks.
    template <int K>
    uint64_t fill_strideK_neon_table(uint64_t bit_lo, uint64_t bit_last) {
        static_assert(K >= 3 && K <= 31);
        constexpr int PHASE_STEP = 64 % K;
        constexpr int BLOCK_STEP = (2 * PHASE_STEP) % K;

        static constexpr auto ms = []() constexpr {
            std::array<uint64_t, K> m = {};
            for (int bit = 0; bit < 64; ++bit) m[bit % K] |= uint64_t(1) << bit;
            return m;
        }();

        static constexpr auto neon_groups = []() constexpr {
            std::array<std::array<uint64_t, 2>, K> g = {};
            for (int p = 0; p < K; ++p) {
                g[p][0] = ms[p];
                g[p][1] = ms[((p - PHASE_STEP) % K + K) % K];
            }
            return g;
        }();

        uint64_t w0 = bit_lo   >> 6;
        uint64_t w1 = bit_last >> 6;

        int r          = (int)(bit_lo % K);
        int q0         = (int)((w0 % K) * (64 % K) % K);
        int phase      = (r - q0 + K * K) % K;
        int q1         = (int)((w1 % K) * (64 % K) % K);
        int last_phase = (r - q1 + K * K) % K;

        uint64_t first_mask = ms[phase] & (~uint64_t(0) << (bit_lo & 63));
        int      hi_bit     = (int)(bit_last & 63);
        uint64_t last_mask  = ms[last_phase] &
                              ((hi_bit == 63) ? ~uint64_t(0)
                                              : (uint64_t(1) << (hi_bit + 1)) - 1);
        uint64_t count = 0;

        if (w0 == w1) {
            uint64_t mask = first_mask & last_mask;
            uint64_t old  = data_[w0];
            count = __builtin_popcountll(mask & ~old);
            data_[w0] = old | mask;
            return count;
        }

        // First word: scalar with first_mask.
        { uint64_t old = data_[w0];
          count += __builtin_popcountll(first_mask & ~old);
          data_[w0] = old | first_mask; }
        int cur_phase = (phase - PHASE_STEP + K) % K;

        // NEON body: pairs of full words in [w0+1, w1-1].
        uint64_t w = w0 + 1;
        while (w + 1 < w1) {
            uint64x2_t mask_vec = vld1q_u64(neon_groups[cur_phase].data());
            uint64x2_t old_vec  = vld1q_u64(data_ + w);
            count += neon_popcount128(vbicq_u64(mask_vec, old_vec));
            vst1q_u64(data_ + w, vorrq_u64(old_vec, mask_vec));
            cur_phase = (cur_phase - BLOCK_STEP + K) % K;
            w += 2;
        }

        // Leftover middle word (when w1-w0 is even).
        if (w < w1) {
            uint64_t old = data_[w];
            count += __builtin_popcountll(ms[cur_phase] & ~old);
            data_[w] = old | ms[cur_phase];
            w++;
        }

        // Last word: scalar with last_mask.
        { uint64_t old = data_[w1];
          count += __builtin_popcountll(last_mask & ~old);
          data_[w1] = old | last_mask; }

        return count;
    }

    // Shift-based NEON variant.
    // Maintains one 128-bit mask register updated per block via cross-lane
    // bit-shift — no groups table; zero cache footprint beyond one register.
    template <int K>
    uint64_t fill_strideK_neon_shift(uint64_t bit_lo, uint64_t bit_last) {
        static_assert(K >= 3 && K <= 31);
        constexpr int PHASE_STEP = 64 % K;
        constexpr int BLOCK_STEP = (2 * PHASE_STEP) % K;

        static constexpr auto ms = []() constexpr {
            std::array<uint64_t, K> m = {};
            for (int bit = 0; bit < 64; ++bit) m[bit % K] |= uint64_t(1) << bit;
            return m;
        }();

        uint64_t w0 = bit_lo   >> 6;
        uint64_t w1 = bit_last >> 6;

        int r          = (int)(bit_lo % K);
        int q0         = (int)((w0 % K) * (64 % K) % K);
        int phase      = (r - q0 + K * K) % K;
        int q1         = (int)((w1 % K) * (64 % K) % K);
        int last_phase = (r - q1 + K * K) % K;

        uint64_t first_mask = ms[phase] & (~uint64_t(0) << (bit_lo & 63));
        int      hi_bit     = (int)(bit_last & 63);
        uint64_t last_mask  = ms[last_phase] &
                              ((hi_bit == 63) ? ~uint64_t(0)
                                              : (uint64_t(1) << (hi_bit + 1)) - 1);
        uint64_t count = 0;

        if (w0 == w1) {
            uint64_t mask = first_mask & last_mask;
            uint64_t old  = data_[w0];
            count = __builtin_popcountll(mask & ~old);
            data_[w0] = old | mask;
            return count;
        }

        // First word: scalar.
        { uint64_t old = data_[w0];
          count += __builtin_popcountll(first_mask & ~old);
          data_[w0] = old | first_mask; }
        int cur_phase = (phase - PHASE_STEP + K) % K;

        // Build initial 128-bit mask: lane0=ms[cur_phase], lane1=ms[next_phase].
        uint64_t init[2] = { ms[cur_phase],
                             ms[((cur_phase - PHASE_STEP) % K + K) % K] };
        uint64x2_t cur_mask = vld1q_u64(init);

        // NEON body.
        uint64_t w = w0 + 1;
        while (w + 1 < w1) {
            uint64x2_t old_vec = vld1q_u64(data_ + w);
            count += neon_popcount128(vbicq_u64(cur_mask, old_vec));
            vst1q_u64(data_ + w, vorrq_u64(old_vec, cur_mask));
            if constexpr (BLOCK_STEP > 0)
                cur_mask = neon_shift_right<BLOCK_STEP>(cur_mask);
            w += 2;
        }

        // Leftover middle word: lane 0 of cur_mask holds the correct full mask.
        if (w < w1) {
            uint64_t full_mask = vgetq_lane_u64(cur_mask, 0);
            uint64_t old = data_[w];
            count += __builtin_popcountll(full_mask & ~old);
            data_[w] = old | full_mask;
            w++;
        }

        // Last word: scalar.
        { uint64_t old = data_[w1];
          count += __builtin_popcountll(last_mask & ~old);
          data_[w1] = old | last_mask; }

        return count;
    }

#endif // __ARM_NEON

    // -----------------------------------------------------------------------
    // AVX-512 implementations of fill_strideK_and_count (W=8, 512-bit).
    // Enabled when __AVX512F__ is defined (i.e. -mavx512f or -march=znver4).
    // Compile with -DUSE_AVX512_SIMD to activate from fill_strideK_and_count<K>.
    // Two variants:
    //   fill_strideK_avx512_sllv  (default): broadcast T + _mm512_sllv_epi64.
    //     72 bytes per K (8 B for T + 64 B for delta_arr); pure computation,
    //     no per-K mask table. Only needs __AVX512F__.
    //   fill_strideK_avx512_table (legacy):  K×64-byte precomputed group table.
    //     Compile with -DUSE_AVX512_TABLE to select.
    // On Zen 4 (-march=znver4), __AVX512VPOPCNTDQ__ is also defined, enabling
    // the fast _mm512_popcnt_epi64 path inside avx512_popcount512.
    // -----------------------------------------------------------------------
#ifdef __AVX512F__

    // Horizontal popcount of a 512-bit register.
    static inline uint64_t avx512_popcount512(__m512i v) {
#ifdef __AVX512VPOPCNTDQ__
        __m512i cnt = _mm512_popcnt_epi64(v);
        __m256i lo  = _mm512_castsi512_si256(cnt);
        __m256i hi  = _mm512_extracti64x4_epi64(cnt, 1);
        __m256i s   = _mm256_add_epi64(lo, hi);
        __m128i s2  = _mm_add_epi64(_mm256_castsi256_si128(s),
                                     _mm256_extracti128_si256(s, 1));
        return (uint64_t)_mm_cvtsi128_si64(s2) +
               (uint64_t)_mm_extract_epi64(s2, 1);
#else
        alignas(64) uint64_t tmp[8];
        _mm512_store_si512((__m512i*)tmp, v);
        return __builtin_popcountll(tmp[0]) + __builtin_popcountll(tmp[1]) +
               __builtin_popcountll(tmp[2]) + __builtin_popcountll(tmp[3]) +
               __builtin_popcountll(tmp[4]) + __builtin_popcountll(tmp[5]) +
               __builtin_popcountll(tmp[6]) + __builtin_popcountll(tmp[7]);
#endif
    }

    // Broadcast-T + sllv AVX-512 variant.
    //
    // Key property: ms[p] = T << p for any p in [0,K), where
    //   T = ms[0] = bits at 0, K, 2K, ..., floor(63/K)*K.
    // (T << p shifts bit at position b to b+p; for p < K all such positions
    // are distinct from T's other bits, so no masking is needed.)
    //
    // Per 8-word block starting at phase cur_phase:
    //   - Lane i covers the word at offset w+i, whose full mask is ms[phase_i]
    //     where phase_i = (cur_phase - i*PHASE_STEP + K) % K.
    //   - Broadcast T into all 8 lanes, then apply _mm512_sllv_epi64 with
    //     a per-lane shift of phase_i.  The shift vector is computed as
    //       phase_vec = clamp_to_positive(cur_phase - delta_arr, K)
    //     where delta_arr[i] = (i * PHASE_STEP) % K (compile-time constant).
    //
    // Memory per K: 8 bytes (T) + 64 bytes (delta_arr) = 72 bytes total,
    // vs K*64 bytes for the table variant (up to 1984 B for K=31).
    // Requires only __AVX512F__ (no AVX512DQ/BW).
    template <int K>
    uint64_t fill_strideK_avx512_sllv(uint64_t bit_lo, uint64_t bit_last) {
        static_assert(K >= 3 && K <= 31, "K must be in [3,31]");
        constexpr int PHASE_STEP = 64 % K;
        constexpr int BLOCK_STEP = (8 * PHASE_STEP) % K;

        // Base mask: bits at 0, K, 2K, ..., floor(63/K)*K.
        // ms[p] = T << p (variable left-shift reconstructs any phase mask).
        static constexpr uint64_t T = []() constexpr -> uint64_t {
            uint64_t t = 0;
            for (int bit = 0; bit < 64; bit += K) t |= uint64_t(1) << bit;
            return t;
        }();

        // delta_vec[i] = (i * PHASE_STEP) % K: phase offset for lane i.
        // All values are compile-time constants (K and PHASE_STEP are template params);
        // the compiler folds _mm512_set_epi64 to an immediate, so no load needed.

        // ms[]: needed for scalar head/tail words.
        static constexpr auto ms = []() constexpr {
            std::array<uint64_t, K> m = {};
            for (int bit = 0; bit < 64; ++bit) m[bit % K] |= (uint64_t(1) << bit);
            return m;
        }();

        uint64_t w0 = bit_lo   >> 6;
        uint64_t w1 = bit_last >> 6;

        int r          = (int)(bit_lo % K);
        int q0         = (int)((w0 % K) * (64 % K) % K);
        int phase      = (r - q0 + K * K) % K;
        int q1         = (int)((w1 % K) * (64 % K) % K);
        int last_phase = (r - q1 + K * K) % K;

        uint64_t first_mask = ms[phase] & (~uint64_t(0) << (bit_lo & 63));
        int      hi_bit     = (int)(bit_last & 63);
        uint64_t last_mask  = ms[last_phase] &
                              ((hi_bit == 63) ? ~uint64_t(0)
                                              : (uint64_t(1) << (hi_bit + 1)) - 1);
        uint64_t count = 0;

        if (w0 == w1) {
            uint64_t mask = first_mask & last_mask;
            uint64_t old  = data_[w0];
            count = __builtin_popcountll(mask & ~old);
            data_[w0] = old | mask;
            return count;
        }

        // First word: scalar.
        { uint64_t old = data_[w0];
          count += __builtin_popcountll(first_mask & ~old);
          data_[w0] = old | first_mask; }
        int cur_phase = (phase - PHASE_STEP + K) % K;

        // Load compile-time constants into SIMD registers once.
        // _mm512_set_epi64 takes lanes high-to-low: (e7,e6,...,e0).
        __m512i T_vec     = _mm512_set1_epi64((int64_t)T);
        __m512i delta_vec = _mm512_set_epi64(
            (7*PHASE_STEP)%K, (6*PHASE_STEP)%K,
            (5*PHASE_STEP)%K, (4*PHASE_STEP)%K,
            (3*PHASE_STEP)%K, (2*PHASE_STEP)%K,
            (1*PHASE_STEP)%K, (0*PHASE_STEP)%K);
        __m512i K_vec     = _mm512_set1_epi64(K);
        __m512i zero      = _mm512_setzero_si512();

        // AVX-512 body: 8 words at a time.
        uint64_t w = w0 + 1;
        while (w + 7 < w1) {
            // per-lane phase: (cur_phase - delta[i] + K) % K.
            // cur_phase ∈ [0,K), delta[i] ∈ [0,K)  →  diff ∈ [-(K-1), K-1].
            __m512i p0_vec    = _mm512_set1_epi64((int64_t)cur_phase);
            __m512i diff      = _mm512_sub_epi64(p0_vec, delta_vec);
            // Add K to lanes where diff < 0 (signed comparison; needs AVX512F only).
            __mmask8 neg      = _mm512_cmpgt_epi64_mask(zero, diff);
            __m512i phase_vec = _mm512_mask_add_epi64(diff, neg, diff, K_vec);
            // Reconstruct each lane's stride mask: ms[phase_i] = T << phase_i.
            __m512i mask_vec  = _mm512_sllv_epi64(T_vec, phase_vec);

            __m512i old_vec  = _mm512_loadu_si512((const void*)(data_ + w));
            __m512i new_bits = _mm512_andnot_si512(old_vec, mask_vec);  // mask & ~old
            count += avx512_popcount512(new_bits);
            _mm512_storeu_si512((void*)(data_ + w), _mm512_or_si512(old_vec, mask_vec));

            cur_phase = (cur_phase - BLOCK_STEP + K) % K;
            w += 8;
        }

        // Scalar tail: remaining full words before the last.
        while (w < w1) {
            uint64_t old = data_[w];
            count += __builtin_popcountll(ms[cur_phase] & ~old);
            data_[w] = old | ms[cur_phase];
            cur_phase = (cur_phase - PHASE_STEP + K) % K;
            ++w;
        }

        // Last word: scalar.
        { uint64_t old = data_[w1];
          count += __builtin_popcountll(last_mask & ~old);
          data_[w1] = old | last_mask; }

        return count;
    }

    // Table-based AVX-512 variant.
    // avx_groups[p][i] = mask for word i within an 8-word block at starting phase p.
    // Inner loop loads one 512-bit group per 8 words; phase cycles every
    // K / gcd(K, BLOCK_STEP) blocks.
    template <int K>
    uint64_t fill_strideK_avx512_table(uint64_t bit_lo, uint64_t bit_last) {
        static_assert(K >= 3 && K <= 31, "K must be in [3,31]");
        constexpr int PHASE_STEP = 64 % K;
        constexpr int BLOCK_STEP = (8 * PHASE_STEP) % K;

        static constexpr auto ms = []() constexpr {
            std::array<uint64_t, K> m = {};
            for (int bit = 0; bit < 64; ++bit) m[bit % K] |= uint64_t(1) << bit;
            return m;
        }();

        // avx_groups[p][i] = ms[phase of the i-th word in a block starting at phase p].
        // Phase of word i = (p - i * PHASE_STEP) mod K.
        static constexpr auto avx_groups = []() constexpr {
            std::array<std::array<uint64_t, 8>, K> g = {};
            for (int p = 0; p < K; ++p)
                for (int i = 0; i < 8; ++i)
                    g[p][i] = ms[((p - i * PHASE_STEP) % K + K * 64) % K];
            return g;
        }();

        uint64_t w0 = bit_lo   >> 6;
        uint64_t w1 = bit_last >> 6;

        int r          = (int)(bit_lo % K);
        int q0         = (int)((w0 % K) * (64 % K) % K);
        int phase      = (r - q0 + K * K) % K;
        int q1         = (int)((w1 % K) * (64 % K) % K);
        int last_phase = (r - q1 + K * K) % K;

        uint64_t first_mask = ms[phase] & (~uint64_t(0) << (bit_lo & 63));
        int      hi_bit     = (int)(bit_last & 63);
        uint64_t last_mask  = ms[last_phase] &
                              ((hi_bit == 63) ? ~uint64_t(0)
                                              : (uint64_t(1) << (hi_bit + 1)) - 1);
        uint64_t count = 0;

        if (w0 == w1) {
            uint64_t mask = first_mask & last_mask;
            uint64_t old  = data_[w0];
            count = __builtin_popcountll(mask & ~old);
            data_[w0] = old | mask;
            return count;
        }

        // First word: scalar with first_mask.
        { uint64_t old = data_[w0];
          count += __builtin_popcountll(first_mask & ~old);
          data_[w0] = old | first_mask; }
        int cur_phase = (phase - PHASE_STEP + K) % K;

        // AVX-512 body: 8 words at a time.
        uint64_t w = w0 + 1;
        while (w + 7 < w1) {
            __m512i mask_vec = _mm512_loadu_si512((const void*)avx_groups[cur_phase].data());
            __m512i old_vec  = _mm512_loadu_si512((const void*)(data_ + w));
            __m512i new_bits = _mm512_andnot_si512(old_vec, mask_vec);  // mask & ~old
            count += avx512_popcount512(new_bits);
            _mm512_storeu_si512((void*)(data_ + w), _mm512_or_si512(old_vec, mask_vec));
            cur_phase = (cur_phase - BLOCK_STEP + K) % K;
            w += 8;
        }

        // Scalar tail: remaining full words before the last.
        while (w < w1) {
            uint64_t old = data_[w];
            count += __builtin_popcountll(ms[cur_phase] & ~old);
            data_[w] = old | ms[cur_phase];
            cur_phase = (cur_phase - PHASE_STEP + K) % K;
            ++w;
        }

        // Last word: scalar with last_mask.
        { uint64_t old = data_[w1];
          count += __builtin_popcountll(last_mask & ~old);
          data_[w1] = old | last_mask; }

        return count;
    }

#endif // __AVX512F__

    // Test and set atomically; returns true if the bit was newly set (was clear).
    bool test_and_set(uint64_t idx) {
        uint64_t& word = data_[idx >> 6];
        uint64_t  mask = uint64_t(1) << (idx & 63);
        bool newly_set = !(word & mask);
        word |= mask;
        return newly_set;
    }

    // Count set bits (Hamming weight)
    uint64_t popcount() const {
        uint64_t count = 0;
        for (uint64_t i = 0; i < num_words_; ++i) {
            count += __builtin_popcountll(data_[i]);
        }
        return count;
    }

    uint64_t num_words() const { return num_words_; }

private:
    uint64_t* data_;
    uint64_t  num_words_;
    uint64_t  capacity_words_ = 0; // actual allocated size

    void allocate(uint64_t words) {
        capacity_words_ = words;
        // 64-byte aligned for cache line efficiency
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, words * sizeof(uint64_t)) != 0) {
            std::abort();
        }
        data_ = static_cast<uint64_t*>(ptr);
    }
};

// ---------------------------------------------------------------------------
// WideBitVector: same interface as BitVector but stores one uint64_t per bit.
// 64x larger footprint — used only for cache-stress benchmarking.
// ---------------------------------------------------------------------------
class WideBitVector {
public:
    WideBitVector() = default;
    explicit WideBitVector(uint64_t num_bits) : data_(num_bits, 0) {}

    void clear() { std::fill(data_.begin(), data_.end(), uint64_t(0)); }
    void set(uint64_t idx)        { data_[idx] = 1; }
    bool test(uint64_t idx) const { return data_[idx] != 0; }

private:
    std::vector<uint64_t> data_;
};
