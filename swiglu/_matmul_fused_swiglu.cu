// matmul_fused_swiglu: single-kernel SwiGLU stage, register-held silu(gate).
//
// Computes  C = silu(gate) * up   where  [up | gate] = A @ W
//   A      [M, K]   bf16, row-major   (tokens × d_model)
//   W      [K, 2N]  bf16, row-major   — left half = W_up, right = W_gate
//                                       up   = A @ W[:, :N]
//                                       gate = A @ W[:, N:]
//   C      [M, N]   bf16, row-major
//
// ── Pipeline ────────────────────────────────────────────────────────────────
//
// The kernel reuses b42's main loop verbatim (BM=128 BN=256 BK=64,
// NS=7, warp 0 = TMA, warp 1 = MMA, 2-CTA MMA via cta_group::2) — just
// run TWICE:
//
//   Pass 0 (gate): TMA streams (A, W[:, N:2N]) tiles; MMA accumulates
//                  into taddr_g (TMEM cols [0, BN)).
//   Pass 1 (up):   TMA streams (A, W[:, 0:N])   tiles; MMA accumulates
//                  into taddr_x (TMEM cols [BN, 2BN)).
//
// The two passes are stitched into one unified K-loop in the TMA and
// MMA warps — total length 2 * (K/BK).  The MMA warp pre-arrives one
// extra mbar (`gate_done`) at the moment the LAST gate MMA commits,
// BEFORE the first up MMA queues.  That mbar is the handoff to the
// epilogue's phase A.
//
// ── Phase A: gate consumer, RUNS IN PARALLEL with pass 1 K-loop ─────────────
//
// Warps 4..7 (on both CTAs) are otherwise idle during the K-loop in
// b42.  Here, they:
//
//   1. mbarrier_wait(gate_done)
//   2. tcgen05.ld  (taddr_g) — read fp32 gate accumulator into regs
//   3. silu(g) in fp32 with __fdividef(g, 1 + __expf(-g))
//   4. cast to bf16 and HOLD in registers across pass 1 K-loop
//
// Critically, the stash uses NO SMEM and NO TMEM — the bf16 silu(gate)
// values live in the silu warps' register file.  That keeps SMEM free
// for the pass-1 ring buffer and TMEM free for the up accumulator.
//
// Register cost (per silu warp lane, BN=256):
//   each warp covers 32 rows × BN cols → 32 lanes × 256 cols
//   → 256 bf16 = 128 b32 registers per lane just for the silu hold,
//   plus ~50 for compute = ~180 regs/thread.  Fits under maxregs=255.
//
// ── Phase B: up consumer + multiply + write ─────────────────────────────────
//
//   B-1.  Warps 4..7 wait all_done, read taddr_x in the same shape as
//         their phase-A read, multiply the up fp32 by the held bf16
//         silu(gate) (cast back to fp32), down-cast to bf16, and
//         write to SMEM staging.
//   B-2.  All 8 warps __syncthreads, then do a coalesced SMEM → GMEM
//         store.  Warps 0..3 stay idle through phase A and B-1; they
//         contribute here, where the work is GMEM-bandwidth limited.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>

#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 1
#endif

constexpr int WARP_SIZE = 32;
constexpr int CTA_GROUP = 2;

// ── mbarrier helpers ────────────────────────────────────────────────────────

__device__ __forceinline__ void mbarrier_init(uint32_t mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(count));
}

__device__ __forceinline__ void mbarrier_wait(uint32_t mb, uint32_t phase) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t"
        "bra WAIT_%=;\n\t"
        "DONE_%=:\n\t"
        "}"
        :: "r"(mb), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t mb, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(bytes) : "memory");
}

__device__ __forceinline__ bool elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t .reg .pred px;\n\t"
        "elect.sync _|px, %1;\n\t"
        "@px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred) : "r"(0xFFFFFFFF));
    return pred;
}

// ── TMA 2D load (cluster mode) ──────────────────────────────────────────────

__device__ __forceinline__ void tma_2d_load(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}

// ── tcgen05 wrappers (cta_group::2) ─────────────────────────────────────────

__device__ __forceinline__ void tcgen05_alloc_g2(uint32_t smem_dst, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_dst), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_dealloc_g2(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_mma_g2(
    uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, bool enable_d
) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "setp.ne.b32 P, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, P;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
           "r"((uint32_t)enable_d) : "memory");
}
__device__ __forceinline__ void tcgen05_commit_mcast_g2(uint32_t smem_bar, int16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
        "[%0], %1;"
        :: "r"(smem_bar), "h"(cta_mask) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}
__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// 32 b32 elements / lane in one instruction.
__device__ __forceinline__ void tcgen05_ld_32x32b_x32(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=f"(out[0]),  "=f"(out[1]),  "=f"(out[2]),  "=f"(out[3]),
          "=f"(out[4]),  "=f"(out[5]),  "=f"(out[6]),  "=f"(out[7]),
          "=f"(out[8]),  "=f"(out[9]),  "=f"(out[10]), "=f"(out[11]),
          "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15]),
          "=f"(out[16]), "=f"(out[17]), "=f"(out[18]), "=f"(out[19]),
          "=f"(out[20]), "=f"(out[21]), "=f"(out[22]), "=f"(out[23]),
          "=f"(out[24]), "=f"(out[25]), "=f"(out[26]), "=f"(out[27]),
          "=f"(out[28]), "=f"(out[29]), "=f"(out[30]), "=f"(out[31])
        : "r"(taddr));
}

// ── MMA descriptors (same as b42) ──────────────────────────────────────────

__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t b = ((SBO) >> 4) & 0x3FFFULL;
    return a | (b << 32) | (1ULL << 46) | (2ULL << 61);
}
__device__ __forceinline__ uint64_t make_desc_K_major(uint32_t smem_addr,
                                                      int block_k_bytes) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a   = ((uint64_t)smem_addr     >> 4) & 0x3FFFULL;
    uint64_t lbo = ((uint64_t)block_k_bytes >> 4) & 0x3FFFULL;
    uint64_t b   = ((SBO)                   >> 4) & 0x3FFFULL;
    return a | (lbo << 16) | (b << 32) | (1ULL << 46) | (2ULL << 61);
}
__device__ __forceinline__ uint32_t make_idesc_bf16_cluster(int M, int N) {
    uint32_t d = 0;
    d |= (1u << 4);                                   // c_format = F32
    d |= (1u << 7);                                   // a_format = BF16
    d |= (1u << 10);                                  // b_format = BF16
    d |= (1u << 16);                                  // B is K-major
    d |= (((uint32_t)(N >> 3) & 0x3F) << 17);         // n_dim = N/8
    d |= (((uint32_t)(M >> 4) & 0x1F) << 24);         // m_dim = M/16
    return d;
}


// ── Kernel ──────────────────────────────────────────────────────────────────

template <int BLOCK_N, int BLOCK_K, int NUM_STAGES, int GROUP_SIZE_M>
__device__ __forceinline__ void matmul_fused_swiglu_impl(
    const CUtensorMap* A_tmap,
    const CUtensorMap* W_tmap,    // [K, 2N]
    __nv_bfloat16* C_ptr,
    int M, int N, int K           // N = output width (= half of W's N-dim)
) {
    constexpr int BLOCK_M       = 128;
    constexpr int BLOCK_N_LOCAL = BLOCK_N / CTA_GROUP;
    constexpr int MMA_K         = 16;
    constexpr int NUM_WARPS     = 8;
    static_assert(BLOCK_K % 64 == 0,            "BK must be a multiple of 64");
    static_assert(NUM_STAGES >= 2,              "NS must be >= 2");
    static_assert(BLOCK_N % CTA_GROUP == 0,     "BN must be divisible by CTA_GROUP");
    static_assert(BLOCK_N_LOCAL % 64 == 0,      "BN_LOCAL must be a multiple of 64");

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int bid     = blockIdx.x;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // ── CTA swizzle (Triton-style, cluster-tile granularity) — same as b42 ──
    constexpr int GROUP_M_INTRA = CTA_GROUP;
    const int grid_m = M / BLOCK_M;
    const int grid_n = N / BLOCK_N;
    const int cluster_id       = bid / GROUP_M_INTRA;
    const int which_in_cluster = bid % GROUP_M_INTRA;
    const int num_cluster_m    = grid_m / GROUP_M_INTRA;

    const int num_cluster_in_group = GROUP_SIZE_M * grid_n;
    const int group_id      = cluster_id / num_cluster_in_group;
    const int first_clust_m = group_id * GROUP_SIZE_M;
    const int gsm        = min(num_cluster_m - first_clust_m, GROUP_SIZE_M);
    const int cluster_m  = first_clust_m + (cluster_id % gsm);
    const int cluster_n  = (cluster_id % num_cluster_in_group) / gsm;

    const int bid_m = cluster_m * GROUP_M_INTRA + which_in_cluster;
    const int bid_n = cluster_n;
    const int off_m = bid_m * BLOCK_M;
    const int off_n = bid_n * BLOCK_N;
    const int off_n_local = off_n + cta_rank * BLOCK_N_LOCAL;

    // ── SMEM layout (per CTA) ───────────────────────────────────────────────
    //   ring[NS]: A_slot[BM*BK*2] + B_slot[BN_LOCAL*BK*2]
    //   stagingC: [BM][BN_PAD] bf16, aliases the start of SMEM after the K-loop
    //
    // No silu(gate) SMEM region — the stash lives in registers in the silu warps.

    constexpr int A_SLOT_BYTES = BLOCK_M       * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr int B_SLOT_BYTES = BLOCK_N_LOCAL * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr int CP_BYTES_PER_TILE_PER_CTA = A_SLOT_BYTES + B_SLOT_BYTES;

    constexpr int BN_PAD = BLOCK_N + 8;

    extern __shared__ __align__(1024) char smem[];
    const uint32_t SMEM_BASE = (uint32_t)__cvta_generic_to_shared(smem);
    #define A_SMEM_SLOT(s_) (SMEM_BASE + (s_) * A_SLOT_BYTES)
    #define B_SMEM_SLOT(s_) (SMEM_BASE + NUM_STAGES * A_SLOT_BYTES + (s_) * B_SLOT_BYTES)

    __shared__ uint64_t tile_ready_mbar[NUM_STAGES];
    __shared__ uint64_t mma_done_mbar[NUM_STAGES];
    __shared__ uint64_t gate_done_mbar;
    __shared__ uint64_t all_done_mbar;
    __shared__ uint32_t tmem_addr_holder[1];

    // ── One-time setup ──────────────────────────────────────────────────────
    if (warp_id == 0 && elect_sync()) {
        #pragma unroll
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&tile_ready_mbar[s]),
                          CTA_GROUP);
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&mma_done_mbar[s]), 1);
        }
        mbarrier_init((uint32_t)__cvta_generic_to_shared(&gate_done_mbar), 1);
        mbarrier_init((uint32_t)__cvta_generic_to_shared(&all_done_mbar),  1);
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                     :: "r"((uint32_t)__cvta_generic_to_shared(&mma_done_mbar[NUM_STAGES - 1]))
                     : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        // 2*BN cols of TMEM: gate cols [0,BN), up cols [BN,2BN).
        tcgen05_alloc_g2((uint32_t)__cvta_generic_to_shared(tmem_addr_holder),
                         2 * BLOCK_N);
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    const uint32_t taddr   = tmem_addr_holder[0];
    const uint32_t taddr_g = taddr;
    const uint32_t taddr_x = taddr + BLOCK_N;
    const uint32_t idesc   = make_idesc_bf16_cluster(BLOCK_M * CTA_GROUP, BLOCK_N);

    // ── Unified K-loop driver: gate (W right half) then up (W left half) ────

    constexpr int K_MMAS = BLOCK_K / MMA_K;
    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;

    const int num_k_per_pass = K / BLOCK_K;
    const int num_iters      = 2 * num_k_per_pass;

#define LOAD_TILE(slot_, k0_, n_half_off_)                                                 \
    do {                                                                                    \
        const int _slot = (slot_);                                                          \
        const uint32_t _mb_local =                                                          \
            (uint32_t)__cvta_generic_to_shared(&tile_ready_mbar[_slot]);                    \
        const uint32_t _mb_cta0  = _mb_local & 0xFEFFFFFFu;                                 \
        _Pragma("unroll")                                                                   \
        for (int _k = 0; _k < BLOCK_K / 64; _k++) {                                         \
            const int _off_k = (k0_) + _k * 64;                                             \
            tma_2d_load(A_SMEM_SLOT(_slot) + _k * BLOCK_M * 128,                            \
                        A_tmap, _off_k, off_m, _mb_cta0);                                   \
        }                                                                                   \
        _Pragma("unroll")                                                                   \
        for (int _n = 0; _n < BLOCK_N_LOCAL / 64; _n++) {                                   \
            tma_2d_load(B_SMEM_SLOT(_slot) + _n * BLOCK_K * 128,                            \
                        W_tmap,                                                             \
                        off_n_local + (n_half_off_) + _n * 64,                              \
                        (k0_),                                                              \
                        _mb_cta0);                                                          \
        }                                                                                   \
        mbarrier_arrive_expect_tx(_mb_cta0, CP_BYTES_PER_TILE_PER_CTA);                     \
    } while (0)

    if (warp_id == 0 && elect_sync()) {
        // TMA warp — on both CTAs.
        uint32_t mma_done_phase[NUM_STAGES] = {};

        #pragma unroll
        for (int s = 0; s < NUM_STAGES - 1; s++) {
            // Prologue: gate pass, right half of W → n_half_off = N.
            LOAD_TILE(s, s * BLOCK_K, N);
        }

        for (int k = 0; k < num_iters - (NUM_STAGES - 1); k++) {
            const int slot = (k + NUM_STAGES - 1) % NUM_STAGES;
            const uint32_t mb =
                (uint32_t)__cvta_generic_to_shared(&mma_done_mbar[slot]);
            mbarrier_wait(mb, mma_done_phase[slot]);

            const int load_k_idx = k + NUM_STAGES - 1;
            const bool is_up = load_k_idx >= num_k_per_pass;
            const int  pass_k_idx = is_up ? (load_k_idx - num_k_per_pass) : load_k_idx;
            const int  off_k = pass_k_idx * BLOCK_K;
            const int  n_half_off = is_up ? 0 : N;
            LOAD_TILE(slot, off_k, n_half_off);
            mma_done_phase[slot] ^= 1;
        }
    } else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
        // MMA warp — CTA 0 only; cta_group::2 fans to both CTAs' TMEM.
        uint32_t tile_ready_phase[NUM_STAGES] = {};
        constexpr int B_SUB_TILE_BYTES = BLOCK_K * 128;

        for (int k = 0; k < num_iters; k++) {
            const int slot = k % NUM_STAGES;
            const uint32_t mb =
                (uint32_t)__cvta_generic_to_shared(&tile_ready_mbar[slot]);

            const bool is_up = (k >= num_k_per_pass);
            const int  pass_k_idx = is_up ? (k - num_k_per_pass) : k;
            const uint32_t d_tmem = is_up ? taddr_x : taddr_g;

            uint64_t a_desc[K_MMAS], b_desc[K_MMAS];
            #pragma unroll
            for (int _kk = 0; _kk < K_MMAS; _kk++) {
                const int _k1 = _kk / 4;
                const int _k2 = _kk % 4;
                a_desc[_kk] = make_desc(
                    A_SMEM_SLOT(slot) + _k1 * BLOCK_M * 128 + _k2 * 32);
                b_desc[_kk] = make_desc_K_major(
                    B_SMEM_SLOT(slot) + (_kk * 16) * 128,
                    B_SUB_TILE_BYTES);
            }

            mbarrier_wait(mb, tile_ready_phase[slot]);
            tcgen05_fence_after_thread_sync();

            const bool reset = (pass_k_idx == 0);
            #pragma unroll
            for (int _kk = 0; _kk < K_MMAS; _kk++) {
                const bool _accumulate = !reset || (_kk > 0);
                tcgen05_mma_g2(d_tmem, a_desc[_kk], b_desc[_kk], idesc, _accumulate);
            }
            tcgen05_commit_mcast_g2(
                (uint32_t)__cvta_generic_to_shared(&mma_done_mbar[slot]),
                cta_mask);
            tile_ready_phase[slot] ^= 1;

            // After the LAST gate MMA commit, fire gate_done.  Up MMAs
            // continue immediately after — no wait in this warp.
            if (k == num_k_per_pass - 1) {
                tcgen05_commit_mcast_g2(
                    (uint32_t)__cvta_generic_to_shared(&gate_done_mbar),
                    cta_mask);
            }
        }

        tcgen05_commit_mcast_g2(
            (uint32_t)__cvta_generic_to_shared(&all_done_mbar),
            cta_mask);
    }

#undef LOAD_TILE

    // ── Phase A: silu warps hold silu(gate) in registers ────────────────────
    //
    // Warps 4..7 cover 32 rows × BN cols each (full BN width per warp).
    // Per lane: BN bf16 = BN/2 b32 regs (BN=256 → 128 b32 regs).
    //
    // We layout the register stash as `__nv_bfloat162 sg[BN/16][8]`
    // (BN/2 packed pairs per lane, grouped by 32-col chunks to match
    // the tcgen05.ld x32 chunks used in phase B).

    constexpr int LD_X         = 32;              // cols per tcgen05.ld
    constexpr int N_CHUNKS_PA  = BLOCK_N / LD_X;  // chunks across BN
    constexpr int PACKS_PER_CHUNK = LD_X / 2;     // 16 bf162 pairs

    __nv_bfloat162 sg[N_CHUNKS_PA][PACKS_PER_CHUNK];   // register stash

    const int lane = tid % WARP_SIZE;
    const int is_silu_warp = (warp_id >= 4);
    const int silu_row_warp = warp_id - 4;        // 0..3 when warp_id ∈ [4,7]
    const int silu_my_row   = silu_row_warp * 32 + lane;
    const uint32_t taddr_row_g =
        taddr_g + (((uint32_t)(cta_rank * BLOCK_M + silu_row_warp * 32)) << 16);

    if (is_silu_warp) {
        mbarrier_wait((uint32_t)__cvta_generic_to_shared(&gate_done_mbar), 0);
        tcgen05_fence_after_thread_sync();

        #pragma unroll
        for (int c = 0; c < N_CHUNKS_PA; c++) {
            const int n = c * LD_X;
            float g[LD_X];
            tcgen05_ld_32x32b_x32(taddr_row_g + (uint32_t)n, g);
            tcgen05_wait_ld();

            #pragma unroll
            for (int i = 0; i < PACKS_PER_CHUNK; i++) {
                const float g0 = g[2 * i + 0];
                const float g1 = g[2 * i + 1];
                // silu(g) = g / (1 + exp(-g)) with fast reciprocal.
                const float s0 = __fdividef(g0, 1.0f + __expf(-g0));
                const float s1 = __fdividef(g1, 1.0f + __expf(-g1));
                sg[c][i] = __floats2bfloat162_rn(s0, s1);
            }
        }
    }

    // ── Wait for the entire up K-loop to drain ──────────────────────────────
    mbarrier_wait((uint32_t)__cvta_generic_to_shared(&all_done_mbar), 0);

    // ── Phase B-1: silu warps read taddr_x, multiply by sg regs, stash bf16 in SMEM ──
    //
    // The A/B ring is dead now; reuse its SMEM as the C staging buffer.
    auto C_sh = reinterpret_cast<__nv_bfloat16 (*)[BN_PAD]>(smem);

    tcgen05_fence_after_thread_sync();

    if (is_silu_warp) {
        const uint32_t taddr_row_x =
            taddr_x + (((uint32_t)(cta_rank * BLOCK_M + silu_row_warp * 32)) << 16);

        #pragma unroll
        for (int c = 0; c < N_CHUNKS_PA; c++) {
            const int n = c * LD_X;
            float x[LD_X];
            tcgen05_ld_32x32b_x32(taddr_row_x + (uint32_t)n, x);
            tcgen05_wait_ld();

            __nv_bfloat162 out[PACKS_PER_CHUNK];
            #pragma unroll
            for (int i = 0; i < PACKS_PER_CHUNK; i++) {
                const float u0 = x[2 * i + 0];
                const float u1 = x[2 * i + 1];
                const float s0 = __bfloat162float(__low2bfloat16(sg[c][i]));
                const float s1 = __bfloat162float(__high2bfloat16(sg[c][i]));
                out[i] = __floats2bfloat162_rn(s0 * u0, s1 * u1);
            }
            // Stage to SMEM: 32 cols × 1 row → 4 int4s.
            #pragma unroll
            for (int j = 0; j < LD_X / 8; j++) {
                *reinterpret_cast<int4*>(&C_sh[silu_my_row][n + j * 8]) =
                    *reinterpret_cast<int4*>(&out[j * 4]);
            }
        }
    }

    __syncthreads();
    if (warp_id == 0) {
        tcgen05_dealloc_g2(taddr, 2 * BLOCK_N);
    }

    // ── Phase B-2: SMEM → GMEM, coalesced (all 8 warps participate) ─────────
    constexpr int CHUNK_BF16 = 8;
    constexpr int N_CHUNKS   = BLOCK_N / CHUNK_BF16;
    constexpr int TB_SIZE    = NUM_WARPS * WARP_SIZE;
    constexpr int STORES_PER_THREAD = (BLOCK_M * BLOCK_N) / (TB_SIZE * CHUNK_BF16);
    static_assert(STORES_PER_THREAD * TB_SIZE * CHUNK_BF16 == BLOCK_M * BLOCK_N,
                  "BM*BN must be a multiple of TB_SIZE*8");

    #pragma unroll
    for (int s = 0; s < STORES_PER_THREAD; s++) {
        const int flat = tid + s * TB_SIZE;
        const int row  = flat / N_CHUNKS;
        const int col  = (flat % N_CHUNKS) * CHUNK_BF16;
        const int gr   = off_m + row;
        const int gc   = off_n + col;
        if (gr < M && gc + 7 < N) {
            *reinterpret_cast<int4*>(&C_ptr[gr * N + gc]) =
                *reinterpret_cast<const int4*>(&C_sh[row][col]);
        }
    }
}


// ── Launchers ───────────────────────────────────────────────────────────────

#define MAKE_LAUNCHER(BN_, BK_, NS_, GSM_)                                            \
extern "C" __global__ __cluster_dims__(CTA_GROUP, 1, 1)                                \
__launch_bounds__(256, LB_MIN_BLOCKS)                                                  \
void matmul_fused_swiglu_bm128_bn##BN_##_bk##BK_##_ns##NS_##_gsm##GSM_(               \
    const __grid_constant__ CUtensorMap A_tmap,                                        \
    const __grid_constant__ CUtensorMap W_tmap,                                        \
    __nv_bfloat16* C_ptr, int M, int N, int K)                                         \
{                                                                                       \
    matmul_fused_swiglu_impl<BN_, BK_, NS_, GSM_>(                                     \
        &A_tmap, &W_tmap, C_ptr, M, N, K);                                             \
}

// Standard b42 main-loop config: BN=256 BK=64 NS=7.  TMEM = 2*BN = 512
// cols (full TMEM cap).  Per-lane silu(gate) stash = BN bf16 = 128 b32
// regs ⇒ ~180 regs/thread total; fits maxregs=255.
MAKE_LAUNCHER(256, 64, 7, 8)
MAKE_LAUNCHER(256, 64, 6, 8)
MAKE_LAUNCHER(256, 64, 5, 8)
MAKE_LAUNCHER(256, 64, 4, 8)
MAKE_LAUNCHER(256, 64, 7, 4)
MAKE_LAUNCHER(256, 64, 7, 16)

#undef MAKE_LAUNCHER
