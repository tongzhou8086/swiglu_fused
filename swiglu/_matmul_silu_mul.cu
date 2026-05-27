// matmul_silu_mul: fused SwiGLU stage built on b42_gsm.
//
// Computes  h = silu(A @ B_gate) * up   where:
//   A      = x        [M, K]   (bf16, row-major)   -- tokens × d_model
//   B_gate = W_gate   [K, N]   (bf16, row-major)   -- d_model × d_ff
//   up     = x@W_up   [M, N]   (bf16, row-major)   -- precomputed elsewhere
//   h                 [M, N]   (bf16, row-major)   -- output
//
// The GEMM mainloop is unchanged from b42_gsm (gate accumulated in TMEM,
// fp32).  The fusion lives in epilogue Phase 1: where b42 down-casts the
// fp32 accumulator to bf16, this kernel instead computes
//   silu(gate) * up  =  (gate * sigmoid(gate)) * up
// in fp32, loading the matching `up` tile straight from GMEM, then writes
// the bf16 result.  `gate` is never materialised to HBM.
//
// --- inherited b42_gsm header ---
// b42_gsm: b41_w8 with GROUP_SIZE_M lifted to a template parameter so the
// Python autotuner can sweep it (1/4/8/16) instead of the b41 hardcoded 8.
// Everything else is identical to b41_w8.
//
// --- original b41_w8 header ---
// b41_w8: b35_sw bumped to 8 warps (256 threads / block).
//
// Same main-loop structure (warp 0 = TMA, warp 1 = MMA, rest idle in
// main loop).  The extra warps participate only in the epilogue:
//   - Phase 1 (TMEM → SMEM): only warps 0..3 (4 warps × 32 lanes =
//     128 rows, matching BM).  Warps 4..7 wait at __syncthreads.
//   - Phase 2 (SMEM → GMEM): all 8 warps participate, halving each
//     thread's GMEM-store count from 32 to 16.
//
// Hypothesis: phase 2 is GMEM-store-throughput bound; more parallel
// stores → faster epilogue → less time stalling between tiles.  The
// main loop is unchanged.
//
// Register cost: 8 warps × ~100 regs/thread ≈ 25K regs/block.  B200
// has 64K-128K regs/SM, so this fits with room to spare for occupancy.
//
// ─────────────────────────────────────────────────────────────────────────
//
// b35_sw: b34_cl2_kn + Triton-style chunked CTA swizzle for L2 reuse.
//
// Only difference vs b34: the bid → (bid_m, bid_n) mapping.  Instead
// of walking N-major across one M-stripe (b34's pattern), this kernel
// walks `GROUP_SIZE_M` cluster-rows in M, then iterates N, then jumps
// to the next M-chunk.  The swizzle is applied at the *cluster-tile*
// granularity so the cluster pairing (bid_m, bid_m+1 in same bid_n)
// stays intact and cta_group::2 MMA still works.
//
// Expected to help at large grids (7168+) where the previous N-major
// walk lost L2 reuse on A.  At small grids, ~neutral.
//
// Inherited from b34_cl2_kn: K-major B, no transpose, same API as b20.
//
// ── Walk pattern (GROUP_SIZE_M=8, CTA_GROUP=2) ─────────────────────────────
//
// A "cluster tile" = CTA_GROUP × BM rows × BN cols = 256 × 256 of output.
// One group block = GROUP_SIZE_M cluster-tiles in M × grid_n in N.  So at
// GROUP_SIZE_M=8: one group covers 16 CTA-tiles (= 2048 M-rows) × grid_n
// CTA-tiles in N.
//
// Within a group, the walk is M-major: visit cluster_m = 0..7 for
// cluster_n=0, then cluster_m = 0..7 for cluster_n=1, etc.  Trace for
// the first group with grid_n=16:
//
//   cluster_id  cluster_m  cluster_n   bid_m (per CTA)
//        0          0          0       0, 1
//        1          1          0       2, 3
//        ...
//        7          7          0       14, 15   ← 16 CTA-rows for n=0 done
//        8          0          1       0, 1     ← back to top, n=1
//        ...
//       15          7          1       14, 15   ← 16 CTA-rows for n=1
//        ...
//      127          7         15       14, 15   ← group 0 done
//      128          8          0       16, 17   ← group 1, next 8 cluster_m
//        ...
//
// L2 reuse logic: within one group, the same 16 M-stripes of A are
// reused for every N-column.  A working set per group = 16 × BM × K × 2
// bytes = 32 MB at K=8192 (BF16), well inside B200's 132 MB L2.  B is
// streamed contiguously per N-column; no L2 reuse expected there.
//
// Larger GROUP_SIZE_M trades more A reuse for a bigger working set:
//   GSM=1   →   2 M-stripes × K   →   ~4 MB A working set, no reuse
//   GSM=4   →   8 M-stripes × K   →  ~16 MB A working set, partial reuse
//   GSM=8   →  16 M-stripes × K   →  ~32 MB A working set, full reuse
//   GSM=16  →  32 M-stripes × K   →  ~64 MB A working set (still fits L2)
//
// Below ~GSM=4 the walk degenerates into ~N-major (no A reuse), above
// ~GSM=16 the working set starts spilling L2 (at K=8192).  The sweet
// spot is shape-dependent; GSM=8 was Triton's autotuner pick for our
// kernel's parameters at most shapes.
//
// The N-axis split is unchanged — each CTA still owns BN/2 contiguous
// N columns.  In K-major, that's a (BK, BN/2) tile per CTA per stage,
// loaded as (BN/2)/64 sub-tiles of (BK, 64) instead of (BK/64) sub-tiles
// of (BN/2, 64).  The MMA B descriptor switches to `make_desc_K_major`
// with LBO = BK*128 (stride between N-sub-tiles in SMEM), and idesc
// has bit 16 set (B is K-major).
//
// Concrete changes vs b20:
//
//   1.  TMA path
//        - B's TMA descriptor: built on (N, K) global with box
//          (BN/CTA_GROUP, BK).  Each CTA loads only its half (cols
//          [cta_rank*BN/2, cta_rank*BN/2 + BN/2)).
//        - SMEM B layout: [NS][BK/64][BN_LOCAL][64] (back to b20's
//          pre-K-major shape, just BN_LOCAL = BN/2 instead of BN).
//        - tile_ready_mbar init count = CTA_GROUP (both CTAs arrive
//          per tile).  Both CTAs' expect_tx routes to CTA 0's mbar
//          via `& 0xFEFFFFFF` (cluster SMEM convention).
//
//   2.  MMA path
//        - idesc MMA_M = BM * CTA_GROUP = 256.
//        - cta_group::2 MMA; only cta_rank==0 issues; tensor cores
//          fan operand reads across the cluster.
//        - Commit uses `multicast::cluster` so both CTAs' mma_done
//          mbars fire from one issue.
//        - TMEM alloc is cta_group::2 (same 256-col footprint per
//          CTA, addressed as one 256-row cluster-wide tile).
//
//   3.  Init / sync
//        - Post-init: `barrier.cluster.arrive/wait` instead of
//          __syncthreads, so peer CTAs see each other's mbars.
//
//   4.  Epilogue + grid
//        - TMEM row addressing: `t_row = cta_rank * BM + warp_id*32`.
//          CTA 1 reads logical rows 128..255 from its OWN local
//          TMEM (hardware maps cluster-wide indices).
//        - Per-CTA GMEM write target unchanged — each CTA already
//          has its own off_m via the bid→(bid_m, bid_n) mapping.
//        - bid → (bid_m, bid_n) uses GROUP_M=2 so the cluster's two
//          CTAs sit on adjacent M-rows of the same N-column.

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

// Cross-cluster arrive (mb may be on a peer CTA via 0xFEFFFFFF mask).
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

// ── TMA 2D load ─────────────────────────────────────────────────────────────
//
// `.cta_group::2` is the key cluster-mode modifier: it tells the TMA
// engine to bookkeep tx-count against a cluster-wide mbar (the cross-
// CTA arrival is what makes both CTAs' loads count toward CTA 0's
// tile_ready_mbar).  Without it, peer-CTA loads silently fail to
// advance the mbar and the kernel deadlocks.

__device__ __forceinline__ void tma_2d_load(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}

// ── tcgen05 PTX wrappers (cta_group::2) ─────────────────────────────────────

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
__device__ __forceinline__ void tcgen05_ld_32x32b_x8(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
        : "r"(taddr));
}

// Wider TMEM load: 32 b32 elements/lane in one instruction — 4× fewer
// loads (and 4× fewer tcgen05.wait::ld) than the x8 form for the epilogue.
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

// Even wider: 64 b32 elements/lane — 2 waits/thread for a 128-col half.
__device__ __forceinline__ void tcgen05_ld_32x32b_x64(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
        :
          "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7]),
          "=f"(out[8]), "=f"(out[9]), "=f"(out[10]), "=f"(out[11]),
          "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15]),
          "=f"(out[16]), "=f"(out[17]), "=f"(out[18]), "=f"(out[19]),
          "=f"(out[20]), "=f"(out[21]), "=f"(out[22]), "=f"(out[23]),
          "=f"(out[24]), "=f"(out[25]), "=f"(out[26]), "=f"(out[27]),
          "=f"(out[28]), "=f"(out[29]), "=f"(out[30]), "=f"(out[31]),
          "=f"(out[32]), "=f"(out[33]), "=f"(out[34]), "=f"(out[35]),
          "=f"(out[36]), "=f"(out[37]), "=f"(out[38]), "=f"(out[39]),
          "=f"(out[40]), "=f"(out[41]), "=f"(out[42]), "=f"(out[43]),
          "=f"(out[44]), "=f"(out[45]), "=f"(out[46]), "=f"(out[47]),
          "=f"(out[48]), "=f"(out[49]), "=f"(out[50]), "=f"(out[51]),
          "=f"(out[52]), "=f"(out[53]), "=f"(out[54]), "=f"(out[55]),
          "=f"(out[56]), "=f"(out[57]), "=f"(out[58]), "=f"(out[59]),
          "=f"(out[60]), "=f"(out[61]), "=f"(out[62]), "=f"(out[63])
        : "r"(taddr));
}

// ── Matrix descriptor (MN-major, gau-nernst convention) ─────────────────────
//
// Same format as upstream v5: SBO = 8*128, swizzle = 128B, LBO implicit
// (no bit 16 — B is N-major; tensor cores step along K and N in the
// natural SMEM order).

__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t b = ((SBO) >> 4) & 0x3FFFULL;
    return a | (b << 32) | (1ULL << 46) | (2ULL << 61);
}

// K-major B descriptor (same as b20).  LBO = BK*128 = stride between
// N-sub-tiles in SMEM.
__device__ __forceinline__ uint64_t make_desc_K_major(uint32_t smem_addr,
                                                      int block_k_bytes) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a   = ((uint64_t)smem_addr     >> 4) & 0x3FFFULL;
    uint64_t lbo = ((uint64_t)block_k_bytes >> 4) & 0x3FFFULL;
    uint64_t b   = ((SBO)                   >> 4) & 0x3FFFULL;
    return a | (lbo << 16) | (b << 32) | (1ULL << 46) | (2ULL << 61);
}

// idesc with MMA_M spanning the cluster (BM*CTA_GROUP) and bit 16 set
// (B is K-major in this kernel).
__device__ __forceinline__ uint32_t make_idesc_bf16_cluster(int M, int N) {
    uint32_t d = 0;
    d |= (1u << 4);                                    // c_format = F32
    d |= (1u << 7);                                    // a_format = BF16
    d |= (1u << 10);                                   // b_format = BF16
    d |= (1u << 16);                                   // B is K-major
    d |= (((uint32_t)(N >> 3) & 0x3F) << 17);          // n_dim = N/8
    d |= (((uint32_t)(M >> 4) & 0x1F) << 24);          // m_dim = M/16
    return d;
}


// ── Kernel ──────────────────────────────────────────────────────────────────

template <int BLOCK_N, int BLOCK_K, int NUM_STAGES, int GROUP_SIZE_M>
__device__ __forceinline__ void matmul_silu_mul_impl(
    const CUtensorMap* A_tmap,
    const CUtensorMap* B_tmap,
    __nv_bfloat16* C_ptr,
    const __nv_bfloat16* UP_ptr,
    int M, int N, int K
) {
    constexpr int BLOCK_M       = 128;
    constexpr int BLOCK_N_LOCAL = BLOCK_N / CTA_GROUP;     // 128 at BN=256
    constexpr int MMA_K         = 16;
    constexpr int NUM_WARPS     = 8;       // b41_w8: doubled from b35's 4
    static_assert(BLOCK_K % 64 == 0,            "BK must be a multiple of 64");
    static_assert(NUM_STAGES >= 2,              "NS must be >= 2");
    static_assert(BLOCK_N % CTA_GROUP == 0,     "BN must be divisible by CTA_GROUP");
    static_assert(BLOCK_N_LOCAL % 64 == 0,      "BN_LOCAL must be a multiple of 64");

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int bid     = blockIdx.x;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // ── Triton-style chunked CTA swizzle at cluster-tile granularity ──
    //
    // Treat the grid as a 2D array of "cluster tiles" of size
    // (2*BM, BN).  Apply Triton's GROUP_SIZE_M chunking on the
    // cluster_id, then map back to per-CTA (bid_m, bid_n) with the
    // cluster's two CTAs landing on adjacent M-rows of the same
    // N-column (so cta_group::2 MMA still works).
    //
    // GROUP_SIZE_M is now a template parameter (b42_gsm) — swept by the
    // Python autotuner.  b41_w8 hardcoded this to 8 (Triton's pick).
    constexpr int GROUP_M_INTRA = CTA_GROUP;   // 2 — fixed by cta_group::2

    const int grid_m = M / BLOCK_M;
    const int grid_n = N / BLOCK_N;

    const int cluster_id       = bid / GROUP_M_INTRA;
    const int which_in_cluster = bid % GROUP_M_INTRA;
    const int num_cluster_m    = grid_m / GROUP_M_INTRA;     // M-stride 2*BM

    // Triton's swizzle, applied to cluster_id.
    const int num_cluster_in_group = GROUP_SIZE_M * grid_n;
    const int group_id      = cluster_id / num_cluster_in_group;
    const int first_clust_m = group_id * GROUP_SIZE_M;
    // Last group may be ragged if num_cluster_m % GROUP_SIZE_M != 0;
    // `gsm` shrinks for that case so the (cluster_id % gsm) wrap stays
    // within the actual M-range.
    const int gsm        = min(num_cluster_m - first_clust_m, GROUP_SIZE_M);
    const int cluster_m  = first_clust_m + (cluster_id % gsm);
    const int cluster_n  = (cluster_id % num_cluster_in_group) / gsm;

    const int bid_m = cluster_m * GROUP_M_INTRA + which_in_cluster;
    const int bid_n = cluster_n;
    const int off_m  = bid_m * BLOCK_M;
    const int off_n  = bid_n * BLOCK_N;
    // Per-CTA N base for B loads: each CTA covers a contiguous slab of
    // BN_LOCAL columns starting here.  Loop-invariant — hoisted out of
    // LOAD_TILE so it isn't recomputed per K-iter.
    const int off_n_local = off_n + cta_rank * BLOCK_N_LOCAL;

    // ── SMEM layout (per CTA) ───────────────────────────────────────────────
    //   A[NS][BK/64][BM][64]
    //   B[NS][BN_LOCAL/64][BK][64]                  ← K-major B, half-width

    constexpr int A_SLOT_BYTES = BLOCK_M       * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr int B_SLOT_BYTES = BLOCK_N_LOCAL * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr int CP_BYTES_PER_TILE_PER_CTA = A_SLOT_BYTES + B_SLOT_BYTES;

    extern __shared__ __align__(1024) char smem[];
    const uint32_t SMEM_BASE = (uint32_t)__cvta_generic_to_shared(smem);
    #define A_SMEM_SLOT(s_) (SMEM_BASE + (s_) * A_SLOT_BYTES)
    #define B_SMEM_SLOT(s_) (SMEM_BASE + NUM_STAGES * A_SLOT_BYTES + (s_) * B_SLOT_BYTES)

    __shared__ uint64_t tile_ready_mbar[NUM_STAGES];
    __shared__ uint64_t mma_done_mbar[NUM_STAGES];
    __shared__ uint64_t all_mmas_done;
    __shared__ uint32_t tmem_addr_holder[1];

    // ── One-time setup ──────────────────────────────────────────────────────
    if (warp_id == 0 && elect_sync()) {
        #pragma unroll
        for (int s = 0; s < NUM_STAGES; s++) {
            // tile_ready count = CTA_GROUP: both CTAs' TMA arrivals required.
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&tile_ready_mbar[s]),
                          CTA_GROUP);
            // mma_done count = 1: one multicast commit per stage from CTA 0.
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&mma_done_mbar[s]), 1);
        }
        mbarrier_init((uint32_t)__cvta_generic_to_shared(&all_mmas_done), 1);
        // Pre-arrive so iter 0's wait on mma_done[NS-1] returns immediately.
        // (Both CTAs execute this on their LOCAL mma_done[NS-1].)
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                     :: "r"((uint32_t)__cvta_generic_to_shared(&mma_done_mbar[NUM_STAGES - 1]))
                     : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        // cta_group::2 alloc — issued cooperatively by both CTAs' warp 1.
        tcgen05_alloc_g2((uint32_t)__cvta_generic_to_shared(tmem_addr_holder),
                         BLOCK_N);
    }

    // Cluster barrier — peers must see each other's mbar init.
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    const uint32_t taddr = tmem_addr_holder[0];

    // idesc: MMA covers (BM*CTA_GROUP, BN) per issue.
    const uint32_t idesc = make_idesc_bf16_cluster(BLOCK_M * CTA_GROUP, BLOCK_N);

    // ── LOAD_TILE ───────────────────────────────────────────────────────────
    //
    // B's TMA coord (K-major: (K, N) global):
    //   x = off_n + cta_rank * BN_LOCAL + n_sub*64    (N position)
    //   y = off_k                                     (K position)
    // Each CTA issues (BN_LOCAL/64) sub-tiles of (BK, 64) — one per
    // N-sub-tile.  The full K range of this stage's tile arrives in
    // one bulk per sub-tile.

#define LOAD_TILE(slot_, k0_)                                                          \
    do {                                                                                \
        const int _slot = (slot_);                                                      \
        const uint32_t _mb_local =                                                      \
            (uint32_t)__cvta_generic_to_shared(&tile_ready_mbar[_slot]);                \
        const uint32_t _mb_cta0  = _mb_local & 0xFEFFFFFFu;                             \
        /* A: BK/64 sub-tiles of (BM, 64). */                                            \
        _Pragma("unroll")                                                               \
        for (int _k = 0; _k < BLOCK_K / 64; _k++) {                                     \
            const int _off_k = (k0_) + _k * 64;                                         \
            tma_2d_load(A_SMEM_SLOT(_slot) + _k * BLOCK_M * 128,                        \
                        A_tmap, _off_k, off_m, _mb_cta0);                               \
        }                                                                               \
        /* B (K-major): BN_LOCAL/64 sub-tiles of (BK, 64), per-CTA N slab. */            \
        _Pragma("unroll")                                                               \
        for (int _n = 0; _n < BLOCK_N_LOCAL / 64; _n++) {                               \
            tma_2d_load(B_SMEM_SLOT(_slot) + _n * BLOCK_K * 128,                        \
                        B_tmap,                                                         \
                        off_n_local + _n * 64,                                          \
                        (k0_),                                                          \
                        _mb_cta0);                                                      \
        }                                                                               \
        mbarrier_arrive_expect_tx(_mb_cta0, CP_BYTES_PER_TILE_PER_CTA);                  \
    } while (0)

    // ── Warp-spec main loop ─────────────────────────────────────────────────

    const int num_tiles = K / BLOCK_K;

    if (warp_id == 0 && elect_sync()) {
        // TMA warp — runs on BOTH CTAs.
        uint32_t mma_done_phase[NUM_STAGES] = {};

        #pragma unroll
        for (int s = 0; s < NUM_STAGES - 1; s++) {
            LOAD_TILE(s, s * BLOCK_K);
        }
        for (int k = 0; k < num_tiles - (NUM_STAGES - 1); k++) {
            const int slot = (k + NUM_STAGES - 1) % NUM_STAGES;
            const uint32_t mb =
                (uint32_t)__cvta_generic_to_shared(&mma_done_mbar[slot]);
            mbarrier_wait(mb, mma_done_phase[slot]);
            
            LOAD_TILE(slot, (k + NUM_STAGES - 1) * BLOCK_K);
            mma_done_phase[slot] ^= 1;
        }
    } else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
        // MMA warp — only CTA 0 issues; cta_group::2 result lands in
        // both CTAs' TMEM.
        uint32_t tile_ready_phase[NUM_STAGES] = {};
        constexpr int K_MMAS = BLOCK_K / MMA_K;
        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;

        for (int k = 0; k < num_tiles; k++) {
            const int slot = k % NUM_STAGES;
            const uint32_t mb =
                (uint32_t)__cvta_generic_to_shared(&tile_ready_mbar[slot]);

            // Hoist descriptors above WAIT (same as b20).  A is MN-major;
            // B is K-major in SMEM as [BN_LOCAL/64][BK][64].  The MMA
            // descriptor's LBO field tells the tensor cores the stride
            // between N-sub-tiles, so the K-step is encoded by adding
            // (16-row * 128 B) to the start address.
            constexpr int B_SUB_TILE_BYTES = BLOCK_K * 128;
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

            #pragma unroll
            for (int _kk = 0; _kk < K_MMAS; _kk++) {
                const bool _accumulate = (k != 0) || (_kk > 0);
                tcgen05_mma_g2(taddr, a_desc[_kk], b_desc[_kk], idesc, _accumulate);
            }
            tcgen05_commit_mcast_g2(
                (uint32_t)__cvta_generic_to_shared(&mma_done_mbar[slot]),
                cta_mask);

            tile_ready_phase[slot] ^= 1;
        }

        // Final deferred-arrive that fires both CTAs' all_mmas_done.
        tcgen05_commit_mcast_g2(
            (uint32_t)__cvta_generic_to_shared(&all_mmas_done),
            cta_mask);
    }

#undef LOAD_TILE

    // All warps on both CTAs wait for the cluster's main loop to drain.
    mbarrier_wait((uint32_t)__cvta_generic_to_shared(&all_mmas_done), 0);

    // ── Epilogue: TMEM → SMEM → coalesced GMEM ──
    //
    // TMEM row addressing is cluster-wide: CTA 1 reads its data via
    // logical rows 128..255 even though the bits are in its OWN local
    // TMEM.

    constexpr int BN_PAD = BLOCK_N + 8;
    auto C_sh = reinterpret_cast<__nv_bfloat16 (*)[BN_PAD]>(smem);

    tcgen05_fence_after_thread_sync();

    const int lane = tid % WARP_SIZE;

    // Phase 1 work split: TMEM is BM=128 rows per CTA, covered by 4
    // row-warps × 32 lanes.  With 8 warps total, split the BN columns
    // in half — row-warps (warp_id % 4) handle rows 0..127, col-warps
    // (warp_id / 4) handle cols 0..BN/2-1 or BN/2..BN-1.  Each warp now
    // does BN/16 iters instead of BN/8.
    const int row_warp = warp_id & 3;       // 0..3 — TMEM row group
    const int col_warp = warp_id >> 2;      // 0..1 — N-half
    const int my_row   = row_warp * 32 + lane;   // 0..BM-1
    const int col_base = col_warp * (BLOCK_N / 2);
    const int col_end  = col_base + (BLOCK_N / 2);

    const uint32_t taddr_row_base =
        taddr + (((uint32_t)(cta_rank * BLOCK_M + row_warp * 32)) << 16);

    // Phase 1: TMEM → (silu·mul fusion) → SMEM.
    //
    // `tmp[8]` holds the fp32 `gate` accumulator for (my_row, n..n+7).
    // The matching `up` values are 8 contiguous bf16 in row (off_m+my_row)
    // at column (off_n+n) — one aligned int4 load.  Exact tiling
    // (M%BM==0, N%BN==0) guarantees these are in-bounds, so no guard.
    const int up_row = off_m + my_row;
    const __nv_bfloat16* up_row_ptr = &UP_ptr[(size_t)up_row * N + off_n];
    #pragma unroll
    for (int n = col_base; n < col_end; n += 32) {
        // One wide TMEM load (32 cols) instead of four x8 loads — 4× fewer
        // tcgen05.wait::ld stalls.  (x64 was tried: flat — register pressure
        // cancels the fewer-waits gain, so x32 is the sweet spot.)  Issue the
        // `up` loads first so their GMEM latency overlaps the TMEM read.
        int4 up_raw[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            up_raw[j] = *reinterpret_cast<const int4*>(&up_row_ptr[n + j * 8]);
        }

        float tmp[32];
        const uint32_t addr = taddr_row_base + (uint32_t)n;
        tcgen05_ld_32x32b_x32(addr, tmp);
        tcgen05_wait_ld();

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const __nv_bfloat16* up_h =
                reinterpret_cast<const __nv_bfloat16*>(&up_raw[j]);
            __nv_bfloat162 packed[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                // silu(g) = g/(1+exp(-g)) with fast reciprocal divide
                // (__fdividef, ~2.5 ulp fp32 — below bf16 rounding).
                const float g0 = tmp[j * 8 + 2 * i];
                const float g1 = tmp[j * 8 + 2 * i + 1];
                const float s0 = __fdividef(g0, 1.0f + __expf(-g0));
                const float s1 = __fdividef(g1, 1.0f + __expf(-g1));
                const float u0 = __bfloat162float(up_h[2 * i]);
                const float u1 = __bfloat162float(up_h[2 * i + 1]);
                packed[i] = __floats2bfloat162_rn(s0 * u0, s1 * u1);
            }
            *reinterpret_cast<int4*>(&C_sh[my_row][n + j * 8]) =
                *reinterpret_cast<int4*>(packed);
        }
    }

    __syncthreads();
    if (warp_id == 0) {
        tcgen05_dealloc_g2(taddr, BLOCK_N);
    }

    // Phase 2: SMEM → GMEM, coalesced.
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
        } else if (gr < M) {
            #pragma unroll
            for (int i = 0; i < 8 && gc + i < N; i++) {
                C_ptr[gr * N + gc + i] = C_sh[row][col + i];
            }
        }
    }
}


// ── Launchers ───────────────────────────────────────────────────────────────

#define MAKE_LAUNCHER(BN_, BK_, NS_, GSM_)                                            \
extern "C" __global__ __cluster_dims__(CTA_GROUP, 1, 1)                                \
__launch_bounds__(256, LB_MIN_BLOCKS)                                                  \
void matmul_silu_mul_bm128_bn##BN_##_bk##BK_##_ns##NS_##_gsm##GSM_(                   \
    const __grid_constant__ CUtensorMap A_tmap,                                        \
    const __grid_constant__ CUtensorMap B_tmap,                                        \
    __nv_bfloat16* C_ptr, const __nv_bfloat16* UP_ptr, int M, int N, int K)            \
{                                                                                       \
    matmul_silu_mul_impl<BN_, BK_, NS_, GSM_>(                                         \
        &A_tmap, &B_tmap, C_ptr, UP_ptr, M, N, K);                                     \
}

// BN must be divisible by CTA_GROUP=2.  Sweep GROUP_SIZE_M ∈ {1,4,8,16}
// across the b41_w8 (BN,BK,NS) configs.
#define MAKE_GSM_SET(BN_, BK_, NS_)  \
    MAKE_LAUNCHER(BN_, BK_, NS_, 1)  \
    MAKE_LAUNCHER(BN_, BK_, NS_, 4)  \
    MAKE_LAUNCHER(BN_, BK_, NS_, 8)  \
    MAKE_LAUNCHER(BN_, BK_, NS_, 16)

MAKE_GSM_SET(256, 64, 7)
MAKE_GSM_SET(256, 64, 6)
MAKE_GSM_SET(256, 64, 5)
MAKE_GSM_SET(256, 64, 4)
MAKE_GSM_SET(256, 128, 3)

#undef MAKE_GSM_SET
#undef MAKE_LAUNCHER
