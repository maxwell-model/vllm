# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.
"""Triton block-scoring and top-k kernels for Maxwell-V1-Flash.

Index queries score each 128-token block (one KV page) of the single-head
index-K cache ``(num_blocks, 128, index_head_dim)``, then the top-k blocks plus
the forced init/local blocks are selected per query token. Both paths are
device-only with a shape-derived split-K count, so the grid is fixed inside a
captured CUDA graph.
"""

import functools

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128


# Bitonic top-k helpers.
@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


# score[h, token, block] = causal max over the 128-token block of
# (idx_q . index_k); each K tile is exactly one page. Sliced mixed-batch
# metadata may be unaligned; these values only use scalar loads.
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens,  # [batch+1] query start offsets
    seq_lens,  # [batch] total K length
    prefix_lens,  # [batch] context length before this chunk's queries
    num_idx_heads,
    head_dim: tl.constexpr,
    sm_scale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_idx_heads
    pid_h = pid_bh % num_idx_heads

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0,), padding_option="zero")
    q_start = prefix_len + pid_q * BLOCK_SIZE_Q

    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    # Causal window: only blocks up to the last query token's position.
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        blk = i // BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = i + off_k
        # KV-cache pages are fully allocated; causal masking removes positions
        # beyond the sequence length after the dot product.
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )
        qk = tl.dot(q, k) * sm_scale_log2e
        if q_start < i + BLOCK_SIZE_K:
            qk = tl.where(off_q[:, None] >= pos[None, :], qk, float("-inf"))
        score = tl.max(qk, axis=1)  # [BLOCK_SIZE_Q]
        s_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + (seq_start + pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            * stride_s_n
            + blk * stride_s_k
        )
        q_store_mask = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < q_len
        tl.store(s_ptrs, score, mask=q_store_mask)


# Top-k selection uses one score row per query token. Sliced mixed-batch
# metadata may be unaligned, but these values use scalar loads.
@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=["BLOCK_SIZE_T"],
)
@triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
def _topk_index_kernel(
    s_ptr,  # [num_heads, total_q, max_block]
    ti_ptr,  # [num_heads, total_q, topk]
    sample_interval: tl.constexpr,  # Always one.
    block_size: tl.constexpr,  # sparse block size (128)
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    topk,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    s_ptrs = (
        s_ptr
        + (seq_start + pid_q * sample_interval) * stride_s_n
        + pid_h * stride_s_h
        + off_k * stride_s_k
    )
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        causal_mask = i + off_k < valid_blocks
        local_mask = i + off_k >= max(0, valid_blocks - local_blocks)
        init_mask = i + off_k < init_blocks
        score = tl.load(s_ptrs, mask=causal_mask, other=-1e30).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        if MASK_INIT:
            score = tl.where(causal_mask & init_mask, score - 1e29, score)
        else:
            score = tl.where(causal_mask & init_mask, 1e30, score)
        if MASK_LOCAL:
            score = tl.where(causal_mask & local_mask, score - 1e28, score)
        else:
            score = tl.where(causal_mask & local_mask, 1e29, score)
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (tl.where(causal_mask, i + off_k + 1, 0), topk_idx)
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx = tl.sum(
        topk_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    ti_ptrs = (
        ti_ptr
        + (block_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    store_mask = off_t < topk
    valid_mask = off_t < valid_blocks
    topk_idx = tl.where(store_mask & valid_mask, topk_idx, -1)
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=store_mask)


# Decode index score, split-K over seq blocks. Queries are flattened
# request-major; decode_query_len maps each back to its request. The chunk
# count depends only on shapes, so the grid is fixed within a CUDA graph.
@triton.jit(do_not_specialize=["num_kv_chunks", "decode_query_len"])
def _decode_index_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    init_blocks,
    local_blocks,
    sm_scale,
    decode_query_len,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    num_kv_chunks,
    USE_PDL: tl.constexpr,
    SCORE_SUM: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_c = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # A block-aligned fixed-count split keeps the grid independent of seq_len.
    chunk_size_blocks = (num_blocks + num_kv_chunks - 1) // num_kv_chunks
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)  # positions within a 128-block
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + req_id * stride_bt_b
    # Force-select init (1e30) and local (1e29, higher priority) blocks.
    local_start = tl.maximum(0, num_blocks - local_blocks)
    q = tl.load(
        q_ptr
        + pid_b * stride_q_n
        + tl.arange(0, num_idx_heads) * stride_q_h
        + off_d[:, None] * stride_q_d,
    )  # [D,H]
    for blk in tl.range(chunk_start_block, chunk_end_block):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos < kv_len
        # KV-cache pages are fully allocated; the mask below removes positions
        # beyond the sequence length.
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[:, None] * stride_ik_pos
            + off_d * stride_ik_d,
        )  # [N,D]
        kq = tl.dot(k, q) * sm_scale_log2e  # [N,H]
        if SCORE_SUM:
            # Masked positions add 0; every block here has a visible token.
            kq = tl.where(pos_mask[:, None], kq, 0.0)
            score = tl.sum(kq, axis=0)  # [H]
        else:
            kq = tl.where(pos_mask[:, None], kq, float("-inf"))
            score = tl.max(kq, axis=0)  # [H]
        is_init = blk < init_blocks
        is_local = (blk >= local_start) & (blk < num_blocks)
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(
            score_ptr
            + tl.arange(0, num_idx_heads) * stride_s_h
            + pid_b * stride_s_n
            + blk * stride_s_k,
            score,
        )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# Decode top-k (split-K): per-chunk partial top-k + merge. Forced init/local
# blocks are already encoded in the scores.
@triton.jit(do_not_specialize=["chunk_blocks", "decode_query_len"])
def _topk_index_partial_jit(
    s_ptr,  # score: [num_idx_heads, total_q, max_block]
    ts_partial_ptr,  # partial scores out: [NUM_TOPK_CHUNKS, num_idx_heads, total_q, T]
    ti_partial_ptr,  # partial idx out (1-indexed global, 0=invalid): same shape
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    chunk_blocks,  # how many score-blocks each chunk owns
    decode_query_len,
    init_blocks,
    local_blocks,
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_PDL: tl.constexpr,
    # Raw scores: force the init/local windows here, as
    # _decode_index_score_kernel encodes them.
    FORCE_WINDOWS: tl.constexpr,
    # Split each request's own pages across the chunks, not the batch bound.
    ADAPTIVE: tl.constexpr = False,
):
    tl.static_assert(topk < BLOCK_SIZE_K)
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    if ADAPTIVE:
        num_chunks = tl.num_programs(2)
        chunk_blocks = (num_blocks + num_chunks - 1) // num_chunks
    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, num_blocks)
    chunk_actual = tl.maximum(chunk_end - chunk_start, 0)

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)

    s_ptrs = (
        s_ptr
        + pid_b * stride_s_b
        + pid_h * stride_s_h
        + (chunk_start + off_k) * stride_s_k
    )

    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2

    # Empty chunks skip the loop and store the -1e30 / 0 sentinels.
    for i in tl.range(0, chunk_actual, BLOCK_SIZE_K):
        mask = off_k < chunk_actual - i
        score = tl.load(s_ptrs, mask=mask, other=-1e30).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        if FORCE_WINDOWS:
            blk = chunk_start + i + off_k
            is_init = blk < init_blocks
            is_local = blk >= tl.maximum(0, num_blocks - local_blocks)
            score = tl.where(
                mask, tl.where(is_local, 1e29, tl.where(is_init, 1e30, score)), score
            )
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (
            tl.where(mask, chunk_start + i + off_k + 1, 0),  # 1-indexed global
            topk_idx,
        )
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    topk_mask_extract = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    final_score = tl.sum(
        topk_mask_extract[:, None]
        * tl.reshape(topk_score, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    final_idx = tl.sum(
        topk_mask_extract[:, None]
        * tl.reshape(topk_idx, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )

    # Invalid slots carry -1e30 / 0 sentinels and lose in the merge.
    ts_ptrs = (
        ts_partial_ptr
        + pid_chunk * stride_ts_c
        + pid_b * stride_ts_b
        + pid_h * stride_ts_h
        + off_t * stride_ts_t
    )
    ti_ptrs = (
        ti_partial_ptr
        + pid_chunk * stride_ti_c
        + pid_b * stride_ti_b
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    tl.store(ts_ptrs, final_score)
    tl.store(ti_ptrs, final_idx)


_topk_index_partial_kernel = triton.heuristics(
    {"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])}
)(
    triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_SIZE_K": 256}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
        ],
        key=["topk"],
    )(_topk_index_partial_jit)
)


@triton.jit
def _top_of_pairs(score, ids, lists: tl.constexpr, T: tl.constexpr):
    """Descending top-``T`` of each adjacent pair of descending lists.

    One list against the other reversed, elementwise max, is a bitonic
    sequence holding the pair's top ``T``; one merge network sorts it.
    """
    shape: tl.constexpr = [lists // 2, 2, T]
    second = tl.arange(0, 2)[None, :, None]
    s = tl.reshape(score, shape)
    i = tl.reshape(ids, shape)
    s_left, s_right = tl.sum(s * (1 - second), 1), tl.flip(tl.sum(s * second, 1), 1)
    i_left, i_right = tl.sum(i * (1 - second), 1), tl.flip(tl.sum(i * second, 1), 1)
    take_left = s_left >= s_right
    s = tl.reshape(tl.where(take_left, s_left, s_right), [lists // 2 * T])
    i = tl.reshape(tl.where(take_left, i_left, i_right), [lists // 2 * T])
    n_t: tl.constexpr = tl.standard._log2(T)
    return _bitonic_merge(s, i.to(tl.int32), n_t, True, n_t)


@triton.jit(do_not_specialize=["decode_query_len"])
def _topk_index_tree_merge_kernel(
    ts_partial_ptr,  # partial scores: [NUM_TOPK_CHUNKS, num_idx_heads, total_q, T]
    ti_partial_ptr,  # partial idx (1-indexed global, 0=invalid): same shape
    ti_final_ptr,  # final idx (0-indexed, -1=invalid): [num_idx_heads, total_q, topk]
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,
    topk: tl.constexpr,
    decode_query_len,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_tif_h,
    stride_tif_b,
    stride_tif_t,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """:func:`_topk_index_merge_kernel` via pairwise merges of sorted
    partials."""
    tl.static_assert(NUM_TOPK_CHUNKS <= 16)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    off = tl.arange(0, NUM_TOPK_CHUNKS * BLOCK_SIZE_T)
    chunk_idx = off // BLOCK_SIZE_T
    in_chunk_idx = off % BLOCK_SIZE_T
    score = tl.load(
        ts_partial_ptr
        + chunk_idx * stride_ts_c
        + pid_h * stride_ts_h
        + pid_b * stride_ts_b
        + in_chunk_idx * stride_ts_t
    ).to(tl.float32)
    score = tl.where(score != score, -1e30, score)
    idx = tl.load(
        ti_partial_ptr
        + chunk_idx * stride_ti_c
        + pid_h * stride_ti_h
        + pid_b * stride_ti_b
        + in_chunk_idx * stride_ti_t
    ).to(tl.int32)

    if NUM_TOPK_CHUNKS >= 2:
        score, idx = _top_of_pairs(score, idx, NUM_TOPK_CHUNKS, BLOCK_SIZE_T)
    if NUM_TOPK_CHUNKS >= 4:
        score, idx = _top_of_pairs(score, idx, NUM_TOPK_CHUNKS // 2, BLOCK_SIZE_T)
    if NUM_TOPK_CHUNKS >= 8:
        score, idx = _top_of_pairs(score, idx, NUM_TOPK_CHUNKS // 4, BLOCK_SIZE_T)
    if NUM_TOPK_CHUNKS >= 16:
        score, idx = _top_of_pairs(score, idx, NUM_TOPK_CHUNKS // 8, BLOCK_SIZE_T)

    off_t = tl.arange(0, BLOCK_SIZE_T)
    topk_idx_final = tl.where(off_t < tl.minimum(topk, num_blocks), idx - 1, -1)
    tl.store(
        ti_final_ptr
        + pid_h * stride_tif_h
        + pid_b * stride_tif_b
        + off_t * stride_tif_t,
        topk_idx_final.to(ti_final_ptr.dtype.element_ty),
        mask=off_t < topk,
    )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


#: One warp keeps the adaptive partial's bitonic network within a warp.
_ADAPTIVE_PARTIAL_WARPS = 1


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
        "BLOCK_SIZE_K": lambda args: triton.next_power_of_2(
            args["num_topk_chunks"] * triton.next_power_of_2(args["topk"])
        ),
    }
)
@triton.jit(do_not_specialize=["num_topk_chunks", "decode_query_len"])
def _topk_index_merge_kernel(
    ts_partial_ptr,  # partial scores: [NUM_TOPK_CHUNKS, num_idx_heads, total_q, T]
    ti_partial_ptr,  # partial idx (1-indexed global, 0=invalid): same shape
    ti_final_ptr,  # final idx (0-indexed, -1=invalid): [num_idx_heads, total_q, topk]
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    decode_query_len,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_tif_h,
    stride_tif_b,
    stride_tif_t,
    num_topk_chunks,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_h = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    # num_topk_chunks * BLOCK_SIZE_T candidates, padded to BLOCK_SIZE_K.
    off = tl.arange(0, BLOCK_SIZE_K)
    chunk_idx = off // BLOCK_SIZE_T
    in_chunk_idx = off % BLOCK_SIZE_T
    valid = chunk_idx < num_topk_chunks

    score_offset = (
        chunk_idx * stride_ts_c
        + pid_h * stride_ts_h
        + pid_b * stride_ts_b
        + in_chunk_idx * stride_ts_t
    )
    idx_offset = (
        chunk_idx * stride_ti_c
        + pid_h * stride_ti_h
        + pid_b * stride_ti_b
        + in_chunk_idx * stride_ti_t
    )

    score = tl.load(ts_partial_ptr + score_offset, mask=valid, other=-1e30).to(
        tl.float32
    )
    score = tl.where(score != score, -1e30, score)
    idx = tl.load(ti_partial_ptr + idx_offset, mask=valid, other=0).to(tl.int32)

    n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
    for j in tl.static_range(1, n_dims):
        score, idx = _bitonic_merge(score, idx.to(tl.int32), j, 2, n_dims)
    score, idx = _bitonic_merge(score, idx.to(tl.int32), n_dims, True, n_dims)

    extract_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx_final = tl.sum(
        extract_mask[:, None]
        * tl.reshape(idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )

    off_t = tl.arange(0, BLOCK_SIZE_T)
    tif_ptrs = (
        ti_final_ptr
        + pid_h * stride_tif_h
        + pid_b * stride_tif_b
        + off_t * stride_tif_t
    )
    store_mask = off_t < topk
    topk_idx_final = tl.where(off_t < tl.minimum(topk, num_blocks), topk_idx_final, -1)
    tl.store(
        tif_ptrs, topk_idx_final.to(ti_final_ptr.dtype.element_ty), mask=store_mask
    )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@functools.cache
def _cute_index_score_available() -> bool:
    """Whether decode scores can use the CuteDSL score kernel."""
    if not current_platform.is_device_capability_family(100):
        return False
    try:
        from vllm.models.maxwell_v1_flash.nvidia.ops.index_decode_score import (  # noqa: F401
            IndexDecodeScoreKernel,
        )
    except ImportError as exc:
        logger.warning_once("CuteDSL index score unavailable (%s); using Triton.", exc)
        return False
    return True


def _cute_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    score: torch.Tensor,
) -> bool:
    """Fill ``score`` with raw block maxima of Q.K via the CuteDSL kernel.

    The scores omit the (order-preserving) softmax scale and the init/local
    sentinels; the top-k partial kernel forces the windows instead.

    Returns:
        False, having written nothing, for an unsupported shape.

    """
    total_q, num_heads, head_dim = idx_q.shape
    if not (
        _cute_index_score_available()
        and idx_q.dtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fn)
        and index_kv_cache.dtype == idx_q.dtype
        and idx_q.is_contiguous()
        and index_kv_cache.shape[1] == SPARSE_BLOCK_SIZE
        and head_dim % 128 == 0
        and num_heads * decode_query_len <= 32
        and block_table.dtype == torch.int32
        and seq_lens.dtype == torch.int32
    ):
        return False
    from vllm.cute_utils import _TORCH_TO_CUTE_DTYPE
    from vllm.models.maxwell_v1_flash.nvidia.ops.index_decode_score import (
        IndexDecodeScoreKernel,
    )

    # A shape-only split count keeps the grid fixed inside a captured graph.
    split_k = 256 if seq_lens.shape[0] <= 8 else 128
    kernel = IndexDecodeScoreKernel.compile(
        _TORCH_TO_CUTE_DTYPE[idx_q.dtype],
        num_heads,
        decode_query_len,
        split_k,
        head_dim,
    )
    kernel(idx_q, index_kv_cache, block_table, score, seq_lens)
    return True


@torch.no_grad()
def maxwell_index_score(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    sm_scale: float,
) -> torch.Tensor:
    """Compute per-token index scores for each visible sparse block.

    Returns:
        ``[num_kv_heads, total_q, max_block]`` per-block maxima.

    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, (
        "Maxwell-V1-Flash expects num_idx_heads == num_kv_heads"
    )
    batch = cu_seqlens_q.shape[0] - 1
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)

    # Keep score strides 16-divisible to avoid Triton recompiles.
    score_block_stride = round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )
    BLOCK_SIZE_Q = 64
    grid_score = (triton.cdiv(max_query_len, BLOCK_SIZE_Q), batch * num_idx_heads)
    _index_block_score_kernel[grid_score](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_idx_heads,
        head_dim,
        sm_scale,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
    )
    return score


@torch.no_grad()
def maxwell_index_topk(
    score: torch.Tensor,  # [num_idx_heads, total_q, max_block]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select index top-k from a precomputed score tensor.

    When ``out`` (``[num_idx_heads, >=total_q, topk]``) is given, the result is
    written into ``out[:, :total_q, :]`` for a stable CUDA-graph address.
    """
    num_idx_heads = score.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    total_q = score.shape[1]
    if out is not None:
        topk_idx = out[:, :total_q, :]
    else:
        topk_idx = torch.empty(
            (num_idx_heads, total_q, topk),
            dtype=torch.int32,
            device=score.device,
        )
    # block_size_q == 1 -> query blocks coincide with query tokens.
    grid_topk = (max_query_len, batch, num_idx_heads)
    _topk_index_kernel[grid_topk](
        score,
        topk_idx,
        1,  # sample_interval (block_size_q)
        SPARSE_BLOCK_SIZE,
        cu_seqlens_q,
        cu_seqlens_q,  # cu_seqblocks_q == cu_seqlens_q when block_size_q == 1
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        MASK_INIT=False,
        MASK_LOCAL=False,
    )
    return topk_idx


@torch.no_grad()
def maxwell_index_decode(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    sm_scale: float,
    decode_query_len: int,
    share_route: bool = False,
    score_sum: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode index block-score + top-k, both split-K (cudagraph-safe).

    ``share_route`` reduces the per-head scores to one row before the top-k and
    broadcasts the selection to every KV head; the forced init/local sentinels
    (``1e30``/``1e29``) survive the reduction. ``score_sum`` collapses a
    block's tokens, and the head rows under ``share_route``, by sum instead of
    max. When ``out`` (``[num_kv_heads, >=total_q, topk]``) is given, the
    result goes into ``out[:, :total_q, :]`` for a stable CUDA-graph address.

    Returns:
        ``[num_kv_heads, total_q, topk]`` 0-indexed block ids, -1 padded.

    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, (
        "Maxwell-V1-Flash expects num_idx_heads == num_kv_heads"
    )
    assert total_q == seq_lens.shape[0] * decode_query_len
    batch = total_q
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    use_pdl = current_platform.is_arch_support_pdl()
    # Some Triton backends reject launch_pdl even when false.
    pdl_launch = {"launch_pdl": True} if use_pdl else {}

    # Keep score strides 16-divisible to avoid Triton recompiles.
    score_block_stride = round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )
    # A shape-derived split-K count keeps the grid fixed in a CUDA graph.
    TARGET_GRID = 4096
    MAX_NUM_KV_CHUNKS = 256
    target = max(
        1, min(MAX_NUM_KV_CHUNKS, TARGET_GRID // max(1, batch * num_idx_heads))
    )
    num_kv_chunks = 1 << (target.bit_length() - 1)
    grid_score = (batch, num_kv_chunks)
    raw_scores = not score_sum and _cute_index_score(
        idx_q, index_kv_cache, block_table, seq_lens, decode_query_len, score
    )
    if not raw_scores:
        _decode_index_score_kernel[grid_score](
            idx_q,
            index_kv_cache,
            score,
            block_table,
            seq_lens,
            num_idx_heads,
            head_dim,
            init_blocks,
            local_blocks,
            sm_scale,
            decode_query_len,
            idx_q.stride(0),
            idx_q.stride(1),
            idx_q.stride(2),
            index_kv_cache.stride(0),
            index_kv_cache.stride(1),
            index_kv_cache.stride(2),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            num_kv_chunks=num_kv_chunks,
            USE_PDL=use_pdl,
            SCORE_SUM=score_sum,
            **pdl_launch,
        )

    return maxwell_index_topk_from_scores(
        score,
        seq_lens,
        topk,
        num_kv_heads,
        max_block,
        decode_query_len,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        share_route=share_route,
        score_sum=score_sum,
        force_windows=raw_scores,
        out=out,
    )


@torch.no_grad()
def maxwell_index_topk_from_scores(
    score: torch.Tensor,  # [num_idx_heads, total_q, >= max_block] fp32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    topk: int,
    num_kv_heads: int,
    max_block: int,
    decode_query_len: int,
    *,
    init_blocks: int = 0,
    local_blocks: int = 0,
    share_route: bool = False,
    score_sum: bool = False,
    force_windows: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """The split-K decode top-k of :func:`maxwell_index_decode`, from scores.

    ``score`` carries the init/local sentinels unless ``force_windows`` is
    set. ``max_block`` is the logical page count, not the padded score width.
    """
    num_idx_heads, total_q, _ = score.shape
    batch = total_q
    use_pdl = current_platform.is_arch_support_pdl()
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    if share_route:
        # Contiguous for the kernel strides; shape-constant for a fixed grid.
        score = (
            score.sum(dim=0, keepdim=True)
            if score_sum
            else score.amax(dim=0, keepdim=True)
        )
    sel_heads = 1 if share_route else num_idx_heads

    dest = None if out is None else out[:, :total_q, :]
    if dest is not None and not share_route:
        topk_idx = dest
    else:
        topk_idx = torch.empty(
            (sel_heads, total_q, topk),
            dtype=torch.int32,
            device=score.device,
        )
    # Shape-constant chunk count, capped to bound the merge candidate set.
    TOPK_TARGET_GRID = 4096
    MAX_NUM_TOPK_CHUNKS = 4
    topk_target = max(
        1, min(MAX_NUM_TOPK_CHUNKS, TOPK_TARGET_GRID // max(1, batch * sel_heads))
    )
    num_topk_chunks = 1 << (topk_target.bit_length() - 1)
    # Small batches split each request's own pages and merge pairwise.
    adaptive = batch * sel_heads <= 64
    if adaptive:
        num_topk_chunks = 8
    block_size_t = triton.next_power_of_2(topk)
    chunk_blocks = (max_block + num_topk_chunks - 1) // num_topk_chunks
    topk_score_partial = torch.empty(
        num_topk_chunks,
        sel_heads,
        batch,
        block_size_t,
        dtype=torch.float32,
        device=score.device,
    )
    topk_idx_partial = torch.empty(
        num_topk_chunks,
        sel_heads,
        batch,
        block_size_t,
        dtype=torch.int32,
        device=score.device,
    )
    partial_args = (
        score,
        topk_score_partial,
        topk_idx_partial,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        chunk_blocks,
        decode_query_len,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
    )
    merge_args = (
        topk_score_partial,
        topk_idx_partial,
        topk_idx,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        decode_query_len,
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
    )
    if adaptive:
        # Pinned config: warmup autotuning would freeze a short-input choice.
        _topk_index_partial_jit[(batch, sel_heads, num_topk_chunks)](
            *partial_args,
            BLOCK_SIZE_K=max(64, 2 * block_size_t),
            BLOCK_SIZE_T=block_size_t,
            USE_PDL=use_pdl,
            FORCE_WINDOWS=force_windows,
            ADAPTIVE=True,
            num_warps=_ADAPTIVE_PARTIAL_WARPS,
            num_stages=2,
            **pdl_launch,
        )
        _topk_index_tree_merge_kernel[(batch, sel_heads)](
            *merge_args,
            NUM_TOPK_CHUNKS=num_topk_chunks,
            BLOCK_SIZE_T=block_size_t,
            USE_PDL=use_pdl,
            num_warps=max(1, num_topk_chunks // 2),
            **pdl_launch,
        )
    else:
        _topk_index_partial_kernel[(batch, sel_heads, num_topk_chunks)](
            *partial_args,
            USE_PDL=use_pdl,
            FORCE_WINDOWS=force_windows,
            **pdl_launch,
        )
        _topk_index_merge_kernel[(batch, sel_heads)](
            *merge_args,
            num_topk_chunks=num_topk_chunks,
            USE_PDL=use_pdl,
            **pdl_launch,
        )
    if share_route:
        # The CSR builder needs a materialized, not stride-zero, broadcast.
        wide = topk_idx.expand(num_kv_heads, -1, -1)
        if dest is None:
            return wide.contiguous()
        dest.copy_(wide)
        return dest
    return topk_idx
