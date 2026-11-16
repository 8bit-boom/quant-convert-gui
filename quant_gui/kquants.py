"""K-quant encoders (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) in pure numpy.

Why this exists: `gguf-py` can only DEcode K-quants (its Q2_K..Q6_K classes have no
`quantize_blocks`), and stock `llama-quantize` refuses image/video architectures. So
K-quant GGUFs of diffusion models normally need a patched llama.cpp build. This module
writes the same blocks directly.

It is a port of llama.cpp's reference quantizers in `ggml/src/ggml-quants.c`
(`quantize_row_q{2,3,4,5,6}_K_ref`, `make_qkx2_quants`, `make_qx_quants`,
`make_q3_quants`), the no-imatrix path `llama-quantize` takes, vectorised across all
blocks. Arithmetic is float32 and sums run in element order like the C code so results
match; tests/test_kquants.py compares the bytes with the real ggml encoder when a built
libggml is available (set GGML_LIB), and otherwise checks round-trip error and that
gguf-py decodes the blocks. Nothing here uses an importance matrix.

Layouts (ggml-common.h, QK_K = 256):
    Q2_K 84 B  scales[16] qs[64] d dmin          Q3_K 110 B hmask[32] qs[64] scales[12] d
    Q4_K 144 B d dmin scales[12] qs[128]         Q5_K 176 B d dmin scales[12] qh[32] qs[128]
    Q6_K 210 B ql[128] qh[64] scales[16] d
"""

from __future__ import annotations

import numpy as np

QK_K = 256
BLOCK_BYTES = {"Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "Q5_K": 176, "Q6_K": 210}
SUPPORTED = tuple(BLOCK_BYTES)
_F = np.float32
_EPS = _F(1e-15)
# superblocks per chunk: bounds temporary memory (~ a few hundred MB at float32)
CHUNK_BLOCKS = 1 << 11


def _seqsum(a: np.ndarray) -> np.ndarray:
    """Sum over the last axis in element order (C loop order), float32 - numpy's own
    sum is pairwise and would differ in the last bits."""
    return np.cumsum(a, axis=-1, dtype=_F)[..., -1]


def _pos_max(a: np.ndarray) -> np.ndarray:
    """max over the last axis, floored at +0 (C starts its running max at 0.f, so a -0.0 min never wins)."""
    m = a.max(axis=-1) if a.ndim == 1 else a.max(axis=1)
    return np.where(m > 0, m, _F(0)).astype(_F)


def _rint(a: np.ndarray) -> np.ndarray:
    """ggml's nearest_int (magic-number rounding): round half to even."""
    return np.rint(a)


def _f16(a: np.ndarray) -> np.ndarray:
    """Round to fp16 and back to fp32 (what d/dmin hold in the block)."""
    return np.asarray(a, dtype=_F).astype(np.float16).astype(_F)


def _first_absmax(x: np.ndarray):
    """(amax, signed value at the first max |x|) along the last axis; C uses a strict `>`."""
    idx = np.abs(x).argmax(axis=-1)
    mx = np.take_along_axis(x, idx[..., None], axis=-1)[..., 0]
    return np.abs(mx), mx


# --------------------------------------------------------------------------- scale search helpers

def _tsum(a: np.ndarray) -> np.ndarray:
    """Element-order float32 sum over axis 0 of an (n, N) array: n contiguous vector adds
    (same association as the C loop; much faster than cumsum over short rows)."""
    acc = a[0].copy()
    for i in range(1, a.shape[0]):
        acc += a[i]
    return acc


def _qkx2(x, w, nmax, rmin, rdelta, nstep, use_mad):
    """make_qkx2_quants over rows of x (N, n): returns scale (N,), the_min (N,), L (N, n) uint8.
    Computed on the transposed (n, N) layout so every per-element loop is a contiguous vector op."""
    xt = np.ascontiguousarray(x.T)
    wt = np.ascontiguousarray(w.T)
    n, N = xt.shape
    mn = xt.min(axis=0)
    mx = xt.max(axis=0)
    sum_w = _tsum(wt)
    sum_x = _tsum(wt * xt)
    mn = np.minimum(mn, _F(0))
    flat = mx == mn                                   # `max == min` -> all-zero block
    rng = np.where(flat, _F(1), mx - mn).astype(_F)
    iscale = (_F(nmax) / rng).astype(_F)
    scale = (_F(1) / iscale).astype(_F)
    L = np.clip(_rint(iscale * (xt - mn)), 0, nmax).astype(_F)
    diff = scale * L + mn - xt
    best = _tsum(wt * (np.abs(diff) if use_mad else diff * diff))
    min_cur = mn.copy()
    for s in range(nstep + 1):
        # the C code divides by (max - min) with the CURRENT min, which an improving step updates
        isc = ((_F(rmin) + _F(rdelta) * _F(s) + _F(nmax)) / np.where(flat, _F(1), mx - min_cur)).astype(_F)
        La = np.clip(_rint(isc * (xt - min_cur)), 0, nmax).astype(_F)
        wl = wt * La
        sum_l = _tsum(wl)
        sum_l2 = _tsum(wl * La)
        sum_xl = _tsum(wl * xt)
        D = sum_w * sum_l2 - sum_l * sum_l
        ok = D > 0
        Dsafe = np.where(ok, D, _F(1))
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / Dsafe
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / Dsafe
        neg = ok & (this_min > 0)
        this_min = np.where(neg, _F(0), this_min)
        this_scale = np.where(neg, sum_xl / np.where(sum_l2 == 0, _F(1), sum_l2), this_scale)
        d = this_scale * La + this_min - xt
        err = _tsum(wt * (np.abs(d) if use_mad else d * d))
        better = ok & (err < best)
        L = np.where(better, La, L)
        best = np.where(better, err, best)
        scale = np.where(better, this_scale, scale)
        min_cur = np.where(better, this_min, min_cur)
    scale = np.where(flat, _F(0), scale).astype(_F)
    L = np.where(flat, _F(0), L)
    return scale, (-min_cur).astype(_F), np.ascontiguousarray(L.T).astype(np.uint8)


def _first_absmax_t(xt):
    """_first_absmax for an (n, N) layout: (|max|, signed value) per column; first max wins like the C `>`."""
    idx = np.abs(xt).argmax(axis=0)
    mx = xt[idx, np.arange(xt.shape[1])]
    return np.abs(mx), mx


def _qx(x, nmax):
    """make_qx_quants(n, nmax, x, L, rmse_type=1, qw=NULL) for Q6_K: scale (N,), L (N, n) in [0, 2*nmax-1]."""
    xt = np.ascontiguousarray(x.T)
    amax, mx = _first_absmax_t(xt)
    zero = amax < _EPS
    mxs = np.where(zero, _F(1), mx)
    iscale = (-_F(nmax) / mxs).astype(_F)
    w = xt * xt
    wx = w * xt

    def evaluate(isc):
        l = np.clip(_rint(isc * xt), -nmax, nmax - 1).astype(_F)
        return l, _tsum(wx * l), _tsum(w * l * l)

    L, sumlx, suml2 = evaluate(iscale)
    safe = np.where(suml2 == 0, _F(1), suml2)
    scale = np.where(suml2 != 0, sumlx / safe, _F(0)).astype(_F)
    best = scale * sumlx
    for k in range(-9, 10):
        if k == 0:
            continue
        isc = (-(_F(nmax) + _F(0.1) * _F(k)) / mxs).astype(_F)
        l2, slx, sl2 = evaluate(isc)
        take = (sl2 > 0) & (slx * slx > best * sl2)
        L = np.where(take, l2, L)
        sc2 = slx / np.where(sl2 == 0, _F(1), sl2)
        scale = np.where(take, sc2, scale).astype(_F)
        best = np.where(take, scale * slx, best)
    scale = np.where(zero, _F(0), scale)
    L = np.where(zero, _F(0), L + nmax)
    return scale.astype(_F), np.ascontiguousarray(L.T).astype(np.int16)


def _q3(x, nmax=4):
    """make_q3_quants(n=16, nmax=4, x, L, do_rmse=true): scale (N,), L (N,16) in [0,7]."""
    xt = np.ascontiguousarray(x.T)
    n, N = xt.shape
    amax, mx = _first_absmax_t(xt)
    zero = amax < _EPS
    mxs = np.where(zero, _F(1), mx)
    iscale = (-_F(nmax) / mxs).astype(_F)
    L = np.clip(_rint(iscale * xt), -nmax, nmax - 1).astype(_F)
    w = xt * xt
    wx = w * xt
    sumlx = _tsum(wx * L)
    suml2 = _tsum(w * L * L)
    for _ in range(5):                                  # a no-change try is a fixed point, so no early break needed
        for i in range(n):
            xi, wi, Li = xt[i], w[i], L[i]
            slx = sumlx - wi * xi * Li
            cand = slx > 0
            sl2 = suml2 - wi * Li * Li
            slx_safe = np.where(cand, slx, _F(1))
            new_l = np.clip(_rint(xi * sl2 / slx_safe), -nmax, nmax - 1).astype(_F)
            chg = cand & (new_l != Li)
            slx2 = slx + wi * xi * new_l
            sl2b = sl2 + wi * new_l * new_l
            acc = chg & (sl2b > 0) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl2b)
            L[i] = np.where(acc, new_l, Li)
            sumlx = np.where(acc, slx2, sumlx)
            suml2 = np.where(acc, sl2b, suml2)
    scale = np.where(suml2 > 0, sumlx / np.where(suml2 > 0, suml2, _F(1)), _F(0)).astype(_F)
    scale = np.where(zero, _F(0), scale)
    L = np.where(zero, _F(0), L + nmax)
    return scale.astype(_F), np.ascontiguousarray(L.T).astype(np.int16)


# --------------------------------------------------------------------------- block encoders (x: (B, 256) float32)

def _f16_bytes(a):
    return np.ascontiguousarray(np.asarray(a, dtype=_F).astype(np.float16)).view(np.uint8).reshape(-1, 2)


def _pack_2bit(L):
    """qs for Q2_K / Q3_K: for each 128-group, qs[l] = L[l] | L[l+32]<<2 | L[l+64]<<4 | L[l+96]<<6."""
    B = L.shape[0]
    g = L.reshape(B, 2, 4, 32).astype(np.uint8)
    return (g[:, :, 0] | (g[:, :, 1] << 2) | (g[:, :, 2] << 4) | (g[:, :, 3] << 6)).reshape(B, 64)


def _q2_k(x):
    B = x.shape[0]
    xs = x.reshape(B * 16, 16)
    scale, mins, L = _qkx2(xs, np.abs(xs), 3, -0.5, 0.1, 15, True)
    scales = scale.reshape(B, 16)
    mins = mins.reshape(B, 16)
    L = L.reshape(B, 256).astype(_F)
    max_scale = _pos_max(scales)
    max_min = _pos_max(mins)
    q4 = _F(15)
    ok_s = max_scale > 0
    isc = np.where(ok_s, q4 / np.where(ok_s, max_scale, _F(1)), _F(0)).astype(_F)
    sc = np.where(ok_s[:, None], _rint(isc[:, None] * scales), 0).astype(np.uint8)
    d = np.where(ok_s, max_scale / q4, _F(0)).astype(_F)
    ok_m = max_min > 0
    imn = np.where(ok_m, q4 / np.where(ok_m, max_min, _F(1)), _F(0)).astype(_F)
    sc = sc | (np.where(ok_m[:, None], _rint(imn[:, None] * mins), 0).astype(np.uint8) << 4)
    dmin = np.where(ok_m, max_min / q4, _F(0)).astype(_F)
    d16, dm16 = _f16(d), _f16(dmin)
    dl = d16[:, None] * (sc & 0xF).astype(_F)
    ml = dm16[:, None] * (sc >> 4).astype(_F)
    dl_e = np.repeat(dl, 16, axis=1)
    ml_e = np.repeat(ml, 16, axis=1)
    nz = dl_e != 0
    L2 = np.clip(_rint((x + ml_e) / np.where(nz, dl_e, _F(1))), 0, 3)
    L = np.where(nz, L2, L)
    out = np.concatenate([sc, _pack_2bit(L), _f16_bytes(d), _f16_bytes(dmin)], axis=1)
    return out


def _q3_k(x):
    B = x.shape[0]
    scale, L = _q3(x.reshape(B * 16, 16))
    scales = scale.reshape(B, 16)
    L = L.reshape(B, 256).astype(np.int32)
    amax_i = np.abs(scales).argmax(axis=1)
    max_scale = scales[np.arange(B), amax_i]
    ok = max_scale != 0
    iscale = np.where(ok, -_F(32) / np.where(ok, max_scale, _F(1)), _F(0)).astype(_F)
    l = np.clip(_rint(iscale[:, None] * scales), -32, 31).astype(np.int32) + 32
    l = np.where(ok[:, None], l, 0)
    sc = np.zeros((B, 12), dtype=np.uint8)
    for j in range(16):
        lj = l[:, j]
        if j < 8:
            sc[:, j] |= (lj & 0xF).astype(np.uint8)
        else:
            sc[:, j - 8] |= ((lj & 0xF) << 4).astype(np.uint8)
        sc[:, j % 4 + 8] |= ((lj >> 4) << (2 * (j // 4))).astype(np.uint8)
    d = np.where(ok, _F(1) / np.where(ok, iscale, _F(1)), _F(0)).astype(_F)
    d16 = _f16(d)
    sc_signed = l - 32                                    # what the decoder recovers
    dl = np.repeat(d16[:, None] * sc_signed.astype(_F), 16, axis=1)
    nz = dl != 0
    L2 = np.clip(_rint(x / np.where(nz, dl, _F(1))), -4, 3).astype(np.int32) + 4
    L = np.where(nz, L2, L)
    hi = (L > 3)
    hmask = np.zeros((B, 32), dtype=np.uint8)
    for bit in range(8):                                  # element j -> hmask[j % 32] bit j // 32
        hmask |= (hi[:, bit * 32:(bit + 1) * 32].astype(np.uint8) << bit)
    L = np.where(hi, L - 4, L)
    return np.concatenate([hmask, _pack_2bit(L), sc, _f16_bytes(d)], axis=1)


def _scales_min_k4(scales, mins):
    """Pack 8 six-bit scales and 8 six-bit mins into the 12-byte K4 layout; scales/mins are (B, 8) ints."""
    B = scales.shape[0]
    out = np.zeros((B, 12), dtype=np.uint8)
    for j in range(8):
        ls, lm = scales[:, j].astype(np.int32), mins[:, j].astype(np.int32)
        if j < 4:
            out[:, j] = ls
            out[:, j + 4] = lm
        else:
            out[:, j + 4] = (ls & 0xF) | ((lm & 0xF) << 4)
            out[:, j - 4] |= ((ls >> 4) << 6).astype(np.uint8)
            out[:, j] |= ((lm >> 4) << 6).astype(np.uint8)
    return out


def _unpack_k4(sc):
    """Inverse of _scales_min_k4 (get_scale_min_k4): (B, 8) scale and min ints."""
    B = sc.shape[0]
    s = np.zeros((B, 8), dtype=np.int32)
    m = np.zeros((B, 8), dtype=np.int32)
    q = sc.astype(np.int32)
    for j in range(8):
        if j < 4:
            s[:, j], m[:, j] = q[:, j] & 63, q[:, j + 4] & 63
        else:
            s[:, j] = (q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4)
            m[:, j] = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
    return s, m


def _q45_k(x, five: bool):
    B = x.shape[0]
    xs = x.reshape(B * 8, 32)
    av = np.sqrt(_seqsum(xs * xs) / _F(32)).astype(_F)
    w = av[:, None] + np.abs(xs)
    nmax = 31 if five else 15
    scale, mins, L = _qkx2(xs, w, nmax, -0.5 if five else -1.0, 0.1, 15 if five else 20, False)
    scales = scale.reshape(B, 8)
    mins = mins.reshape(B, 8)
    L = L.reshape(B, 256).astype(_F)
    max_scale = _pos_max(scales)
    max_min = _pos_max(mins)
    inv_s = np.where(max_scale > 0, _F(63) / np.where(max_scale > 0, max_scale, _F(1)), _F(0)).astype(_F)
    inv_m = np.where(max_min > 0, _F(63) / np.where(max_min > 0, max_min, _F(1)), _F(0)).astype(_F)
    ls = np.minimum(63, _rint(inv_s[:, None] * scales)).astype(np.int32)
    lm = np.minimum(63, _rint(inv_m[:, None] * mins)).astype(np.int32)
    sc = _scales_min_k4(ls, lm)
    d = (max_scale / _F(63)).astype(_F)
    dmin = (max_min / _F(63)).astype(_F)
    s_dec, m_dec = _unpack_k4(sc)
    dl = np.repeat(_f16(d)[:, None] * s_dec.astype(_F), 32, axis=1)
    ml = np.repeat(_f16(dmin)[:, None] * m_dec.astype(_F), 32, axis=1)
    nz = dl != 0
    L2 = np.clip(_rint((x + ml) / np.where(nz, dl, _F(1))), 0, nmax)
    L = np.where(nz, L2, L).astype(np.uint8)
    g = L.reshape(B, 4, 2, 32)                              # 64-groups: low nibble = first 32, high = next 32
    if not five:
        qs = (g[:, :, 0] | (g[:, :, 1] << 4)).reshape(B, 128)
        return np.concatenate([_f16_bytes(d), _f16_bytes(dmin), sc, qs], axis=1)
    hi = g > 15
    lo = np.where(hi, g - 16, g)
    qs = (lo[:, :, 0] | (lo[:, :, 1] << 4)).reshape(B, 128)
    qh = np.zeros((B, 32), dtype=np.uint8)
    for n in range(4):                                      # group n: bit 2n for the first 32, bit 2n+1 for the next
        qh |= (hi[:, n, 0].astype(np.uint8) << (2 * n))
        qh |= (hi[:, n, 1].astype(np.uint8) << (2 * n + 1))
    return np.concatenate([_f16_bytes(d), _f16_bytes(dmin), sc, qh, qs], axis=1)


def _q6_k(x):
    B = x.shape[0]
    scale, L = _qx(x.reshape(B * 16, 16), 32)
    scales = scale.reshape(B, 16)
    L = L.reshape(B, 256).astype(np.int32)
    amax_i = np.abs(scales).argmax(axis=1)
    max_scale = scales[np.arange(B), amax_i]
    zero_block = np.abs(max_scale) < _EPS
    iscale = np.where(zero_block, _F(0), -_F(128) / np.where(zero_block, _F(1), max_scale)).astype(_F)
    d = np.where(zero_block, _F(0), _F(1) / np.where(zero_block, _F(1), iscale)).astype(_F)
    sc = np.minimum(127, _rint(iscale[:, None] * scales)).astype(np.int32)
    sc = np.where(zero_block[:, None], 0, sc)
    dl = np.repeat(_f16(d)[:, None] * sc.astype(_F), 16, axis=1)
    nz = dl != 0
    L2 = np.clip(_rint(x / np.where(nz, dl, _F(1))), -32, 31).astype(np.int32) + 32
    L = np.where(nz, L2, L)
    L = np.where(zero_block[:, None], 0, L).astype(np.uint8)
    g = L.reshape(B, 2, 4, 32)                              # per 128: q1..q4 = elements 0/32/64/96 of the group
    q1, q2, q3, q4 = (g[:, :, k] for k in range(4))
    ql = np.concatenate([(q1 & 0xF) | ((q3 & 0xF) << 4), (q2 & 0xF) | ((q4 & 0xF) << 4)], axis=2)   # (B,2,64)
    qh = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6)                      # (B,2,32)
    return np.concatenate([ql.reshape(B, 128), qh.reshape(B, 64), sc.astype(np.int8).view(np.uint8),
                           _f16_bytes(d)], axis=1)


_ENCODERS = {"Q2_K": _q2_k, "Q3_K": _q3_k, "Q4_K": lambda x: _q45_k(x, False),
             "Q5_K": lambda x: _q45_k(x, True), "Q6_K": _q6_k}


def quantize_blocks(x: np.ndarray, qtype: str, threads: int = 1) -> np.ndarray:
    """x: float array whose size is a multiple of 256 -> uint8 (n_blocks, BLOCK_BYTES[qtype]).

    `threads` > 1 encodes chunks of blocks in parallel (numpy releases the GIL on the large
    array ops); callers that already run several tensors at once should leave it at 1."""
    if qtype not in _ENCODERS:
        raise ValueError(f"{qtype} is not a supported K-quant ({', '.join(SUPPORTED)})")
    flat = np.ascontiguousarray(x, dtype=_F).reshape(-1)
    if flat.size % QK_K:
        raise ValueError(f"K-quants need the element count to be a multiple of {QK_K}, got {flat.size}")
    blocks = flat.reshape(-1, QK_K)
    out = np.empty((blocks.shape[0], BLOCK_BYTES[qtype]), dtype=np.uint8)
    enc = _ENCODERS[qtype]

    def work(s: int) -> None:
        with np.errstate(all="ignore"):
            out[s:s + CHUNK_BLOCKS] = enc(blocks[s:s + CHUNK_BLOCKS])

    starts = range(0, blocks.shape[0], CHUNK_BLOCKS)
    if threads > 1 and len(starts) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(threads, len(starts))) as pool:
            list(pool.map(work, starts))
    else:
        for s in starts:
            work(s)
    return out


def quantize(x: np.ndarray, qtype: str, threads: int = 1) -> np.ndarray:
    """Quantize a (rows, cols) float array (cols % 256 == 0) to (rows, cols // 256 * block_bytes) uint8,
    the same shape contract as gguf.quants.quantize."""
    x = np.asarray(x)
    if x.shape[-1] % QK_K:
        raise ValueError(f"row length {x.shape[-1]} is not a multiple of {QK_K}")
    out = quantize_blocks(x, qtype, threads)
    return out.reshape(*x.shape[:-1], (x.shape[-1] // QK_K) * BLOCK_BYTES[qtype])
