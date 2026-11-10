"""总入口：与参考 pcode 的 ``OUT = bayoma(IN)`` 契约对齐的批量识别接口。

本模块把数据预处理、自动选带、单模态与多模态识别、后验协方差汇总成一个调用，
输入输出字段与 ``Contents_bayoma.m``（pcode 参考包，见 ``docs/REUSE_SEARCH.md``）
的说明逐条对应，便于其余软件与智能体直接调用。

字段约定
--------
输入
    ``tdata``   原始环境振动数据，``nt x nch``，默认按加速度计（g）
    ``fs``      采样频率（Hz）
    ``f1f2``    识别频带，``nb x 2``，每行一个频带上下界（Hz）
    ``f0``      各频带内固有频率初值，长度 ``nb`` 的序列，每项为一维数组
    ``mute``    1 表示不输出中间信息，0 表示输出（缺省 0）
    ``ichan``   参与计算的通道索引（0 基），缺省全部
    ``iband``   参与计算的频带索引（0 基），缺省全部
    ``maxiter`` MPV 迭代上限，缺省 500
    ``f1f2d``   需从似然函数排除的频带，``nd x 2``，缺省空
    ``p``       原始数据各通道乘以 ``(i*2*pi*f)**p`` 的指数，标量或长度 ``nch``
    ``q``       模态频响函数除以 ``(i*2*pi*f_k)**q`` 的指数，缺省 0

输出
    ``f``       各阶频率（Hz）
    ``z``       各阶阻尼比
    ``Sii``     各阶模态力自谱（单边）
    ``Se``      各阶通道噪声谱（单边）
    ``phi``     振型，每列一阶，欧氏范数为 1
    ``rms``     模态加速度均方根
    ``S``       各频带的模态力谱矩阵
    ``coefv``   各参数的后验变异系数
    ``cmat``    各频带的无量纲参数协方差矩阵
    ``sn``      模态信噪比
    ``time_req`` 各频带求 MPV 与后验协方差的耗时（s）
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from . import multi_mode, multi_setup, posterior, results, single_mode
from ._util import LOG
from .auto_band import select_bands
from .data import build_p_vector, sample_psd_welch, scaled_fft

__all__ = ["bayoma"]


def _as_band_list(f1f2) -> np.ndarray:
    """把频带输入整理成 ``nb x 2`` 的浮点数组。"""
    arr = np.atleast_2d(np.asarray(f1f2, dtype=float))
    if arr.shape[1] != 2:
        raise ValueError("f1f2 需为 nb x 2 的数组，每行一个频带上下界")
    return arr


def _as_f0_list(f0, nb: int) -> list:
    """把固有频率初值整理成长度 ``nb`` 的列表，每项为一维浮点数组。"""
    if f0 is None:
        return [np.empty(0, dtype=float) for _ in range(nb)]
    seq = list(f0)
    if len(seq) != nb:
        raise ValueError("f0 的长度需与频带数一致")
    return [np.atleast_1d(np.asarray(item, dtype=float)).ravel() for item in seq]


def _mask_excluded(freqs: np.ndarray, fft: np.ndarray, f1f2d) -> tuple[np.ndarray, np.ndarray]:
    """剔除 ``f1f2d`` 各频带内的频率点。"""
    if f1f2d is None:
        return freqs, fft
    bands = np.atleast_2d(np.asarray(f1f2d, dtype=float))
    if bands.size == 0:
        return freqs, fft
    keep = np.ones(freqs.shape, dtype=bool)
    for lo, hi in bands:
        keep &= ~((freqs >= lo) & (freqs <= hi))
    return freqs[keep], fft[:, keep]


def _single_band(
    data: np.ndarray,
    fs: float,
    band: tuple[float, float],
    f_init: np.ndarray,
    *,
    p_vec: np.ndarray | None,
    q: int,
    maxiter: int,
    f1f2d,
    mute: int,
) -> dict[str, Any]:
    """对一个频带做识别，返回该频带的参数与后验信息。"""
    spec = scaled_fft(data, fs, band, p=p_vec)
    freqs, fft = _mask_excluded(spec.freqs, spec.fft, f1f2d)
    if freqs.size < 4:
        raise ValueError(f"频带 {band} 内可用频率点过少，无法识别")
    n_mode = int(max(f_init.size, 1))
    dhat = np.einsum("ak,bk->abk", fft, fft.conj()).real
    d_total = float(np.sum(np.abs(fft) ** 2))

    t_mpv = time.perf_counter()
    if n_mode == 1:
        res = single_mode.mpv(
            freqs,
            dhat,
            d_total,
            q=q,
            f0=float(f_init[0]) if f_init.size else None,
            max_iter=maxiter,
        )
        f = np.array([res.f])
        z = np.array([res.z])
        S = np.array([[res.S]])
        Se = np.array([res.Se])
        phi = res.phi[:, None]
        cov = single_mode.posterior_covariance(res, freqs, dhat, d_total, q=q)
        coefv = single_mode.coefficients_of_variation(res, cov)
        cmat = cov
        nch_loc = int(res.phi.size)
        cov_phi = [np.asarray(cov[4:, 4:], dtype=float)]
        sizes = [nch_loc - 1]
        n_off = 0
        index = results.single_mode_index(nch_loc)
    else:
        res = multi_mode.mpv(
            freqs,
            fft,
            n_mode,
            q=q,
            f0=f_init if f_init.size else None,
            max_iter=maxiter,
        )
        f = np.asarray(res.f, dtype=float)
        z = np.asarray(res.z, dtype=float)
        S = np.asarray(res.S)
        Se = np.full(f.size, float(res.Se))
        phi = np.asarray(res.phi)
        post = posterior.multi_mode_covariance(res, freqs, fft, q=q)
        coefv = dict(post.coefv_named)
        cmat = post.cov
        cov_phi = list(post.cov_phi)
        m_loc = f.size
        n_off = m_loc * (m_loc - 1) // 2
        sizes = results.shape_block_sizes(phi, subspace_dim=int(np.asarray(res.basis).shape[1]))
        index = None
    t_mpv = time.perf_counter() - t_mpv

    if not mute:
        LOG.info(
            "频带 %.4f-%.4f Hz：%d 阶，f=%s，z=%s",
            band[0],
            band[1],
            f.size,
            np.array2string(f, precision=5),
            np.array2string(z, precision=5),
        )
    return {
        "f": f,
        "z": z,
        "S": S,
        "Se": Se,
        "phi": phi,
        "coefv": coefv,
        "cmat": cmat,
        "cov_phi": cov_phi,
        "sizes": sizes,
        "n_off": n_off,
        "complex": bool(np.iscomplexobj(phi)),
        "f0": np.asarray(f_init, dtype=float).ravel(),
        "freqs": freqs,
        "index": index,
        "time_mpv": t_mpv,
    }


def bayoma(
    tdata: np.ndarray,
    fs: float,
    f1f2=None,
    f0=None,
    *,
    mute: int = 0,
    ichan=None,
    iband=None,
    maxiter: int = 500,
    f1f2d=None,
    p=None,
    q: int = 0,
    geometry=None,
    figure_dir=None,
    spectra: bool = False,
) -> dict[str, Any]:
    """按参考 pcode 的契约执行自动贝叶斯模态识别。

    参数见模块文档。返回字典的键与 ``Contents_bayoma.m`` 的输出字段一致。

    Notes
    -----
    ``f1f2`` 与 ``f0`` 均缺省时，先用奇异值谱自动选带并给出初值；给出 ``f1f2``
    而未给 ``f0`` 时，初值由各频带内的奇异值谱峰自动确定。

    ``geometry`` 给出各通道的位置、方向与连接时，结果另带 ``recognition.spatial``
    与 ``recognition.shape_uncertainty``；``figure_dir`` 指定后每阶振型存一张 PNG。
    ``spectra=True`` 时结果另带 PSD 谱与 SV 谱（含频带与频率初值标注）。
    """
    data = np.asarray(tdata, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    nch_all = data.shape[1]

    chan = np.arange(nch_all) if ichan is None else np.atleast_1d(np.asarray(ichan, dtype=int)).ravel()
    data = data[:, chan]
    nch = data.shape[1]

    p_vec = build_p_vector(p, nch)

    if f1f2 is None:
        # 自动选带必须在平均谱上做：单次实现的 Re(F_k F_k^*) 秩为 1，高阶奇异值
        # 恒为零，噪声基线被估成零，宽带噪声起伏会被误判为谱峰。
        wfreqs, wpsd = sample_psd_welch(data, fs)
        sel = select_bands(wfreqs, wpsd)
        bands = np.asarray(sel.bands, dtype=float)
        f0 = [np.asarray([pk], dtype=float) for pk in sel.peaks]
        if not mute:
            LOG.info("自动选带：%d 个频带", bands.shape[0])
    else:
        bands = _as_band_list(f1f2)

    if iband is not None:
        idx = np.atleast_1d(np.asarray(iband, dtype=int)).ravel()
        bands = bands[idx]
        if f0 is not None:
            f0 = [f0[i] for i in idx]

    nb = bands.shape[0]
    f0_list = _as_f0_list(f0, nb)

    if not mute:
        LOG.info("识别通道数 %d，频带数 %d", nch, nb)

    freqs_out: list[np.ndarray] = []
    z_out: list[np.ndarray] = []
    S_list: list[np.ndarray] = []
    Se_out: list[np.ndarray] = []
    phi_cols: list[np.ndarray] = []
    cmat_list: list[np.ndarray] = []
    coefv_rows: dict[str, list[float]] = {
        "f": [],
        "z": [],
        "Sii": [],
        "Se": [],
        "phi": [],
        "rms": [],
    }
    time_req = np.zeros((2, nb), dtype=float)
    band_blocks: list[dict] = []
    n_modes_per_band: list[int] = []
    offset = 0

    for i in range(nb):
        band = (float(bands[i, 0]), float(bands[i, 1]))
        t0 = time.perf_counter()
        out = _single_band(
            data,
            fs,
            band,
            f0_list[i],
            p_vec=p_vec,
            q=q,
            maxiter=maxiter,
            f1f2d=f1f2d,
            mute=mute,
        )
        time_req[0, i] = out["time_mpv"]
        time_req[1, i] = time.perf_counter() - t0

        freqs_out.append(out["f"])
        z_out.append(out["z"])
        S_list.append(out["S"])
        Se_out.append(out["Se"])
        phi_cols.append(out["phi"])
        cmat_list.append(out["cmat"])

        n_modes_per_band.append(int(out["f"].size))
        band_blocks.append(
            {
                "cov": out["cmat"],
                "cov_phi": out["cov_phi"],
                "m": int(out["f"].size),
                "n_off": int(out["n_off"]),
                "sizes": list(out["sizes"]),
                "band_index": i,
                "offset": offset,
                "complex": bool(out["complex"]),
                "nch": nch,
                "channel_index": chan.tolist(),
                "index": out["index"],
            }
        )
        offset += int(out["f"].size)

        cv = out["coefv"]
        n_modes = out["f"].size
        for name, key in (("f", "f"), ("z", "z"), ("S", "Sii"), ("Se", "Se")):
            vals = cv.get(name)
            if np.isscalar(vals) or vals is None:
                coefv_rows[key].extend([float(vals)] * n_modes)
            else:
                coefv_rows[key].extend(np.atleast_1d(np.asarray(vals, dtype=float)).ravel().tolist())
        phi_cv = cv.get("phi", 0.0)
        coefv_rows["phi"].extend([float(phi_cv)] * n_modes)

    f_all = np.concatenate(freqs_out) if freqs_out else np.empty(0)
    z_all = np.concatenate(z_out) if z_out else np.empty(0)
    Se_all = np.concatenate(Se_out) if Se_out else np.empty(0)
    # 每个频带的模态力谱矩阵 S 的对角元逐阶给出该阶的模态力谱密度；一个频带含多阶模态时
    # 必须逐阶展开，否则与 f、z 的长度不一致。
    Sii_all = np.concatenate(
        [np.real(np.diag(mat)).ravel() for mat in S_list]
    ) if S_list else np.empty(0)
    phi_all = np.hstack(phi_cols) if phi_cols else np.empty((nch, 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        rms_all = np.sqrt(np.pi * Sii_all * f_all / (4.0 * z_all))
        sn_all = Sii_all / (4.0 * Se_all * z_all**2)
        coefv_rows["rms"] = (0.5 * np.sqrt(
            np.asarray(coefv_rows["f"], dtype=float) ** 2
            + np.asarray(coefv_rows["z"], dtype=float) ** 2
            + np.asarray(coefv_rows["Sii"], dtype=float) ** 2
        )).tolist()

    coefv = {name: np.asarray(vals, dtype=float) for name, vals in coefv_rows.items()}

    geo = results.geometry_from_spec(geometry)
    figure_path = None if figure_dir is None else str(Path(figure_dir))
    recognition = results.build_recognition(
        f=f_all,
        z=z_all,
        phi=phi_all,
        bands=bands,
        f0_list=f0_list,
        n_modes_per_band=n_modes_per_band,
        band_blocks=band_blocks,
        data=data if spectra else None,
        fs=float(fs) if spectra else None,
        channel_index=chan.tolist(),
        geometry=geo,
        figure_dir=figure_path,
        with_spectra=bool(spectra),
    )

    return {
        "f": f_all,
        "z": z_all,
        "Sii": Sii_all,
        "Se": Se_all,
        "phi": phi_all,
        "rms": rms_all,
        "S": S_list,
        "coefv": coefv,
        "cmat": cmat_list,
        "sn": sn_all,
        "time_req": time_req,
        "bands": bands,
        "nch": nch,
        "fs": float(fs),
        "recognition": recognition,
    }
