"""识别结果的整理与导出（频带对应、模态区分度、联合后验协方差、谱、空间可视化）。

本模块不参与识别计算，只把各识别模块已经算出的量整理成「可追溯」的结构：每一项新增
结果都带索引，索引与参数名、模态阶次、通道、频带一一对应。设计的出发点是让结果文件
自身说明「这个数对应哪一阶、哪一段频率、哪个通道」，读结果的人不必回看识别过程。

索引约定
--------
- 模态阶次 ``order``：全频带连续编号，从 0 起；
- 频带 ``band_index``：调用方给出的频带顺序，从 0 起；
- 通道 ``channel``：参与识别的通道在输入数据中的列号；
- 参数 ``index``：联合协方差矩阵的行列号，从 0 起。

后验协方差的组装
----------------
不同频带使用互不重叠的频点，负对数似然按 ``L = sum_b L_b`` 相加，因此 Hessian 关于
各频带参数是分块对角的，联合协方差即各频带协方差的分块对角放置。这不是近似，而是
似然可加性的直接结果；跨频带参数之间的相关性为零。同一频带内的模态由联合似然同时
确定，其相关性保留在各频带协方差块内。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ._util import LOG

# 模态区分度判据：两阶频率间隔小于若干倍半功率带宽时判为密集模态。
# 半功率带宽取 ``2 zeta f``，故 ``separation_in_bw = df / max(2 z_i f_i, 2 z_j f_j)``。
DENSE_MODE_BW_THRESHOLD = 2.0


def _f(value) -> float:
    """转成 Python 浮点，非有限值转 ``None``。"""
    out = float(value)
    return out if math.isfinite(out) else None


def shape_block_sizes(phi: np.ndarray, subspace_dim: int | None = None) -> list[int]:
    """各阶模态形状在切空间中的实维数。

    实形状每阶 ``n - 1``；复形状按 ``[Re; Im]`` 展开后每阶 ``2n - 2``（单位范数去掉
    径向、复相位去掉一个方向）。当形状在 ``m0`` 维子空间内表示时（第 13 章的
    ``U = B0 alpha``），维数按 ``m0`` 计。
    """
    arr = np.asarray(phi)
    n = arr.shape[0]
    m = arr.shape[1]
    dim = int(subspace_dim) if subspace_dim is not None else n
    if np.iscomplexobj(arr):
        return [max(2 * dim - 2, 0)] * m
    return [max(dim - 1, 0)] * m


def spectral_index(m: int, n_off: int) -> list[dict]:
    """谱参数在协方差矩阵中的位置描述。

    排列与 :func:`bayoma.posterior.pack_spectrum` 一致：
    ``[f (m), z (m), diag(S) (m), Re(offdiag S) (n_off), Im(offdiag S) (n_off), Se (1)]``。
    """
    items: list[dict] = []
    iu = np.triu_indices(m, 1)
    for i in range(m):
        items.append({"name": f"f[{i}]", "kind": "f", "mode": i, "symbol": "f", "unit": "Hz"})
    for i in range(m):
        items.append({"name": f"z[{i}]", "kind": "z", "mode": i, "symbol": "zeta", "unit": "-"})
    for i in range(m):
        items.append({"name": f"S[{i}][{i}]", "kind": "S_diag", "mode": i, "symbol": "S", "unit": "-"})
    for k in range(n_off):
        items.append(
            {
                "name": f"ReS[{iu[0][k]}][{iu[1][k]}]",
                "kind": "S_off_re",
                "mode": None,
                "modes": [int(iu[0][k]), int(iu[1][k])],
                "symbol": "Re S",
                "unit": "-",
            }
        )
    for k in range(n_off):
        items.append(
            {
                "name": f"ImS[{iu[0][k]}][{iu[1][k]}]",
                "kind": "S_off_im",
                "mode": None,
                "modes": [int(iu[0][k]), int(iu[1][k])],
                "symbol": "Im S",
                "unit": "-",
            }
        )
    items.append({"name": "Se", "kind": "Se", "mode": None, "symbol": "S_e", "unit": "-"})
    return items


def shape_index(m: int, sizes: list[int], nch: int, complex_shape: bool) -> list[dict]:
    """模态形状切空间坐标的位置描述。"""
    items: list[dict] = []
    for i in range(m):
        for k in range(sizes[i]):
            items.append(
                {
                    "name": f"phi[{i}].theta[{k}]",
                    "kind": "phi_tangent",
                    "mode": i,
                    "coord": k,
                    "symbol": "theta",
                    "unit": "-",
                    "channel": None,
                }
            )
    return items


def band_table(
    bands: np.ndarray,
    f0_list: list,
    n_modes_per_band: list[int],
    peaks: list | None = None,
    excluded: list | None = None,
) -> list[dict]:
    """频带表：每段的上下界、模态阶数、代入计算的频率初值。"""
    out: list[dict] = []
    for i, band in enumerate(np.asarray(bands, dtype=float)):
        rec = {
            "band_index": i,
            "lower": _f(band[0]),
            "upper": _f(band[1]),
            "bandwidth": _f(band[1] - band[0]),
            "n_modes": int(n_modes_per_band[i]) if i < len(n_modes_per_band) else 0,
            "f0": [float(v) for v in np.asarray(f0_list[i], dtype=float).ravel()] if i < len(f0_list) else [],
        }
        if peaks is not None and i < len(peaks):
            rec["peak"] = _f(peaks[i])
        if excluded:
            rec["excluded"] = [
                {"lower": _f(lo), "upper": _f(hi)} for lo, hi in excluded
            ]
        out.append(rec)
    return out


def mode_band_map(
    f: np.ndarray,
    band_of_band: list[int],
    offset_of_band: list[int],
    bands: np.ndarray | None = None,
    f0_list: list | None = None,
) -> list[dict]:
    """每阶模态落在哪个频带、在频带内的次序、以及该频带的上下界与频率初值。

    识别频率、频带下界、上界与代入计算的初值放在同一条记录里，便于按阶次直接查回
    所用的频带，不需要再关联 ``bands`` 表。
    """
    band_mat = None if bands is None else np.asarray(bands, dtype=float)
    out: list[dict] = []
    for order, freq in enumerate(np.asarray(f, dtype=float)):
        bi = band_of_band[order]
        item = {
            "order": order,
            "frequency": _f(freq),
            "band_index": bi,
            "order_in_band": order - offset_of_band[bi],
        }
        if band_mat is not None and 0 <= bi < band_mat.shape[0]:
            item["band_lower"] = _f(band_mat[bi, 0])
            item["band_upper"] = _f(band_mat[bi, 1])
        if f0_list is not None and 0 <= bi < len(f0_list):
            item["f0"] = [_f(v) for v in np.atleast_1d(f0_list[bi])]
        out.append(item)
    return out


def separation_table(
    f: np.ndarray,
    z: np.ndarray,
    phi: np.ndarray,
    band_index: np.ndarray,
    joint_cov: np.ndarray | None = None,
    f_positions: list[int] | None = None,
) -> dict:
    """密集模态的频率区分度。

    对同一频带内的模态两两给出：频率间隔、半功率带宽、以半功率带宽计的间隔
    ``separation_in_bw = df / max(2 z_i f_i, 2 z_j f_j)``、振型 MAC，以及（若给出联合
    协方差）频率估计的相关系数。``separation_in_bw < DENSE_MODE_BW_THRESHOLD`` 判为
    密集模态。
    """
    f = np.asarray(f, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    band_index = np.asarray(band_index, dtype=int).ravel()
    m = f.size
    pairs: list[dict] = []

    def _mac(a, b) -> float:
        a = np.asarray(a, dtype=complex).ravel()
        b = np.asarray(b, dtype=complex).ravel()
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 0.0 or nb <= 0.0:
            return float("nan")
        return float(abs(np.vdot(a, b)) ** 2 / (na * nb) ** 2)

    corr = None
    if joint_cov is not None and f_positions is not None:
        cov = np.asarray(joint_cov, dtype=float)
        idx = np.asarray(f_positions, dtype=int)
        sub = cov[np.ix_(idx, idx)]
        sd = np.sqrt(np.maximum(np.diag(sub), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = sub / np.outer(sd, sd)
            corr[~np.isfinite(corr)] = 0.0

    for i in range(m):
        for j in range(i + 1, m):
            if band_index[i] != band_index[j]:
                continue
            df = abs(f[j] - f[i])
            bw_i = 2.0 * abs(z[i]) * f[i]
            bw_j = 2.0 * abs(z[j]) * f[j]
            bw = max(bw_i, bw_j)
            sep = df / bw if bw > 0.0 else float("inf")
            rec = {
                "modes": [i, j],
                "band_index": int(band_index[i]),
                "df": _f(df),
                "half_power_bw": [_f(bw_i), _f(bw_j)],
                "separation_in_bw": _f(sep),
                "dense": bool(sep < DENSE_MODE_BW_THRESHOLD),
                "mac": _f(_mac(phi[:, i], phi[:, j])),
            }
            if corr is not None:
                rec["f_correlation"] = _f(corr[i, j])
            pairs.append(rec)

    dense_pairs = [p for p in pairs if p["dense"]]
    return {
        "criterion": "separation_in_bw = df / max(2 z_i f_i, 2 z_j f_j)",
        "threshold": DENSE_MODE_BW_THRESHOLD,
        "pairs": pairs,
        "dense_pairs": [p["modes"] for p in dense_pairs],
        "has_dense_modes": bool(dense_pairs),
    }


def single_mode_index(nch: int) -> list[dict]:
    """单模态频带的参数排列描述，与 ``single_mode.posterior_covariance`` 一致。

    排列为 ``[f, z, S, Se, phi_1 ... phi_n]``，共 ``n + 4`` 项；模态形状分量只在
    ``(I - phi phi^T)`` 子空间内有意义，此处仍按通道逐项列出，便于追溯。
    """
    items = [
        {"name": "f[0]", "kind": "f", "mode": 0, "symbol": "f", "unit": "Hz"},
        {"name": "z[0]", "kind": "z", "mode": 0, "symbol": "zeta", "unit": "-"},
        {"name": "S[0][0]", "kind": "S_diag", "mode": 0, "symbol": "S", "unit": "-"},
        {"name": "Se", "kind": "Se", "mode": None, "symbol": "S_e", "unit": "-"},
    ]
    for r in range(nch):
        items.append(
            {
                "name": f"phi[0].c[{r}]",
                "kind": "phi_component",
                "mode": 0,
                "coord": r,
                "symbol": "phi",
                "unit": "-",
                "channel": r,
            }
        )
    return items


def _prepare_entries(blk: dict) -> list[dict]:
    """取一个协方差块的参数描述，优先使用块自带的显式索引。"""
    explicit = blk.get("index")
    if explicit is not None:
        return [dict(item) for item in explicit]
    m = int(blk["m"])
    n_off = int(blk["n_off"])
    sizes = list(blk["sizes"])
    return spectral_index(m, n_off) + shape_index(
        m, sizes, int(blk.get("nch", 0)), bool(blk.get("complex", False))
    )


def joint_covariance(
    blocks: list[dict],
    mode_offsets: list[int] | None = None,
) -> dict:
    """把各频带（或各测组）的协方差块拼成联合协方差矩阵，并给出索引。

    ``blocks[i]`` 需含 ``cov``（该块的协方差矩阵）、``band_index``、``offset``（该块
    首阶在全局的序号）。块的参数描述可取两种来源之一：给出 ``index``（显式列表，
    每项含 ``name`` 与 ``kind`` 等）时直接使用；否则按 ``m``、``n_off``、``sizes``
    由 :func:`spectral_index` 与 :func:`shape_index` 推导。
    """
    total = 0
    for blk in blocks:
        total += int(np.asarray(blk["cov"]).shape[0])
    joint = np.zeros((total, total), dtype=float)
    index: list[dict] = []
    pos = 0
    for blk in blocks:
        cov = np.asarray(blk["cov"], dtype=float)
        k = cov.shape[0]
        joint[pos : pos + k, pos : pos + k] = 0.5 * (cov + cov.T)
        off = int(blk.get("offset", 0))
        entries = _prepare_entries(blk)
        if len(entries) != k:
            raise ValueError(
                f"频带 {blk.get('band_index')} 的参数描述 {len(entries)} 项与协方差维数 {k} 不符"
            )
        for local, item in enumerate(entries):
            rec = dict(item)
            rec["index"] = pos + local
            rec["band_index"] = int(blk.get("band_index", 0))
            if rec.get("mode") is not None:
                rec["order"] = off + int(rec["mode"])
            else:
                rec["order"] = None
            if rec.get("modes") is not None:
                rec["orders"] = [off + int(v) for v in rec["modes"]]
            if rec.get("channel") is not None and blk.get("channel_index") is not None:
                rec["channel"] = int(blk["channel_index"][int(rec["channel"])])
            index.append(rec)
        pos += k
    return {
        "joint": joint,
        "index": index,
        "size": total,
        "structure": "block_diagonal",
        "note": "不同频带使用互不重叠的频点，似然可加，Hessian 关于频带分块对角；"
        "同一频带内的模态由联合似然同时确定，其相关性保留在块内。",
    }


def shape_uncertainty(
    cov_phi: list[np.ndarray],
    phi: np.ndarray,
    channel_index: list[int],
) -> list[dict]:
    """各阶振型的识别不确定性。

    每阶给出：单位范数振型的变异系数（通道协方差矩阵迹的平方根，与课本 11.3.3 节
    的定义一致）、逐通道标准差、逐通道 95% 区间半宽、以及该阶的协方差矩阵。
    """
    phi = np.asarray(phi)
    m = phi.shape[1]
    out: list[dict] = []
    for i in range(m):
        shape = phi[:, i]
        cov = np.asarray(cov_phi[i]) if i < len(cov_phi) else np.zeros((shape.size, shape.size))
        cplx = np.iscomplexobj(shape)
        if cplx:
            n = shape.size
            if cov.shape[0] == 2 * n:
                cov_re = cov[:n, :n]
                cov_im = cov[n:, n:]
            else:
                cov_re = np.zeros((n, n))
                cov_im = np.zeros((n, n))
            var_ch = np.maximum(np.diag(cov_re), 0.0) + np.maximum(np.diag(cov_im), 0.0)
        else:
            var_ch = np.maximum(np.diag(cov), 0.0)
        trace = float(np.sum(var_ch))
        std = np.sqrt(var_ch)
        out.append(
            {
                "order": i,
                "complex": bool(cplx),
                "coefv": _f(math.sqrt(trace)) if trace > 0.0 else 0.0,
                "channel_index": list(channel_index),
                "std_per_channel": [_f(v) for v in std],
                "ci95_halfwidth_per_channel": [_f(1.96 * v) for v in std],
                "shape": shape,
                "cov": cov,
            }
        )
    return out


def spectra_table(
    data: np.ndarray,
    fs: float,
    bands: np.ndarray,
    f0_list: list,
    *,
    max_points: int = 1024,
    nblock: int | None = None,
) -> dict:
    """时程数据的 PSD 谱与 SV 谱，并标注频带与代入计算的频率初值。

    谱按 ``stride`` 抽稀后写入，抽稀倍数与原始频点数一并记录，便于复算；抽稀采用块内
    平均而非简单抽样，避免丢掉窄峰。
    """
    from .auto_band import sv_curves
    from .data import sample_psd_welch

    arr = np.asarray(data, dtype=float)
    wfreqs, wpsd = sample_psd_welch(arr, fs, nblock=nblock)
    diag = np.real(np.einsum("aak->ak", wpsd)) if wpsd.ndim == 3 else np.asarray(wpsd).T
    sv = sv_curves(wpsd)
    nf = wfreqs.size
    stride = max(1, int(math.ceil(nf / float(max_points))))

    def _reduce(mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat)
        if stride == 1:
            return mat
        keep = (nf // stride) * stride
        red = mat[:, :keep].reshape(mat.shape[0], -1, stride).mean(axis=2)
        return red

    freqs_red = wfreqs[: (nf // stride) * stride].reshape(-1, stride).mean(axis=1) if stride > 1 else wfreqs
    marks = []
    for i, band in enumerate(np.asarray(bands, dtype=float)):
        in_band = (wfreqs >= band[0]) & (wfreqs <= band[1])
        marks.append(
            {
                "band_index": i,
                "lower": _f(band[0]),
                "upper": _f(band[1]),
                "f0": [float(v) for v in np.asarray(f0_list[i], dtype=float).ravel()] if i < len(f0_list) else [],
                "n_freq_points": int(np.sum(in_band)),
                "peak_freq": _f(wfreqs[in_band][int(np.argmax(diag[0][in_band]))]) if np.any(in_band) else None,
            }
        )
    return {
        "freqs": np.asarray(freqs_red, dtype=float),
        "psd_diag": _reduce(diag),
        "sv": _reduce(sv),
        "stride": int(stride),
        "n_freq_full": int(nf),
        "n_freq_kept": int(np.asarray(freqs_red).size),
        "reduction": "块内平均",
        "bands": marks,
        "note": "psd_diag 为各通道自谱（Welch 平均），sv 为奇异值谱（按降序）；"
        "行列顺序与 channel_index、模态阶次一一对应。",
    }


def validate_geometry(geometry: dict, nch: int, channel_index: list[int]) -> dict:
    """校验并规范化通道几何：位置、方向、连接。"""
    chans = geometry.get("channels")
    if not chans:
        raise ValueError("geometry.channels 不能为空")
    if len(chans) != nch:
        raise ValueError(f"geometry.channels 的长度应为 {nch}，收到 {len(chans)}")
    positions = np.zeros((nch, 3), dtype=float)
    directions = np.zeros((nch, 3), dtype=float)
    names = []
    for i, ch in enumerate(chans):
        pos = np.asarray(ch.get("position", [0.0, 0.0, 0.0]), dtype=float).ravel()
        if pos.size != 3:
            raise ValueError(f"通道 {i} 的 position 必须是长度 3 的数组")
        dirv = np.asarray(ch.get("direction", [0.0, 0.0, 1.0]), dtype=float).ravel()
        if dirv.size != 3:
            raise ValueError(f"通道 {i} 的 direction 必须是长度 3 的数组")
        nrm = float(np.linalg.norm(dirv))
        if nrm <= 0.0:
            raise ValueError(f"通道 {i} 的 direction 不能为零向量")
        positions[i] = pos
        directions[i] = dirv / nrm
        names.append(str(ch.get("name", f"ch{i}")))
    conns = []
    for pair in geometry.get("connections", []):
        pair = list(pair)
        if len(pair) != 2:
            raise ValueError("connections 的每项必须是两个通道下标")
        a, b = int(pair[0]), int(pair[1])
        if not (0 <= a < nch and 0 <= b < nch):
            raise ValueError(f"连接 ({a}, {b}) 的通道下标越界")
        conns.append([a, b])
    return {
        "channels": [
            {
                "index": int(channel_index[i]),
                "local_index": i,
                "name": names[i],
                "position": [float(v) for v in positions[i]],
                "direction": [float(v) for v in directions[i]],
            }
            for i in range(nch)
        ],
        "connections": conns,
        "positions": positions,
        "directions": directions,
    }


def _plot_mode_shape(
    name: str,
    shape: np.ndarray,
    positions: np.ndarray,
    directions: np.ndarray,
    connections: list,
    out_path: Path,
    title: str,
) -> Path | None:
    """画一阶振型的空间形状（未变形虚线 + 变形实线/箭头）。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - 环境缺 matplotlib 时跳过
        LOG.warning("未安装 matplotlib，跳过空间可视化：%s", exc)
        return None

    shape = np.asarray(shape)
    cplx = np.iscomplexobj(shape)
    panels = 2 if cplx else 1
    fig = plt.figure(figsize=(5.2 * panels, 5.0))
    amp = float(np.max(np.abs(shape))) or 1.0
    scale = 0.25 * float(np.max(np.ptp(positions, axis=0))) if positions.size else 1.0
    scale = scale / amp if amp > 0 else 1.0

    for p in range(panels):
        ax = fig.add_subplot(1, panels, p + 1, projection="3d")
        comp = shape.real if (not cplx or p == 0) else shape.imag
        label = "实部" if (cplx and p == 0) else ("虚部" if cplx else "幅值")
        for a, b in connections:
            ax.plot(
                [positions[a, 0], positions[b, 0]],
                [positions[a, 1], positions[b, 1]],
                [positions[a, 2], positions[b, 2]],
                color="0.75",
                linestyle="--",
                linewidth=0.9,
            )
        for a, b in connections:
            ax.plot(
                [positions[a, 0] + scale * comp[a] * directions[a, 0], positions[b, 0] + scale * comp[b] * directions[b, 0]],
                [positions[a, 1] + scale * comp[a] * directions[a, 1], positions[b, 1] + scale * comp[b] * directions[b, 1]],
                [positions[a, 2] + scale * comp[a] * directions[a, 2], positions[b, 2] + scale * comp[b] * directions[b, 2]],
                color="C0",
                linewidth=1.6,
            )
        vmax = float(np.max(np.abs(comp))) or 1.0
        colors = plt.cm.coolwarm((comp / vmax + 1.0) / 2.0) if hasattr(plt, "cm") else None
        ax.scatter(
            positions[:, 0] + scale * comp * directions[:, 0],
            positions[:, 1] + scale * comp * directions[:, 1],
            positions[:, 2] + scale * comp * directions[:, 2],
            c=comp if colors is None else colors,
            s=34,
            depthshade=False,
        )
        ax.set_title(f"{name} · {title}（{label}，缩放 {scale:.3g}）", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.tick_params(labelsize=7)
        try:
            ax.set_box_aspect((1, 1, 0.7))
        except Exception:  # pragma: no cover - 老版本 matplotlib
            pass
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def spatial_figures(
    phi: np.ndarray,
    geometry: dict,
    figure_dir: str | Path | None,
    prefix: str = "mode",
) -> dict:
    """在给定几何下展示各阶振型的空间形状。

    ``geometry`` 由 :func:`validate_geometry` 规范化后传入。未给 ``figure_dir`` 时只返回
    可用于绘图的几何与振型数据（``figures`` 为空、``available`` 为 ``False``），不落盘。
    """
    nch = np.asarray(phi).shape[0]
    geo = validate_geometry(geometry, nch, list(range(nch)))
    result = {
        "available": False,
        "reason": None,
        "channel_names": [c["name"] for c in geo["channels"]],
        "connections": geo["connections"],
        "figures": [],
    }
    if figure_dir is None:
        result["reason"] = "未指定 figure_dir，仅返回几何与振型数据"
        return result

    out_dir = Path(figure_dir)
    for i in range(np.asarray(phi).shape[1]):
        path = out_dir / f"{prefix}_{i:02d}.png"
        written = _plot_mode_shape(
            f"阶 {i + 1}",
            np.asarray(phi)[:, i],
            geo["positions"],
            geo["directions"],
            geo["connections"],
            path,
            prefix,
        )
        if written is None:
            result["reason"] = "未安装 matplotlib"
            return result
        result["figures"].append({"order": i, "path": str(written), "complex": bool(np.iscomplexobj(phi))})
    result["available"] = True
    return result


def conditional_decomposition(cov: np.ndarray, shape_idx, theta_idx) -> dict:
    """把振型协方差按「由其他参数解释」与「其余」作条件方差分解。

    ``C_uu`` 为振型的边际协方差，``C_uθ C_θθ^{-1} C_θu`` 为可由 ``θ`` 线性解释的
    部分，余下为偏协方差。两项都半正定，在高斯近似下分别对应「组装阶段引入的
    不确定性」与「识别本身的其余不确定性」。``shape_idx`` 与 ``theta_idx`` 为
    参数下标序列（不要求连续）。
    """
    cov = np.asarray(cov, dtype=float)
    su = np.asarray(list(shape_idx), dtype=int)
    st = np.asarray(list(theta_idx), dtype=int)
    uu = cov[np.ix_(su, su)]
    if st.size == 0:
        return {
            "total": uu,
            "from_assembly": np.zeros_like(uu),
            "residual": uu,
            "trace_total": float(np.trace(uu)),
            "trace_from_assembly": 0.0,
            "trace_residual": float(np.trace(uu)),
        }
    tt = cov[np.ix_(st, st)]
    ut = cov[np.ix_(su, st)]
    try:
        tt_inv = np.linalg.pinv(0.5 * (tt + tt.T), rcond=1.0e-12)
        explained = ut @ tt_inv @ ut.T
    except Exception:  # pragma: no cover - 数值异常时退回零解释
        explained = np.zeros_like(uu)
    explained = 0.5 * (explained + explained.T)
    residual = 0.5 * ((uu - explained) + (uu - explained).T)
    return {
        "total": uu,
        "from_assembly": explained,
        "residual": residual,
        "trace_total": float(np.trace(uu)),
        "trace_from_assembly": float(np.trace(explained)),
        "trace_residual": float(np.trace(residual)),
    }


def save_spectra_csv(spectra: dict, path: str | Path) -> Path:
    """把抽稀后的谱写成 CSV，便于外部绘图。首列为频率。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    freqs = np.asarray(spectra["freqs"], dtype=float)
    sv = np.asarray(spectra["sv"], dtype=float)
    diag = np.asarray(spectra["psd_diag"], dtype=float)
    header = ["freq"] + [f"sv{i}" for i in range(sv.shape[0])] + [f"psd{i}" for i in range(diag.shape[0])]
    mat = np.column_stack([freqs, sv.T, diag.T])
    np.savetxt(target, mat, delimiter=",", header=",".join(header), comments="")
    return target


def shape_covariances_from_tangent(cov, start: int, bases: list, channel_basis: list) -> list:
    """把切空间坐标块的协方差映射回通道空间，逐阶给出协方差矩阵。

    ``start`` 为第一个形状块在 ``cov`` 中的起始下标；``channel_basis[i]`` 为
    ``None`` 时表示形状向量本身即通道空间向量，否则按子空间基映射。
    """
    from .posterior import _channel_map

    cov = np.asarray(cov, dtype=float)
    out = []
    pos = int(start)
    for i, base in enumerate(bases):
        width = int(base.shape[1])
        blk = cov[pos : pos + width, pos : pos + width]
        mat = _channel_map(base, channel_basis[i])
        out.append(mat @ blk @ mat.T)
        pos += width
    return out


def to_jsonable(value):
    """把结果分组里的数组与嵌套结构转成可直接 ``json.dumps`` 的形式。

    规则与结果文件的其余部分一致：实数数组转嵌套列表（一维仍为列表），复数数组转
    ``{"re": …, "im": …}``，``None`` 与标量原样返回，字典与序列递归处理。
    """
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"re": np.real(value).tolist(), "im": np.imag(value).tolist()}
        return np.asarray(value, dtype=float).tolist()
    if isinstance(value, np.generic):
        return to_jsonable(np.asarray(value))
    return value


def asynchronous_multi_layout(m: int, ng: int, nch: int, cov_size: int, n_pairs: int) -> dict:
    """由协方差维数反推异步多阶的自由参数排布。

    ``n_spec = 3m + n_off + ng``，其中 ``n_off = 2 * m(m-1)/2``（单测组且多阶时
    ``S`` 取复 Hermitian）或 ``m(m-1)/2``（其余情形）；形状块总宽为
    ``cov_size - n_spec - 2 * npair * n_pairs``，单测组多阶时每阶 ``2 m_0 - 2``、
    其余每阶 ``nch - 1``，由此定出每阶宽度与子空间维数 ``m_0``。
    """
    ntri = m * (m - 1) // 2
    complex_s = bool(ng == 1 and m > 1)
    n_off = 2 * ntri if complex_s else ntri
    n_spec = 3 * m + n_off + ng
    npair = ng * (ng - 1) // 2
    shape_total = int(cov_size) - n_spec - 2 * npair * int(n_pairs)
    if shape_total < 0:
        raise ValueError(
            f"协方差维数 {cov_size} 小于谱参数与相干参数所占的 {n_spec + 2 * npair * n_pairs}"
        )
    if complex_s:
        if m == 0 or shape_total % (2 * m) != 0:
            raise ValueError(f"单测组多阶的形状块宽度 {shape_total} 不能被 2m = {2 * m} 整除")
        m0 = shape_total // (2 * m) + 1
        widths = [2 * m0 - 2] * m
        subspace_dim = int(m0)
    else:
        if m == 0 or shape_total % m != 0:
            raise ValueError(f"形状块宽度 {shape_total} 不能被阶数 {m} 整除")
        widths = [shape_total // m] * m
        subspace_dim = int(widths[0] + 1)
    return {
        "m": int(m),
        "ng": int(ng),
        "nch": int(nch),
        "n_off": int(n_off),
        "complex": complex_s,
        "n_spec": int(n_spec),
        "npair": int(npair),
        "n_tan": widths,
        "subspace_dim": subspace_dim,
        "shape_start": int(n_spec),
    }


def geometry_from_spec(spec_geometry: dict | None) -> dict | None:
    """从描述文件的 ``geometry`` 段取几何；缺省返回 ``None``。"""
    if not spec_geometry:
        return None
    return spec_geometry


def asynchronous_single_index(nch: int, ng: int) -> list[dict]:
    """异步单阶协方差的参数排列描述。

    排列与 :func:`bayoma.asynchronous.posterior_covariance` 一致：
    ``[f, z, S, Se_1..Se_ng, theta (nch-1 维), {Re v_ij, Im v_ij} (每对测组 2 项)]``。
    """
    items = [
        {"name": "f[0]", "kind": "f", "mode": 0, "symbol": "f", "unit": "Hz"},
        {"name": "z[0]", "kind": "z", "mode": 0, "symbol": "zeta", "unit": "-"},
        {"name": "S[0][0]", "kind": "S_diag", "mode": 0, "symbol": "S", "unit": "-"},
    ]
    for g in range(ng):
        items.append({"name": f"Se[{g}]", "kind": "Se", "mode": None, "group": g, "symbol": "S_e", "unit": "-"})
    for k in range(max(nch - 1, 0)):
        items.append(
            {
                "name": f"phi[0].theta[{k}]",
                "kind": "phi_tangent",
                "mode": 0,
                "coord": k,
                "symbol": "theta",
                "unit": "-",
            }
        )
    for i in range(ng):
        for j in range(i + 1, ng):
            items.append(
                {
                    "name": f"Rev[{i}][{j}]",
                    "kind": "v_re",
                    "mode": None,
                    "group": i,
                    "group_pair": j,
                    "symbol": "Re v",
                    "unit": "-",
                }
            )
            items.append(
                {
                    "name": f"Imv[{i}][{j}]",
                    "kind": "v_im",
                    "mode": None,
                    "group": i,
                    "group_pair": j,
                    "symbol": "Im v",
                    "unit": "-",
                }
            )
    return items


def asynchronous_multi_index(
    m: int,
    ng: int,
    n_off: int,
    complex_s: bool,
    n_tan: list[int],
    pairs: list,
) -> list[dict]:
    """异步多阶协方差的参数排列描述。

    排列与 :func:`bayoma.asynchronous_multi.posterior_covariance` 一致：
    ``[f (m), z (m), diag(S) (m), S_off (n_off；复情形再翻一倍), Se (ng),
    theta_i (各阶 n_tan_i), 每阶对的 {模 (npair), 相位 (npair)}]``。
    """
    items: list[dict] = []
    iu = np.triu_indices(m, 1)
    for i in range(m):
        items.append({"name": f"f[{i}]", "kind": "f", "mode": i, "symbol": "f", "unit": "Hz"})
    for i in range(m):
        items.append({"name": f"z[{i}]", "kind": "z", "mode": i, "symbol": "zeta", "unit": "-"})
    for i in range(m):
        items.append({"name": f"S[{i}][{i}]", "kind": "S_diag", "mode": i, "symbol": "S", "unit": "-"})
    ntri = int(len(iu[0]))
    if complex_s:
        for k in range(ntri):
            items.append(
                {
                    "name": f"ReS[{iu[0][k]}][{iu[1][k]}]",
                    "kind": "S_off_re",
                    "mode": None,
                    "modes": [int(iu[0][k]), int(iu[1][k])],
                    "symbol": "Re S",
                    "unit": "-",
                }
            )
        for k in range(ntri):
            items.append(
                {
                    "name": f"ImS[{iu[0][k]}][{iu[1][k]}]",
                    "kind": "S_off_im",
                    "mode": None,
                    "modes": [int(iu[0][k]), int(iu[1][k])],
                    "symbol": "Im S",
                    "unit": "-",
                }
            )
    else:
        for k in range(n_off):
            items.append(
                {
                    "name": f"S[{iu[0][k]}][{iu[1][k]}]",
                    "kind": "S_off",
                    "mode": None,
                    "modes": [int(iu[0][k]), int(iu[1][k])],
                    "symbol": "S",
                    "unit": "-",
                }
            )
    for g in range(ng):
        items.append({"name": f"Se[{g}]", "kind": "Se", "mode": None, "group": g, "symbol": "S_e", "unit": "-"})
    for i in range(m):
        for k in range(int(n_tan[i])):
            items.append(
                {
                    "name": f"phi[{i}].theta[{k}]",
                    "kind": "phi_tangent",
                    "mode": i,
                    "coord": k,
                    "symbol": "theta",
                    "unit": "-",
                }
            )
    # 每阶对的排布为「先 npair 个模、再 npair 个相位」，与 posterior_covariance 的
    # 取段方式 `blk[:npair]` / `blk[npair:]` 逐项对应。
    for key in pairs:
        i, j = int(key[0]), int(key[1])
        for g in range(ng):
            for h in range(g + 1, ng):
                items.append(
                    {
                        "name": f"|v|[{i}][{j}][{g}][{h}]",
                        "kind": "v_mod",
                        "mode": None,
                        "modes": [i, j],
                        "group": g,
                        "group_pair": h,
                        "symbol": "|v|",
                        "unit": "-",
                    }
                )
        for g in range(ng):
            for h in range(g + 1, ng):
                items.append(
                    {
                        "name": f"phase v[{i}][{j}][{g}][{h}]",
                        "kind": "v_phase",
                        "mode": None,
                        "modes": [i, j],
                        "group": g,
                        "group_pair": h,
                        "symbol": "arg v",
                        "unit": "rad",
                    }
                )
    return items


# ---------------------------------------------------------------------------
# 结果组装：把上面各函数的结果拼成最终结果文件里的两个分组
# ---------------------------------------------------------------------------


def build_recognition(
    *,
    f: np.ndarray,
    z: np.ndarray,
    phi: np.ndarray,
    bands: np.ndarray,
    f0_list: list,
    n_modes_per_band: list[int],
    band_blocks: list[dict],
    data: np.ndarray | None = None,
    fs: float | None = None,
    channel_index: list[int] | None = None,
    geometry: dict | None = None,
    figure_dir: str | Path | None = None,
    with_spectra: bool = False,
    peaks: list | None = None,
    excluded: list | None = None,
) -> dict:
    """识别结果分组：频带、阶次、区分度、联合后验协方差、振型不确定性、谱、可视化。

    ``band_blocks`` 逐频带给出 ``{cov, cov_phi, m, n_off, sizes, band_index, offset,
    complex}``，用于拼联合协方差与逐阶振型不确定性。
    """
    f = np.asarray(f, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    phi = np.asarray(phi)
    m = f.size
    nch = phi.shape[0]
    channel_index = list(range(nch)) if channel_index is None else list(channel_index)

    band_of_band = [int(blk["band_index"]) for blk in band_blocks]
    offset_of_band = {int(blk["band_index"]): int(blk["offset"]) for blk in band_blocks}
    nb = int(np.asarray(bands).shape[0])
    nb = max(nb, max(band_of_band, default=0) + 1)
    per_order_band = [0] * m
    offsets = [0] * nb
    for bi, off in offset_of_band.items():
        if 0 <= bi < nb:
            offsets[bi] = off
    for blk in band_blocks:
        off = int(blk["offset"])
        for k in range(int(blk["m"])):
            if off + k < m:
                per_order_band[off + k] = int(blk["band_index"])

    tables = band_table(bands, f0_list, n_modes_per_band, peaks=peaks, excluded=excluded)
    modes = mode_band_map(f, per_order_band, offsets, bands, f0_list)
    order_band = np.asarray(per_order_band, dtype=int)
    joint = joint_covariance(band_blocks)
    f_positions = [item["index"] for item in joint["index"] if item["kind"] == "f"]
    separation = separation_table(f, z, phi, order_band, joint["joint"], f_positions)
    shapes = []
    for blk in band_blocks:
        off = int(blk["offset"])
        for k, cov_phi in enumerate(blk.get("cov_phi", [])):
            if off + k >= m:
                break
            shapes.append((off + k, np.asarray(cov_phi)))
    shapes.sort(key=lambda item: item[0])
    shape_unc = shape_uncertainty([item[1] for item in shapes], phi, channel_index)

    out = {
        "bands": tables,
        "modes": modes,
        "separation": separation,
        "joint_covariance": {k: v for k, v in joint.items() if k != "joint"},
        "joint_covariance_matrix": joint["joint"],
        "shape_uncertainty": shape_unc,
        "spectra": None,
        "spatial": None,
    }
    if with_spectra and data is not None and fs is not None:
        spec = spectra_table(data, fs, bands, f0_list)
        out["spectra"] = {k: v for k, v in spec.items() if k not in ("freqs", "psd_diag", "sv")}
        out["spectra"]["freqs"] = spec["freqs"]
        out["spectra"]["psd_diag"] = spec["psd_diag"]
        out["spectra"]["sv"] = spec["sv"]
    if geometry is not None:
        out["spatial"] = spatial_figures(phi, geometry, figure_dir, prefix="mode")
    return out


def build_assembly(
    *,
    phi: np.ndarray,
    cov: np.ndarray,
    index: list[dict],
    channel_index: list[int] | None = None,
    geometry: dict | None = None,
    figure_dir: str | Path | None = None,
    setups: list[dict] | None = None,
    order_labels: list[str] | None = None,
    extra: dict | None = None,
    shape_covariances: list | None = None,
    shape_bases: list | None = None,
    shape_channel_basis: list | None = None,
) -> dict:
    """组装结果分组：全局振型、识别与组装不确定性的合成、后验协方差与索引。

    ``index`` 为协方差矩阵的参数描述（每项含 ``name`` 与 ``kind``），来自
    :func:`asynchronous_single_index` 或 :func:`asynchronous_multi_index` 等。振型
    对应 ``kind == "phi_tangent"`` 的项，其余参数记为 ``theta``。

    全局振型的边际协方差取 ``C_uu``；其中可由 ``theta`` 线性解释的部分记为组装阶段
    引入的不确定性，余下为识别本身的贡献。两项都是协方差矩阵，迹用于标量汇总。这是
    高斯近似下的条件方差分解，不是方差分量估计。
    """
    phi = np.asarray(phi)
    nch = phi.shape[0]
    channel_index = list(range(nch)) if channel_index is None else list(channel_index)
    cov = np.asarray(cov, dtype=float)
    if len(index) != cov.shape[0]:
        raise ValueError(f"参数描述 {len(index)} 项与协方差维数 {cov.shape[0]} 不符")

    shape_idx = [i for i, item in enumerate(index) if item.get("kind") == "phi_tangent"]
    theta_idx = [i for i, item in enumerate(index) if item.get("kind") != "phi_tangent"]
    dec = conditional_decomposition(cov, shape_idx, theta_idx)

    entries = []
    for pos, item in enumerate(index):
        rec = dict(item)
        rec["index"] = pos
        if rec.get("mode") is not None and order_labels is not None:
            mode = int(rec["mode"])
            if 0 <= mode < len(order_labels):
                rec["order_label"] = order_labels[mode]
        entries.append(rec)

    shape_cov = dec["total"]
    total_trace = max(dec["trace_total"], 0.0)

    # 逐阶给出通道空间的振型协方差：按各阶切空间宽度切块，再用子空间基映射回通道空间。
    per_mode: list[dict] = []
    shape_covs: list | None = None
    if shape_bases:
        from .posterior import _channel_map

        pos = 0
        shape_covs = []
        for i, base in enumerate(shape_bases):
            width = int(np.asarray(base).shape[1])
            blk = dec["total"][pos : pos + width, pos : pos + width]
            cb = None if shape_channel_basis is None else shape_channel_basis[i]
            shape_covs.append(_channel_map(np.asarray(base), cb) @ blk @ _channel_map(np.asarray(base), cb).T)
            pos += width
    if shape_covs is None and shape_covariances is not None:
        shape_covs = [np.asarray(mat, dtype=float) for mat in shape_covariances]
    if shape_covs:
        for i, mat in enumerate(shape_covs):
            mat = np.asarray(mat, dtype=float)
            if mat.shape[0] == 2 * nch:
                # 复形状：协方差按「实部在前、虚部在后」排布，逐通道取两块对角之和
                var = np.maximum(np.diag(mat)[:nch] + np.diag(mat)[nch:], 0.0)
            else:
                var = np.maximum(np.diag(mat), 0.0)
            m_std = np.sqrt(var)
            per_mode.append(
                {
                    "order": i,
                    "complex": bool(mat.shape[0] == 2 * nch),
                    "coefv": _f(math.sqrt(max(float(np.trace(mat)), 0.0))),
                    "channel_index": list(channel_index),
                    "std_per_channel": [_f(v) for v in m_std],
                    "ci95_halfwidth_per_channel": [_f(1.96 * v) for v in m_std],
                    "shape": None,
                    "cov": mat,
                }
            )
    else:
        per_mode.append(
            {
                "order": 0,
                "complex": False,
                "coefv": _f(math.sqrt(total_trace)),
                "channel_index": list(channel_index),
                "std_per_channel": [0.0] * nch,
                "ci95_halfwidth_per_channel": [0.0] * nch,
                "shape": None,
                "cov": shape_cov,
            }
        )

    # 顶层逐通道标准差按各阶方差的算术平均给出：单阶即该阶的标准差，多阶为各阶的
    # 平均不确定性，便于调用方在不区分阶次时也有一组通道级数字。
    chan_var = np.zeros(nch, dtype=float)
    for item in per_mode:
        vals = item["std_per_channel"]
        if len(vals) == nch:
            chan_var += np.square(np.asarray([0.0 if v is None else v for v in vals], dtype=float))
    if per_mode:
        chan_var = chan_var / float(len(per_mode))
    top_std = np.sqrt(np.maximum(chan_var, 0.0))

    out = {
        "phi": phi,
        "channel_index": channel_index,
        "shape_uncertainty": {
            "coefv": _f(math.sqrt(total_trace)),
            "std_per_channel": [_f(v) for v in top_std],
            "ci95_halfwidth_per_channel": [_f(1.96 * v) for v in top_std],
            "channel_index": channel_index,
            "per_mode": per_mode,
        },
        "uncertainty": {
            "method": "高斯近似下的条件方差分解：C_uu = C_uθ C_θθ^+ C_θu + 余项",
            "total_coefv": _f(math.sqrt(total_trace)),
            "assembly_coefv": _f(math.sqrt(max(dec["trace_from_assembly"], 0.0))),
            "recognition_coefv": _f(math.sqrt(max(dec["trace_residual"], 0.0))),
            "trace_total": _f(dec["trace_total"]),
            "trace_assembly": _f(dec["trace_from_assembly"]),
            "trace_recognition": _f(dec["trace_residual"]),
            "note": "组装项为可由其余参数（测组噪声谱、相干矩阵等）线性解释的振型方差；"
            "识别项为余下方差。两项按迹相加等于总方差。",
        },
        "covariance_total": dec["total"],
        "covariance_from_assembly": dec["from_assembly"],
        "covariance_recognition": dec["residual"],
        "posterior_covariance": cov,
        "index": entries,
        "setups": setups,
        "spatial": None,
    }
    if geometry is not None:
        # 单阶传一维、多阶传 n×m 矩阵，统一成列矩阵后再画，避免三维广播
        shapes = phi[:, None] if phi.ndim == 1 else phi
        out["spatial"] = spatial_figures(shapes, geometry, figure_dir, prefix="assembly")
    if extra:
        out.update(extra)
    return out
