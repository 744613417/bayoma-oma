"""机器可读接口层：以 JSON 描述任务，供其余软件与智能体直接调用。

本模块把总入口 :func:`bayoma.api.bayoma` 包装成「读一份 JSON 描述、写一份
JSON 结果」的形式。数据本身不进 JSON（记录长度可达百万点），JSON 只给出数据
文件路径与读取方式，避免描述文件膨胀。

描述文件的字段见 :func:`run_from_dict` 的文档；结果文件的字段见
:func:`result_to_dict`。两套实现（Python 与 MATLAB）使用同一份描述与同一套
结果字段，便于交叉比对。

命令行用法::

    python -m bayoma.io_json job.json
    python -m bayoma.io_json job.json --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .api import bayoma
from .asynchronous import mpv as mpv_async
from .asynchronous import posterior_covariance as async_covariance
from .asynchronous_multi import mpv as mpv_async_multi
from .asynchronous_multi import posterior_covariance as async_multi_covariance
from .data import scaled_fft
from . import results

__all__ = ["load_data", "run_from_dict", "result_to_dict", "run_file"]


def _as_complex_json(value: Any) -> Any:
    """把复数（含复数数组）转成 ``{"re":…, "im":…}`` 形式，便于 JSON 承载。"""
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        return {
            "re": np.real(arr).tolist(),
            "im": np.imag(arr).tolist(),
        }
    return arr.tolist()


def load_data(spec: dict) -> np.ndarray:
    """按描述读取数据矩阵。

    Parameters
    ----------
    spec : dict
        字段：

        - ``path``：数据文件路径；
        - ``format``：``"csv"``、``"npy"``、``"mat"`` 之一，缺省按扩展名推断；
        - ``key``：``mat`` 格式下变量名，缺省 ``"tdata"``；
        - ``delimiter``：``csv`` 格式的分隔符，缺省逗号；
        - ``skip_header``：``csv`` 跳过的行数，缺省 0。
    """
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"数据文件不存在：{path}")
    fmt = spec.get("format") or path.suffix.lower().lstrip(".")
    fmt = {"txt": "csv", "dat": "csv"}.get(fmt, fmt)

    if fmt == "csv":
        data = np.loadtxt(
            path,
            delimiter=spec.get("delimiter", ","),
            skiprows=int(spec.get("skip_header", 0)),
        )
    elif fmt == "npy":
        data = np.load(path)
    elif fmt == "mat":
        import scipy.io as sio

        key = spec.get("key", "tdata")
        content = sio.loadmat(str(path))
        if key not in content:
            raise KeyError(f"mat 文件 {path} 中没有变量 {key}")
        data = np.asarray(content[key], dtype=float)
    else:
        raise ValueError(f"不支持的数据格式：{fmt}")

    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("数据必须是二维矩阵，行对应时刻、列对应通道")
    return data


def run_from_dict(spec: dict) -> dict:
    """按描述执行识别，返回结果字典。

    Parameters
    ----------
    spec : dict
        字段：

        - ``data``：数据来源，见 :func:`load_data`；
        - ``fs``：采样频率（Hz）；若 ``data`` 内已给出则以 ``data`` 为准；
        - ``bands``：识别频带，``[[下界, 上界], …]``；缺省时自动选带；
        - ``f0``：各频带内固有频率初值，``[[f1, f2], …]``；
        - ``options``：``q``、``p``、``maxiter``、``mute``、``ichan``、``iband``、
          ``f1f2d``，含义见 :func:`bayoma.api.bayoma`；
        - ``multi_setup``：多测组识别描述（可选），字段 ``n_dof``、``setups``，
          每项含 ``data``、``dofs``、``bands``、``f0``；给出时按多测组处理；
        - ``asynchronous``：异步数据识别描述（可选），字段 ``data``、``groups``、
          ``bands``、``f0``、``q``；给出时按第 10.4 节的异步模型处理；
        - ``output``：``{"path": …}``，给出时把结果写入该文件。
    """
    if "asynchronous" in spec and spec["asynchronous"]:
        raw = _run_asynchronous(spec)
        out = result_to_dict(raw)
        data = load_data(spec["asynchronous"]["data"])
        fs_a = float((spec["asynchronous"].get("data") or {}).get(
            "fs", spec.get("fs", 0.0)
        ))
        out["input"] = {
            "fs": fs_a,
            "nch": int(data.shape[1]),
            "nt": int(data.shape[0]),
            "duration": float(data.shape[0] / fs_a) if fs_a > 0 else 0.0,
            "n_group": len(raw["groups"]),
        }
        bands_a = (spec["asynchronous"] or {}).get("bands") or spec.get("bands")
        if bands_a:
            f0_a = (spec["asynchronous"] or {}).get("f0") or spec.get("f0") or [[]]
            n_modes_a = int((spec["asynchronous"] or {}).get("n_modes", 1))
            out.setdefault("recognition", {})
            out["recognition"]["bands"] = results.to_jsonable(
                results.band_table(
                    np.asarray(bands_a, dtype=float),
                    [np.atleast_1d(np.asarray(v, dtype=float)) for v in f0_a],
                    [n_modes_a] * len(np.atleast_2d(bands_a)),
                )
            )
            out["recognition"]["modes"] = [
                {"order": i, "band_index": 0, "order_in_band": i}
                for i in range(out.get("n_modes", 0))
            ]
            # 异步识别同样给出区分度、联合后验协方差与逐阶振型不确定性：单频带时
            # 频带内各阶共用一张协方差，直接取自组装分组的同一组参数描述。
            asm = out.get("assembly") or {}
            f_a = np.asarray(out["f"], dtype=float)
            z_a = np.asarray(out["z"], dtype=float)
            phi_a = np.asarray(raw["phi"], dtype=float)
            out["recognition"]["separation"] = results.to_jsonable(
                results.separation_table(
                    f_a, z_a, phi_a, np.zeros(f_a.size, dtype=int)
                )
            )
            out["recognition"]["joint_covariance"] = {
                "index": results.to_jsonable(asm.get("index", [])),
                "size": int(np.asarray(asm.get("posterior_covariance", [[]])).shape[0])
                if asm
                else 0,
                "structure": "whole",
                "note": "异步数据在单频带内一次给出全部参数，协方差为整体矩阵，不分块。",
            }
            if asm:
                out["recognition"]["joint_covariance_matrix"] = asm[
                    "posterior_covariance"
                ]
                out["recognition"]["shape_uncertainty"] = results.to_jsonable(
                    asm.get("shape_uncertainty", {}).get("per_mode", [])
                )
        target = (spec.get("output") or {}).get("path")
        if target:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            Path(target).write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return out
    if "multi_setup" in spec and spec["multi_setup"]:
        # 多测组描述的采样频率可写在顶层 fs，也可写在任一个测组的数据来源里
        fs_ms = float(spec.get("fs", 0.0))
        if fs_ms <= 0.0:
            for item in spec["multi_setup"]["setups"]:
                cand = item.get("data") or {}
                fs_ms = float(cand.get("fs", 0.0))
                if fs_ms > 0.0:
                    break
        if fs_ms <= 0.0:
            raise ValueError("多测组识别必须给出正的采样频率 fs")
        raw = _run_multi_setup(spec, fs_ms)
        out = result_to_dict(raw)
        nt = 0
        for item in spec["multi_setup"]["setups"]:
            nt = max(nt, int(load_data(item["data"]).shape[0]))
        out["input"] = {
            "fs": fs_ms,
            "nch": int(raw["nch"]),
            "nt": nt,
            "duration": float(nt / fs_ms) if fs_ms > 0 else 0.0,
            "n_setup": int(raw.get("n_setup", len(spec["multi_setup"]["setups"]))),
        }
        target = (spec.get("output") or {}).get("path")
        if target:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            Path(target).write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return out

    data_spec = spec["data"]
    data = load_data(data_spec)
    fs = float(data_spec.get("fs", spec.get("fs", 0.0)))
    if fs <= 0.0:
        raise ValueError("必须给出正的采样频率 fs")

    opts = dict(spec.get("options") or {})
    figure_dir = (spec.get("figures") or {}).get("dir")
    if figure_dir:
        Path(figure_dir).mkdir(parents=True, exist_ok=True)
    result = bayoma(
        data,
        fs,
        f1f2=spec.get("bands"),
        f0=spec.get("f0"),
        geometry=spec.get("geometry"),
        figure_dir=figure_dir,
        spectra=bool(opts.pop("spectra", spec.get("spectra", False))),
        **opts,
    )
    out = result_to_dict(result)
    out["input"] = {
        "fs": fs,
        "nch": int(data.shape[1]),
        "nt": int(data.shape[0]),
        "duration": float(data.shape[0] / fs),
    }
    target = (spec.get("output") or {}).get("path")
    if target:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _run_multi_setup(spec: dict, fs: float) -> dict:
    """按描述执行多测组识别，返回与单测组同构的结果字典。"""
    from .data import scaled_fft
    from .multi_setup import SetupData, mpv
    from .posterior import multi_setup_covariance

    ms = spec["multi_setup"]
    n_dof = int(ms["n_dof"])
    setups = []
    for item in ms["setups"]:
        arr = load_data(item["data"])
        bands = item.get("bands")
        if bands is None:
            raise ValueError("多测组的每个测组都需要给出识别频带")
        lo, hi = float(bands[0][0]), float(bands[0][1])
        sp = scaled_fft(arr, fs, (lo, hi))
        setups.append(SetupData(freqs=sp.freqs, fft=sp.fft, dofs=item["dofs"]))

    q = int((spec.get("options") or {}).get("q", 0))
    res = mpv(setups, n_dof, q=q)
    cov = multi_setup_covariance(res, setups, q=q)
    n_setup = int(np.asarray(res.f).size)
    coefv_f = np.asarray(cov.coefv_named["f"], dtype=float).reshape(-1)
    coefv_z = np.asarray(cov.coefv_named["z"], dtype=float).reshape(-1)
    coefv_s = np.asarray(cov.coefv_named["S"], dtype=float).reshape(-1)
    coefv_se = np.asarray(cov.coefv_named["Se"], dtype=float).reshape(-1)
    f_arr = np.asarray(res.f, dtype=float)
    z_arr = np.asarray(res.z, dtype=float)
    s_arr = np.real(np.asarray(res.S, dtype=complex))
    coefv_rms = 0.5 * np.sqrt(
        coefv_f ** 2 + coefv_s ** 2 + coefv_z ** 2
    )
    return {
        "f": f_arr,
        "z": z_arr,
        "Sii": s_arr,
        "Se": np.asarray(res.Se, dtype=float),
        "phi": np.asarray(res.phi, dtype=float)[:, None],
        "rms": np.sqrt(np.pi * s_arr * f_arr / (4.0 * z_arr)),
        "S": [np.atleast_2d(np.asarray(s, dtype=complex)) for s in np.atleast_1d(res.S)],
        "coefv": {"f": coefv_f, "z": coefv_z, "S": coefv_s, "Se": coefv_se,
                  "phi": float(cov.coefv_named["phi"]), "rms": coefv_rms},
        "cmat": [np.asarray(cov.cov, dtype=float)],
        "sn": s_arr / (4.0 * np.asarray(res.Se) * z_arr ** 2),
        "time_req": np.zeros((2, 1)),
        "bands": np.array([[float(ms["setups"][0]["bands"][0][0]),
                            float(ms["setups"][0]["bands"][0][1])]]),
        "nch": n_dof,
        "fs": fs,
        "n_setup": n_setup,
    }


def _run_asynchronous(spec: dict) -> dict:
    """按描述执行异步数据识别，返回与单测组同构的结果字典。

    描述块内给出 ``n_modes``（缺省 1）时，``n_modes > 1`` 走第 10.4 节的多阶
    模型（各阶对共享独立的相干矩阵），否则走单阶模型。两条路径的输出字段与
    单测组同步识别同构，便于调用方统一处理。
    """
    raw = spec["asynchronous"]
    data = load_data(raw["data"])
    fs = float((raw.get("data") or {}).get("fs", spec.get("fs", 0.0)))
    if fs <= 0.0:
        raise ValueError("异步数据识别必须给出正的采样频率 fs")
    bands = raw.get("bands")
    if bands is None:
        raise ValueError("异步数据识别需要给出识别频带 bands")
    lo, hi = float(bands[0][0]), float(bands[0][1])
    groups = [[int(i) for i in g] for g in raw["groups"]]
    q = int(raw.get("q", (spec.get("options") or {}).get("q", 0)))

    sp = scaled_fft(data, fs, (lo, hi))
    f0 = raw.get("f0")
    if f0 is None:
        f0_list = []
    else:
        f0_list = [float(v) for v in np.atleast_1d(np.asarray(f0, dtype=float)).ravel()]
    n_modes = int(raw.get("n_modes", len(f0_list) if len(f0_list) > 1 else 1))
    if n_modes < 1:
        raise ValueError("异步数据识别的模态阶数 n_modes 必须为正整数")
    if n_modes > 1:
        return _run_asynchronous_multi(
            sp, data, fs, lo, hi, groups, q, f0_list, n_modes, spec
        )
    f0_one = f0_list[0] if f0_list else None
    res = mpv_async(sp.freqs, sp.fft, groups, q=q, f0=f0_one)
    cov, coefv = async_covariance(res, sp.freqs, sp.fft, groups, q=q)

    f_arr = np.array([float(res.f)])
    z_arr = np.array([float(res.z)])
    s_arr = np.array([float(res.S)])
    se_arr = np.asarray(res.Se, dtype=float).reshape(-1)
    coefv_f = float(np.atleast_1d(coefv["f"])[0])
    coefv_z = float(np.atleast_1d(coefv["z"])[0])
    coefv_s = float(np.atleast_1d(coefv["S"])[0])
    nch = int(data.shape[1])
    cov = np.asarray(cov, dtype=float)
    index = results.asynchronous_single_index(nch, len(groups))
    from .posterior import tangent_basis

    # 协方差的形状块在 ``u`` 的切空间里（nch-1 维），组装分组要按通道空间给逐通道标准差，
    # 因此把切空间基一并传入，由 ``build_assembly`` 映射回通道空间。
    assembly = results.build_assembly(
        phi=np.asarray(res.phi, dtype=float)[:, None],
        cov=cov,
        index=index,
        channel_index=list(range(nch)),
        geometry=results.geometry_from_spec(spec.get("geometry")),
        figure_dir=(spec.get("figures") or {}).get("dir"),
        setups=[{"group_index": g, "channels": list(groups[g])} for g in range(len(groups))],
        shape_bases=[tangent_basis(np.asarray(res.phi, dtype=float))],
        shape_channel_basis=[None],
    )
    assembly["coherence"] = results.to_jsonable(np.asarray(res.coherence, dtype=complex))
    return {
        "f": f_arr,
        "z": z_arr,
        "Sii": s_arr,
        "Se": se_arr,
        "phi": np.asarray(res.phi, dtype=float)[:, None],
        "rms": np.sqrt(np.pi * s_arr * f_arr / (4.0 * z_arr)),
        "S": [np.atleast_2d(s_arr.astype(complex))],
        "coefv": {
            # 与同步路径一致，各参数一律给一维数组，便于调用方统一按下标取值
            "f": np.atleast_1d(coefv_f),
            "z": np.atleast_1d(coefv_z),
            "S": np.atleast_1d(coefv_s),
            "Se": np.asarray(coefv["Se"], dtype=float).reshape(-1),
            "phi": np.atleast_1d(float(coefv["phi"])),
            "v": np.atleast_1d(float(coefv["v"])),
            "rms": np.atleast_1d(
                0.5 * np.sqrt(coefv_f ** 2 + coefv_z ** 2 + coefv_s ** 2)
            ),
        },
        "cmat": [cov],
        "sn": s_arr / (4.0 * se_arr.mean() * z_arr ** 2),
        "time_req": np.zeros((2, 1)),
        "bands": np.array([[lo, hi]]),
        "nch": nch,
        "fs": fs,
        "groups": groups,
        "coherence": _as_complex_json(np.asarray(res.coherence, dtype=complex)),
        "assembly": assembly,
    }


def _run_asynchronous_multi(sp, data, fs, lo, hi, groups, q, f0_list, n_modes, spec):
    """异步数据多阶识别，返回与单测组同构的结果字典。"""
    res = mpv_async_multi(sp.freqs, sp.fft, groups, n_modes, q=q,
                          f0=np.asarray(f0_list, dtype=float) if f0_list else None)
    cov, coefv = async_multi_covariance(res, sp.freqs, sp.fft, groups, q=q)

    f_arr = np.asarray(res.f, dtype=float).reshape(-1)
    z_arr = np.asarray(res.z, dtype=float).reshape(-1)
    smat = np.asarray(res.S, dtype=complex)
    se_arr = np.asarray(res.Se, dtype=float).reshape(-1)
    # 单测组且多阶时 ``res.phi`` 为复矩阵，多测组时为实矩阵；不强制 dtype，让
    # ``result_to_dict`` 按实际类型决定是否按 ``{"re":…, "im":…}`` 序列化。
    s_off = coefv.get("S_off")
    if s_off is None:
        s_off = np.zeros(max(n_modes * (n_modes - 1) // 2, 0), dtype=float)
    else:
        s_off = np.asarray(s_off).reshape(-1)
    v_cov = coefv.get("v") or {}
    # 单测组（ng = 1）时相干矩阵只有对角块，组间元个数为 0，``mod`` 为空数组；
    # 此处跳过空项，避免对零长数组取最大值。
    v_mods = [
        float(np.max(np.atleast_1d(item["mod"])))
        for item in v_cov.values()
        if np.size(item["mod"]) > 0
    ]
    v_cov_mean = float(np.mean(v_mods)) if v_mods else 0.0
    cov = np.asarray(cov, dtype=float)
    nch = int(data.shape[1])
    layout = results.asynchronous_multi_layout(
        m=n_modes,
        ng=len(groups),
        nch=nch,
        cov_size=int(cov.shape[0]),
        n_pairs=len(res.coherence) if hasattr(res.coherence, "__len__") else 0,
    )
    index = results.asynchronous_multi_index(
        m=n_modes,
        ng=len(groups),
        n_off=layout["n_off"],
        complex_s=layout["complex"],
        n_tan=layout["n_tan"],
        pairs=sorted(res.coherence) if isinstance(res.coherence, dict) else [],
    )
    from .posterior import tangent_basis

    shape_base = np.asarray(res.alpha, dtype=complex) if layout["complex"] else np.asarray(res.phi)
    if layout["complex"]:
        bases = [tangent_basis(shape_base[:, i]) for i in range(n_modes)]
        channel_basis = [np.asarray(res.basis, dtype=float)] * n_modes
    else:
        bases = [tangent_basis(shape_base[:, i]) for i in range(n_modes)]
        channel_basis = [None] * n_modes
    cov_phi = results.shape_covariances_from_tangent(
        cov, layout["shape_start"], bases, channel_basis
    )
    assembly = results.build_assembly(
        phi=np.asarray(res.phi),
        cov=cov,
        index=index,
        channel_index=list(range(nch)),
        geometry=results.geometry_from_spec(spec.get("geometry")),
        figure_dir=(spec.get("figures") or {}).get("dir"),
        setups=[{"group_index": g, "channels": list(groups[g])} for g in range(len(groups))],
    )
    assembly["shape_uncertainty_per_mode"] = [
        {
            "order": i,
            "coefv": float(
                np.sqrt(max(float(np.trace(np.asarray(cov_phi[i], dtype=float))), 0.0))
            ),
        }
        for i in range(n_modes)
    ]
    return {
        "f": f_arr,
        "z": z_arr,
        "Sii": np.real(np.diag(smat)),
        "Se": se_arr,
        "phi": np.asarray(res.phi),
        "rms": np.sqrt(np.pi * np.real(np.diag(smat)) * f_arr / (4.0 * z_arr)),
        "S": [smat],
        "coefv": {
            "f": np.asarray(coefv["f"], dtype=float).reshape(-1),
            "z": np.asarray(coefv["z"], dtype=float).reshape(-1),
            "S": np.real(np.diag(np.asarray(coefv["S"], dtype=float))),
            "S_off": s_off,
            "Se": np.asarray(coefv["Se"], dtype=float).reshape(-1),
            "phi": np.asarray(coefv["phi"], dtype=float).reshape(-1),
            "v": np.atleast_1d(v_cov_mean),
            "rms": 0.5 * np.sqrt(
                np.asarray(coefv["f"], dtype=float).reshape(-1) ** 2
                + np.asarray(coefv["z"], dtype=float).reshape(-1) ** 2
                + np.real(np.diag(np.asarray(coefv["S"], dtype=float))) ** 2
            ),
        },
        "cmat": [cov],
        "sn": np.real(np.diag(smat)) / (4.0 * se_arr.mean() * z_arr ** 2),
        "time_req": np.zeros((2, 1)),
        "bands": np.array([[lo, hi]]),
        "nch": nch,
        "fs": fs,
        "groups": groups,
        "coherence": _as_complex_json(
            np.asarray([res.coherence[key] for key in sorted(res.coherence)], dtype=complex)
        ),
        "assembly": assembly,
    }


def result_to_dict(result: dict) -> dict:
    """把总入口的返回字典转成可直接序列化的结构。

    复数矩阵按 ``{"re": …, "im": …}`` 形式给出；一维数组转为列表。
    """
    out: dict[str, Any] = {}
    for key in ("f", "z", "Sii", "Se", "rms", "sn"):
        out[key] = np.asarray(result[key], dtype=float).ravel().tolist()

    # 模态形状可能是复向量（异步单测组多阶、第 13 章复模态形状），按复数序列化，
    # 否则 dtype=float 会静默丢弃虚部。
    out["phi"] = _as_complex_json(np.asarray(result["phi"]))
    out["S"] = [_as_complex_json(mat) for mat in result["S"]]
    out["cmat"] = [np.asarray(mat, dtype=float).tolist() for mat in result["cmat"]]

    cv = result.get("coefv") or {}
    out["coefv"] = {
        name: (
            float(cv[name])
            if np.isscalar(cv[name])
            else _as_complex_json(np.asarray(cv[name]).ravel())
        )
        # 同步识别的总入口把模态力谱的变异系数命名为 ``Sii``（与输出字段同名），
        # 异步与多测组路径内部用 ``S``；两者都列出，按实际存在的键取值。
        for name in ("f", "z", "Sii", "S", "S_off", "Se", "phi", "rms", "v")
        if name in cv
    }
    out["bands"] = np.asarray(result["bands"], dtype=float).tolist()
    out["time_req"] = np.asarray(result["time_req"], dtype=float).tolist()
    out["nch"] = int(result["nch"])
    out["fs"] = float(result["fs"])
    out["n_modes"] = int(np.asarray(result["f"]).size)
    for extra in ("groups", "coherence"):
        if extra in result:
            out[extra] = result[extra]
    if result.get("recognition") is not None:
        out["recognition"] = results.to_jsonable(result["recognition"])
    if result.get("assembly") is not None:
        out["assembly"] = results.to_jsonable(result["assembly"])
    return out


def run_file(spec_path: str | Path, out_path: str | Path | None = None) -> dict:
    """读取描述文件、执行识别，并按需写出结果文件。"""
    t0 = time.perf_counter()
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if out_path is not None:
        spec.setdefault("output", {})["path"] = str(out_path)
    result = run_from_dict(spec)
    result["elapsed"] = time.perf_counter() - t0
    target = (spec.get("output") or {}).get("path")
    if target:
        Path(target).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        prog="bayoma.io_json",
        description="按 JSON 描述执行自动贝叶斯模态识别，结果写成 JSON。",
    )
    parser.add_argument("spec", help="描述文件路径")
    parser.add_argument("--out", default=None, help="结果文件路径，覆盖描述中的 output.path")
    args = parser.parse_args(argv)

    try:
        result = run_file(args.spec, args.out)
    except Exception as exc:  # noqa: BLE001 - 命令行需要把任何失败转成非零退出码
        print(f"识别失败：{exc}", file=sys.stderr)
        return 1

    print(f"识别完成：{result['n_modes']} 阶模态，用时 {result.get('elapsed', 0.0):.2f} s")
    for i, (freq, damp) in enumerate(zip(result["f"], result["z"]), start=1):
        print(f"  第 {i} 阶：f = {freq:.6f} Hz，z = {damp:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
