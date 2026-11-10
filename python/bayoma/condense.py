"""参考自由度凝聚与全局最小二乘（课本 14.2 节，式 14.1--14.20）。

本模块提供两类功能。

第一类用于多测组识别的全局振型求解。当同一自由度被两个以上通道测到时，
这些通道的局部振型在全局振型中的分量相等，全局振型的非参考部分可按
式 14.17 写成各测组识别振型的缩放组合，从而把未知量从 n 个降到
``n00 + q`` 个（式 14.17--14.20）。该式用于第 14.3.3 节贝叶斯算法的
第 2 步，对式 14.40 的矩阵作同样的子空间投影。

第二类给出 14.2 节的全局最小二乘解，用于以课本 Example 14.2 的已知结果
校核实现，也供不需要贝叶斯推断时直接调用。

符号与课本一致：``u`` 为全局振型（单位欧氏范数），``t_r = L_r u`` 为第 r
个测组的局部振型，``L_r`` 为由测点自由度编号确定的选择矩阵。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "reference_dofs",
    "condensation_matrix",
    "project_condensed",
    "global_least_squares",
]


def reference_dofs(dofs_per_setup):
    """返回被两个以上通道测到的自由度编号（0 起计，升序）。

    参数
    ----
    dofs_per_setup : sequence of sequence of int
        每个测组的自由度编号列表，0 起计。

    返回
    ----
    list of int
        参考自由度编号；无重复测点时返回空列表。
    """
    counts = {}
    for dofs in dofs_per_setup:
        for dof in np.asarray(list(dofs), dtype=int).ravel():
            counts[int(dof)] = counts.get(int(dof), 0) + 1
    return sorted([dof for dof, cnt in counts.items() if cnt > 1])


def condensation_matrix(dofs_per_setup, n_dof):
    """构造式 14.17 的凝聚矩阵 ``L_c``。

    全局振型按 ``u = [u_R; u0_1; ...; u0_q]`` 排列，``u_R`` 为参考自由度
    分量（按 ``reference_dofs`` 的升序），``u0_r`` 为第 r 个测组的非参考
    自由度分量（按该测组内出现顺序）。凝聚参数 ``u_c = [u_R; a_1; ...; a_q]``，
    其中 ``a_r`` 复制第 r 个测组的非参考分量。

    参数
    ----
    dofs_per_setup : sequence of sequence of int
        每个测组的自由度编号列表，0 起计。
    n_dof : int
        结构全部自由度个数。

    返回
    ----
    L_c : ndarray, shape (n_dof, n_condensed)
        满足 ``L_c.T @ L_c = I``。
    info : dict
        ``reference`` 为参考自由度列表，``non_reference`` 为按测组顺序拼接的
        非参考自由度列表，``n_condensed`` 为凝聚参数个数。
    """
    n_dof = int(n_dof)
    dofs_per_setup = [np.asarray(list(dofs), dtype=int).ravel() for dofs in dofs_per_setup]
    reference = reference_dofs(dofs_per_setup)
    ref_set = set(reference)

    columns = []
    for dof in reference:
        col = np.zeros(n_dof)
        col[dof] = 1.0
        columns.append(col)

    non_reference = []
    for dofs in dofs_per_setup:
        extra = [int(dof) for dof in dofs if int(dof) not in ref_set]
        non_reference.extend(extra)
        for dof in extra:
            col = np.zeros(n_dof)
            col[dof] = 1.0
            columns.append(col)

    if columns:
        matrix = np.column_stack(columns)
    else:
        matrix = np.zeros((n_dof, 0))

    info = {
        "reference": reference,
        "non_reference": non_reference,
        "n_condensed": int(matrix.shape[1]),
    }
    return matrix, info


def project_condensed(matrix, basis):
    """在凝聚子空间内求解最小特征值问题。

    输入 ``matrix`` 为式 14.40 的 ``B``（对称），``basis`` 为 ``L_c``。
    返回最小特征值对应的特征向量 ``u``（已归一化）与特征值。

    参数
    ----
    matrix : ndarray, shape (n_dof, n_dof)
        待求最小特征向量的对称矩阵。
    basis : ndarray, shape (n_dof, n_c)
        凝聚矩阵。

    返回
    ----
    u : ndarray, shape (n_dof,)
        单位欧氏范数的全局振型。
    lam : float
        对应特征值。
    """
    matrix = np.asarray(matrix, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if basis.shape[1] == 0:
        values, vectors = np.linalg.eigh(matrix)
        u = vectors[:, int(np.argmin(values))]
        return u / np.linalg.norm(u), float(np.min(values))
    reduced = basis.T @ matrix @ basis
    reduced = 0.5 * (reduced + reduced.T)
    values, vectors = np.linalg.eigh(reduced)
    order = int(np.argmin(values))
    u = basis @ vectors[:, order]
    norm = float(np.linalg.norm(u))
    if norm > 0.0:
        u = u / norm
    return u, float(values[order])


def _smallest_real_eigenvector(matrix, n_head):
    """取实特征值最小者的特征向量前 ``n_head`` 个分量。"""
    values, vectors = np.linalg.eig(matrix)
    scale = np.maximum(np.abs(values.real), 1.0)
    mask = np.abs(values.imag) <= 1e-8 * scale
    index = np.flatnonzero(mask)
    if index.size == 0:
        index = np.arange(values.size)
    order = index[int(np.argmin(values.real[index]))]
    return np.real(vectors[:n_head, order])


def global_least_squares(tilde_shapes, selection, weights=None, condense=True, max_iter=200, tol=1e-10):
    """式 14.3--14.14 的全局最小二乘解，可按式 14.17--14.20 凝聚。

    参数
    ----
    tilde_shapes : sequence of ndarray
        各测组识别并归一的局部振型 ``~t_r``，长度 ``n_r``。
    selection : sequence of ndarray
        各测组的选择矩阵 ``L_r``，形状 ``(n_r, n)``。
    weights : sequence of float, optional
        权重 ``w_r``，缺省取等权 ``1/q``。
    condense : bool, optional
        是否启用参考自由度凝聚，缺省启用。
    max_iter : int, optional
        迭代上限，缺省 200。
    tol : float, optional
        相邻两次迭代全局振型的最大分量差，缺省 ``1e-10``。

    返回
    ----
    dict
        键 ``u``（全局振型）、``c``（各测组范数平方）、``lam``（各测组乘子）、
        ``n_iter``、``converged``、``condensed``（是否凝聚）、``info``。
    """
    tilde_list = [np.asarray(t, dtype=float).ravel() for t in tilde_shapes]
    select_list = [np.asarray(sel, dtype=float) for sel in selection]
    n_setup = len(tilde_list)
    if len(select_list) != n_setup:
        raise ValueError("tilde_shapes 与 selection 的长度必须一致")
    n_dof = int(select_list[0].shape[1])
    for sel, tilde in zip(select_list, tilde_list):
        if sel.shape[0] != tilde.size:
            raise ValueError("每个选择矩阵的行数必须等于对应局部振型的长度")

    if weights is None:
        w = np.full(n_setup, 1.0 / n_setup)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != n_setup:
            raise ValueError("weights 的长度必须等于测组数")
        w = w / float(np.sum(w))

    # 式 14.14：A0 的最小特征向量作为初值
    a0 = np.zeros((n_dof, n_dof))
    for wr, sel, tilde in zip(w, select_list, tilde_list):
        a0 += wr * (sel.T @ (np.eye(tilde.size) - np.outer(tilde, tilde)) @ sel)
    values, vectors = np.linalg.eigh(0.5 * (a0 + a0.T))
    u = vectors[:, int(np.argmin(values))]
    u = u / np.linalg.norm(u)

    basis = None
    info = {"reference": [], "non_reference": [], "n_condensed": n_dof}
    if condense:
        dofs_per_setup = [np.flatnonzero(sel.any(axis=0)) for sel in select_list]
        basis, info = condensation_matrix(dofs_per_setup, n_dof)

    c = np.zeros(n_setup)
    lam = np.zeros(n_setup)
    converged = False
    n_iter = 0
    for n_iter in range(1, int(max_iter) + 1):
        # 式 14.6：由当前 u 更新 c_r 与 lambda_r
        for idx, (wr, sel, tilde) in enumerate(zip(w, select_list, tilde_list)):
            t = sel @ u
            c[idx] = float(t @ t)
            norm_t = np.sqrt(c[idx])
            lam[idx] = wr * (1.0 - abs(float(tilde @ t)) / norm_t) if norm_t > 0.0 else wr
        lam = np.clip(lam, 0.0, w)

        # 式 14.8--14.9：构造 A 与 b。局部振型的整体正负号不可辨识，
        # 每次迭代按当前的 u 把它们对齐到 t~_r^T L_r u > 0，否则 b 项符号相反。
        a_mat = np.zeros((n_dof, n_dof))
        b_vec = np.zeros(n_dof)
        for idx, (wr, sel, tilde) in enumerate(zip(w, select_list, tilde_list)):
            t_cur = sel @ u
            sgn = 1.0 if float(tilde @ t_cur) >= 0.0 else -1.0
            a_mat += (wr + lam[idx]) * (sel.T @ sel)
            b_vec += -wr * np.sqrt(max(c[idx], 0.0)) * (sel.T @ (sgn * tilde))

        # 式 14.12 或 14.20：最小实特征值问题
        if basis is not None:
            a_red = basis.T @ a_mat @ basis
            b_red = basis.T @ b_vec
            size = a_red.shape[0]
            d_mat = np.zeros((2 * size, 2 * size))
            d_mat[:size, :size] = a_red
            d_mat[:size, size:] = np.outer(b_red, b_red)
            d_mat[size:, :size] = np.eye(size)
            d_mat[size:, size:] = a_red
            u_new = basis @ _smallest_real_eigenvector(d_mat, size)
        else:
            d_mat = np.zeros((2 * n_dof, 2 * n_dof))
            d_mat[:n_dof, :n_dof] = a_mat
            d_mat[:n_dof, n_dof:] = np.outer(b_vec, b_vec)
            d_mat[n_dof:, :n_dof] = np.eye(n_dof)
            d_mat[n_dof:, n_dof:] = a_mat
            u_new = _smallest_real_eigenvector(d_mat, n_dof)

        norm_u = float(np.linalg.norm(u_new))
        if norm_u > 0.0:
            u_new = u_new / norm_u
        step = float(np.max(np.abs(u_new - u)))
        u = u_new
        if step < tol:
            converged = True
            break

    result = {
        "u": u,
        "c": c,
        "lam": lam,
        "n_iter": n_iter,
        "converged": converged,
        "condensed": bool(basis is not None),
        "info": info,
    }
    return result
