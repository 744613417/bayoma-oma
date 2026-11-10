"""参考自由度凝聚与全局最小二乘的校核（课本式 14.17--14.20 与 Example 14.2）。

运行：``python python\\tests\\test_condense.py``
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bayoma import condense  # noqa: E402


def _selection(rows, n_dof):
    matrix = np.zeros((len(rows), n_dof))
    for index, dof in enumerate(rows):
        matrix[index, dof] = 1.0
    return matrix


def test_reference_dofs():
    dofs = [[5, 2, 3], [5, 2, 1], [4, 1], [4, 0]]
    found = condense.reference_dofs(dofs)
    assert found == [1, 2, 4, 5], found
    print("[OK ] 参考自由度编号  %s" % found)


def test_condensation_matrix():
    dofs = [[5, 2, 3], [5, 2, 1], [4, 1], [4, 0]]
    matrix, info = condense.condensation_matrix(dofs, 6)
    err = float(np.max(np.abs(matrix.T @ matrix - np.eye(matrix.shape[1]))))
    assert err < 1e-12, err
    assert info["n_condensed"] == 6, info
    print("[OK ] 凝聚矩阵正交性  err = %.1e，凝聚维数 %d" % (err, info["n_condensed"]))


def test_example_14_2():
    """课本 Example 14.2：四测组共六个自由度，参考自由度为 2、3、5、6。"""
    selection = [_selection([5, 2, 3], 6), _selection([5, 2, 1], 6),
                 _selection([4, 1], 6), _selection([4, 0], 6)]
    tilde = [np.array([-0.642, -0.454, -0.618]),
             np.array([0.804, 0.420, 0.421]),
             np.array([0.883, 0.470]),
             np.array([0.959, 0.284])]
    expect = np.array([0.162, 0.286, 0.315, 0.477, 0.546, 0.516])
    expect = expect / np.linalg.norm(expect)

    results = {}
    for flag in (False, True):
        out = condense.global_least_squares(tilde, selection, condense=flag)
        u = out["u"] if out["u"][0] >= 0.0 else -out["u"]
        mac = float((u @ expect) ** 2 / ((u @ u) * (expect @ expect)))
        assert mac > 0.9999, (flag, mac)
        assert float(np.max(np.abs(u - expect))) < 1e-3
        results[flag] = (u, mac, out["n_iter"], out["converged"])
        print("[OK ] Example 14.2 condense=%-5s MAC = %.6f，最大分量差 %.1e，%d 次迭代"
              % (flag, mac, float(np.max(np.abs(u - expect))), out["n_iter"]))

    gap = float(np.max(np.abs(results[False][0] - results[True][0])))
    assert gap < 1e-8, gap
    print("[OK ] 凝聚与非凝聚结果一致  最大分量差 %.1e" % gap)


def main():
    test_reference_dofs()
    test_condensation_matrix()
    test_example_14_2()
    print("\n全部通过")


if __name__ == "__main__":
    main()
