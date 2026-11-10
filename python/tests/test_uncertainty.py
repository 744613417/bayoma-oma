"""第 16 章不确定度定律的校核：表 16.2 的 Fisher 信息矩阵与表 15.1 的边际定律。

核对三项：

1. 表 16.2 的 4×4 Fisher 信息矩阵求逆后，边际变异系数与表 15.1 的零阶定律一致
   （差异来自表 15.1 已把 ``q_{ζS}`` 解析吸收进 ``B_ζ``/``B_S``）；
2. 表 16.4 的阻尼比与模态力谱方差闭式与 ``1/(L^{xx}(1-q²_{ζS}))`` 一致；
3. 振型变异系数与表 15.1 的 ``B_φ = (2/π)tan⁻¹κ`` 完全一致。
"""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bayoma
from bayoma.auto_band import uncertainty_law_cov
from bayoma.uncertainty import (
    check_table_16_4,
    fisher_block,
    mode_shape_cov,
    posterior_matrix,
)


class TestFisherBlock(unittest.TestCase):
    def test_symmetric_positive_definite(self) -> None:
        info = fisher_block(1.0, 0.01, 1.0, 1.0, 10.0, 2400.0, 3)
        self.assertEqual(info.shape, (4, 4))
        self.assertLess(np.max(np.abs(info - info.T)), 1e-12)
        self.assertTrue(np.all(np.linalg.eigvalsh(info) > 0.0))

    def test_scaling_with_duration(self) -> None:
        """Fisher 信息随记录长度线性增长，故变异系数按 1/√T 收缩。"""
        a = posterior_matrix(1.0, 0.01, 1.0, 1.0, 10.0, 600.0, 3)
        b = posterior_matrix(1.0, 0.01, 1.0, 1.0, 10.0, 2400.0, 3)
        for name in ("f", "z", "S", "Se"):
            self.assertAlmostEqual(a["coefv"][name] / b["coefv"][name], 2.0, places=9)

    def test_matches_zeroth_order_law(self) -> None:
        """与表 15.1 的零阶定律逐项比较。"""
        f, z, S, Se, kappa, dur, nch = 1.0, 0.01, 1.0, 1.0, 10.0, 2400.0, 3
        mat = posterior_matrix(f, z, S, Se, kappa, dur, nch)
        law = uncertainty_law_cov(f, z, S, Se, kappa, dur, nch)
        for name in ("f", "z"):
            self.assertAlmostEqual(mat["coefv"][name] / law[name], 1.0, places=2)
        # 模态力谱的相关性最强，表 15.1 的 B_S 已吸收一部分，差异在 2% 以内。
        self.assertLess(abs(mat["coefv"]["S"] / law["S"] - 1.0), 0.03)
        self.assertAlmostEqual(mat["coefv"]["Se"] / law["Se"], 1.0, places=3)
        self.assertAlmostEqual(mode_shape_cov(f, z, S, Se, kappa, dur, nch) / law["phi"], 1.0, places=9)

    def test_table_16_4_closed_forms(self) -> None:
        """表 16.4 的阻尼比与模态力谱方差闭式。"""
        for kappa in (3.0, 5.0, 10.0, 20.0):
            chk = check_table_16_4(1.0, 0.01, 1.0, 1.0, kappa, 2400.0, 3)
            self.assertLess(chk["rel_z"], 1e-12)
            self.assertLess(chk["rel_S"], 1e-12)
            self.assertLess(chk["q_zeta_S"], 0.0)

    def test_correlation_decays_with_kappa(self) -> None:
        """κ 越大，阻尼比与模态力谱的相关性越弱。"""
        vals = [
            abs(check_table_16_4(1.0, 0.01, 1.0, 1.0, k, 2400.0, 3)["q_zeta_S"])
            for k in (3.0, 5.0, 10.0, 20.0)
        ]
        self.assertTrue(all(a > b for a, b in zip(vals, vals[1:])))

    def test_q_only_affects_shape(self) -> None:
        """表 16.2 的谱参数块与 q 无关，q 只通过 ν 进入振型。"""
        a = posterior_matrix(1.0, 0.01, 1.0, 1.0, 10.0, 2400.0, 3, 0)
        b = posterior_matrix(1.0, 0.01, 1.0, 1.0, 10.0, 2400.0, 3, 2)
        for name in ("f", "z", "S", "Se"):
            self.assertAlmostEqual(a["coefv"][name], b["coefv"][name], places=12)
        ratio = mode_shape_cov(1.0, 0.01, 1.0, 1.0, 10.0, 2400.0, 3, 2) / mode_shape_cov(
            1.0, 0.01, 1.0, 1.0, 10.0, 2400.0, 3, 0
        )
        self.assertAlmostEqual(ratio, (2.0 * np.pi) ** 2, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
