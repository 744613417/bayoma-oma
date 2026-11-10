"""BAYOMA：基于《Operational Modal Analysis》的自动贝叶斯模态识别。

本包按课本第 10–16 章实现单模态、多模态、多测组与异步数据的贝叶斯识别，并提供自动选带、
不确定性定律预测与批量处理接口。全部数值以缩放 FFT 为数据基础，似然函数与
后验协方差的公式编号与课本一致。
"""

from __future__ import annotations

from .api import bayoma
from .asynchronous import AsynchronousResult, AsynchronousSetup, coherence_matrix
from .asynchronous import mpv as mpv_asynchronous
from .asynchronous import posterior_covariance as asynchronous_covariance
from .asynchronous import simulate_asynchronous
from .asynchronous_multi import AsynchronousMultiResult
from .asynchronous_multi import mpv as mpv_asynchronous_multi
from .asynchronous_multi import posterior_covariance as asynchronous_multi_covariance
from .asynchronous_multi import simulate_asynchronous_multi
from .auto_band import BandSelection, optimal_duration, select_bands, sv_curves, uncertainty_law_cov
from .uncertainty import check_table_16_4, fisher_block, mode_shape_cov, posterior_matrix
from .data import (
    SpectralData,
    mac_matrix,
    sample_psd_welch,
    scaled_fft,
    singular_value_spectrum,
)
from .multi_mode import MultiModeResult
from .multi_mode import mpv as mpv_multi
from .multi_setup import MultiSetupResult, SetupData, selection_matrix
from .multi_setup import asymptotic_weights
from .multi_setup import mpv as mpv_multi_setup
from .posterior import (
    PosteriorResult,
    covariance_from_nllf,
    multi_mode_covariance,
    multi_setup_covariance,
    pack_spectrum,
    tangent_basis,
    unpack_spectrum,
)
from .single_mode import SingleModeResult, coefficients_of_variation, posterior_covariance
from .single_mode import mpv as mpv_single
from .synth import simulate_multi_setup_data, simulate_oma_data

from . import io_json

__all__ = [
    # 总入口
    "bayoma",
    # 数据与谱分析
    "SpectralData",
    "scaled_fft",
    "sample_psd_welch",
    "singular_value_spectrum",
    "mac_matrix",
    # 单模态
    "SingleModeResult",
    "mpv_single",
    "posterior_covariance",
    "coefficients_of_variation",
    # 多模态
    "MultiModeResult",
    "mpv_multi",
    # 多测组
    "SetupData",
    "MultiSetupResult",
    "selection_matrix",
    "mpv_multi_setup",
    "asymptotic_weights",
    # 异步数据
    "AsynchronousSetup",
    "AsynchronousResult",
    "coherence_matrix",
    "mpv_asynchronous",
    "asynchronous_covariance",
    "simulate_asynchronous",
    # 异步数据多阶
    "AsynchronousMultiResult",
    "mpv_asynchronous_multi",
    "asynchronous_multi_covariance",
    "simulate_asynchronous_multi",
    # 后验协方差
    "PosteriorResult",
    "pack_spectrum",
    "unpack_spectrum",
    "tangent_basis",
    "covariance_from_nllf",
    "multi_mode_covariance",
    "multi_setup_covariance",
    # 自动选带与不确定度定律
    "BandSelection",
    "sv_curves",
    "select_bands",
    "uncertainty_law_cov",
    "optimal_duration",
    # 第 16 章：Fisher 信息矩阵与参数间相关性
    "fisher_block",
    "posterior_matrix",
    "mode_shape_cov",
    "check_table_16_4",
    # 合成数据
    "simulate_oma_data",
    "simulate_multi_setup_data",
    # 机器可读接口
    "io_json",
]

__version__ = "0.1.0"
