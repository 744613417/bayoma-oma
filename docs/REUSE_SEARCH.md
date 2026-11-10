# 复用检索记录

按全局规则「实现前优先复用既有方案」，本工程在动手实现前对可复用的既有实现做了
检索，以下记录检索范围、候选验证结果与最终判定。

## 1. 检索范围

| 渠道 | 检索方式 | 关键词 |
| --- | --- | --- |
| PyPI | `pip index versions` 与 PyPI JSON 接口 | `bayoma`、`pyoma`、`pyoma2` |
| 本机 Python 环境 | `pip list` 过滤 | `oma`、`modal`、`bayes`、`pymc`、`arviz`、`emcee` |
| 本机既有代码 | 目录遍历 | `bayoma`、`Bayesian`、`modal identification` |
| 参考 pcode 包 | 目录与文件清单 | 全部 `.p` 文件 |

## 2. 候选与验证

### 2.1 PyPI 上的 `pyoma`

- 存在版本（0.15.0 等），但下载 wheel 解包后确认其内容是 `pyoma/browser/*`、
  `pyoma/inference/*`、`pyoma/orthoxml/*`，即 OMA（Orthologous MAtrix）生物信息
  数据库的 HDF5 访问库（DessimozLab / omabrowser），与运行模态分析无关。
- 判定：**不可复用**，同名属误配。

### 2.2 PyPI 上的 `pyoma2`、`bayoma`

- `pip index versions pyoma2` 返回 `No matching distribution found for pyoma2`。
- `pip index versions bayoma` 返回 `No matching distribution found for bayoma`。
- 判定：**不存在**。

### 2.3 本机 Python 环境

- `pip list` 过滤 `oma|modal|bayes|pymc|arviz|emcee` 仅命中 `Automat`、
  `roman-numerals` 两个无关包。
- 判定：**无可复用的已装库**。

### 2.4 本机既有代码

- 课本作者提供的 pcode 参考包中，函数体已加密为 `.p`，只能调用不能阅读。
- 该参考包的 `WJY` 目录下存在同一作者的 MATLAB 实现，可作为功能对照与接口约定的参考，
  但不能直接作为 Python 实现的基础。
- 判定：**可作为对照基准，不能直接复用**。

## 3. 判定与依据

结论：**没有可直接复用的 Python 实现**，因此 Python 侧自行实现；MATLAB 侧同样自行
实现，与参考 pcode 的输入输出契约保持一致，便于用户已有的调用脚本平滑迁移。

依据：

1. 可复用的候选要么不存在（`pyoma2`、`bayoma`），要么是无关领域的同名包（`pyoma`）。
2. 参考 pcode 是加密的 `.p` 文件，无法移植。
3. 用户既有 MATLAB 代码可作为接口约定与结果对照的来源，但本工程需要在两套语言上
   给出一致的、可被外部软件与智能体调用的接口，故重新实现并逐项校核。

## 4. 复用的部分

以下内容直接沿用既有方案，未重新发明：

| 内容 | 来源 | 用法 |
| --- | --- | --- |
| 输入输出字段与单位约定 | 参考 pcode 的 `Contents_bayoma.m` | 字段名、单位、量纲、`coefv` 与 `cmat` 的组织方式完全对齐 |
| 合成数据的生成方式 | 参考 pcode 的 `job01_gen_data.m` | 噪声标准差取 `sqrt(Se·fs/2)`，与单边谱密度的定义一致 |
| 课本公式 | 课本第 10–16 章 | 逐式实现，见 [ALGORITHM.md](ALGORITHM.md) |
| 数值优化 | SciPy 的 `L-BFGS-B`、MATLAB 的 `fmincon`/`fminunc` | 谱参数子问题用有界拟牛顿法求解 |
| 峰值检测 | SciPy 的 `find_peaks` | 自动选带，MATLAB 侧自写等价实现 |
