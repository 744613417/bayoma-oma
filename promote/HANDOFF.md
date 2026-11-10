# 交接文件：自动贝叶斯模态识别（BAYOMA）双实现工程

本文件供后续接入的智能体或开发者使用。读完本文件即可掌握工程的目标、结构、接口、
已实现范围、校核结论与待办事项，无需回溯历史对话。

## 1. 工程目标

按《Operational Modal Analysis: Modeling, Bayesian Inference, Uncertainty Laws》
（Springer，下称课本）第 10–16 章，用 Python 与 MATLAB 各实现一套完整的自动贝叶斯
模态识别（BAYOMA）流程。两套实现独立编写、互为校核，输入输出契约一致。

要求来自用户 2026-11-06 的七条指令：

| 编号 | 要求 | 落实位置 |
| --- | --- | --- |
| 1 | 两套实现（Python + MATLAB），各有详细使用说明 | `python/bayoma/`、`matlab/+bayoma/`、`docs/USAGE.md` |
| 2 | 课本涉及 BAYOMA 的部分完整实现，不简化（密集模态、多测组组合等） | 第 12–15 章逐节实现，见 `docs/ALGORITHM.md` |
| 3 | 输入输出格式与说明文件详细，超过参考 pcode 的说明文件 | `docs/IO_SPEC.md`、`docs/USAGE.md` |
| 4 | 保存结构清晰，自动化功能完整（如自动选带） | `python/bayoma/auto_band.py`、`matlab/+bayoma/select_bands.m` |
| 5 | 两套接口可供其余软件或智能体直接读取 | `python/bayoma/io_json.py`、`matlab/+bayoma/io_json.m` |
| 6 | 形成实现要求、实现过程、实现效果的交接文件 | `promote/IMPLEMENTATION.md`、`promote/RESULTS.md`、本文件 |
| 7 | 形成对应课本的技能与插件 | `skill/bayoma-oma/`、`plugin/` |

## 2. 目录结构

```text

├─ python/bayoma\        Python 实现（包，可直接 import）
├─ matlab/+bayoma\       MATLAB 实现（包，addpath 后调用）
├─ docs/                 输入输出规范、使用说明、分步教程、算法对照、校核记录、复用检索
├─ examples/             两套实现的调用示例（合成数据，可直接运行）
├─ promote/              本交接文件与实现过程、实现效果说明
├─ skill/bayoma-oma\     与课本对应的技能包
├─ plugin/               在 DeepSeek-Harness 中加载的插件
└─ _work/                勘察与调试过程的临时脚本（可删除，不属于交付物）
```

## 3. 环境

- Python：`python`，3.13.11。依赖 numpy 2.3.5、scipy 1.16.3。
- MATLAB：R2022b，`matlab`。需要 Optimization Toolbox
  （`fmincon`、`fminunc`）。
- 两套实现不依赖任何商业工具箱中的模态分析函数，也不依赖参考 pcode。

## 4. 接口速查

### 4.1 Python

```python
import sys
sys.path.insert(0, r"python")
import bayoma

out = bayoma.bayoma(tdata, fs, f1f2=[[0.9, 1.1]], f0=[[1.0]])
# out 的键：f, z, Sii, Se, phi, rms, S, coefv, cmat, sn, time_req, bands, nch, fs,
#           recognition（结果分组，见 docs/IO_SPEC.md 第 8 节）
```

`bayoma.bayoma` 另有三个可选关键字：`geometry`（通道位置、方向与连接，用于振型空间
可视化）、`figure_dir`（图件输出目录）、`spectra`（是否把 PSD 谱与 SV 谱写进结果，
缺省 `False`，长记录时建议打开以核对频带与初值）。异步数据识别在 JSON 接口下另给出
`assembly` 分组。

不给出 `f1f2` 时自动选带。多测组用 `bayoma.io_json` 的描述文件，或直接调
`bayoma.mpv_multi_setup`。

### 4.2 MATLAB

```matlab
addpath(genpath('matlab'));
IN.tdata = tdata;  IN.fs = 100;  IN.f1f2 = [0.9 1.1];  IN.f0 = {{1.0}};
OUT = bayoma.bayoma(IN);
```

### 4.3 机器可读接口（两套实现共用同一份 JSON 描述）

```powershell
python -m bayoma.io_json job.json --out result.json
```

```matlab
result = bayoma.io_json('job.json');
```

描述文件与结果文件的字段见 `docs/IO_SPEC.md` 第 6 节。多测组的自由度编号从 0 起计，
两套实现共用该约定。

字段的合法取值与形状由 `schema/job.schema.json`、`schema/result.schema.json` 给出，
两者都是 JSON Schema 2020-12，顶层 `additionalProperties` 为 `false`。接入方不必解析
`docs/IO_SPEC.md` 即可校验：先校验描述文件，跑完再校验结果文件。同一字段在三种识别
路径下形状一致——`f`、`z`、`Sii`、`Se`、`rms`、`sn` 一律一维数组（只有一阶也写 `[x]`），
`phi` 一律二维，`coefv` 各项沿用同一形状，`input` 内的数量字段是数字。MATLAB 的
`jsonencode` 无法区分标量与长度 1 的数组，侧写盘前先经 `bayoma.to_json_tree` 显式化形状、
再由 `bayoma.encode_json` 生成文本，故两侧产物对同一份 schema 校验均为 0 错误。
该结论由 `python/tests/check_matlab_schema.py` 对 MATLAB 写出的异步单阶与异步多阶两份
结果文件复核，不靠人工比对。

### 4.4 技能与插件

`skill/bayoma-oma/SKILL.md` 是课本第 10–16 章的技能正文，含每个环节的式号与实现约束。
两种安装方式，任选其一：

```powershell
# 账号级技能目录，立即生效
Copy-Item -Recurse -Force 'skill/bayoma-oma' "$env:USERPROFILE\.dsh\skills/"
```

```powershell
# Harness 插件，重启后生效
plugin_manager action=install_bundle target=<本机绝对路径>\plugin/dsh-bayoma-oma
```

插件 `plugin/dsh-bayoma-oma/` 由 `cordis.patch.yml` 插入一个 `bayoma-oma` 条目，
`lib/startup.js` 读取自带的 `skills/bayoma-oma/SKILL.md`，校验 frontmatter 后调用
`ctx.skills.register` 注册。两份内容相同，装任一份即可。

三处副本（工程 `skill/`、账号级 `%USERPROFILE%\.dsh\skills/`、插件 `skills/`）现均为
22098 B、SHA256 前缀 `0e04cd5fb1c60c97`（385 行）。改动技能正文后按下列命令同步三处：

```powershell
$src = 'skill/bayoma-oma\SKILL.md'
Copy-Item $src '%USERPROFILE%\.dsh\skills/bayoma-oma\SKILL.md' -Force
Copy-Item $src 'plugin/dsh-bayoma-oma\skills/bayoma-oma\SKILL.md' -Force
```

技能正文第 11 节为使用说明（安装、调用、调用前确认的四项、判读与交付、常见误用、
参考文档），插件 `README.md` 另有安装确认表与一次完整使用的四步流程，
`docs/TUTORIAL.md` 是从零开始的分步教程。

## 5. 已实现范围

| 模块 | 课本章节 | Python | MATLAB |
| --- | --- | --- | --- |
| 缩放 FFT 与谱矩阵 | 10.2 | `data.py` | `scaled_fft.m`、`psd_matrices.m` |
| 单模态最可能值 | 12.2–12.3 | `single_mode.py` | `single_mode.m` |
| 单模态后验协方差 | 12.4 | `single_mode.posterior_covariance` | `single_mode_cov.m` |
| 多模态最可能值 | 13.2–13.5 | `multi_mode.py` | `multi_mode.m` |
| 多模态后验协方差 | 13.7 | `posterior.multi_mode_covariance` | `multi_mode.m` 内 `covariance_estimate` |
| 多测组最可能值 | 14.2–14.3 | `multi_setup.py` | `multi_setup.m` |
| 多测组后验协方差 | 14.4 | `posterior.multi_setup_covariance` | `multi_setup_cov.m` |
| 参考自由度凝聚 | 14.2.4 | `condense.py` | `condensation_matrix.m`、`project_condensed.m`、`global_least_squares.m` |
| 异步数据最可能值 | 10.4 | `asynchronous.py` | `asynchronous_single_mode.m` |
| 异步数据后验协方差 | 10.4 | `asynchronous.posterior_covariance` | `asynchronous_single_mode.m` 内 `posterior_cov` |
| 异步数据多阶最可能值 | 10.4（式 10.44–10.49） | `asynchronous_multi.py` | `asynchronous_multi_mode.m` |
| 异步数据多阶后验协方差 | 10.4 | `asynchronous_multi.posterior_covariance` | `asynchronous_multi_mode.m` 内 `covariance_estimate` |
| 自动选带 | 7.3、7.7 | `auto_band.py` | `select_bands.m` |
| 不确定度定律 | 15.2 | `auto_band.uncertainty_law_cov` | `uncertainty_law.m` |
| Fisher 信息矩阵与参数相关性 | 16.1–16.4 | `uncertainty.py` | `uncertainty_matrix.m` |
| 合成数据 | 10.2 | `synth.py` | `simulate_oma_data.m` |
| 机器可读接口 | — | `io_json.py` | `io_json.m` |
| 结果分组（识别与组装） | — | `results.py` | `result_recognition.m`、`result_assembly.m` |

逐式对照见 `docs/ALGORITHM.md`。

## 6. 校核结论（摘要）

详见 `docs/VERIFICATION.md`，其中记录了全部数值与数据条件。

- 单模态：解析 Hessian 与中心差分一致到千分之一；与第 15 章不确定度定律的比值
  0.97/0.86/1.00/1.00/1.00；Python 与 MATLAB 在同一数据上逐位一致（相对差 < 1e-6）。
- 多模态：分离模态下频率与阻尼比恢复真值；密集模态（1.00/1.02 Hz、相干 0.5）在
  9600 s 记录下 `f = 1.000894 / 1.019513`、MAC 0.9997；模态力互谱的模收敛到真值。
- 多测组：单测组时与单模态实现逐位一致；四测组联合识别的全局振型 MAC 0.99999；
  逐测组谱参数恢复各自真值。
- 后验协方差：单模态与多测组五项参数的变异系数与定律一致到 1%–7%；密集模态下
  振型的变异系数显著大于定律，已在文档中如实记录。
- 异步数据（单阶）：式 (10.42) 的快速式与逐频率直接式相对差 4.4e-16；`q = 1` 时与
  同步单模态的相对差 1e-7–1e-5；两测组算例的 `f` 与真值差 0.034%、`z` 差 0.11%、
  振型 MAC 0.99999、相干模 0.533 对真值 0.5。
- 异步数据（多阶）：单阶退化到单阶异步实现相对差 `f` 2e-9、`z` 1.9e-5；降维恒等式与
  逐频率直接计算相对差 3e-15；两阶两测组的 `f` 误差 7.3e-4 / 3.0e-4、MAC 0.999988 /
  0.999985、相干模差 0.038、相位差 0.074 rad。单测组退化到第 13 章按模型类嵌套关系
  校核（`f` 1.6e-7、`2|ΔL| = 5.5e-3` 对 `χ²₈` 的 20.09）。
- 机器可读接口：同一份四测组描述文件由两套实现执行，结果一致到 1e-4，协方差矩阵
  同为 21×21。异步多阶走同一接口，两侧频率一致到 2e-4、阻尼比与谱密度一致到 1e-2。
- 现场实测数据：pcode 附带的 12 通道、1800 s、64 Hz 记录识别出六个模态，频率全部
  落在给定频带内，阻尼比 0.7%–2.9%，振型范数为 1。这组数据不含真值，核对的是自洽性；
  它暴露出两处合成数据未覆盖的缺陷（逐阶输出长度不一致、谱参数量级极小时后验协方差
  退化为零），均已修正，详见 `docs/VERIFICATION.md` 第 8 节。
- 结果分组：`recognition` 与 `assembly` 两组的结构由 `python/tests/test_results.py`
  逐项断言（四组共 60 项），并与 `schema/result.schema.json` 对齐；`test_schema.py`
  另对四类算例做「描述 → 识别 → 结果过 schema → 文件读回」的端到端校验（17 项），
  `check_matlab_schema.py` 再用同一份 schema 复核 MATLAB 写出的两份结果文件。
  联合协方差的分块对角结构（跨频带相关系数为零）以似然可加性为依据，非近似；组装分组
  的不确定性分解满足 `trace_recognition + trace_assembly = trace_total`（实测
  3.883286e-04 对 3.883286e-04）。

## 7. 已知边界

1. 密集模态且模态力互谱相干较弱时，短记录下 `S` 的非对角元估计方差大，可能收敛到
   模与真值相近而相位不同的值。这是两阶模态形状在有限记录下难以分离的直接后果，
   与课本 13.6 节的说明一致，不属于实现缺陷。
2. 密集模态下振型的后验变异系数大于不确定度定律的预测（实测 23.8 倍）。定律按单阶
   模态独立推导，未计入两阶模态形状之间的相关性。
3. 后验协方差求逆前作雅可比预条件（对角元构造尺度后求伪逆、再还原），两侧变异系数
   一致到 1e-6；不作预条件时振型方向会被截断阈值判成零曲率，见 `docs/VERIFICATION.md`
   1.3 节。
4. 多模态交替极小化在最优点附近存在慢漂移，`maxiter` 小于 40 时可能以「达到迭代上限」
   结束而非满足收敛判据；结果仍在最优值附近。
5. 多测组全局振型的变异系数高于单个测组按本组通道数算出的定律值，原因是各自由度的有效
   约束长度不同（被测到的各测组记录长度之和）。以总约束时长作参照时两者一致：全覆盖
   情形 0.001083 对六通道 9600 s 定律 0.001078，比值 1.004。参考自由度凝聚开关不改变
   该数值，见 `docs/VERIFICATION.md` 3.3 节。
6. 异步多阶在单测组（`ng = 1`）时已放开复模态形状，其模型类**包含**第 13 章同步多阶的
   `φ = B₀α`（`B₀` 实正交基、`α` 复坐标）：后者两阶形状的实部与虚部共用同一个 `m₀` 维
   实子空间，前者在整个 `Cⁿ` 单位球上取形状，多出 `m(2n−2) − m(n−1)` 个形状实自由度。
   因此两者的振型点估计不会相等：实值 MAC 的 `1 − MAC` 为 1e-8 量级、复内积 MAC 为
   6.4e-6，而同步解与真值的 `1 − MAC` 本身就在 8e-4 量级（1.00/1.02 Hz 密集模态、间距约
   半功率带宽），两个这样的估计之间互差 6.4e-6 属 `O(1/N)` 抽样差，需约六倍记录时长
   才能压到 1e-6。谱参数不受影响：`f`、`z`、`S` 与同步多阶的相对差为 1.6e-7、6.8e-5、
   2.0e-5，似然比 `2|ΔL| = 2.43` 对 `χ²₄` 的 13.28（MATLAB 侧），故判据按嵌套关系写，
   见 `docs/VERIFICATION.md` 4.6 节。
7. 异步多阶的组间相干矩阵 `V_ij` 精度受带内独立平均次数限制。固定其余参数对 `V_12`
   做二维剖面搜索，4800 s、两测组下不同种子的极大似然点相对真值散布为模 0.018–0.084、
   相位 0.000–0.157 rad。三个及以上测组不可靠：待估复非对角元随 `q(q−1)/2` 增长，
   实测 `q = 3`、4800 s、seed 7 时最大模偏差 0.1420、最大相位偏差 0.3787 rad；
   单通道测组（`n_r = 1`）的噪声谱与相干无法从单通道互谱中稳定估计。

## 7.1 本轮打磨的结论

1. 单模态估计量无系统偏差：100 个种子的均值与真值之比为 1.000106 / 0.9884 / 1.0015 /
   0.9951，偏差均在 1.5 个标准误以内。
2. 后验区间比实测离散度宽 5%–15%，方向偏保守，不改变点估计与相对精度排序。
3. 多模态与多测组的最可能值已达驻点：延长迭代与增加重启对 NLLF 的改善小于 1e-6
   相对量级，参数变化小于 1e-5。
4. 参考自由度凝聚已实现并以课本 Example 14.2 校核，Python 与 MATLAB 均为 MAC 0.999999。
5. 两侧后验协方差统一采用雅可比预条件后求伪逆，五个参数的变异系数相对差小于 1e-4。

## 8. 待办与可扩展方向

1. 后验协方差在完整 n 维切空间内计算，不在参考自由度凝聚子空间内计算：凝聚子空间把
   非参考自由度写成识别振型的等比缩放，等价于假定这些分量完全准确，会低估方差。实测
   开关凝聚后变异系数不变（0.002729 对 0.002729），确认当前做法正确，无需改动。
2. 课本第 16 章已实现表 16.2 的闭式 Fisher 信息矩阵、表 16.3 的敏感度量级与表 16.4 的
   零阶后验方差/相关系数（`uncertainty.py` / `uncertainty_matrix.m`）；式 (16.43)–(16.44)
   的一阶信噪比修正未单独实现，因为表 15.1 的 `α_x/c` 已给出等价结果，两处同时施加会
   重复计数。第 16 章的带外模态影响（16.6 节之后的讨论）未展开。
3. 异步模型的单阶（式 10.42–10.51）与多阶（式 10.44–10.49，各阶对独立的相干矩阵）
   均已实现；尚缺两项：多阶情形下组间相干矩阵的简化参数化（当前每个非对角元各自取
   模与相位，测组数多时参数增长快）、以及异步多阶的参考自由度凝聚。
4. `_work/` 目录是勘察与调试脚本，不属于交付物，可按需删除。
5. 结果分组的两处可扩展方向：`recognition.spectra` 的抽稀上限固定为 1024 点，若需
   在结果文件内做细致谱形比对，可提高上限或改为分频带写出；`assembly.spatial` 只画
   变形后的振型，未叠加组装前的各测组局部振型，若需核对组装环节的几何一致性，可增加
   一张含各测组局部振型的对照图。
