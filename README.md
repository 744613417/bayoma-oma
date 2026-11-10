# 自动贝叶斯模态识别（BAYOMA）双实现工程

本工程按《Operational Modal Analysis: Modeling, Bayesian Inference, Uncertainty Laws》
（Springer，以下称课本）第 10–16 章的内容，实现环境振动数据的自动贝叶斯模态识别。
Python 与 MATLAB 两套实现独立编写、互为校核，输入输出契约一致，均可被其余软件或
智能体直接调用。

[![python-tests](https://github.com/744613417/bayoma-oma/actions/workflows/python-tests.yml/badge.svg)](https://github.com/744613417/bayoma-oma/actions/workflows/python-tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 交付内容

| 内容 | 位置 | 说明 |
| --- | --- | --- |
| Python 实现 | `python/bayoma/` | 单模态、多模态、多测组（含参考自由度凝聚）、异步数据（单阶与多阶）、后验协方差、自动选带、不确定度定律、Fisher 信息矩阵、合成数据 |
| MATLAB 实现 | `matlab/+bayoma/` | 与 Python 侧一一对应的同名函数 |
| 输入输出规范 | `docs/IO_SPEC.md` | 字段、单位、量纲、缺省值、失败情形、结果分组（识别与组装） |
| 机器可读契约 | `schema/job.schema.json`、`schema/result.schema.json` | 任务描述与结果文件的 JSON Schema，供其他软件或智能体直接校验 |
| 使用说明 | `docs/USAGE.md` | 两套实现的安装、调用、示例、常见问题 |
| 分步教程 | `docs/TUTORIAL.md` | 从环境确认到报告写作的六步上手流程 |
| 算法与公式对照 | `docs/ALGORITHM.md` | 每个模块对应课本的章节与式号 |
| 校核记录 | `docs/VERIFICATION.md` | 交叉校核的数值证据与已知偏差 |
| 复用检索记录 | `docs/REUSE_SEARCH.md` | 实现前的既有方案检索范围与判定 |
| 示例 | `examples/` | 合成数据与现场数据的完整调用脚本 |
| 交接材料 | `promote/` | 智能体接入用的任务说明、实现过程、实现效果 |
| 课本技能 | `skill/` | 与课本对应的技能包 |
| 插件 | `plugin/` | 在 Harness 中加载的插件 |

## 快速开始

Python：

```python
import sys
sys.path.insert(0, r"python")
import bayoma

out = bayoma.bayoma(tdata, fs, f1f2=[[0.9, 1.1]], f0=[[1.0]])
print(out["f"], out["z"], out["phi"])
```

MATLAB：

```matlab
addpath(genpath('matlab'));
IN.tdata = tdata;  IN.fs = 100;  IN.f1f2 = [0.9 1.1];  IN.f0 = {{1.0}};
OUT = bayoma.bayoma(IN);
disp(OUT.f);  disp(OUT.z);  disp(OUT.phi);
```

机器可读接口（两套实现同一份 JSON 描述），在 `python/` 目录下运行：

```powershell
cd python
python -m bayoma.io_json job.json --out result.json
```

```matlab
result = bayoma.io_json('job.json');
```

## 已实现与校核范围

- 单模态问题（课本第 12 章）：降维负对数似然、交替极小化求最可能值、解析后验协方差。
- 多模态问题（课本第 13 章）：模态形状子空间表示、超角旋转、结构化模态力谱矩阵、
  复值投影似然、数值后验协方差。
- 多测组问题（课本第 14 章）：逐测组独立谱参数、全局模态形状的部分最可能值迭代、
  选择矩阵与全局最小二乘。参考自由度凝聚（14.2.4 节）已实现，见 `python/bayoma/condense.py`
  与 `matlab/+bayoma/condensation_matrix.m`，最可能值的全局振型求解缺省在凝聚子空间内进行。
- 异步数据（课本 10.4 节、式 10.42–10.51）：按测组把模态形状分块，估计组间复相干矩阵与
  各测组的预测误差谱，见 `python/bayoma/asynchronous.py` 与
  `matlab/+bayoma/asynchronous_single_mode.m`；`q = 1` 时退化到同步单模态，实测振型
  MAC 为 1.0000000000。多阶模态的异步识别（式 10.44–10.49，各阶对独立的相干矩阵）见
  `python/bayoma/asynchronous_multi.py` 与 `matlab/+bayoma/asynchronous_multi_mode.m`。
- 自动选带：奇异值谱、噪声基线、峰值检测、模态数判定。
- 不确定度定律（课本第 15 章）：零阶定律与一阶修正、最优记录长度。
- Fisher 信息矩阵与参数间相关性（课本第 16 章）：表 16.2 的闭式矩阵、表 16.4 的
  零阶后验方差与相关系数、`ζ`–`S` 相关性的解析形式。
- 合成数据：按理论谱矩阵在频域抽取，可生成单模态、多模态、多测组与异步数据。
- 结果分组：识别结果与异步数据组装结果分别输出到 `recognition` 与 `assembly`。识别分组
  给出每个识别频率所属频带与代入的频率初值、密集模态区分度（频率间隔与半功率带宽之比
  及两阶振型的模态置信因子）、参数联合后验协方差矩阵及其参数索引、逐阶振型的识别
  不确定性（逐通道标准差与 95% 置信半宽）、时程数据的 PSD 谱与 SV 谱、以及给定通道
  几何后的振型空间可视化。组装分组给出组装后振型的最终不确定性（识别环节与组装环节
  的条件方差分解）、后验协方差矩阵与索引、逐阶振型不确定性与空间可视化。见
  `python/bayoma/results.py`、`docs/IO_SPEC.md` 第 8 节。

两套实现在同一份数据上的单模态结果逐位一致（频率、阻尼比、模态力谱、噪声谱相对差
小于 1e-6），多模态与多测组结果的差异在记录长度引起的抽样起伏量级内。除合成数据外，
还用 pcode 附带的 12 通道、1800 s 现场记录核对过：识别出六个模态，频率全部落在给定
频带内，前两阶相差 0.005 Hz 仍被分成两个峰。详见 `docs/VERIFICATION.md`。

自检命令：

```powershell
python python/tests/test_smoke.py          # 七项，约 2 分钟
python python/tests/test_condense.py       # 参考自由度凝聚
python python/tests/test_asynchronous.py   # 异步数据（课本 10.4 节）
python python/tests/test_async_json.py     # 异步数据的 JSON 接口
python python/tests/test_async_multi_json.py  # 异步多阶的 JSON 导出，约 40 s
python python/tests/test_async_json_cross.py  # 异步 JSON 两侧对照，约 3 分钟
python python/tests/test_schema.py     # 两份 JSON Schema 与四类算例，约 6 分钟
python python/tests/test_results.py    # 结果分组（识别与组装），约 8 分钟
python python/tests/check_matlab_schema.py  # MATLAB 结果文件对同一份 schema
python python/tests/test_field_data.py     # 现场实测数据（需 pcode 数据目录）
python python/tests/test_monte_carlo.py 20 # 统计校核，约 205 s
python python/tests/test_cross_law.py      # 两侧定律逐项对照
python python/tests/test_uncertainty.py    # 第 16 章 Fisher 矩阵与边际定律一致
python python/tests/test_tutorial_snippets.py  # 教程片段核对，约 40 s
```

```matlab
addpath(genpath('matlab'));
cd('matlab/tests'); run_all          % 十四项，约 30 分钟
```

## 已知边界

- 密集模态且模态力相关系数较弱时，模态力谱矩阵的非对角元在短记录下方差很大，
  这是问题的固有性质（课本 13.6 节亦有说明），不属于实现缺陷；文档中如实记录，
  不宣称该情形下的互谱估计与真值一致。
- 后验协方差在雅可比预条件下求伪逆，谱参数与振型分量的量级差异不致使振型方向被
  截断阈值判成零曲率；Python 与 MATLAB 的变异系数一致到 1e-6，见
  `docs/VERIFICATION.md` 1.3 节。
- 多测组全局振型的后验变异系数高于单个测组按本组通道数算出的定律值：各自由度的有效
  约束长度不同，取各测组对该自由度的记录长度之和；以总约束时长作参照时两者一致（全覆盖
  情形比值 1.004）。参考自由度凝聚开关不改变该数值，见 `docs/VERIFICATION.md` 3.3 节。
- 异步多阶的组间相干矩阵按每个非对角元各自取模与相位，测组数增大时自由参数个数按
  `m(m−1)/2 × q(q−1)/2 × 2` 增长。实测两测组时相干模偏差 0.018–0.084、相位偏差
  0–0.157 rad，三测组时模差 0.044、相位差 0.077 rad。单个测组只有一个通道时该组与
  其它组的相干无法稳定估计，需保证每个测组至少两个通道。

## 技能与插件

`skill/bayoma-oma/SKILL.md` 是把课本第 10–16 章的建模假设、公式、实现要点与判读经验
压缩成的一份技能正文，供智能体在遇到模态识别任务时直接调用。正文只写可直接执行的
内容：每个环节对应的式号、必须守住的实现约束（复值投影、子空间不可锁死、相位基准
统一、局部振型不得单独归一）、以及各参数的合理量级。

两种安装方式：

- 账号级技能目录（立即生效，无需重启）：

```powershell
Copy-Item -Recurse -Force 'skill/bayoma-oma' "$env:USERPROFILE\.dsh\skills"
```

- DSH 插件（随 Harness 启动注册，重启后生效）：

```powershell
# 在 Harness 中执行
plugin_manager action=install_bundle target=<本机绝对路径>\plugin\dsh-bayoma-oma
```

`plugin/dsh-bayoma-oma/` 是一个最小 bundle：`cordis.patch.yml` 插入一个 `bayoma-oma`
条目，`lib/startup.js` 读取自带的 `skills/bayoma-oma/SKILL.md`，校验 frontmatter 后
调用 `ctx.skills.register`。插件与账号级技能内容相同，二者装任一份即可。

三处副本（工程内 `skill/`、账号级 `%USERPROFILE%\.dsh\skills\`、插件内 `skills/`）内容一致，
同步命令：

```powershell
$src = 'skill/bayoma-oma/SKILL.md'
Copy-Item $src "$env:USERPROFILE\.dsh\skills\bayoma-oma\SKILL.md" -Force
Copy-Item $src 'plugin\dsh-bayoma-oma\skills\bayoma-oma\SKILL.md' -Force
```

技能正文第 11 节为使用说明，插件的 `README.md` 另有安装确认与一次完整使用的四步流程。
分步教程见 `docs/TUTORIAL.md`。

## 环境

- Python：`python`（3.13.11），依赖 numpy、scipy。
- MATLAB：R2022b，使用 Optimization Toolbox 的 `fmincon`/`fminunc`。
