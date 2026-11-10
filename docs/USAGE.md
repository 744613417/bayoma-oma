# 使用说明

本文件给出两套实现的安装、调用、示例与常见问题。字段与单位的完整定义见
[IO_SPEC.md](IO_SPEC.md)，公式对照见 [ALGORITHM.md](ALGORITHM.md)。

## 1. 环境

### 1.1 Python

- 解释器：`python`（3.13.11）。
- 依赖：`numpy`、`scipy`。安装检查：

```powershell
python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
```

- 加入搜索路径（任选其一）：

```powershell
$env:PYTHONPATH = "$PWD\python"
```

或在脚本内：

```python
import sys
sys.path.insert(0, r"python")
```

### 1.2 MATLAB

- 版本：R2022b，路径 `matlab`。
- 需要的工具箱：Optimization Toolbox（`fmincon`、`fminunc`）。检查：

```matlab
ver('optim')
```

- 加入搜索路径：

```matlab
addpath(genpath('matlab'));
```

`+bayoma` 是包目录，函数以 `bayoma.` 前缀调用。

## 2. Python 调用

### 2.1 最小示例

```python
import sys
sys.path.insert(0, r"python")
import numpy as np
import bayoma

# 生成一段合成记录：单阶模态，f = 1.0 Hz，z = 1%，三通道
data = bayoma.simulate_oma_data(
    freqs_hz=[1.0], damping=[0.01],
    phi=np.array([[1.0], [2.0], [2.0]]) / 3.0,
    S=np.array([[1.0]]), Se=1.0,
    fs=100.0, duration=1200.0, seed=7,
)

out = bayoma.bayoma(data, 100.0, f1f2=[[0.9, 1.1]], f0=[[1.0]])
print(out["f"], out["z"], out["phi"].ravel())
print(out["coefv"]["f"], out["coefv"]["phi"])
```

### 2.2 多模态频带

给出 `f0` 的多个初值即走多模态分支：

```python
out = bayoma.bayoma(data, 100.0, f1f2=[[0.9, 1.1]], f0=[[1.0, 1.02]],
                    maxiter=40)
```

### 2.3 自动选带

不给 `f1f2` 与 `f0` 时按奇异值谱自动选带：

```python
out = bayoma.bayoma(data, 100.0)
print(out["bands"])       # 识别到的频带
print(out["f"], out["z"])
```

### 2.4 多测组

```python
from bayoma import SetupData, mpv_multi_setup

setups = []
for dofs, arr in [([0, 1, 2], data1), ([2, 3, 4], data2)]:
    sp = bayoma.scaled_fft(arr, 100.0, (2.9, 3.1))
    setups.append(SetupData(freqs=sp.freqs, fft=sp.fft, dofs=dofs))

res = mpv_multi_setup(setups, n_dof=6)
print(res.f, res.z, res.phi.ravel())
```

### 2.5 异步数据（组内同步、组间不同步）

```python
from bayoma import mpv_asynchronous, asynchronous_covariance

sp = bayoma.scaled_fft(data, 100.0, (0.9, 1.1))
groups = [[0, 1], [2, 3]]          # 通道分组，0 起计，互不重叠且覆盖全部通道
res = mpv_asynchronous(sp.freqs, sp.fft, groups, q=0)
cov, coefv = asynchronous_covariance(res, sp.freqs, sp.fft, groups, q=0)
print(res.f, res.z, res.Se, abs(res.coherence[0, 1]))
```

`res.coherence` 是组间复相干矩阵，`res.Se` 是各组的通道噪声谱密度。上式适用于一个频带
内只有一阶模态的情形；频带内有多阶模态时改用多阶模型（式 10.44–10.49），各阶对各自
带一个相干矩阵：

```python
from bayoma import mpv_asynchronous_multi, asynchronous_multi_covariance

res = mpv_asynchronous_multi(sp.freqs, sp.fft, groups, n_mode=2, q=0, f0=[1.0, 1.4])
cov, coefv = asynchronous_multi_covariance(res, sp.freqs, sp.fft, groups, q=0)
print(res.f, res.z, res.phi.shape)          # phi 形状 (nch, n_mode)，逐组局部振型取 res.local
```

`res.coherence` 为各阶对的相干矩阵字典，键是阶对 `(i, j)`。

### 2.6 单模态直接调用

需要完全控制似然计算时，可跳过总入口：

```python
from bayoma import single_mode

sp = bayoma.scaled_fft(data, 100.0, (0.9, 1.1))
D = sp.psd_matrices                      # 形状 (nch, nch, nf)
res = single_mode.mpv(sp.freqs, D, sp.d_total, q=0, f0=1.0)
cov = single_mode.posterior_covariance(res, sp.freqs, D, sp.d_total, q=0)
cv = single_mode.coefficients_of_variation(res, cov)
```

### 2.7 JSON 接口

在 `python/` 目录下运行 `-m` 形式，或先把包目录加入 `PYTHONPATH`：

```powershell
cd python
python -m bayoma.io_json job.json --out result.json
```

描述文件与结果文件的字段见 [IO_SPEC.md](IO_SPEC.md) 第 6 节。在 Python 内调用：

```python
from bayoma import io_json
result = io_json.run_file("job.json", "result.json")
```

### 2.8 精度预估与参数相关性

识别之后想判断结果够不够用，或试验之前想估算需要多长的记录：

```python
from bayoma import uncertainty_law_cov, optimal_duration, posterior_matrix

# 第 15 章：各参数变异系数的边际预测
law = uncertainty_law_cov(f=1.0, z=0.01, S=1.0, Se=1.0,
                          kappa=10.0, duration=2400.0, nch=3, q=0)
print(law)                       # {'f': ..., 'z': ..., 'S': ..., 'Se': ..., 'phi': ...}

# 满足给定频率变异系数所需的最短记录时长
print(optimal_duration(1.0, 0.01, 1.0, 1.0, 10.0, 3, target=1e-3))

# 第 16 章：四参数的后验相关结构
mat = posterior_matrix(1.0, 0.01, 1.0, 1.0, 10.0, 2400.0, 3)
print(mat["coefv"], mat["corr"])
```

`posterior_matrix` 返回 `fisher`、`cov`、`corr`、`coefv`、`names`；后验相关系数只有
`ζ` 与 `S` 之间显著（κ = 10 时约 +0.525），其余参数对都在 0.03 以内。表 16.2 求逆的
边际值比表 15.1 的零阶定律小 0–1.6%，两者不一致时以第 15 章做精度预估、以第 16 章做
相关性分析。

### 2.9 结果分组与振型可视化

结果字典除基础字段外另有 `recognition` 一组，给出频带对应、密集模态区分度、联合后验
协方差与索引、逐阶振型不确定性。三个关键字控制可选内容：

```python
geom = {"channels": [{"name": "ch1", "position": [0.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0]},
                     {"name": "ch2", "position": [0.0, 0.0, 0.6], "direction": [1.0, 0.0, 0.0]},
                     {"name": "ch3", "position": [0.0, 0.0, 1.2], "direction": [1.0, 0.0, 0.0]}],
        "connections": [[0, 1], [1, 2]]}

out = bayoma.bayoma(data, 100.0, f1f2=[[0.9, 1.1]], f0=[[1.0]],
                    geometry=geom, figure_dir="figs", spectra=True)

rec = out["recognition"]
print(rec["modes"][0])                  # 频率所属频带与代入的 f0
print(rec["separation"]["has_dense_modes"])
print(rec["joint_covariance"]["index"][:3], rec["joint_covariance_matrix"].shape)
print(rec["shape_uncertainty"][0]["coefv"], rec["shape_uncertainty"][0]["std_per_channel"])
print(rec["spectra"]["bands"][0])       # 该频带的峰频与频点数
print(rec["spatial"]["figures"])        # figs/mode_00.png
```

`geometry` 缺省时 `recognition["spatial"]["available"]` 为 `False`，`reason` 说明原因；
`figure_dir` 缺省时不写图件；`spectra=False` 时 `recognition["spectra"]` 为 `None`。

异步数据另给 `assembly` 一组（组装后振型、识别与组装的不确定性分解、总后验协方差、
逐阶振型不确定性、空间图）：

```python
asm = out["assembly"]
print(asm["uncertainty"]["total_coefv"],
      asm["uncertainty"]["recognition_coefv"],
      asm["uncertainty"]["assembly_coefv"])
print(asm["uncertainty"]["trace_total"])          # 与两项迹之和相等
print(len(asm["index"]), asm["posterior_covariance"].shape)
```

`assembly_coefv` 占比过半时，不确定度主要来自测组之间相干矩阵的估计，加长测组的同步
记录比延长单个测组更有效。字段级说明见 `docs/IO_SPEC.md` 第 8 节。

## 3. MATLAB 调用

### 3.1 最小示例

```matlab
addpath(genpath('matlab'));

tdata = bayoma.simulate_oma_data([1.0], [0.01], [1; 2; 2] / 3, 1.0, 1.0, ...
    100.0, 1200.0, struct('seed', 7));

IN = struct('tdata', tdata, 'fs', 100.0, ...
    'f1f2', [0.9, 1.1], 'f0', {{1.0}}, 'mute', 1);
OUT = bayoma.bayoma(IN);

disp(OUT.f);  disp(OUT.z);  disp(OUT.phi);
disp(OUT.coefv);
```

注意 `f0` 是 cell，`{{1.0}}` 的双层花括号不能省。

### 3.2 多模态频带

```matlab
IN.f0 = {{[1.0, 1.02]}};
IN.maxiter = 40;
OUT = bayoma.bayoma(IN);
```

### 3.3 自动选带

```matlab
IN = struct('tdata', tdata, 'fs', 100.0, 'mute', 1);
OUT = bayoma.bayoma(IN);
disp(OUT.bands);
```

### 3.4 多测组

```matlab
setups = repmat(struct('freqs', [], 'fft', [], 'dofs', []), 1, 2);
S1 = bayoma.scaled_fft(data1, 100.0, [2.9, 3.1]);
setups(1).freqs = S1.freqs;  setups(1).fft = S1.fft;  setups(1).dofs = [1 2 3];
S2 = bayoma.scaled_fft(data2, 100.0, [2.9, 3.1]);
setups(2).freqs = S2.freqs;  setups(2).fft = S2.fft;  setups(2).dofs = [3 4 5];

OUT = bayoma.multi_setup(setups, 6, struct('q', 0));
disp(OUT.f);  disp(OUT.phi);

cov = bayoma.multi_setup_cov(OUT, setups, struct('q', 0));
disp(cov.coefv);
```

`setups` 必须是结构体数组（`repmat(struct(...), 1, n)`），不能是 cell 数组。
`dofs` 用 MATLAB 的 1 起计编号；JSON 描述文件中的 `dofs` 用 0 起计，由 `io_json`
在读入时换算。

### 3.5 异步数据

```matlab
sA = bayoma.scaled_fft(data, 100.0, [0.9, 1.1]);
groups = {[1 2], [3 4]};                 % MATLAB 内部用 1 起计
out = bayoma.asynchronous_single_mode(sA.freqs, sA.fft, groups, struct('q', 0));
disp(out.f);  disp(out.Se.');  disp(abs(out.v));
```

JSON 描述文件中的 `groups` 用 0 起计，`io_json` 在读入时换算、写出时还原。

### 3.6 JSON 接口

```matlab
result = bayoma.io_json('job.json');
% 或指定输出路径
result = bayoma.io_json('job.json', 'result.json');
```

### 3.7 精度预估与参数相关性

```matlab
law = bayoma.uncertainty_law(1.0, 0.01, 1.0, 1.0, 10.0, 2400.0, 3, 0);
fprintf('f %.3e  z %.3e  S %.3e  Se %.3e  phi %.3e\n', ...
    law.f, law.z, law.S, law.Se, law.phi);

mat = bayoma.uncertainty_matrix(1.0, 0.01, 1.0, 1.0, 10.0, 2400.0, 3, 0);
disp(mat.coefv.');
disp(mat.corr);
```

`mat` 的字段与 Python 侧 `posterior_matrix` 相同：`fisher`、`cov`、`corr`、`coefv`、
`phi`、`q_zeta_S`、`names`。

### 3.8 结果分组与振型可视化

MATLAB 侧的关键字与 Python 一致，放在 `IN` 结构体里：

```matlab
IN = struct();
IN.tdata = tdata; IN.fs = 100.0; IN.f1f2 = [0.9 1.1]; IN.f0 = {{1.0}};
IN.geometry = struct( ...
    'channels', {{ struct('name', 'ch1', 'position', [0 0 0], 'direction', [1 0 0]), ...
                   struct('name', 'ch2', 'position', [0 0 0.6], 'direction', [1 0 0]), ...
                   struct('name', 'ch3', 'position', [0 0 1.2], 'direction', [1 0 0]) }}, ...
    'connections', [0 1; 1 2]);
IN.figureDir = 'figs';
IN.spectra = true;
OUT = bayoma.bayoma(IN);

rec = OUT.recognition;
disp(rec.modes{1});                        % 频率所属频带与代入的 f0
disp(rec.separation.has_dense_modes);
disp(numel(rec.joint_covariance.index));   % 与矩阵维数一致
disp(rec.shape_uncertainty{1}.coefv);
disp(rec.spatial.figures);                 % figs/mode_00.png
```

`IN.geometry` 缺省时 `rec.recognition.spatial.available` 为假；`IN.figureDir` 缺省时不写
图件；`IN.spectra` 缺省为假。异步识别的 `OUT.assembly` 与 Python 侧同构。

## 4. 示例脚本

| 脚本 | 内容 |
| --- | --- |
| `examples/py_single_mode.py` | Python 单模态：合成数据、识别、与真值比对、后验变异系数 |
| `examples/py_multi_mode.py` | Python 多模态：密集模态、模态力谱矩阵、振型 MAC |
| `examples/py_multi_setup.py` | Python 多测组：四个测组、全局振型比对 |
| `examples/py_json_job.py` | Python JSON 接口端到端 |
| `examples/matlab_single_mode.m` | MATLAB 单模态 |
| `examples/matlab_multi_mode.m` | MATLAB 多模态 |
| `examples/matlab_multi_setup.m` | MATLAB 多测组 |
| `examples/matlab_json_job.m` | MATLAB JSON 接口端到端 |

## 5. 自检

改动代码后先跑自检，确认核心路径未被破坏。

Python（七项，约 2 分钟，全部通过返回 0）：

```powershell
python python/tests/test_smoke.py
```

MATLAB（十四项，约 30 分钟，任一失败不中断后续）：

```matlab
addpath(genpath('matlab'));
cd('matlab/tests');
run_all
```

`matlab/tests/` 下还有三个数据生成脚本：`gen_for_python.m`（生成单模态交叉校核数据）、
`gen_multi_case.m`（生成多模态数据）、`gen_dense_case.m`（生成密集模态数据）。
`python/tests/gen_cross_case.py` 生成 MATLAB 多模态交叉校核用的 `py_multi_case.mat`。
这些数据文件不随工程分发，缺失时相应校核脚本会自动调用生成脚本现做一份。

另有参考自由度凝聚、异步数据、不确定度定律、第 16 章 Fisher 矩阵、两侧定律对照、
统计性质与现场实测数据七项校核，未并入快速自检：

```powershell
python python/tests/test_condense.py        # 课本 Example 14.2
python python/tests/test_asynchronous.py    # 异步数据，四个算例
python python/tests/test_uncertainty.py     # 第 16 章 Fisher 矩阵
python python/tests/test_cross_law.py       # 两侧定律逐项对照
python python/tests/test_tutorial_snippets.py  # 教程片段可执行性
python python/tests/test_monte_carlo.py 20  # 20 个种子，约 205 s
```

异步数据的 JSON 接口与跨语言核对：

```powershell
python python/tests/test_async_json.py
python python/tests/test_async_multi_json.py  # 异步多阶的 JSON 导出
python python/tests/test_async_json_cross.py  # 调用 MATLAB，约 3 分钟
python python/tests/test_schema.py            # 两份 JSON Schema，约 6 分钟
```

`test_schema.py` 校验 `schema/job.schema.json` 与 `schema/result.schema.json` 自身可用，
再对三条反例断言被拒、对四类真实算例走「描述过 schema → 识别 → 结果过 schema」的完整
链路，共 17 项。未安装 `jsonschema` 时该脚本跳过校验并提示，不判为失败。

MATLAB 侧写出的结果文件用同一份 schema 核对，需先跑过 `matlab/tests/test_results.m`
（它把结果写入系统临时目录）：

```powershell
python python/tests/check_matlab_schema.py
```

该脚本读取临时目录中的异步单阶与异步多阶结果文件，逐份校验并打印顶层不符的字段路径。

现场数据核对依赖 pcode 附带的数据目录：

```powershell
python python/tests/test_field_data.py
```

`run_all` 偶发在某个脚本中途退出，标准输出末行为
`Exit Status: 0xc0000374, Heap corruption`。这是 MATLAB R2022b 的偶发问题，同一命令
重跑通常可过；需要定位时把该脚本单独运行一次即可。

## 6. 常见问题

**识别出的阻尼比明显偏离真值。** 先看记录长度。阻尼比的标准差与记录长度的平方根
成反比，600 s 记录在 f = 1 Hz、z = 1% 时阻尼比的标准差可达 20%。把记录加长到
2400 s 以上再比较。

**模态力谱矩阵的非对角元每次都不一样。** 密集模态（频率差小于带宽）且记录较短时，
互谱的可辨识性很弱，这是问题的固有性质。加长记录可以减小方差，但短记录下不应把
互谱估计与真值逐项比较。

**自动选带把噪声起伏判成模态。** 检查 `peak_factor` 与噪声基线。自动选带基于 Welch
平均谱，单次实现的谱矩阵秩为 1，高阶奇异值为 0，不能用于判模态数。

**多测组的模态力谱比真值大若干倍。** 生成合成数据时，局部振型必须取全局振型的对应
分量 `t_r = L_r·u`，不能对每个测组的局部振型单独归一化；否则 `S = S₀/‖t_r‖²` 会
按 `1/‖u(dofs)‖²` 的比例偏离真值。

**MATLAB 报「无法识别的字段名称 cmat」。** 早期版本的 `single_mode.m` 不返回
`coefv` 与 `cmat`，总入口会因此失败。当前版本已补齐，若仍报错请确认
`+bayoma/single_mode.m` 末尾调用了 `bayoma.single_mode_cov`。

**Python 子进程输出中文报 `UnicodeEncodeError`。** 本机控制台为 GBK 编码，重定向
子进程输出时需指定 `encoding="utf-8", errors="replace"`，或只打印 ASCII 行。

**MATLAB 启动时有 `pageinv` 名称冲突警告。** 与本工程无关，可忽略。

**异步识别报「此类型的变量不支持使用花括号进行索引」。** `jsondecode` 把 `[[0,1],[2,3]]`
解成数值矩阵而不是元胞，`io_json` 已用 `to_groups_cell` 统一接住两种形态；若自行写描述
文件，`groups` 仍应是「每项一个通道编号列表」的嵌套数组。

## 7. 分步教程

第一次使用建议从 `docs/TUTORIAL.md` 开始，按「环境确认 → 合成数据跑通 → 换成自己的数据 →
判断结果能不能用 → 多测组 → 交给其他软件调用 → 写报告」六步走，每步给出可直接复制的
命令与预期输出。该文件里的调用片段由 `python/tests/test_tutorial_snippets.py` 逐条执行核对：

```powershell
python python/tests/test_tutorial_snippets.py
```
