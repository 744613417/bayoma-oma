# 从零开始：第一次跑出模态参数

本文件按「拿到一批数据 → 跑出结果 → 判断结果能不能用」的顺序走一遍，命令可直接复制。
每一步都给出预期输出片段，便于对照。全部命令在本机执行。

字段与单位的完整定义见 [IO_SPEC.md](IO_SPEC.md)，公式对照见 [ALGORITHM.md](ALGORITHM.md)，
校核数值见 [VERIFICATION.md](VERIFICATION.md)。

---

## 第 0 步：确认环境能用

Python 侧只需要 numpy 与 scipy：

```powershell
python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
```

预期输出形如 `2.3.5 1.16.3`。

MATLAB 侧需要 Optimization Toolbox：

```matlab
addpath(genpath('matlab'));
ver('optim')
```

输出里应出现 `Optimization Toolbox` 及其版本号。

跑一遍自检确认工程本身没坏：

```powershell
cd bayoma-oma
python python/tests/test_smoke.py
```

预期末行为 `全部通过`，退出码 0，约 1 分钟。

---

## 第 1 步：先用合成数据跑通，不碰自己的数据

合成数据的真值已知，能立刻判断结果对错。这一步的目的是确认调用路径通、参数含义清楚。

```powershell
cd bayoma-oma
python examples/py_single_mode.py
```

预期输出（数值会有小差异，因为随机种子固定为 7，实际应逐位接近）：

```text
数据：240000 样本、3 通道、2400 s

识别结果
  频率    : 1.001457 Hz（真值 1.000000）
  阻尼比  : 0.009591（真值 0.010000）
  模态力谱: 0.997193（真值 1.000000）
  噪声谱  : 0.937170（真值 1.000000）
  信噪比  : 2892.1
  振型    : [0.335138 0.665849 0.666579]，MAC = 0.999996

后验变异系数
  频率    : 0.000854
  阻尼比  : 0.097365
  模态力谱: 0.054101
  噪声谱  : 0.032242
  振型    : 0.002256
```

**怎么读这组数**：频率误差 0.15%、阻尼比误差 4%、振型 MAC 0.999996，都在记录长度对应的
抽样起伏范围内。阻尼比的变异系数 0.097 是频率的 114 倍，这是问题的性质，不是实现的问题。

MATLAB 侧对应命令：

```matlab
addpath(genpath('matlab'));
cd('examples'); matlab_single_mode
```

预期结果与 Python 侧同量级（`f≈0.9995`、`z≈0.0111`、`Sii≈0.982`、`Se≈0.956`、
`sn≈2093`、`MAC≈0.99999`）。两侧都用种子 7，但 MATLAB 与 numpy 的随机数序列不同，
数据本身不同，数值不会逐位相同。

---

## 第 2 步：换成自己的数据

### 2.1 数据要求

- 二维实数矩阵，行是时刻、列是通道，各通道同一时基同步采样；
- 平稳环境振动，去趋势后即可用（工程内部会去均值与线性趋势）；
- 采样频率 `fs` 已知且恒定。

从文件读入：

```python
import sys
sys.path.insert(0, r"python")
import numpy as np
import bayoma

# CSV：每列一个通道，无表头
tdata = np.loadtxt(r"D:\my_data\record.csv", delimiter=",")
fs = 100.0
print(tdata.shape)          # 期望形如 (240000, 12)：24 万个时刻、12 个通道
```

MATLAB：

```matlab
addpath(genpath('matlab'));
S = load('D:\my_data\record.mat');   % 内含 tdata 与 fs
```

### 2.2 第一次识别：交给自动选带

不给频带与初值，由程序按奇异值谱自己找模态：

```python
out = bayoma.bayoma(tdata, fs, mute=1)
print("频带:", out["bands"])
print("频率:", out["f"])
print("阻尼比:", out["z"])
```

MATLAB：

```matlab
IN = struct('tdata', tdata, 'fs', 100.0, 'mute', 1);
OUT = bayoma.bayoma(IN);
disp(OUT.bands);  disp(OUT.f);  disp(OUT.z);
```

**先把 `bands` 看一眼**。自动选带给出的是「有峰且峰足够突出」的区间；如果某个已知模态
没有被选进任何频带，说明它的峰不够突出（多见于模态力弱、测点远离振型极值），需要手工指定。

### 2.3 手工指定频带

确认了要识别哪几阶之后，把它们写死，结果更可控：

```python
out = bayoma.bayoma(
    tdata, fs,
    f1f2=[[0.90, 1.10], [2.80, 3.20]],   # 两个频带
    f0=[[1.00], [3.00, 3.02]],           # 第一个频带 1 阶，第二个频带 2 阶（密集模态）
    mute=1,
)
```

`f0` 的每个内层列表长度决定该频带的模态阶数：长度 1 走单模态分支，长度 ≥ 2 走多模态分支。
初值不必精确，±5% 以内即可，但密集模态要给出两阶各自的初值。

---

## 第 3 步：判断结果能不能用

按下面顺序看，任何一项不过就回到第 2 步调整频带或加长记录。

### 3.1 信噪比

```python
for i in range(len(out["f"])):
    print(f"阶 {i+1}: f={out['f'][i]:.4f}  z={out['z'][i]:.4f}  sn={out['sn'][i]:.1f}")
```

- `sn > 100`：频率与振型可靠，阻尼比看下一条；
- `10 < sn < 100`：频率可用，阻尼比的不确定度较大；
- `sn < 10`：该阶在这批数据里信噪比不足，不宜作为结论。

### 3.2 变异系数

```python
cv = out["coefv"]
print("f   :", cv["f"])
print("z   :", cv["z"])
print("phi :", cv["phi"])
```

经验量级（`fs` 与记录长度典型取值下）：

| 参数 | 典型变异系数 | 说明 |
| --- | --- | --- |
| `f` | 0.1%–1% | 最容易识别 |
| `z` | 5%–30% | 由半功率带宽决定，对谱形状敏感 |
| `phi` | 0.2%–2% | 单位范数下的整体不确定度 |
| `Sii` | 3%–15% | 模态力幅值 |
| `Se` | 3%–10% | 通道噪声 |

阻尼比变异系数远大于频率是正常的。**报告阻尼比时必须同时给出变异系数与记录长度**，
单独一个 0.01 没有解释力。

### 3.3 与不确定度定律对照

定律给出这批数据在理论上能达到的精度，用来判断结果是否已经到顶：

```python
law = bayoma.uncertainty_law_cov(
    f=out["f"][0], z=out["z"][0], S=out["Sii"][0], Se=out["Se"][0],
    kappa=10.0, duration=tdata.shape[0] / fs, nch=tdata.shape[1], q=0,
)
print("定律预测 z 的变异系数:", law["z"])
print("实际:", out["coefv"]["z"][0])
```

两者比值在 0.8–1.2 之间即说明识别已接近该记录长度下的极限；比值大于 2 说明还有改进空间
（通常是频带没覆盖完整的半功率带宽）。

要同时看四个参数之间的相关性（例如把结果交给模型修正、需要联合置信域），用第 16 章的
Fisher 信息矩阵：

```python
mat = bayoma.posterior_matrix(
    f=out["f"][0], z=out["z"][0], S=out["Sii"][0], Se=out["Se"][0],
    kappa=10.0, duration=tdata.shape[0] / fs, nch=tdata.shape[1], q=0,
)
print("相关系数矩阵:\n", mat["corr"])
```

只有阻尼比与模态力谱之间显著相关（κ = 10 时后验相关系数约 +0.53），其余参数对都在
0.03 以内；相关性的强度随频带宽度增大而减弱。注意这里的正号是**后验相关系数**的符号；
课本式 (16.35) 的敏感度系数 `q_{ζS}` 取负，两者相差一个负号（`ρ_{ζS} ≈ −q_{ζS}`）。

### 3.4 试验前估算记录长度

反过来用同一个定律：给定频率的目标精度，算出需要多长记录。

```python
need = bayoma.optimal_duration(
    f=1.0, z=0.01, S=1.0, Se=1.0, kappa=10.0, nch=3, target=1.0e-3,
)
print(f"频率变异系数达到 0.1% 需要 {need:.0f} s")
```

`target` 判定的是**频率**的变异系数；阻尼比的变异系数约大两个量级，要控制阻尼比精度时
把目标值按比例换算后再代入，或直接用 3.3 节的 `uncertainty_law_cov` 试算几个记录长度比较。

---

## 第 4 步：多测组（分几次测、每次测不同测点）

场景：现场分批测量，`setup1` 测 1–3 号测点，`setup2` 测 3–5 号测点，3 号点为参考自由度，
两组共用同一个全局振型。

```python
from bayoma import SetupData, mpv_multi_setup, scaled_fft

setups = []
for path, dofs in ((r"D:\data\setup1.csv", [0, 1, 2]),
                   (r"D:\data\setup2.csv", [2, 3, 4])):
    arr = np.loadtxt(path, delimiter=",")
    sp = scaled_fft(arr, 100.0, (2.9, 3.1))
    setups.append(SetupData(freqs=sp.freqs, fft=sp.fft, dofs=dofs))

res = mpv_multi_setup(setups, n_dof=6)      # 全局共 6 个自由度
print("逐测组频率:", res.f)
print("逐测组阻尼比:", res.z)
print("全局振型:", res.phi.ravel())
```

要点：

- `dofs` 是**全局自由度编号**（Python 侧 0 起计），两组都能测到的编号即参考自由度；
- 每个测组可以有独立的 `f`、`z`、`S`、`Se`，全局振型只有一套；
- 参考自由度凝聚缺省开启，全局振型在凝聚子空间内求解，无需额外设置。

MATLAB 侧见 [USAGE.md](USAGE.md) 3.4 节，`dofs` 换成 1 起计。

### 4.1 异步采集（组内同步、组间不同步）

场景：两台采集仪各自同步，互相之间没有共同时基。这时把通道按采集仪分组，组内的振型按
同步模型处理，组间用复相干矩阵描述模态力的相关性。

```python
from bayoma import mpv_asynchronous, asynchronous_covariance, scaled_fft

arr = np.loadtxt(r"D:\data\async.csv", delimiter=",")   # 4 通道
sp = scaled_fft(arr, 100.0, (0.9, 1.1))
groups = [[0, 1], [2, 3]]        # 两台仪器各 2 通道，0 起计

res = mpv_asynchronous(sp.freqs, sp.fft, groups, q=0)
cov, coefv = asynchronous_covariance(res, sp.freqs, sp.fft, groups, q=0)
print("f, z =", res.f, res.z)
print("逐组噪声谱 =", res.Se)
print("组间相干 =", abs(res.coherence[0, 1]), "相位 =", np.angle(res.coherence[0, 1]))
print("相干模的变异系数 =", coefv["v"])
```

要点：

- `groups` 必须互不重叠且覆盖全部通道，组内通道必须共用时基；
- 上式只用于一个频带内只有一阶模态的情形。频带内有多阶时用多阶模型：
  `mpv_asynchronous_multi(sp.freqs, sp.fft, groups, n_mode=2, q=0, f0=[1.0, 1.4])`，返回的
  `phi` 为全部通道上的各阶振型，`local` 给出逐测组的局部振型，`coherence` 是各阶对的
  相干矩阵字典；
- 相干矩阵的**模**可靠；相位与各阶振型的整体相位基准耦合，比较相位时先统一振型的
  主要分量为正实数；
- 记录长度按 4800 s 量级取，阻尼比相对误差的标准差在 4.4% 左右（2400 s 时为 7.8%）。

JSON 描述见 [IO_SPEC.md](IO_SPEC.md) 6.1 节的 `asynchronous` 段。

---

## 第 5 步：交给其他软件或智能体调用

不写 Python 或 MATLAB 代码，用一份 JSON 描述文件驱动：

`job.json`：

```json
{
  "data": {"path": "D:/data/record.csv", "format": "csv", "fs": 100.0},
  "bands": [[0.9, 1.1]],
  "f0": [[1.0]],
  "options": {"q": 0, "mute": 1},
  "output": {"path": "D:/data/result.json"}
}
```

执行（先把包目录加进 `PYTHONPATH`，或在 `python/` 目录下运行）：

```powershell
$env:PYTHONPATH = "$PWD/python"
python -m bayoma.io_json D:\data\job.json
```

MATLAB 侧读同一份描述：

```matlab
result = bayoma.io_json('D:\data\job.json');
```

结果文件包含 `f`、`z`、`Sii`、`Se`、`phi`、`rms`、`coefv`、`cmat`、`sn`、`bands`、
`n_modes`、`elapsed` 等字段，复数矩阵写作 `{"re": …, "im": …}`。失败时返回非零退出码，
标准错误给出原因。完整字段见 [IO_SPEC.md](IO_SPEC.md) 第 6 节。

交给别的软件或智能体之前，建议先用工程里的两份 JSON Schema 校验，避免字段形状对不上：

```powershell
python python/tests/test_schema.py    # 工程自检，约 6 分钟
```

自己调用时只需两行（`schema/job.schema.json` 校验描述文件、`schema/result.schema.json`
校验结果文件）：

```python
import json, jsonschema
jsonschema.validate(spec,   json.load(open("schema/job.schema.json",    encoding="utf-8")))
jsonschema.validate(result, json.load(open("schema/result.schema.json", encoding="utf-8")))
```

同一字段在三种识别路径下形状一致：`f`、`z`、`Sii`、`Se`、`rms`、`sn` 一律一维数组
（只有一阶也写 `[x]`），`phi` 一律二维，`input` 内的 `fs`/`nch`/`nt`/`duration` 是数字。

多测组也可用 JSON 驱动，见 `IO_SPEC.md` 6.1 节的 `multi_setup` 段；异步采集用同一份
描述文件的 `asynchronous` 段，结果另给 `groups` 与 `coherence`。

---

## 第 6 步：把结果写进报告

一条模态的完整报告应包含：

| 项 | 取自 | 示例 |
| --- | --- | --- |
| 频率 | `out["f"][i]` | 1.0015 Hz |
| 阻尼比 | `out["z"][i]` | 0.96% |
| 阻尼比变异系数 | `out["coefv"]["z"][i]` | 9.7% |
| 模态力谱 | `out["Sii"][i]` | 0.997 g²/Hz |
| 通道噪声谱 | `out["Se"][i]` | 0.937 g²/Hz |
| 模态信噪比 | `out["sn"][i]` | 2892 |
| 振型 | `out["phi"][:, i]` | 单位范数，与参考振型的 MAC |
| 记录长度与频带 | 输入 | 2400 s，0.90–1.10 Hz |

振型只给单位范数形式与 MAC，不给符号或整体相位——这两者在模型里不可辨识。

---

## 常见坑速查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 阻尼比忽大忽小 | 记录太短 | 见 3.4 节估算所需长度，加长到 2400 s 以上 |
| 自动选带漏掉已知模态 | 该阶峰不够突出 | 手工给 `f1f2` 与 `f0` |
| 自动选带把噪声判成模态 | 用单次 FFT 判模态数 | 程序已改用 Welch 平均谱；若自写代码需注意 |
| 阻尼比系统性偏大 | `p` 与 `q` 不匹配 | 原始数据加速度时 `p=0`，模型侧加速度时 `q=0` |
| 变异系数全为 0 | 谱参数量级与振型相差过多 | 程序已按对角元预条件；自写协方差需注意 |
| 多测组 `S` 比预期大若干倍 | 合成数据时局部振型被单独归一化 | 取 `t_r = L_r·u` 的对应分量 |
| 密集模态 `S` 的非对角元每次不同 | 短记录下互谱不可辨识 | 加长记录；报告中说明该情形 |
| MATLAB 报 `pageinv` 冲突警告 | 与本工程无关 | 忽略 |
| 异步数据报「不支持使用花括号索引」 | `groups` 被 `jsondecode` 解成数值矩阵 | 程序已统一接住两种形态；自写脚本需按「每项一个编号列表」给 |

---

## 命令速查

```powershell
# 自检（约 1 分钟）
cd bayoma-oma
python python/tests/test_smoke.py

# 参考自由度凝聚校核（课本 Example 14.2）
python python/tests/test_condense.py

# 异步数据校核（四个算例）与 JSON 端到端
python python/tests/test_asynchronous.py
python python/tests/test_async_json.py

# 统计校核（20 个种子，约 205 s）
python python/tests/test_monte_carlo.py 20

# 第 16 章 Fisher 矩阵与表 15.1 的一致性与闭式核对（秒级）
python python/tests/test_uncertainty.py

# 两份 JSON Schema 与四类算例（约 6 分钟）
python python/tests/test_schema.py

# 现场实测数据核对（依赖 pcode 数据目录）
python python/tests/test_field_data.py

# 示例
python examples/py_single_mode.py
python examples/py_multi_mode.py
python examples/py_multi_setup.py
python examples/py_json_job.py

# MATLAB 全量自检（十四个脚本，约 30 分钟）
```

```matlab
addpath(genpath('matlab'));
cd('matlab/tests'); run_all
```
