# dsh-bayoma-oma

DeepSeek-Harness 插件包，把 `bayoma-oma` 技能注册到当前 profile。技能内容对应
《Operational Modal Analysis: Modeling, Bayesian Inference, Uncertainty Laws》
（Springer）第 10–16 章的自动贝叶斯模态识别方法（单模态、多模态含密集模态、多测组含
参考自由度凝聚、异步数据（单阶与多阶）、后验协方差、自动选带、不确定度定律、Fisher
信息矩阵，以及分「识别结果」与「组装结果」两组的结果输出），配套实现在本仓库根目录
下的 `python/` 与 `matlab/`。

## 组成

| 文件 | 作用 |
| --- | --- |
| `package.json` | 包清单，`dsh.bundle.patch` 指向 `cordis.patch.yml` |
| `cordis.patch.yml` | 组合补丁，插入一条 `bayoma-oma` 条目并注入 `skills` |
| `lib/startup.js` | 读取 `skills/bayoma-oma/SKILL.md`，校验 frontmatter 后注册为运行时技能 |
| `skills/bayoma-oma/SKILL.md` | 技能正文 |
| `tests/check_skill_payload.mjs` | 自检：解析技能正文并核对章节与注册载荷 |

## 安装

在工作区的包目录上执行：

```text
plugin_manager action=install_bundle target=<本机绝对路径>\plugin/dsh-bayoma-oma
```

安装后插件进入当前 profile 的 `dsh.profile.bundles`，对所有会话生效并在重启后保留。
同一包名不能重复安装；改动后需先移除再装，或提升版本号并换新包名。

## 卸载

```text
plugin_manager action=list_plugins
plugin_manager action=set_plugin target=dsh-bayoma-oma enabled=false
```

## 使用说明

### 什么时候会用到

会话里出现「模态识别、工作模态、OMA、频率与阻尼比识别、振型识别、多测组、异步采集、
后验协方差、不确定度定律、Fisher 信息矩阵与参数相关性」一类任务时，模型按技能名
`bayoma-oma` 调用，读入技能正文，再按正文第 11 节给出的命令调用
仓库根目录下 `python/bayoma/` 与 `matlab/+bayoma/` 的两套实现。

插件只负责把技能送进技能目录；计算由工作区里的 Python 与 MATLAB 代码完成，两者可以
分开部署。

### 安装后怎么确认生效

| 检查项 | 操作 | 期望结果 |
| --- | --- | --- |
| 插件已装 | `plugin_manager action=list_plugins` | 列表中有 `dsh-bayoma-oma`，状态启用 |
| 技能已注册 | 新开会话问「列出可用技能」 | 列表中出现 `bayoma-oma` |
| 技能可执行 | 说「用 bayoma-oma 技能识别 `D:\data\record.csv`」 | 模型读技能后给出调用命令与结果解读 |
| 载荷完好 | `node plugin/dsh-bayoma-oma/tests/check_skill_payload.mjs` | 打印 `content len` 与各章节命中 `true` |

启动时插件会打印一行 `[dsh-bayoma-oma] registered bayoma-oma`；若打印
`[dsh-bayoma-oma] ctx.skills unavailable`，说明当前 profile 未提供技能服务，此时改用
账号级技能目录安装。

### 一次完整使用的四步

1. **给出数据与采样频率**：告诉模型数据文件路径、`fs`、通道顺序与单位（g 或 m/s²）。
2. **说明测量组织方式**：单次全通道同步测量；分多个测组分批测量（要给出各测组的全局
   自由度编号，同一编号代表同一个测点）；还是组内同步、组间不同步的异步采集（要给出
   通道分组）。
3. **确认频带策略**：让程序自动选带，还是指定频带与模态阶数初值。密集模态要给出
   频带内各阶的初值。要振型空间图时再给出各通道的空间位置、方向与连接关系。
4. **读结果**：模型给出频率、阻尼比、模态力谱、通道噪声谱、模态信噪比、振型与各自的
   变异系数（异步采集另给组间相干矩阵），并按技能正文第 10 节判读可靠性。

### 结果文件里的两组字段

结果文件在基础字段（`f`、`z`、`Sii`、`Se`、`phi`、`coefv` 等）之外分两组，读结果时
直接引用，不要自己另建索引：

| 组 | 什么时候有 | 内容 |
| --- | --- | --- |
| `recognition` | 所有识别任务 | 每个频率所属频带与代入的初值、密集模态区分度、联合后验协方差及其行列索引、逐阶振型不确定性、PSD 与 SV 谱、振型空间图 |
| `assembly` | 异步数据 | 组装后的振型、识别与组装两个环节的不确定性分解、总后验协方差及其索引、逐阶振型不确定性、组间相干矩阵、空间图 |

读法见技能正文第 10.1 节；字段级说明见 `docs/IO_SPEC.md` 第 8 节。

### 交给其他软件或智能体

结果与描述文件的字段形状由 `schema/` 下两份 JSON Schema 给出
（`job.schema.json`、`result.schema.json`，均为 2020-12）。调用方无需读技能正文即可先
校验描述文件、跑完再校验结果文件，再按字段取值；技能正文第 11.2 节给出命令行写法。

### 与账号级技能的关系

插件注册的技能与账号级技能目录 `%USERPROFILE%\.dsh\skills/bayoma-oma\SKILL.md` 内容相同。
两者任选其一即可；同时存在时以插件注册的为准（`source: runtime`）。

## 说明

本插件只注册技能，不注册工具、不修改会话事件、不注入样式。技能正文不依赖插件运行时，
即使插件未安装，把 `skills/bayoma-oma/SKILL.md` 复制到账号级技能目录同样可用。

配套实现在仓库根目录：`python/bayoma/` 为 Python 实现，`matlab/+bayoma/`
为 MATLAB 实现，`docs/` 含输入输出规范、算法对照、使用说明、校核记录与分步教程，
`schema/` 含机器可读契约，`promote/` 含实现记录与交接文件。
