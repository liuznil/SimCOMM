# 毫米波阻塞与切换的可预测性：DeepMIMO 仿真 → Lumos5G 真实网络

实证研究代码：只用 DeepMIMO 射线追踪仿真训练阻塞预测模型，**零样本**检验其能否预测 Lumos5G 商用 28 GHz 5G 网络中的真实事件（信号跌落、5G 掉线、水平切换、吞吐量骤降）。
本项目**不提出新方法**，目标是给出一个透明的评估流程和可预测性的限度。

---

## 1. 目录与文件

```
scripts/                      # 所有 .py 放同一目录（脚本互相 import）
├── common.py                 # 共用：窗口、特征、事件标签定义（仿真与真实两侧共用同一份）
├── download_data.py          # 下载/定位数据
├── sim_dataset.py            # 步骤1：DeepMIMO 网格 → 仿真轨迹与阻塞事件
├── real_dataset.py           # 步骤2：Lumos5G CSV → 1 Hz 段与真实事件
├── train_eval.py             # 步骤3：训练 + 零样本评估 + 对照 + 域偏移
├── profile_scenarios.py      # 选城市：只看仿真侧统计
├── calibrate_sim.py          # 校准仿真平滑度/遮挡体（只用真实信号统计，不用标签）
├── summarize_runs.py         # 汇总多次运行为敏感性表
├── event_metrics.py          # 固定误报率下的精确率、事件召回、提前量
├── decompose_drop.py         # 拆解真实"信号跌落"里有多少是 5G 掉线
├── single_city_robustness.py # 单城市 / 留一城市训练的稳健性
├── requirements.txt
├── data/                     # 自动创建：lumos5g/、sim_trajs.pkl、real_segs.pkl 等
└── results/                  # 自动创建：results.csv、图、事件级结果等
```

`data/` 和 `results/` 都位于脚本所在目录下（`scripts/data`、`scripts/results`），与运行时所在目录无关。`summarize_runs.py` 默认把汇总写到当前目录下的 `results/sensitivity.csv`。

## 2. 环境

- Python 3.11 或 3.12 推荐（3.14 上曾出现 NumPy 导入错误，见第 6 节）。
- `pip install -r requirements.txt`（numpy、pandas、scikit-learn≥1.2、matplotlib、deepmimo≥4.0、requests）。
- 下载数据需要联网。

## 3. 核心概念

| 概念 | 说明 |
|---|---|
| 共同表示 | 只用窗口 W=10 s 内**相对**信号历史（相对当前值的 dB 差、差分的最后值/均值/最小值/标准差、距窗口最大值的跌幅）和速度；**不用绝对电平**（−路径损耗与 RSRP 不可比） |
| 视野 H | 预测未来 2、5、10 s |
| 标签 | `drop`（两侧通用：未来 H 内下降 ≥10 dB）、`drop_geo`（仿真：不含随机遮挡体）、`los2nlos`（仿真：LOS→NLOS）、`nrloss`（真实：5G→4G）、`hho`（真实：tower_id 变化）、`tput`（真实：吞吐量降至近期中位数 30% 以下，起点 ≥100 Mbps） |
| 零样本 | 模型只在仿真上训练，直接在真实数据上评分 |
| 评估 | 真实数据按 run 分组（同一 run 的多个 segment 不跨折）；AUROC、AP、lift、FPR=10% 召回；置信区间按 run 自助抽样，另报告相对启发式的配对 AUROC 差值区间 |

## 4. 各脚本说明与用法

下面命令均在项目根目录（含 `scripts/`）执行。

### 4.1 `download_data.py`
```bash
python scripts/download_data.py --search                  # 列出 mmWave 室外 DeepMIMO 场景
python scripts/download_data.py --scenario <场景名>       # 下载一个 DeepMIMO 场景
python scripts/download_data.py                           # 打印 Lumos5G 手动下载说明
python scripts/download_data.py --lumos-url <直链>        # 若有直接链接
```
Lumos5G 需手动下载（lumos5g.umn.edu 或 IEEE DataPort），把 CSV 放入 `scripts/data/lumos5g/`。

### 4.2 `profile_scenarios.py`（选城市）
```bash
python scripts/profile_scenarios.py --scenario city_0_newyork_28 city_2_chicago_28 ...
```
输出每个场景的网格大小、可用格点比例、LOS 占比、LOS→NLOS 与几何跌落事件率 → `data/scenario_profile.csv`。
**只用仿真侧统计选城市**，不要依据真实数据结果调整，以免选择偏差。`error` 列非空的场景需排除（如 `high <= 0` 表示该基站下无可用格点）。室内场景（`i2_28b`、`i3_60`）不适用。

### 4.3 `sim_dataset.py`（步骤 1：仿真）
```bash
python scripts/sim_dataset.py --scenario city_0_newyork_28 city_9_sanfrancisco_28 \
    city_2_chicago_28 city_3_houston_28 city_17_seattle_28 --n-train 200 --n-test 80
python scripts/sim_dataset.py --synthetic        # 冒烟测试，无需下载
```
主要参数：

| 参数 | 默认 | 作用 |
|---|---|---|
| `--scenario` | — | 一个或多个场景，轨迹池化；每条轨迹带 `scen` 城市标签 |
| `--tx` | 0 | 使用的基站序号（每场景只用一个基站） |
| `--n-train` / `--n-test` | 400 / 150 | 每个场景的轨迹数 |
| `--steps` | 300 | 每条轨迹的步数（1 步 = 1 s） |
| `--blocker-rate` | 0.01 | 瞬时遮挡体（人/车）每秒发生率 |
| `--dip-min` / `--dip-max` | 8 / 25 | 遮挡体深度 (dB) |
| `--smooth` | 1.0 | 指数平滑系数 α∈(0,1]；1 = 不平滑 |
| `--seed` | 0 | 随机种子 |

仿真设定：随机游走（步行 0.8–2 m/s 或驾车 4–15 m/s）、AR(1) 阴影、测量噪声、信号量化到 1 dB，地图右侧 30% 作空间留出测试集。输出 `data/sim_trajs.pkl`。

### 4.4 `real_dataset.py`（步骤 2：真实数据）
```bash
python scripts/real_dataset.py --csv-dir scripts/data/lumos5g
python scripts/real_dataset.py --synthetic       # 冒烟测试（生成假 CSV）
```
- 自动检测列名并**打印列映射和 `nrStatus` 取值**，请核对。
- `--nr-connected-value`：5G 已连接状态的取值，默认 `CONNECTED`。
- 列名不同则修改脚本中的 `CAND` 字典；`movingSpeed` 按 m/s 处理，若是 km/h 需换算。
- 按 run 和时间间断切分为 1 Hz 段；NR 断开时信号填 −140 dBm。
- 输出 `data/real_segs.pkl`，并打印各视野下的样本数与事件发生率。

### 4.5 `calibrate_sim.py`（可选：校准仿真真实感）
```bash
python scripts/calibrate_sim.py --scenario city_0_newyork_28 city_9_sanfrancisco_28 ...
```
扫描平滑度 × 遮挡体频率 × 遮挡体深度，输出仿真-真实域分类器 AUROC 和 `d_min/d_std/below_max` 的单特征 AUROC → `data/calibration.csv`。只用真实**信号统计**，不用标签。
注意：域分类器分数**不是**迁移效果的可靠代理（本研究中平滑反而使零样本变差），详见论文。

### 4.6 `train_eval.py`（步骤 3：训练与评估）
```bash
python scripts/train_eval.py                 # 完整运行（含速度特征）
python scripts/train_eval.py --no-speed      # 把速度特征置零（消融）
python -W ignore::UserWarning scripts/train_eval.py   # 屏蔽 sklearn/joblib 提示
```
| 参数 | 默认 | 作用 |
|---|---|---|
| `--horizons` | 2 5 10 | 视野 |
| `--targets` | drop tput nrloss hho | 真实评估目标 |
| `--k-list` | 2 5 10 | 对照实验中使用的真实 run 数 |
| `--boot` | 100 | 自助法次数（0 = 不算置信区间，更快） |
| `--max-sim` | 100000 | 仿真训练样本上限 |
| `--real-weight` | 0.3 | 仿真+少量真实训练时真实样本的权重 |
| `--no-speed` | 关 | 置零速度特征 |

对比的方法：`heuristic`（−最近一步差分）、`sim_drop`、`sim_dropgeo`、`sim_los2nlos`（零样本）、`sim+real_k`、`realonly_k`（对照：相同 k 个真实 run、不含仿真）、`real_only`（分组交叉验证上界）。
`sim+real_k` 当前实现为一次加权联合拟合：仿真样本使用 `drop` 标签，真实样本使用目标事件标签；它不是先仿真预训练、再用真实数据微调。
输出：
- `results/results.csv`：全部指标（含仿真留出区自检 `SIM-holdout:*` 与 `domain-classifier` 行）
- `results.csv` 中 `delta_vs_heuristic` 及 `delta_ci_lo/hi` 为相对启发式基线的差值和配对 run-bootstrap 95% 区间。
- `results/feature_shift_H*.csv`：每个特征的仿真/真实可分度（AUROC，真实=1）
- `results/auroc_H*.png`
- 域分类器按仿真轨迹和真实 run 分组留出验证，避免重叠窗口同时进入训练集与验证集。

### 4.7 `summarize_runs.py`（敏感性表）
```bash
python scripts/summarize_runs.py A_speed=results_1.csv B_nospeed=results_nospeed.csv C_smooth=results_smooth.csv
```
把多次 `results.csv` 汇总：打印不同配置下各事件的零样本 AUROC [CI] 及域分类器 AUROC，并写入**当前工作目录**下的 `results/sensitivity.csv`（需有写权限）。运行前把不同配置的 `results.csv` 改名保存；也可用 `--out` 指定输出路径。

### 4.8 `event_metrics.py`（事件级指标）
```bash
python scripts/event_metrics.py --no-speed --real-only
```
参数：`--horizons`（默认 5 10）、`--targets`、`--fprs`（默认 0.05 0.10）、`--real-only`（加入真实训练参考）。
输出 `results/event_metrics.csv`：精确率、事件召回、提前量中位数、每小时误报次数。
定义与注意：
- 阈值按真实**负样本**的 (1−FPR) 分位数校准，是 oracle 工作点，仅用于在相同误报率下比较方法。
- 事件召回 = 至少有一次报警的预警段（连续正样本段）占比。
- 提前量对 `nrloss`、`hho` 是准确的事件前时间；对 `drop`、`tput` 只是预警窗口内的剩余时间（近似）。

### 4.9 `decompose_drop.py`（拆解"信号跌落"）
```bash
python scripts/decompose_drop.py --no-speed
```
输出 `results/drop_composition.csv`（真实 drop 正样本中含 5G 掉线 / 仅水平切换 / 纯跌落的占比）和 `results/drop_decomposition.csv`（在原标签、`drop_pure`、`drop_pure_on5G_only` 上的零样本 AUROC；含 5G 掉线的窗口被排除，不计为负样本）。

### 4.10 `single_city_robustness.py`（单城市稳健性）
```bash
python scripts/single_city_robustness.py --no-speed
```
需要 `sim_trajs.pkl` 由多场景版 `sim_dataset.py` 生成。分别用：全部城市、单个城市、留一城市训练，在真实事件上零样本评估。输出 `results/single_city.csv`，并打印各城市 AUROC 的最小/中位/最大值。

## 5. 推荐流程

```bash
pip install -r scripts/requirements.txt
# 1) 数据
python scripts/download_data.py --search
python scripts/download_data.py --scenario <场景名>          # 逐个下载所需城市
#    手动下载 Lumos5G，CSV 放入 scripts/data/lumos5g/
# 2) 选城市（只看仿真侧）
python scripts/profile_scenarios.py --scenario <候选城市...>
# 3) 生成数据
python scripts/sim_dataset.py --scenario <选定城市...> --n-train 200 --n-test 80
python scripts/real_dataset.py --csv-dir scripts/data/lumos5g
# 4) 主实验（建议 A/B/C 多配置都跑，改名保存 results.csv）
python scripts/train_eval.py                 # A：含速度
python scripts/train_eval.py --no-speed      # B：无速度
#    C：无速度 + 平滑仿真（先 sim_dataset.py --smooth 0.7 重新生成，再 train_eval.py --no-speed）
# 5) 附加分析
python scripts/event_metrics.py --no-speed --real-only
python scripts/decompose_drop.py --no-speed
python scripts/single_city_robustness.py --no-speed
python scripts/summarize_runs.py A=... B=... C=...
```

`train_eval.py` 与 `calibrate_sim.py` 的域分类器 AUROC 使用按轨迹/run 分组的留出集；对已有缓存数据重跑即可更新这一项，无需重新下载数据。
`train_eval.py` 为每个视野和事件目标使用独立的固定随机种子，因此增加或重排其他视野/目标不会改变该组合的抽样结果。
注意：每次重新运行 `sim_dataset.py` 都会覆盖 `data/sim_trajs.pkl`；换配置前如需保留旧数据，先自行备份。

## 6. 常见问题

- **导入 NumPy 报 `cannot convert longdouble infinity to integer`**：多见于新版 Python 与 NumPy 组合。换用 Python 3.11/3.12 并建虚拟环境，或 `pip install -U --force-reinstall numpy`。
- **sklearn `UserWarning: sklearn.utils.parallel.delayed ...`**：版本提示，不影响结果；可用 `python -W ignore::UserWarning ...` 屏蔽。
- **`boston5g_28` 报 `high <= 0`**：该基站下无可用格点，先排除或换 `--tx`。
- **`real_dataset.py` 报找不到列**：对照打印的列名，修改 `CAND`。
- **运行很慢**：减小 `--max-sim`、`--boot`、`--horizons`，或减少 `--n-train`。

## 7. 局限（使用结果前请阅读）

- 仿真城市不是 Lumos5G 的测量区域；每个场景只用一个基站；仿真信号是 −路径损耗而非 RSRP；Lumos5G 无 CSI。
- 步行/驾车速度、阴影、噪声、遮挡体与平滑系数是研究者设定的，不是实测值；结果对速度特征和平滑度敏感。
- 去掉速度是在看到真实数据结果后决定的，应当作为消融实验报告，并汇报**所有**配置。
- 标签阈值（10 dB、30%、100 Mbps、−140 dBm 填充）是人为选择；置信区间不含仿真随机种子带来的波动；事件级阈值为 oracle 校准。
- 列名、`nrStatus` 取值和速度单位需自行核对。
- 本代码只在合成数据与研究者提供的结果文件上检验过流程，未保证在其他数据版本上无需修改。
