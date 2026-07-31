# Factor Miner — 强化学习 + 遗传规划 A股量化因子挖掘系统

## 项目简介

Factor Miner 是一套面向量化研究的因子挖掘系统，核心目标是在海量候选表达式空间中寻找具有稳健截面预测能力的α因子。系统综合采用遗传规划（Genetic Programming）和强化学习（PPO）两套引擎并行搜索，共享同一套算子库和评估体系，互相补充。

### 算法原理

**遗传规划引擎**将因子表达式编码为语法树，在种群中通过锦标赛选择、子树交叉、多种变异（子树/点/hoist）逐代演化，以训练段的截面预测性能为适应度，同时施加简约压力和因子池相关性惩罚，自动筛选简洁且独特的新因子。

**强化学习引擎**采用 AlphaGen 式的逆波兰（RPN）逐令牌生成策略：小型 Transformer 策略网络在合法的动作空间内逐步构造因子表达式，以终端预测性能为奖励信号，通过 PPO 算法（自研，仅依赖 PyTorch）训练网络参数。动作掩码保证每一步生成的表达式均语法合法，已见表达式给予重复惩罚以鼓励探索。

### 评估体系

所有候选因子统一经过：

- **预处理**：MAD 去极值 → 可选的行业与市值中性化 → 截面 z-score 标准化
- **多周期多时段评估**：1 / 5 / 10 / 20 日四个预测周期 × 训练 / 验证 / 观察三个时段，全量指标矩阵
- **核心指标**：IC均值、IC标准差、**IC/IR**（= IC均值 ÷ IC标准差，信息比率，Grinold & Kahn 教材标准定义）、多空年化收益与夏普比率、换手率代理等
- **可配置准入管道**：支持任意指标 × 任意周期 × 任意时段 × 任意大小关系组合为"与"逻辑的筛选规则，可在 Web 界面在线编辑、保存即生效

### 因子库与可视化

挖掘出的因子持久化存储在本地因子库中（SQLite 管理元数据 + Parquet 存储截面值矩阵），配套 Streamlit 本地 Web 界面：

- **因子库总览**：全部因子列表，默认按 IC/IR 绝对值降序，表达式以课本数学格式渲染（正体函数名、下标窗口、分数线等）
- **因子详情**：定义式、四周期 × 三时段完整指标矩阵、累计 IC 曲线、月度热力图、分层分组收益、删改操作
- **筛选标准设置**：在线增删改准入规则，保存后对正在运行的挖掘进程即时生效
- **挖掘监控**：GP 各代适应度曲线、RL 奖励曲线、近期入库因子流水
- **相关性矩阵**：库内活跃因子的截面相关性热力图

## 安装与部署

本项目自带预构建的特征矩阵数据，**部署过程不需要连接任何外部数据库**。

### 1. 安装 Git LFS（一次性）

Windows 的 Git 自带 `git-lfs.exe`，只需注册一次：
```bash
git lfs install
```

Mac：
```bash
brew install git-lfs
git lfs install
```

验证：
```bash
git lfs version          # 显示 git-lfs/x.x.x 即成功
```

### 2. 克隆项目并安装依赖

```bash
git clone https://github.com/ProgrammerYJW/Factor-Miner.git
cd FactorMiner
```

创建虚拟环境并安装依赖：

**Windows：**
```bat
python -m venv .venv
.venv\Scripts\pip install --no-cache-dir -r requirements.txt
```

**Mac：**
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 3. 启动

PyCharm 打开项目，解释器选 `.venv` 下的 python，即可使用右上角运行配置直接启动挖掘和 Web 界面。项目自带预构建的特征矩阵（`data_cache/features/`），无需额外构建步骤。

## 使用方法

### 启动挖掘引擎

PyCharm 右上角运行配置下拉选择：
- **4-GP挖掘因子** → 点 ▶ 开始遗传规划挖掘
- **5-RL挖掘因子** → 点 ▶ 开始强化学习挖掘

GP 和 RL 可以同时运行，互不影响。每次启动会自动从上次的检查点续跑。

### 查看因子库

运行配置选择 **6-Web因子库界面** → 点 ▶，浏览器打开 `http://localhost:8501`。

### 命令行启动（可选）

```bash
# Windows
.venv\Scripts\python.exe scripts\run_gp.py      # GP 挖掘
.venv\Scripts\python.exe scripts\run_rl.py      # RL 挖掘
scripts\start_webapp.bat                        # Web 界面

# Mac
.venv/bin/python scripts/run_gp.py
.venv/bin/python scripts/run_rl.py
```

## 项目结构

```
FactorMiner/
├── config/settings.toml         # 全部可调参数（连接信息、股票池、时间切分、门槛、GP/RL 超参）
├── scripts/                     # 入口脚本与 .bat 启动器
├── .run/                        # PyCharm 共享运行配置（打开即用）
├── factor_miner/
│   ├── data/                    # 数据访问、同步器、特征构建
│   ├── expression/              # 算子库（31 个逐元素/时序/截面算子）、表达式 AST、解析器、
│   │                            #   课本格式数学渲染器
│   ├── evaluation/              # 因子预处理、IC/ICIR/分层回测/换手等指标、评估器
│   ├── engines/gp/              # GP 遗传规划引擎（种群演化 + 进程池并行评估）
│   ├── engines/rl/              # RL 强化学习引擎（RPN Token 空间 + Transformer 策略 + 自研 PPO）
│   ├── library/                 # 因子库持久化、准入管道、可配置筛选规则
│   └── webapp/                  # Streamlit 五页面 Web 交互界面
├── data_cache/features/         # 预构建的特征矩阵（parquet，通过 Git LFS 分发）
├── tests/                       # pytest 单元测试（43 项）
└── README.md
```

## 核心指标说明

- **IC** = 日度截面 RankIC 的均值
- **IC/IR** = IC 均值 ÷ IC 标准差（信息比率，Grinold & Kahn 教材标准定义）

因子列表和详情页中两列并排置顶显示，筛选标准支持按任意指标（IC 均值、IC/IR、多空夏普、秩自相关、与已有因子相关性等 11 种）设置门槛。

## FAQ

**Q: 项目能搬到别的文件夹或别的电脑吗？**

A: 可以，整个 `FactorMiner` 文件夹拖走即用。`.bat` 启动器和 PyCharm 运行配置均使用相对路径自动定位。

**Q: 如何在另一台机器上从零部署？**

A: 本项目仓库包含完整源码和预构建特征数据，按上方"安装与部署"三步操作即可，不需要连接外部数据库。

**Q: 如何调整挖掘强度或筛选标准？**

A: `config/settings.toml` 可调 GP 种群大小、代数、RL 更新轮数等全部参数；筛选标准在 Web 界面"筛选标准设置"页在线编辑，保存即生效。

**Q: GP 和 RL 能同时运行吗？**

A: 可以，两个引擎互不冲突，因子库使用 WAL 模式保证并发安全。建议 PyCharm 中同时启动两个运行配置。
