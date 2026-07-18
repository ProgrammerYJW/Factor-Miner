# FactorMiner — 强化学习 + 遗传规划 A股因子挖掘系统

## 安装与部署（无需连接数据库）

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
git clone <你的仓库URL>
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

### 3. 构建特征矩阵

**Windows：**
```bat
.venv\Scripts\python.exe scripts\build_features.py
```

**Mac：**
```bash
.venv/bin/python scripts/build_features.py
```

此步骤读取项目自带的 `data_cache/features/`，不连接任何数据库。完成后 PyCharm 打开项目，解释器选 `.venv` 下的 python，即可使用右上角运行配置启动挖掘和 Web 界面。

## 使用方法

### 启动挖掘引擎

PyCharm 右上角运行配置下拉选择：
- **4-GP挖掘因子** → 点 ▶ 开始遗传规划挖掘
- **5-RL挖掘因子** → 点 ▶ 开始强化学习挖掘

GP 和 RL 可以同时运行，互不影响。每次启动会自动从上次的检查点续跑。

### 查看因子库

运行配置选择 **6-Web因子库界面** → 点 ▶，浏览器打开 `http://localhost:8501`。左侧导航：
- **因子库总览**：全部因子列表，默认按 IC/IR 降序，支持筛选搜索
- **因子详情**：课本格式定义式、各周期指标矩阵、累计 IC 曲线、删改操作
- **筛选标准设置**：在线修改因子入库门槛，保存即生效
- **挖掘监控**：GP/RL 实时进度曲线
- **相关性矩阵**：库内因子相关性热力图

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
├── config/settings.toml         # 全部可调参数
├── scripts/                     # 入口脚本与 .bat 启动器
├── .run/                        # PyCharm 运行配置（打开即用）
├── factor_miner/
│   ├── data/                    # 数据库客户端、同步器、特征构建
│   ├── expression/              # 算子库、AST、解析器、课本格式渲染
│   ├── evaluation/              # 预处理、指标计算、评估器
│   ├── engines/gp/              # GP 遗传规划引擎
│   ├── engines/rl/              # RL 强化学习引擎（PPO）
│   ├── library/                 # 因子库、准入管道、筛选规则
│   └── webapp/                  # Streamlit 五页面 Web 界面
├── data_cache/features/         # 预构建的特征矩阵（parquet）
├── tests/                       # pytest 单元测试
└── README.md
```

## IC/IR 指标说明

- **IC** = 日度 RankIC 的均值
- **IC/IR** = IC 均值 ÷ IC 标准差（信息比率，Grinold & Kahn 教材标准定义）

因子列表和详情页中两列并排置顶显示，筛选标准支持按 IC/IR 设定门槛。

## 常见问题

**Q: 运行 1-环境自检 或 2-同步聚源数据 报连接失败？**
A: 确认已关闭代理/梯子（内网 `192.168.219.222` 不走公网）。数据已缓存在本地，日常使用不需要这两个步骤。

**Q: 项目能搬到别的文件夹吗？**
A: 可以，整个 `FactorMiner` 文件夹拖走即用。`.bat` 启动器和 PyCharm 配置均使用相对路径自动定位。
