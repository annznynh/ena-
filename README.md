# 知迹：学生答题预测与 ENA 学情预警

按《知迹_项目可行性与开发规格.md》的 MVP 路线实现：全量数据契约与清洗 → 固定学生划分 → 常数/逻辑回归/XGBoost → 消融评价 → 正式 rENA → FastAPI → Streamlit 教师端。

已在本机完成全量实验、模型保存重载、学生级 bootstrap、ENA 窗口与阈值敏感性分析。研究结果和限制见 [实验与审查报告](reports/实验与审查报告.md)。LSTM 属于文档第二阶段，本次未实现，也没有用旧 GRU 结果冒充 LSTM。

## 本机启动

在 VS Code 中打开项目根目录后：

```bash
# 一条命令启动 API（8000）与网页（8501），Ctrl+C 停止
scripts/run zhiji demo

# 一条命令运行 Ruff、格式、mypy、pytest 与已安装 R 的端到端测试
scripts/test

# 真实离线特征与服务预测一致性审查，需要本地实验产物
scripts/run python scripts/audit_serving.py
```

打开 [教师端](http://127.0.0.1:8501) 或 [API 文档](http://127.0.0.1:8000/docs)。网页只调用 API，启动不会训练模型。API 缺少模型或 ENA 产物时返回 503。

## 新电脑配置（macOS）

Python 3.11；依赖以 `uv.lock` 为准。文档原先要求的 XGBoost 3.4 不支持 Python 3.11，已经按实际安装结果修正，当前锁定 XGBoost 3.2.0。

```bash
python3 -m venv .bootstrap
.bootstrap/bin/python -m pip install uv
export LC_ALL=en_US.UTF-8
.bootstrap/bin/uv python install 3.11
.bootstrap/bin/uv sync --locked
```

安装 [R 4.5.3 Apple Silicon 官方包](https://cran.r-project.org/bin/macosx/big-sur-arm64/base/R-4.5.3-arm64.pkg)，再恢复依赖：

```bash
LC_ALL=en_US.UTF-8 R --vanilla -q -e 'install.packages("renv", repos="https://cloud.r-project.org"); renv::restore(project=".", prompt=FALSE)'
```

`renv.lock` 锁定 rENA 0.3.1。`scripts/run` 设置 macOS UTF-8 区域和锁定 PyTorch 附带的 OpenMP 动态库搜索路径；避免中文目录启动失败和 XGBoost 缺少 libomp。锁文件中的镜像用于改善下载速度，不存储认证信息。

## 全量复现顺序

将已有 `2012-2013-data-with-predictions-4-final.csv` 放在根目录。下载命令会优先以只读符号链接复用已有文件，计算 SHA-256；没有本地文件时才使用 KaggleHub 下载。

```bash
scripts/run zhiji data download --config configs/full.yaml
scripts/run zhiji data validate --config configs/debug.yaml
scripts/run zhiji data ingest --config configs/full.yaml
scripts/run zhiji data clean --config configs/full.yaml
scripts/run zhiji data split --config configs/full.yaml
scripts/run zhiji data build-examples --config configs/full.yaml
scripts/run zhiji data features --config configs/full.yaml
scripts/run zhiji train baseline --config configs/full.yaml
scripts/run zhiji train xgb --config configs/full.yaml
scripts/run zhiji evaluate --model xgb --split test --config configs/full.yaml
scripts/run zhiji ena export --config configs/full.yaml
LC_ALL=en_US.UTF-8 Rscript --vanilla analysis/ena/run_ena.R configs/full.yaml
scripts/run zhiji ena validate-artifacts --config configs/full.yaml
scripts/run python scripts/finalize_artifacts.py
scripts/run zhiji build-demo
scripts/run zhiji demo
```

导入和模型产物不会静默覆盖：复现实验请使用干净 checkout 或新的输出目录，保留已有划分与测试报告。`debug.yaml` 的 5,000 行只用于字段验证；`data ingest` 始终导入全量，不能用原文件前几千行冒充完整学生序列。

敏感性分析使用相同 ENA CSV 的窗口 2、4、6；情感阈值 0.33、0.67 必须重新调用 `export_ena_input` 生成对应 CSV，再传给 R 脚本。自动运行命令见 `scripts/ena_sensitivity.py`。R 图形导出使用 Python SVG，不需要 XQuartz。

## 数据与研究边界

- 主文件实际有 6,123,270 行，清洗后 6,032,040 条交互、46,325 名学生。
- 按学生 70/15/15 划分，固定种子 42；训练 32,427、验证 6,949、测试 6,949 名学生。
- 原始数据、模型、Parquet、学生划分与演示明细保留在本地，默认不上传 GitHub。代码、环境锁、匿名聚合指标及审查报告可提交。
- 本地原文件的上游 Kaggle 版本尚未核实；manifest 如实记录为未知。发布正式论文前需补充来源版本和情感检测论文引用。
- 当前是单次学生划分实验，没有多 seed 或跨学校外部验证，也尚未开展真实教师访谈；不能声称干预有效或泛化到真实学校。
- 四个情感字段是系统预测置信度；ENA 是群体共现描述，不是因果诊断。没有自动教学决策或消息发送功能。
- 原始时间没有时区。API 演示记录的 `Z` 只用于统一传输格式，不能解释为经过真实时区校正的 UTC 时间。
- `model.py`、`zhiji_pipeline.py`、`requirements.txt` 是保留的历史探索材料；正式维护入口为 `src/zhiji`，历史脚本不属于本次可复现运行入口。
