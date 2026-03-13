# AutoInsight

A LangGraph-based automated data mining agent system. Upload a CSV file and get data profiling, EDA charts, model evaluation, and a generated analysis report.

## Requirements

- Python >= 3.11

## Setup

**Option A — uv (recommended)**

```bash
uv venv
uv pip install -e .
```

**Option B — pip**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

## 分支策略（贡献者须知）

本项目采用 **功能分支工作流**，请勿直接向 `main` 提交代码。

**操作步骤：**

1. 从 `main` 创建个人功能分支：
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. 在该分支上开发并提交：
   ```bash
   git add .
   git commit -m "feat: 描述你的改动"
   ```

3. 推送分支并发起 Pull Request：
   ```bash
   git push origin feature/your-feature-name
   ```
   然后在 GitHub 上向 `main` 发起 PR，等待 review 后合并。

**分支命名建议：**

| 类型 | 前缀 | 示例 |
|---|---|---|
| 新功能 | `feature/` | `feature/eda-node` |
| Bug 修复 | `fix/` | `fix/profiling-nulls` |
| 文档 | `docs/` | `docs/update-readme` |

## Usage

```bash
python app/main.py --file path/to/data.csv --target target_column_name
```

Outputs are written to:
- `outputs/charts/` — EDA visualizations
- `outputs/reports/analysis_report.md` — final analysis report