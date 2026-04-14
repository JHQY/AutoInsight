# AutoInsight 前端设计 Spec

> 版本：v1.0
> 日期：2026-04-14
> 状态：已确认，待实现

---

## 1. 目标

为 AutoInsight CLI 工具增加一个本地 Web 前端，让用户通过浏览器完成数据上传、参数配置、实时进度查看和报告下载，无需使用命令行。

---

## 2. 技术栈

| 层级 | 方案 |
|---|---|
| 后端 | FastAPI + Uvicorn（新增 `app/server.py`） |
| 前端 | 单页 HTML/CSS/JS（`app/static/index.html`） |
| 进度推送 | SSE（Server-Sent Events），后端逐节点推送状态 |
| 启动方式 | `python app/server.py`，浏览器访问 `http://localhost:8000` |
| 新增依赖 | `fastapi`、`uvicorn`、`python-multipart` |

不引入前端构建工具（无 Node.js、无 webpack/vite）。

---

## 3. 文件结构变更

```
app/
  server.py          ← 新增：FastAPI 应用入口
  static/
    index.html       ← 新增：单页前端
  graph.py           ← 不变
  main.py            ← 不变（CLI 仍可用）
  state.py           ← 不变
```

---

## 4. 后端 API 设计

### 4.1 `POST /analyze`

接收表单数据，启动分析任务，返回 `task_id`。

**请求（multipart/form-data）**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | File | 是 | CSV 文件 |
| `prompt` | str | 是 | 业务问题描述 |
| `target` | str | 否 | 目标列名，默认空字符串 |
| `level` | str | 否 | `"general"`（默认）或 `"expert"` |
| `tune` | bool | 否 | 默认 `false` |

**响应**

```json
{ "task_id": "uuid4-string" }
```

### 4.2 `GET /progress/{task_id}`

SSE 流，每当一个节点完成时推送一条事件。

**事件格式（`data:` 字段为 JSON）**

```json
{ "node": "profiling", "status": "running" }
{ "node": "profiling", "status": "done" }
{ "node": "intent_routing", "status": "running" }
...
{ "node": "reporting", "status": "done" }
{ "event": "complete", "task_type": "regression", "best_model": "XGBoost", "elapsed": 42 }
{ "event": "error", "message": "错误描述" }
```

节点名称与顺序（与 `graph.py` 一致）：

| 顺序 | node key | 前端显示名 |
|---|---|---|
| 1 | `profiling` | 数据剖析 |
| 2 | `intent_routing` | 意图识别 |
| 3 | `processing` | 数据清洗 |
| 4 | `eda` | 探索性分析（EDA） |
| 5 | `model_routing` | 模型路由 |
| 6 | `modeling` | 建模 |
| 7 | `evaluation` | 模型评估 |
| 8 | `reporting` | 报告生成 |

> `correlation_analysis` 任务会跳过 `modeling` 和 `evaluation`，后端需为跳过的节点推送 `status: "skipped"`，前端将跳过节点显示为灰色（不变色）。

### 4.3 `GET /report/{task_id}`

下载生成的 Markdown 报告文件（`Content-Disposition: attachment`）。

---

## 5. 前端设计

### 5.1 整体布局

单列居中，最大宽度 680px，深色主题（背景 `#0f172a`）。页面分为两个阶段：**输入阶段**和**分析阶段**（输入区折叠，进度区展开）。

### 5.2 输入阶段

**页面标题区**
- 标题：`⚡ AutoInsight`
- 副标题：`自动化数据挖掘 · 只需描述你的业务问题`

**Prompt 输入**
- 标签：`你想从数据中得到什么？`
- Textarea，`min-height: 90px`，可拖动调整高度
- 占位符文字：`例如：我想知道哪些因素最影响房价，以及能否预测未来的房价走势`
- 辅助说明：`请用自然语言描述你的业务问题，系统会自动判断分析类型`

**文件上传**
- 标签：`上传数据文件`
- 拖拽区（dashed border），支持点击和拖拽
- 说明文字：`仅支持 CSV 格式 · 建议不超过 200,000 行`
- 上传后显示：文件名 + 大小 + 绿色 ✓ 确认
- 文件类型验证：前端检查扩展名为 `.csv`，非 CSV 提示错误，**阻止提交**

**分析选项（选项卡片）**
- 报告风格 toggle（两选）：
  - `通俗易懂`：副标题"我对数据分析不熟悉，请用业务语言解释"→ 对应 `level=general`
  - `专业详细`：副标题"我有数据分析背景，请包含技术细节和指标"→ 对应 `level=expert`
  - 默认选中`通俗易懂`
- 超参调优 toggle（开关）：
  - 默认关闭
  - 说明文字：`开启后自动搜索更优参数，分析时间会增加约 2-5 分钟`

**开始按钮**
- 文字：`开始分析`
- 禁用条件：prompt 为空 **或** 未上传有效 CSV
- 禁用时按钮置灰，下方显示提示文字
- 点击后切换到分析阶段

### 5.3 分析阶段

输入区整体折叠（显示一行摘要：文件名 + prompt 前 30 字）。

**节点链（垂直列表）**

每个节点由圆形图标 + 名称 + 状态文字 + 竖向连接线组成。

| 状态 | 图标 | 边框色 | 背景色 | 文字色 | 状态文字 |
|---|---|---|---|---|---|
| pending | `○` | `#374151` | `#1f2937` | `#6b7280` | （无） |
| running | `⟳`（CSS旋转动画） | `#eab308` | `#713f12` | `#fde047` | 正在运行… |
| done | `✓` | `#22c55e` | `#14532d` | `#86efac` | 已完成 |
| skipped | `—` | `#374151` | `#1f2937` | `#4b5563` | 已跳过 |
| error | `✕` | `#ef4444` | `#7f1d1d` | `#fca5a5` | 出错 |

节点连接线颜色随上方节点状态变化：done → 绿色，running → 黄色，其余 → 灰色。

**完成态**

所有节点变绿后，底部显示完成摘要卡片：
- 摘要一行：`任务类型：{task_type} · 最优模型：{best_model} · 用时 {elapsed} 秒`
- 主按钮：`⬇ 下载分析报告（Markdown）`（绿色，点击调用 `/report/{task_id}`）
- 次按钮：`重新分析`（透明描边，点击重置页面回输入阶段）

**错误态**

若任一节点收到 `error` 事件：当前节点变红，显示错误信息，下方出现`重新分析`按钮。

---

## 6. 进度推送实现方案

LangGraph 的 `graph.invoke()` 是同步阻塞调用，无法原生中途推送。采用以下方案：

1. `server.py` 维护全局内存字典 `TASK_QUEUES: dict[str, Queue]`，key 为 `task_id`。
2. `POST /analyze` 收到请求后：
   - 创建 `task_id`（uuid4）和对应 Queue，写入 `TASK_QUEUES`
   - 将分析任务放入后台线程（`asyncio.run_in_executor`），并将 Queue 对象**作为普通参数**传给包装函数（不放入 AgentState）
3. 包装函数在调用每个 node 前后向 Queue 写入状态事件；`graph.invoke()` 完成后写入终止标记。
4. `GET /progress/{task_id}` 用 SSE 持续读取对应 Queue，直到收到终止标记后关闭流并清理 `TASK_QUEUES`。

> Queue 对象不进入 AgentState（LangGraph 会序列化 state，Queue 不可序列化）。

---

## 7. 启动方式

```bash
# 激活虚拟环境后
python app/server.py
# 浏览器访问 http://localhost:8000
```

CLI 方式（`python app/main.py`）继续可用，互不影响。

---

## 8. 不在本次范围内

- 用户认证 / 多用户隔离
- 历史记录持久化
- 图表在页面内预览（报告以 Markdown 文件下载）
- 深色/浅色主题切换
- 移动端适配
