# Text2CAD CadQuery Prototype

这是一个最小科研原型：

```text
自然语言描述 -> text LLM -> CadQuery Python 代码 -> 本地执行 -> 导出 STEP/STL
```

当前版本先做导师说的轻量原型，不做训练；已经包含一个简单 agent loop：如果 CadQuery 执行失败，会把 traceback 发回 LLM，请它修复代码后重试。

## 1. 选择 VS Code 解释器

请先选择一个已经安装 CadQuery 的 Python 环境。在终端中可以用下面的命令确认：

```bash
python -c "import cadquery; print(cadquery.__version__)"
```

在 VS Code 里可以：

1. 打开命令面板 `Ctrl+Shift+P`
2. 选择 `Python: Select Interpreter`
3. 选择已经安装 CadQuery 的 Python 解释器

如果 VS Code 没显示它，也可以先用终端命令跑。

## 2. 先不用 API，跑通 CadQuery 导出

```bash
python -m src.text2cad.main \
  "a rectangular plate with a circular hole in the center" \
  --mock
```

成功后会在 `outputs/` 下面生成一个带时间戳的文件夹，里面包括：

- `generated_model.py`: 生成出来的 CadQuery 代码
- `model.step`: 可编辑/交换的 CAD 文件
- `model.stl`: 可预览/网格文件
- `run.json`: 本次运行的元数据
- `agent_run.json`: agent 循环摘要

## 3. 接入 LLM API

复制环境变量模板：

```bash
cp .env.example .env
```

然后编辑 `.env`，填入你的 API key、base URL 和模型名。也可以直接在终端设置：

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
```

运行：

```bash
python -m src.text2cad.main \
  "a 60 mm by 40 mm by 8 mm rectangular plate with four 5 mm corner holes"
```

这个项目使用 OpenAI-compatible `/chat/completions` 接口，所以很多提供兼容 API 的模型都能接。

默认会做最多 2 次错误修复：如果 CadQuery 执行失败，程序会把 traceback 和上一版代码发回 LLM，请它返回修复后的完整代码。可以用 `--max-repairs` 调整：

```bash
python -m src.text2cad.main \
  "a bracket with two bolt holes" \
  --max-repairs 3
```

API 使用 SSE 流式接收，GUI 日志会持续显示已收到的字符数。请求默认最多等待 300 秒；遇到临时断连、连接重置、超时、限流或常见服务端错误时，会自动重试 3 次，等待间隔为 2、4、8 秒。若流在完成前断开，已收到的残缺内容会被丢弃并从头重试。API 网络重试不计入 CadQuery 代码修复次数。CadQuery 本地执行的默认超时为 180 秒。

### Baseline 与 Reference-assisted (RAG)

Baseline 不注入额外参考资料：

```bash
python -m src.text2cad.main \
  "an open-top box with 3 mm wall thickness" \
  --max-repairs 0
```

Reference-assisted 会根据 prompt 检索最相关的 CadQuery 知识片段：

```bash
python -m src.text2cad.main \
  "an open-top box with 3 mm wall thickness" \
  --max-repairs 0 \
  --rag
```

知识库位于 `knowledge/`，包含人工整理模式、官方指南主题和由本机 CadQuery API 自动生成的完整索引。每次 RAG 运行会在 run 文件夹保存 `retrieval.json`，记录命中的条目、分数、内容和来源。

GUI 的 `Knowledge mode` 有三档：

- `Baseline (no RAG)`：不注入知识库。
- `Lightweight RAG (8 entries)`：只使用最初的 `cadquery_reference.json`。
- `Full RAG (all entries)`：使用人工条目、官方指南和完整 API 索引。

命令行分别使用 `--rag-mode lightweight` 和 `--rag-mode full`。旧参数 `--rag` 仍保留，并等同于 `--rag-mode full`。

CadQuery 升级后可以重新生成 API 与官方指南索引：

```bash
python scripts/build_cadquery_reference.py
```

## 4. 打开 GUI

运行：

```bash
python -m src.text2cad.gui
```

GUI 里可以直接输入想生成的 CAD 描述，然后点 `Generate and render`。成功后会显示：

- 渲染预览图
- 生成的 CadQuery 代码
- STEP/STL 输出路径
- 本次 agent 尝试次数
- 如果勾选 `Open 3D in CQ-editor after success`，成功后会自动打开 CQ-editor，并自动运行代码展示 3D 模型

GUI 会额外生成一个 `cq_editor_view.py`。这个文件不是 LLM 输出，而是给 CQ-editor 用的显示包装脚本：它读取 `generated_model.py` 的 `result`，再调用 `show_object(result)`。

GUI 还会生成一个 `launch_cq_editor_autorun.py`。这个文件负责启动 CQ-editor、加载 `cq_editor_view.py`，然后自动触发一次 Render，所以通常不需要你再手动点 Run。

### GLM-5.2 / Z.AI 配置示例

如果你要调用 GLM-5.2，把 `.env` 改成：

```bash
OPENAI_API_KEY=你的Z.AI_API_KEY
OPENAI_BASE_URL=https://api.z.ai/api/paas/v4
OPENAI_MODEL=glm-5.2
CADQUERY_PYTHON=/path/to/your/cadquery/python
```

注意：`OPENAI_BASE_URL` 填到 `/v4` 即可，不要把 `/chat/completions` 写进去；程序会自动补上。

## 5. 代码结构

```text
src/text2cad/
  main.py       命令行入口
  gui.py        本地输入和预览界面
  agent.py      生成-执行-失败修复循环
  llm.py        调 OpenAI-compatible API
  prompts.py    给 LLM 的系统提示词和代码清洗
  runner.py     执行 CadQuery 代码并导出 STEP/STL
  renderer.py   把 STL 渲染成 GUI 预览图
```

## 6. 第一阶段科研可以怎么扩展

建议按这个顺序推进：

1. 记录 Valid Syntax Rate：生成的代码能不能成功执行。
2. 保存 prompt、代码、错误信息、导出文件，方便复现实验。
3. 统计修复前后成功率：first-pass success 和 after-repair success 分开记录。
4. 再考虑更强的视觉反馈，例如多视角渲染和 VLM 评价。
