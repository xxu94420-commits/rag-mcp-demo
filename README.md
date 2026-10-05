# rag-mcp-demo

一个**小型、可验证、本地可跑**的 RAG（检索增强生成）+ MCP（Model Context Protocol）演示项目：
把文档索引进向量库，再通过 MCP 服务器把检索与问答能力暴露给任意 MCP 客户端（Claude Desktop / Cursor 等）。

设计目标与大量"演示优先"的 demo 不同：

- **无外部依赖也能完整跑通**：默认 TF-IDF 本地嵌入 + 抽取式降级回答，不需要任何 API Key，CI 与测试完全离线可复现。
- **可插拔升级**：配置 `OPENAI_BASE_URL` / `OPENAI_API_KEY` 后自动切换为 OpenAI 兼容的嵌入模型与对话模型。
- **诚实降级**：没有 LLM 时输出明确标注"降级抽取回答"，绝不假装是生成式回答。

## 功能

- 递归分块（段落 → 句子 → 字符三级，带重叠窗口）
- TF-IDF 向量检索（L2 归一化 + 余弦相似度）/ OpenAI 兼容嵌入
- 带引用编号的问答（`[1][2]` 溯源 + 来源片段与相似度分数）
- MCP 服务器（stdio 传输），暴露 3 个工具：
  - `list_documents` — 列出已索引文档及分块数
  - `search_documents` — 检索相关分块，返回打分排序
  - `ask_document` — 基于文档问答，返回答案与引用来源

## 在线演示

公网演示（Render 免费实例，首次访问可能需等待冷启动）：
**https://rag-mcp-demo.onrender.com**

打开即可对内置示例文档提问，回答附检索来源与相似度分数，无需任何 API Key。

## 快速开始

```bash
pip install -r requirements.txt

# 0. 或直接在浏览器里体验（启动时自动索引 sample_docs）
uvicorn rag_mcp_demo.webapp:app --reload   # 打开 http://127.0.0.1:8000

# 1. 导入文档（示例语料已随仓库提供）
python -m rag_mcp_demo.cli ingest sample_docs
# 已导入 N 个分块，索引保存在 .ragindex/

# 2. 命令行问答
python -m rag_mcp_demo.cli ask "Agent 的核心模块有哪些？"

# 3. 启动 MCP 服务器（供客户端连接）
python -m rag_mcp_demo.cli serve
```

## 接入 MCP 客户端

以 Claude Desktop 为例，在配置中加入：

```json
{
  "mcpServers": {
    "rag-demo": {
      "command": "python",
      "args": ["-m", "rag_mcp_demo.cli", "serve"],
      "cwd": "/path/to/rag-mcp-demo",
      "env": { "RAG_INDEX_DIR": ".ragindex" }
    }
  }
}
```

## 配置 LLM（可选）

| 环境变量 | 作用 | 默认值 |
|---|---|---|
| `OPENAI_BASE_URL` | OpenAI 兼容接口地址 | 未设置则用本地降级 |
| `OPENAI_API_KEY` | API Key | 未设置则用本地降级 |
| `EMBEDDING_MODEL` | 嵌入模型名 | `text-embedding-3-small` |
| `CHAT_MODEL` | 对话模型名 | `gpt-4o-mini` |

## 项目结构

```
rag_mcp_demo/
├── chunking.py      # 递归分块
├── embeddings.py    # TF-IDF 本地嵌入 / OpenAI 兼容嵌入
├── store.py         # 向量存储与余弦检索
├── llm.py           # OpenAI 兼容对话 / 抽取式降级回答
├── pipeline.py      # RAG 主流程（导入、检索、问答、索引持久化）
├── mcp_tools.py     # MCP 工具层（与传输解耦，可单测）
├── server.py        # MCP stdio 服务器
├── webapp.py        # FastAPI 单页 Web 演示（自动索引 sample_docs，内存运行）
└── cli.py           # 命令行入口
tests/               # 分块 / 检索 / 端到端 / 工具层 / Web 共 23 项测试
sample_docs/         # 示例语料（Agent 笔记 + MCP 简介）
```

## 验证

```bash
pytest -q        # 20 passed
```

GitHub Actions 对 Python 3.12 跑完整测试 + CLI 冒烟（导入 → 问答）。

## 演示说明（诚实声明）

- TF-IDF 只是为了让项目零依赖可跑，检索质量不代表生产水平；真实场景请换用嵌入模型。
- 降级抽取回答仅用于验证检索链路，效果上限是"找到相关句子"，不代表 RAG 的真实能力上限。
- 索引存于本地 `.ragindex/`（已 gitignore），仓库本身不含任何构建产物。
