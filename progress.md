# DiagRAG 项目进度记录

> 记录日期：2026-05-25
> 项目描述：基于 Milvus 向量数据库 + DashScope（Qwen LLM / 文本嵌入）+ FastAPI 的医学诊断 RAG 系统

---

## 一、项目初始化

### 1.1 创建 docker-compose.yml
- 编写 `docker-compose.yml`，包含 4 个服务：
  - `etcd`（Milvus 元数据存储）
  - `minio`（Milvus 对象存储）
  - `milvus-standalone`（Milvus 向量数据库主服务）
  - `rag-app`（RAG 应用容器）
- 所有服务使用 `milvus` bridge 网络
- `rag-app` 依赖 `milvus-standalone`，等待其健康检查通过后启动
- `.env` 文件注入环境变量

### 1.2 创建配置文件 config.yml
- 定义了以下配置项：
  - `milvus`：`host`（默认 `milvus-standalone`）、`port`（19530）、`collection_name`（`medical_chunks`）
  - `embedding`：`model_name`（`text-embedding-v1`）、`dashscope_api_key`（`${DASHSCOPE_API_KEY}` 占位符）
  - `llm`：`model_name`（`qwen-max`）、`temperature`（0.1）、`max_tokens`（1000）
  - `retrieval`：`top_k`（5）、`score_threshold`（0.7）
  - `chunking`：`chunk_size`（500）、`chunk_overlap`（50）
- 环境变量使用 `${VAR}` 语法占位，由 `config_loader.py` 运行时替换

### 1.3 创建 .env.example
- 包含以下环境变量：
  - `MILVUS_HOST`（默认 `milvus-standalone`）
  - `MILVUS_PORT`（默认 `19530`）
  - `MINIO_ENDPOINT`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY`
  - `DASHSCOPE_API_KEY`
  - `APP_PORT`、`LOG_LEVEL`

### 1.4 创建 src/config_loader.py
- `load_config(config_path)`：读取 `config.yml` 并返回 Python dict
- `_substitute_env_vars(value)`：用正则替换 `${VAR}` / `${VAR:-default}` 占位符，从 `os.environ` 取值
- `_walk_and_substitute(obj)`：递归遍历 dict / list / str，对所有字符串执行替换
- 用法：`cfg = load_config()`，`cfg["embedding"]["dashscope_api_key"]` 即为注入后的真实值

### 1.5 创建 requirements.txt
- 依赖清单：`dashscope>=1.20.0`、`pymilvus>=2.4.0`、`langchain>=0.3.0`、`langchain-community>=0.3.0`、`pypdf>=4.0.0`、`sentence-transformers>=3.0.0`、`pyyaml>=6.0`、`python-dotenv>=1.0.0`、`fastapi>=0.115.0`、`uvicorn>=0.30.0`、`tiktoken>=0.7.0`

### 1.6 创建 Dockerfile
- 基于 `python:3.10-slim`
- 设置工作目录 `/app`，复制 `requirements.txt` 安装依赖，复制整个项目
- 暴露端口 8000
- 启动命令：`uvicorn src.api:app --host 0.0.0.0 --port 8000`

### 1.7 创建 .dockerignore
- 排除 `.git`、`venv`、`.env`、`__pycache__`、`Dockerfile`、`*.md`、`*.pdf`、`logs`、`*.log`、`agent-transcripts` 等无关文件，减小构建上下文

---

## 二、目录结构搭建

### 2.1 创建完整目录结构
按照 RAG 系统职责划分，创建了以下子包（均含 `__init__.py`）：

```
DiagRAG/
├── src/
│   ├── ingestion/        # 文档摄取与预处理
│   │   ├── loaders.py    # PDF / 纯文本文档加载
│   │   └── chunking.py   # 文本分块（固定大小 / 句子 / 语义分块）
│   ├── embedding/        # 向量嵌入生成与缓存
│   │   ├── dashscope.py  # DashScope API 客户端（embedding + generation）
│   │   └── cache.py      # 嵌入结果缓存（内存 / 磁盘）
│   ├── vectorstore/      # 向量数据库操作
│   │   ├── milvus.py    # Milvus 连接与 CRUD
│   │   └── schema.py    # Collection schema 定义
│   ├── retrieval/       # 检索增强组件
│   │   ├── search.py    # 向量相似度搜索与混合检索策略
│   │   └── rerank.py   # 重排序与结果后处理
│   ├── generation/      # LLM 生成与提示词管理
│   │   ├── llm.py       # Qwen LLM / DashScope 生成封装
│   │   └── prompts.py  # 医学诊断查询提示词模板
│   ├── api/             # FastAPI 应用入口
│   │   ├── app.py       # FastAPI 实例与中间件
│   │   ├── routes.py    # HTTP 路由处理（query / ingest / health）
│   │   └── schemas.py   # Pydantic 请求 / 响应模型
│   ├── utils/           # 共享工具
│   │   ├── logger.py    # 结构化日志配置
│   │   ├── metrics.py  # 指标收集（Prometheus / OpenTelemetry）
│   │   └── file_utils.py # JSON / YAML / 文本文件 I/O 辅助函数
│   └── config_loader.py # 配置加载与环境变量注入
├── scripts/             # 独立运维脚本
│   ├── ingest.py        # 文档摄取脚本
│   └── evaluate.py      # 评估脚本
├── config/              # 配置 schema 与验证工具
│   └── schemas.py
├── config.yml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

### 2.2 已实现的模块

#### src/embedding/embedder.py（已实现）
- `Embedder` 类，封装 DashScope `TextEmbedding` API
- 支持单条 `embed_single()` 和批量 `embed_batch()` 向量化
- 内置 LRU 内存缓存 + 可选磁盘持久化缓存（TTL 支持）
- 使用 tenacity 实现自动重试（指数退避，3 次重试）
- 批量嵌入时跳过已命中缓存的文本，减少 API 调用

---

## 三、Docker 服务部署

### 3.1 首次启动遇到的问题
- **问题**：Docker Hub 网络访问超时（`dial tcp 69.63.176.143:443`），拉取 `python:3.10-slim` 镜像失败
- **解决**：配置 Docker Desktop → Settings → Docker Engine → 添加国内镜像加速源（如 `https://docker.m.daocloud.io`），Apply & Restart 后重新 `docker-compose up -d`

### 3.2 镜像构建与容器启动
- Milvus 三件套（etcd / minio / milvus-standalone）拉取并启动成功，健康检查全部通过
- `rag-app` 镜像构建成功（构建耗时约 8 分钟，含 Python 依赖安装）
- 4 个容器均以 `milvus` 网络互通运行

### 3.3 服务发现与端口映射修复
- **问题 1**：`Dockerfile` 的 `CMD` 原为 `tail -f /dev/null`（调试模式），FastAPI 未启动
  - 修复：改为 `CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]`
- **问题 2**：`docker-compose.yml` 中 `rag-app` 缺少 `ports` 映射，宿主机无法访问
  - 修复：添加 `ports: "8000:8000"` 端口映射
- **问题 3**：`docker-compose.yml` 包含已废弃的 `version` 字段
  - 修复：移除 `version` 字段

### 3.4 最终服务状态
所有容器均正常运行：

| 容器 | 状态 | 说明 |
|---|---|---|
| `milvus-etcd` | ✅ Healthy | Milvus 元数据存储 |
| `milvus-minio` | ✅ Healthy | Milvus 对象存储 |
| `milvus-standalone` | ✅ Healthy | Milvus 向量数据库 |
| `rag-app` | ✅ Running | FastAPI 应用，端口 8000 映射至宿主机 |

---

## 四、FastAPI 服务实现

### 4.1 当前 API 端点（src/api/__init__.py）

| 端点 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 存活探针，返回 `{"status": "healthy"}` |
| `/ready` | GET | 就绪探针，检查 Milvus 连通性并返回配置加载状态 |
| `/query?q=...` | GET | RAG 查询接口（当前为 mock 模式） |
| `/ingest` | POST | 文档摄入接口，接受 `file_url` 或 `text` 参数 |

- 应用启动时通过 `lifespan` 上下文管理器自动加载 `config.yml` 并注入环境变量
- 启用 CORS 中间件，允许跨域请求
- `/ready` 端点实时连接 Milvus 验证向量数据库可用性

### 4.2 服务访问地址
- **Swagger 文档**：`http://localhost:8000/docs`
- **健康检查**：`http://localhost:8000/health`
- **就绪检查**：`http://localhost:8000/ready`

---

## 五、后续待完成事项

> 以下模块目前为占位文件（docstring 骨架），待后续逐步实现：

1. **src/vectorstore/milvus.py** — Milvus 连接、Collection 创建、向量插入与查询
2. **src/vectorstore/schema.py** — 向量 Collection schema 定义（字段映射、索引类型）
3. **src/vectorstore/indexer.py** — 批量向量索引与 flush 管理
4. **src/ingestion/loaders.py** — PDF / 纯文本文档加载器（pypdf / langchain-community）
5. **src/ingestion/chunking.py** — 文本分块策略（固定大小重叠 / 句子级 / 语义分割）
6. **src/retrieval/search.py** — 向量相似度搜索（top_k 召回、分数过滤）
7. **src/retrieval/rerank.py** — Cross-Encoder 重排序与结果后处理
8. **src/generation/llm.py** — DashScope Qwen LLM 调用封装
9. **src/generation/generator.py** — RAG 生成管道（检索 → 构建上下文 → 调用 LLM）
10. **src/generation/prompts.py** — 医学诊断领域提示词模板
11. **src/retrieval/retriever.py** — 检索器（封装搜索 + 重排序逻辑）
12. **src/utils/logger.py** — 结构化日志（dictConfig / JSON 格式）
13. **src/utils/metrics.py** — Prometheus / OpenTelemetry 指标埋点
14. **src/utils/file_utils.py** — 文件 I/O 辅助函数
15. **src/api/routes.py** — 路由处理逻辑（当前已在 `__init__.py` 中以简化形式实现）
16. **src/api/schemas.py** — Pydantic 模型定义
17. **scripts/ingest.py** — 独立文档摄取脚本
18. **scripts/evaluate.py** — RAG 评估脚本（召回率、精确率、答案质量）
19. **config/schemas.py** — 配置 schema 验证工具

---

## 六、常用运维命令

```powershell
# 启动所有服务
docker-compose up -d

# 重新构建并启动（代码变更后）
docker-compose up -d --build

# 查看所有容器状态
docker ps

# 查看 rag-app 日志
docker logs rag-app -f

# 停止所有服务
docker-compose down

# 测试健康检查
Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing

# 访问 Swagger 文档
# 浏览器打开：http://localhost:8000/docs
```

✅ 步骤 2.3：文档加载器 src/document_loader.py 已就绪。
✅ 步骤 2.4：文本分割器 src/text_splitter.py 已就绪。
✅ 步骤 2.5：整合管道 src/chunk_pipeline.py 已就绪。
✅ 步骤 2.6：命令行脚本 scripts/generate_chunks.py 已就绪。
✅ 步骤 2.7：测试文件 tests/test_chunking.py 已就绪。
✅ 步骤 2.8：运行 generate_chunks.py，共处理 1 个文件，生成 3 个 chunk，保存至 data/processed/chunks.json。

---

## 七、配置加载与验证

### 3.2 增强 config_loader.py（步骤 3.2）

在原有 `load_config()` 基础上补充以下能力：

- **`ConfigError` 异常类**：专属异常类型，用于配置加载 / 解析 / 键缺失等错误，便于调用方统一捕获。
- **缺失文件处理**：`load_config()` 检测到文件不存在时抛出 `ConfigError("Config file not found: ...")`。
- **YAML 解析错误处理**：`yaml.YAMLError` 和 `OSError` 均被捕获并包装为 `ConfigError`。
- **空配置文件处理**：检测到 `yaml.safe_load` 返回 `None` 时抛出明确错误。
- **`get_milvus_config()`**：返回 `config["milvus"]`，键缺失时抛出 `ConfigError`。
- **`get_embedding_config()`**：返回 `config["embedding"]`，键缺失时抛出 `ConfigError`。
- **环境变量替换**（原有功能保留）：`${VAR}` 和 `${VAR:-default}` 语法在所有层级递归生效。

✅ 步骤 3.2：配置加载器支持 milvus 和 embedding 配置。

✅ 步骤 3.3：embedding_client.py 封装千问 Embedding API。

✅ 步骤 3.4：milvus_client.py 实现 Milvus 操作封装。

✅ 步骤 3.5：构建脚本 scripts/build_milvus.py 完成。

✅ 步骤 3.6：测试文件 tests/test_milvus.py 完成。

✅ 步骤 3.7：运行 build_milvus.py，成功插入 3 条向量到 Milvus collection medical_chunks。

✅ 步骤 3.8：为 medical_chunks 增加 BM25 稀疏向量支持，支持关键词检索。

- Schema 新增 `sparse_vector` 字段（SPARSE_FLOAT_VECTOR），配合 SPARSE_INVERTED_INDEX 索引
- `milvus_client.py` 新增工具函数：
  - `_tokenize_chinese()` — 中文分词
  - `_compute_idf()` — 语料库 IDF 计算
  - `compute_bm25_sparse_vector()` — BM25 稀疏向量生成
  - `_rrf_fusion()` — 倒数排名融合
- `milvus_client.py` 新增检索方法：
  - `bm25_search(query_text, top_k)` — 纯关键词检索
  - `hybrid_search(query_text, query_vector, top_k)` — 语义+关键词混合检索（RRF 融合）
- `scripts/build_milvus.py` 同步更新，插入时自动计算 BM25 稀疏向量
- rebuild 后 collection 现有 3 条实体，含稠密向量 + 稀疏向量双索引

✅ 步骤 3.8：检索功能验证通过，可成功返回相关 chunks。

✅ 步骤 4.2：llm_client.py 封装千问 LLM 调用。

✅ 步骤 4.3：prompts.py 定义医疗系统提示词。

✅ 步骤 4.4：rag_chain.py 实现 RAG 完整流程。

✅ 步骤 4.5：config_loader.py 增加 retrieval 和 llm 配置读取。

✅ 步骤 4.6：命令行交互脚本 cli.py 完成。

- 导入配置、`DashScopeEmbeddingClient`、`MilvusClient`、`DashScopeLLMClient`、`RAGChain`，启动时一次性初始化全部客户端和 RAG 链
- 提供循环交互界面，输入 `exit`/`quit`/`q` 或 `Ctrl+C`/`Ctrl+Z` 退出
- 每个问题调用 `rag_chain.answer()`，打印回答内容及引用来源（文档名、页码、相似度分数）
- 捕获 `KeyboardInterrupt`、`EOFError`、`RAGChainError` 及所有未预期异常
- 启动时打印欢迎信息

用法：`python cli.py`

✅ 步骤 4.7：测试文件 tests/test_rag.py 完成。

✅ 步骤 4.8：RAG 系统交互测试通过，能正常回答医疗问题并引用来源。

✅ 步骤 4.9：FastAPI 接口 api.py 完成（可选）。
