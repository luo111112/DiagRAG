# 医智援（DiagRAG）—— 医学诊断检索增强生成系统

> 基于 Milvus 向量数据库与阿里云 DashScope（通义千问 Qwen）的医学智能问答系统

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [详细启动教程](#详细启动教程)
- [API 接口说明](#api-接口说明)
- [配置说明](#配置说明)
- [核心模块详解](#核心模块详解)

---

## 项目简介

**医智援（DiagRAG）** 是一个基于检索增强生成（RAG）技术的医学诊断问答系统。它通过结合 Milvus 向量数据库的混合检索能力与阿里云 DashScope 通义千问（Qwen LLM），为用户提供专业、可靠的医学知识问答服务。

### 核心能力

| 能力 | 说明 |
|------|------|
| 医学知识问答 | 基于医学文献进行循证问答 |
| 混合检索 | 向量语义搜索 + BM25 关键词搜索，RRF 融合 |
| **智能重排序** | LLM（Qwen）驱动的 CrossEncoder 重排序，提升检索相关性 |
| 流式输出 | 实时流式 SSE 响应，用户体验友好 |
| 智能引用 | 答案附带文献来源，可追溯可验证 |
| 安全提醒 | 急危重症自动提示就医，诊断分寸感强 |

### 技术栈

| 类别 | 技术选型 |
|------|----------|
| 前端 | Streamlit |
| 后端 | FastAPI + Uvicorn |
| 向量数据库 | Milvus 2.4 |
| LLM | 阿里云 DashScope（通义千问 Qwen） |
| Embedding | DashScope TextEmbedding |
| 文档存储 | MinIO（S3 兼容） |
| 容器化 | Docker + Docker Compose |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    Streamlit 前端 (streamlit_app.py)             │
│                   医学问答聊天界面，实时流式输出                  │
│                   访问地址：http://localhost:8501                │
│                                                                 │
└──────────────────────────────┬────────────────────────────────────┘
                               │ HTTP / SSE 流式
┌──────────────────────────────▼────────────────────────────────────┐
│                                                                    │
│                    FastAPI 后端 (src/api.py)                      │
│                 http://localhost:8000/docs (Swagger)              │
│                                                                    │
│   端点: POST /ask (同步)  |  POST /ask/stream (流式)  |  GET /health  │
│                                                                    │
└──────────────────────────────┬────────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────────┐
│                                                                    │
│                    RAG Chain (src/rag_chain.py)                    │
│                                                                    │
│   Step 1: Embedding Client ──► 将用户问题转换为向量                 │
│                │                                                    │
│                ▼                                                    │
│   Step 2: Milvus Client ────► 混合检索 (向量 + BM25 + RRF融合)      │
│                │                                                    │
│                ▼                                                    │
│   Step 3: LLM Reranker ───────► LLM 语义重排序，过滤噪声             │
│                │                                                    │
│                ▼                                                    │
│   Step 4: Prompt Assembly ──► 拼接 RAG 提示词模板                   │
│                │                                                    │
│                ▼                                                    │
│   Step 4: LLM Client ───────► 调用 Qwen 生成答案                    │
│                │                                                    │
│                ▼                                                    │
│   Step 5: Response ─────────► 返回答案 + 引用来源                    │
│                                                                    │
└────────────────────────────┬───────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌──────▼─────────┐
│                │  │                  │  │                 │
│ Embedding      │  │    Milvus        │  │    LLM          │
│ Client         │  │    Client        │  │    Client       │
│ (DashScope     │  │                  │  │    (DashScope   │
│  TextEmbed)    │  │  向量检索         │  │     Qwen)       │
│                │  │  BM25 搜索       │  │                 │
│  语义相似度     │  │  RRF 融合        │  │  答案生成        │
│                │  │                  │  │                 │
└────────────────┘  └──────────────────┘  └─────────────────┘
```

### RAG 工作流程详解

```
用户提问: "急性心肌梗死的典型症状是什么？"
     │
     │ 1. Embedding
     ▼
向量 [0.23, -0.45, 0.87, ...]  (1536维)
     │
     │ 2. Hybrid Search in Milvus
     ▼
┌──────────────────────────────────────────┐
│  稠密向量搜索 (内积/余弦相似度)            │
│  +                                        │
│  BM25 稀疏向量搜索 (关键词匹配)            │
│  =                                        │
│  RRF 融合 (倒数排名融合, k=60)             │
│  → 取 top_k=5 最相关文档块                 │
└──────────────────────────────────────────┘
     │
     │ 3. LLM 语义重排序（CrossEncoder 风格）
     ▼
医智援系统提示词 + [上下文: 检索到的5条文档] + 用户问题
     │
     │ 5. LLM 生成
     ▼
"根据检索到的医学文献，急性心肌梗死的典型症状包括：
  1. 胸骨后压榨性疼痛...
  2. 可放射至左臂...
  来源: [文档1], [文档3]
  
  ⚠️ 重要提示：若出现上述症状，请立即就医！"
```

---

## 目录结构

```
DiagRAG/
│
├── .env / .env.example         # 环境变量配置（API密钥等）
├── config.yml                  # 主配置文件（Milvus/LLM/Embedding参数）
├── requirements.txt            # Python 依赖清单
├── docker-compose.yml          # Docker 服务编排（Milvus + App）
├── Dockerfile                 # 应用容器镜像构建
│
├── streamlit_app.py            # Streamlit 前端界面
│
├── src/                        # ──────────── 核心源代码 ────────────
│   ├── api.py                 # FastAPI 应用，定义 REST 端点
│   ├── config_loader.py       # YAML 配置加载器（支持 ${VAR} 语法）
│   ├── llm_client.py          # DashScope Qwen LLM 客户端（同步/流式）
│   ├── embedding_client.py    # DashScope TextEmbedding 客户端
│   ├── milvus_client.py       # Milvus 向量数据库操作封装
│   ├── rag_chain.py           # RAG 检索-生成链核心实现
│   ├── document_loader.py     # PDF/文本 文档加载器
│   ├── text_splitter.py       # 文本分割器（支持重叠）
│   ├── chunk_pipeline.py      # 文档分块处理全流程管道
│   │
│   ├── ingestion/             # 文档摄取模块
│   │   ├── loaders.py        # 多格式文档加载（PDF/TXT/DOC）
│   │   ├── chunking.py       # 分块策略实现
│   │   └── chunker.py        # 通用分块器基类
│   │
│   ├── embedding/             # 向量嵌入模块
│   │   ├── embedder.py        # 嵌入器抽象接口
│   │   ├── dashscope.py       # DashScope 具体实现（含重试逻辑）
│   │   └── cache.py          # 嵌入结果缓存
│   │
│   ├── vectorstore/           # 向量存储模块
│   │   ├── milvus.py         # Milvus 存储接口
│   │   ├── schema.py         # Collection Schema 定义
│   │   └── indexer.py        # 索引构建（IVF/HNSW）
│   │
│   ├── retrieval/             # 检索增强模块
│   │   ├── search.py         # 搜索策略实现
│   │   ├── retriever.py      # 检索器抽象
│   │   └── rerank.py         # 重排序策略
│   │
│   ├── generation/            # LLM 生成模块
│   │   ├── llm.py            # LLM 抽象接口
│   │   ├── generator.py      # 生成器实现
│   │   └── prompts.py        # 医学专用提示词模板
│   │
│   └── utils/                # 工具模块
│       ├── logger.py         # 日志工具
│       ├── metrics.py        # 性能指标
│       └── file_utils.py     # 文件操作
│
├── scripts/                    # ──────────── 运维脚本 ────────────
│   ├── build_milvus.py       # Milvus Collection 构建与索引创建
│   ├── generate_chunks.py    # 文档分块与向量化入库脚本
│   └── evaluate.py           # RAG 系统评估脚本
│
├── tests/                      # ──────────── 测试套件 ────────────
│   ├── test_rag.py           # RAG 链单元测试（Mock LLM/Embedding）
│   ├── test_chunking.py      # 分块策略测试
│   └── test_milvus.py        # Milvus 集成测试
│
└── progress.md                # 项目开发进度文档
```

---

## 快速开始

### 前置要求

- Docker & Docker Compose
- 阿里云 DashScope API Key（[申请地址](https://dashscope.console.aliyun.com/)）
- 至少 4GB 可用内存（Milvus 最低要求）

### 方式一：Docker 一键启动（推荐）

```powershell
# 1. 克隆项目后，在项目根目录创建 .env 文件
#    （复制 .env.example 并填入你的 DashScope API Key）
copy .env.example .env

# 2. 编辑 .env，填入你的 API Key
#    DASHSCOPE_API_KEY=sk-your-actual-api-key-here

# 3. 一键启动所有服务
docker-compose up -d

# 4. 验证服务状态
docker ps

# 5. 访问服务
#    - Streamlit 前端:  http://localhost:8501
#    - FastAPI Swagger: http://localhost:8000/docs
#    - 健康检查:        http://localhost:8000/health
```

### 方式二：本地开发启动

```powershell
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 配置环境变量
#    创建 .env 文件，参考 .env.example

# 3. 启动 Milvus 服务（Docker 方式）
docker run -d \
  --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v ./volumes/milvus:/var/lib/milvus \
  milvusdb/milvus:v3.0.0-rc18

# 4. 构建 Milvus Collection（首次必须执行）
python scripts/build_milvus.py

# 5. 启动 FastAPI 后端（终端 1）
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# 6. 启动 Streamlit 前端（新终端 2）
streamlit run streamlit_app.py
```

---

## 详细启动教程

### Step 1：获取阿里云 DashScope API Key

1. 访问 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)
2. 登录阿里云账号（如无账号请先注册）
3. 在左侧菜单进入 **"API-KEY 管理"**
4. 点击 **"创建新的 API-KEY"**
5. 复制生成的 API Key，格式为 `sk-xxxxxxxxxxxxxxxxxxxxxxxx`

> **注意**：请妥善保管 API Key，不要泄露给他人或提交到代码仓库。

### Step 2：配置环境变量

在项目根目录创建 `.env` 文件：

```bash
# ========== Milvus 配置 ==========
MILVUS_HOST=milvus-standalone
MILVUS_PORT=19530

# ========== MinIO 配置 ==========
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# ========== DashScope API（必须） ==========
DASHSCOPE_API_KEY=sk-your-api-key-here

# ========== 应用配置 ==========
APP_PORT=8000
LOG_LEVEL=INFO
```

> 如果是本地开发（非 Docker），将 `MILVUS_HOST` 改为 `localhost`。

### Step 3：Docker 方式启动服务

```powershell
# 启动所有服务（Milvus + MinIO + FastAPI + Streamlit）
docker-compose up -d

# 查看所有容器运行状态
docker ps

# 查看应用日志
docker logs rag-app -f

# 查看 Milvus 日志
docker logs milvus-standalone -f
```

**服务启动后的访问地址**：

| 服务 | 地址 |
|------|------|
| Streamlit 前端 | http://localhost:8501 |
| FastAPI 后端 | http://localhost:8000 |
| Swagger API 文档 | http://localhost:8000/docs |
| ReDoc 文档 | http://localhost:8000/redoc |
| 健康检查 | http://localhost:8000/health |

### Step 4：构建 Milvus 向量索引（首次必须）

如果使用 Docker 部署，首次启动后需要构建 Collection 和索引：

```powershell
# 进入容器执行
docker exec -it rag-app python scripts/build_milvus.py

# 或直接在本地执行（确保 Milvus 正在运行）
python scripts/build_milvus.py
```

### Step 5：摄入医学文档（可选）

将 PDF 医学文档放入 `data/` 目录后执行：

```powershell
# 分块并向量化入库
python scripts/generate_chunks.py --input data/medical_docs/ --batch-size 100
```

### 常见问题排查

| 问题 | 解决方案 |
|------|----------|
| `ConnectionError: Milvus server not healthy` | 确保 Milvus 容器已启动：`docker ps \| grep milvus` |
| `AuthenticationError: Invalid API Key` | 检查 `.env` 中的 `DASHSCOPE_API_KEY` 是否正确 |
| `Port 8501 already in use` | 停止其他占用端口的程序，或修改 `docker-compose.yml` 中的端口映射 |
| `Milvus 连接超时` | 如果是非 Docker 部署，确认 `MILVUS_HOST=localhost` |
| 检索结果为空 | 执行 `python scripts/build_milvus.py` 重建索引 |

---

## API 接口说明

### 接口概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/ask` | POST | RAG 问答（同步返回） |
| `/ask/stream` | POST | RAG 问答（流式 SSE） |

### 1. 健康检查

```
GET /health
```

**响应示例**：
```json
{
  "status": "healthy",
  "milvus": "connected",
  "llm": "ready",
  "embedding": "ready"
}
```

### 2. 同步问答

```
POST /ask
Content-Type: application/json

{
  "question": "急性心肌梗死的典型症状是什么？",
  "top_k": 5
}
```

**响应示例**：
```json
{
  "answer": "根据检索到的医学文献，急性心肌梗死（AMI）的典型症状包括：\n\n1. **胸骨后压榨性疼痛**：最典型的症状，疼痛可持续超过20分钟...\n\n⚠️ 重要提示：若怀疑心肌梗死，请立即拨打急救电话！\n\n来源：[文档1] 内科学 P245, [文档3] 心血管指南 V2.1",
  "sources": [
    {
      "text": "急性心肌梗死的典型症状为胸骨后压榨性疼痛，可向左肩、左臂、颈部放射...",
      "metadata": {
        "source_file": "内科学.txt",
        "page": 245,
        "chunk_id": 23
      },
      "score": 0.9523
    },
    {
      "text": "心血管疾病诊疗指南指出，AMI患者中约80%表现为典型胸痛症状...",
      "metadata": {
        "source_file": "心血管指南.txt",
        "page": 12,
        "chunk_id": 45
      },
      "score": 0.8941
    }
  ],
  "retrieved_chunks_count": 5,
  "model": "qwen-max",
  "latency_ms": 1234
}
```

### 3. 流式问答

```
POST /ask/stream
Content-Type: application/json

{
  "question": "糖尿病的饮食原则有哪些？",
  "top_k": 5
}
```

**响应（SSE 流式）**：

```
data: {"token": "根据", "done": false}
data: {"token": "检索", "done": false}
data: {"token": "到的", "done": false}
...
data: {"token": "来源：糖尿病防治指南。", "done": false}
data: {"done": true, "full_text": "...完整答案..."}
```

### Swagger 测试

启动服务后，访问 http://localhost:8000/docs，可使用 Swagger UI 在线调试所有 API 接口。

---

## 配置说明

### config.yml 主配置

```yaml
# Milvus 向量数据库配置
milvus:
  host: ${MILVUS_HOST:-localhost}        # Milvus 服务地址
  port: ${MILVUS_PORT:-19530}             # Milvus 服务端口
  collection_name: medical_chunks          # Collection 名称
  vector_dim: 1536                        # 向量维度（text-embedding-v1 为 1536）
  metric_type: IP                          # 度量类型：IP（内积）/ L2（欧氏距离）
  index_type: IVF_FLAT                     # 索引类型：IVF_FLAT / HNSW / ANNOY

# 向量嵌入配置
embedding:
  model: text-embedding-v1                # DashScope 嵌入模型
  batch_size: 25                           # 批量嵌入大小
  api_key: ${DASHSCOPE_API_KEY}           # 从环境变量读取

# 大语言模型配置
llm:
  model: "qwen-max"                        # Qwen 模型：qwen-max / qwen-plus / qwen-turbo
  temperature: 0.1                         # 温度参数（0=确定性，1=创造性）
  max_tokens: 1500                         # 最大生成 token 数
  top_p: 0.95                              # Top-p 采样
  api_key: ${DASHSCOPE_API_KEY}           # 从环境变量读取

# 检索配置
retrieval:
  top_k: 5                                  # 检索返回的文档块数量
  score_threshold: 0.0                      # 相似度分数阈值（0=不过滤）
  enable_rerank: false                      # 是否启用重排序
  rerank_top_k: 3                           # 重排序后传给 LLM 的最终文档数
  rerank_mode: score                       # 评分模式：rank=仅排序 / score=数值评分 / score_with_reason=评分+理由
  enable_bm25_blend: false                 # 是否启用 BM25 混合重排序
  rerank_fusion: rrf                       # 融合策略：rrf=倒数排名融合 / linear=线性加权
  max_docs_per_call: 10                    # 单次 LLM 调用最多处理的文档数

# 文档分块配置
chunking:
  chunk_size: 500                           # 每个文本块的字符数
  chunk_overlap: 50                        # 相邻块之间的重叠字符数
  min_chunk_length: 20                      # 最小块长度（过滤短块）
  separators: ["\n\n", "\n", "。", "？", "!"]  # 分块分隔符（按优先级）
```

### Milvus Collection Schema

| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| `id` | INT64 | 自增主键 |
| `vector` | FLOAT_VECTOR(1536) | 稠密向量（语义搜索） |
| `sparse_vector` | SPARSE_FLOAT_VECTOR | 稀疏向量（BM25 搜索） |
| `text` | VARCHAR(4096) | 原始文本内容 |
| `metadata` | JSON | 元数据（来源文件、页码、章节等） |

---

## 核心模块详解

### 1. RAG Chain (`src/rag_chain.py`)

RAG Chain 是系统的核心编排层，负责串联检索与生成的全流程：

```python
# 伪代码流程
def answer(question: str) -> dict:
    # 1. 问题向量化
    query_vector = embed(question)

    # 2. 混合检索
    chunks = milvus.hybrid_search(query_vector, top_k=5)

    # 3. 构建上下文
    context = "\n\n---\n\n".join(chunks)

    # 4. 填充提示词模板
    prompt = RAG_PROMPT.format(context=context, question=question)

    # 5. LLM 生成
    answer = qwen.generate(prompt, system_prompt=MEDICAL_SYSTEM_PROMPT)

    # 6. 返回结果
    return {"answer": answer, "sources": chunks}
```

### 2. 医学专用提示词 (`src/generation/prompts.py`)

系统内置医学专用提示词模板，包含以下核心原则：

| 规则 | 说明 |
|------|------|
| 循证原则 | 严格基于检索到的知识片段回答，禁止编造 |
| 知识边界意识 | 证据不足时诚实说明，不确定性明确告知 |
| 诊断分寸 | 不给出最终诊断，只提供参考建议和方向 |
| 引用规范 | 答案末尾注明来源，便于用户溯源 |
| 安全警告 | 涉及急危重症时自动提示立即就医 |

### 3. 混合检索策略 (`src/milvus_client.py`)

系统采用**三路检索 + RRF 融合**的混合检索策略：

```
查询向量 [0.23, -0.45, ...]
     │
     ├──► 稠密向量搜索 ──► Top 50 结果（按内积相似度排序）
     │                         │
     │                         │  赋予排名得分
     ├──► BM25 稀疏搜索 ──► Top 50 结果（按词频-逆文档频率排序）
     │                         │
     │                         │  赋予排名得分
     └──► RRF 融合算法 ──────────────────► 最终 Top 5 结果
                              (k=60, 倒数排名融合)
```

**RRF（Reciprocal Rank Fusion）公式**：

$$
\text{RRF\_score}(d) = \sum_{i} \frac{1}{k + \text{rank}_i(d)}
$$

其中 $k=60$ 为融合参数，$\text{rank}_i(d)$ 为文档 $d$ 在第 $i$ 路检索结果中的排名位置。

### 4. LLM 客户端 (`src/llm_client.py`)

```python
class DashScopeLLMClient:
    def generate(prompt, system_prompt) -> str:
        """同步生成：等待完整答案后返回"""

    def stream_generate(prompt, system_prompt) -> Generator[str, None, None]:
        """流式生成：逐 token 返回（SSE）"""

    def _retry_with_backoff(max_retries=3, base_delay=1.0):
        """指数退避重试：1s → 2s → 4s"""
```

### 5. 文档摄取管道 (`scripts/generate_chunks.py`)

```
医学文档 (PDF/TXT)
       │
       ▼
┌─────────────────┐
│  DocumentLoader │  支持格式：PDF, TXT, DOC, DOCX
│    文档加载器    │
└────────┬────────┘
         │ List[Document(id, page_content, metadata)]
         ▼
┌─────────────────┐
│  TextSplitter   │  策略：RecursiveCharacterTextSplitter
│    文本分割器    │  chunk_size=500, overlap=50
└────────┬────────┘
         │ List[Document(chunk)]
         ▼
┌─────────────────┐
│   Embedder      │  调用 DashScope TextEmbedding API
│    向量化嵌入    │  批量处理，每批 25 条
└────────┬────────┘
         │ List[(chunk, vector)]
         ▼
┌─────────────────┐
│   Milvus        │
│    向量入库      │  upsert 写入 Collection
└─────────────────┘
```

---

## 后续扩展

- [ ] 支持更多医学文献格式（Word、HTML、医学影像报告）
- [ ] 接入医学知识图谱，实现多跳推理问答
- [x] 实现检索结果重排序（Reranker）
- [ ] 增加对话历史管理（Multi-turn RAG）
- [ ] 支持更多 LLM 后端（Claude、GPT-4）
- [ ] 增加评估指标（Bleu、Recall、F1、医生评分）
