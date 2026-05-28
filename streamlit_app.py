"""DiagRAG Streamlit Frontend.

Connects to the FastAPI backend (src/api.py) and provides a clean
medical-chat UI for the RAG pipeline with streaming output.
"""

from __future__ import annotations

import streamlit as st

API_BASE = "http://localhost:8000"


# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DiagRAG 医学问答",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.html("""
<style>
  :root {
    --primary:       #1a6ee8;
    --primary-dark: #1557c0;
    --primary-soft: #e8f0fd;
    --accent:       #00a896;
    --bg:           #f0f4f8;
    --surface:      #ffffff;
    --text:         #1a2332;
    --muted:        #6b7280;
    --border:       #e2e8f0;
    --radius:       14px;
    --shadow:       0 4px 20px rgba(0,0,0,0.07);
    --font:         'PingFang SC', 'Microsoft YaHei', 'Segoe UI', sans-serif;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .stApp { background: var(--bg); font-family: var(--font); }

  /* ── Header ── */
  .app-header {
    background: linear-gradient(135deg, #1a6ee8 0%, #0d47a1 100%);
    border-radius: var(--radius);
    padding: 28px 32px 24px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 6px 28px rgba(26,110,232,0.28);
  }
  .app-header h1 {
    font-size: 26px;
    font-weight: 700;
    margin: 0 0 6px;
    letter-spacing: 0.5px;
  }
  .app-header p {
    margin: 0;
    opacity: 0.82;
    font-size: 14px;
  }

  /* ── Chat area ── */
  .chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 16px;
    max-width: 860px;
    margin: 0 auto 16px;
  }

  /* User bubble */
  .user-row {
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
    gap: 10px;
  }
  .user-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: var(--primary);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .user-bubble {
    background: linear-gradient(135deg, #1a6ee8, #1557c0);
    color: white;
    border-radius: var(--radius) 4px var(--radius) var(--radius);
    padding: 12px 18px;
    max-width: 72%;
    font-size: 15px;
    line-height: 1.7;
    box-shadow: var(--shadow);
  }

  /* Assistant bubble */
  .asst-row {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 10px;
  }
  .asst-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: var(--accent);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .asst-bubble {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 4px var(--radius) var(--radius) var(--radius);
    padding: 14px 20px;
    max-width: 72%;
    font-size: 15px;
    line-height: 1.8;
    box-shadow: var(--shadow);
  }
  .asst-bubble strong { color: var(--primary); }
  .asst-bubble em     { color: var(--accent); font-style: normal; font-weight: 600; }
  .asst-bubble code {
    background: var(--primary-soft);
    color: var(--primary-dark);
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 13px;
  }
  .asst-bubble pre {
    background: #1e2535;
    color: #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.6;
    margin: 10px 0;
  }

  /* Streaming cursor */
  .streaming-cursor::after {
    content: '▊';
    color: var(--primary);
    animation: blink 0.9s infinite;
    font-weight: 700;
  }
  @keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
  }

  /* Thinking indicator */
  .thinking-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    color: var(--muted);
    font-size: 13px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    max-width: 72%;
    box-shadow: var(--shadow);
  }
  .thinking-dots span {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--primary);
    animation: dot-bounce 1.4s infinite;
  }
  .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
  .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes dot-bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40%            { transform: translateY(-6px); opacity: 1; }
  }

  /* Source card */
  .src-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: var(--muted);
  }
  .src-card .src-text {
    color: var(--text);
    line-height: 1.6;
    margin-bottom: 4px;
  }
  .src-card .src-meta {
    font-size: 11px;
    color: var(--muted);
  }

  /* Sources expander */
  .sources-wrap {
    margin-top: 8px;
    border-top: 1px dashed var(--border);
    padding-top: 8px;
  }
  .sources-label {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 5px;
  }

  /* Input area */
  .input-wrap {
    position: sticky;
    bottom: 0;
    background: linear-gradient(to top, var(--bg) 80%, transparent);
    padding: 20px 0 8px;
    max-width: 860px;
    margin: 0 auto;
  }
  .input-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.09);
    display: flex;
    gap: 10px;
    align-items: flex-end;
  }
  .input-card textarea {
    flex: 1;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    resize: none !important;
    padding: 4px 0 !important;
    color: var(--text) !important;
    background: transparent !important;
  }
  .input-card textarea::placeholder { color: #adb5bd !important; }
  .send-btn {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.1s !important;
    flex-shrink: 0;
    height: 42px;
  }
  .send-btn:hover { background: var(--primary-dark) !important; transform: translateY(-1px); }
  .send-btn:active { transform: translateY(0) !important; }

  /* Sidebar */
  .sidebar-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
    margin-bottom: 14px;
    box-shadow: var(--shadow);
  }
  .sidebar-card h4 {
    margin: 0 0 8px;
    font-size: 13px;
    font-weight: 700;
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #c1ccd8; border-radius: 3px; }
</style>
""")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_sources(sources: list) -> None:
    """Render retrieved source cards."""
    for i, src in enumerate(sources, 1):
        meta = src.get("metadata") or {}
        filename = meta.get("source_file", "未知来源")
        chunk_id = meta.get("chunk_index", "?")
        score = src.get("score")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        st.markdown(
            f"""
            <div class="src-card">
              <div class="src-text">{src.get("text", "")}</div>
              <div class="src-meta">📄 {filename} · chunk {chunk_id} · score {score_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_assistant_message(content: str, sources: list | None = None) -> None:
    """Render a single assistant message bubble with optional sources."""
    st.markdown(f'<div class="asst-bubble">{content}</div>', unsafe_allow_html=True)
    if sources:
        with st.expander("📚 引用来源", expanded=False):
            _render_sources(sources)


def _stream_to_placeholder(
    question: str,
    placeholder: st.delta_generator.DeltaGenerator,
    sources_placeholder: st.delta_generator.DeltaGenerator,
    top_k: int | None,
) -> tuple[str, list[dict]]:
    """Fetch the SSE stream and write tokens to the placeholder in real-time.

    Returns:
        A tuple of (full_answer_text, sources_list).
    """
    import json as _json
    import requests

    payload: dict = {"question": question}
    if top_k is not None:
        payload["top_k"] = top_k

    try:
        resp = requests.post(
            f"{API_BASE}/ask/stream",
            json=payload,
            headers={"Accept": "text/event-stream"},
            timeout=120,
            stream=True,
        )
        if resp.status_code != 200:
            placeholder.markdown(
                f"❌ 后端返回错误 {resp.status_code}: {resp.text}"
            )
            return f"❌ 后端返回错误 {resp.status_code}", []

        buffer = ""
        sources_data: list[dict] = []
        event_buf = ""

        for raw in resp.iter_content(chunk_size=None):
            if not raw:
                continue
            decoded = raw.decode("utf-8", errors="replace")
            event_buf += decoded

            while "\n\n" in event_buf:
                event_text, event_buf = event_buf.split("\n\n", 1)
                ev_type, ev_payload = None, None

                for line in event_text.splitlines():
                    if line.startswith("event: "):
                        ev_type = line[7:].strip()
                    elif line.startswith("data: "):
                        ev_payload = line[6:].strip()
                        try:
                            ev_payload = _json.loads(ev_payload)
                        except _json.JSONDecodeError:
                            pass

                if ev_type == "sources" and ev_payload:
                    sources_data = ev_payload if isinstance(ev_payload, list) else []
                    with sources_placeholder:
                        if sources_data:
                            with st.expander("📚 引用来源", expanded=False):
                                _render_sources(sources_data)
                elif ev_type == "token" and ev_payload:
                    token = ev_payload if isinstance(ev_payload, str) else ""
                    buffer += token
                    placeholder.markdown(
                        f'<div class="asst-bubble"><span class="streaming-cursor">'
                        f"{buffer}</span></div>",
                        unsafe_allow_html=True,
                    )
                elif ev_type == "done":
                    placeholder.markdown(
                        f'<div class="asst-bubble">{buffer}</div>',
                        unsafe_allow_html=True,
                    )
                    return buffer, sources_data

        return buffer, sources_data

    except requests.exceptions.ConnectionError:
        placeholder.markdown(
            '<div class="asst-bubble">❌ 无法连接后端服务，请确保 FastAPI 服务已在运行 '
            "(<code>uvicorn src.api:app</code>)。</div>",
            unsafe_allow_html=True,
        )
        return "❌ 无法连接后端服务。", []
    except requests.exceptions.Timeout:
        placeholder.markdown(
            '<div class="asst-bubble">⏱ 请求超时，LLM 响应时间较长，请稍后重试。</div>',
            unsafe_allow_html=True,
        )
        return "⏱ 请求超时。", []
    except Exception as exc:
        placeholder.markdown(
            f'<div class="asst-bubble">❌ 未知错误: {exc}</div>',
            unsafe_allow_html=True,
        )
        return f"❌ 未知错误: {exc}", []


@st.cache_data(ttl=0, show_spinner=False)
def _check_health() -> dict | None:
    """Ping the backend health endpoint."""
    import requests
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ── Session state defaults ──────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []
if "pending_sources" not in st.session_state:
    st.session_state.pending_sources: list = []


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🩺 DiagRAG")
    st.caption("医学诊断 RAG 问答系统")

    health = _check_health()
    if health:
        st.success("✅ 后端已连接")
    else:
        st.error("⚠️ 后端未连接")
        st.caption("请先运行 `uvicorn src.api:app`")

    st.divider()

    # Settings card
    with st.container():
        st.markdown('<div class="sidebar-card"><h4>⚙️ 参数设置</h4>', unsafe_allow_html=True)

        top_k = st.slider(
            "🔎 检索文档数 (top_k)",
            min_value=1,
            max_value=20,
            value=5,
            help="每次问答从向量库中召回的最多文档块数量",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    if st.button("🗑 清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    with st.container():
        st.markdown('<div class="sidebar-card"><h4>🛠 技术栈</h4>', unsafe_allow_html=True)
        st.markdown(
            "- Milvus 向量数据库  \n"
            "- DashScope Embedding  \n"
            "- DashScope Qwen LLM  \n"
            "- FastAPI + Streamlit",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
      <h1>🩺 DiagRAG · 医学诊断问答</h1>
      <p>基于检索增强生成的智能医学诊断助手 | Powered by Qwen + Milvus RAG</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Chat history ───────────────────────────────────────────────────────────────

st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        col1, col2 = st.columns([1, 12])
        with col2:
            st.markdown(
                f'<div class="user-row"><div class="user-bubble">{msg["content"]}</div>'
                '<div class="user-avatar">👤</div></div>',
                unsafe_allow_html=True,
            )
    else:
        col1, col2 = st.columns([12, 1])
        with col1:
            _render_assistant_message(msg["content"], msg.get("sources"))

st.markdown("</div>", unsafe_allow_html=True)


# ── Input area ─────────────────────────────────────────────────────────────────

st.markdown('<div class="input-wrap">', unsafe_allow_html=True)

input_col, btn_col = st.columns([1, 0.08])
with input_col:
    question = st.text_area(
        "输入问题",
        placeholder="例如：患者出现持续性胸痛伴出汗，可能的诊断方向有哪些？",
        label_visibility="collapsed",
        key="question_input",
        height=68,
    )
with btn_col:
    submitted = st.button("发送", help="发送问题")

st.markdown("</div>", unsafe_allow_html=True)

if submitted and question.strip():
    q = question.strip()

    # Record user message
    st.session_state.messages.append({"role": "user", "content": q, "sources": []})

    # Render user bubble immediately
    st.markdown(
        f'<div class="user-row"><div class="user-bubble">{q}</div>'
        '<div class="user-avatar">👤</div></div>',
        unsafe_allow_html=True,
    )

    # Thinking indicator
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown(
        '<div class="asst-row">'
        '<div class="asst-avatar">🤖</div>'
        '<div class="thinking-bar">'
        '<div class="thinking-dots"><span></span><span></span><span></span></div>'
        '正在检索知识库并生成回答 ...'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Response area
    response_placeholder = st.empty()
    sources_placeholder = st.empty()

    # Stream the answer (returns full text + sources so we only call the LLM once)
    full_answer, sources_data = _stream_to_placeholder(
        question=q,
        placeholder=response_placeholder,
        sources_placeholder=sources_placeholder,
        top_k=top_k,
    )

    # Clear thinking indicator
    thinking_placeholder.empty()

    # Persist to session state (single LLM call total)
    st.session_state.messages.append(
        {"role": "assistant", "content": full_answer, "sources": sources_data}
    )
    st.rerun()

elif submitted:
    st.warning("请输入问题后再发送。")
