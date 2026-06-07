"""最小手动缓存验证脚本。

用途：
    直接调用本地 API，验证语义缓存是否生效。

使用前提：
    1. 本地 FastAPI 服务已启动（默认 http://127.0.0.1:8000）
    2. Redis / Milvus / MySQL 等依赖已启动
    3. 知识库中已有可检索内容

运行方式：
    python scripts/manual_test_cache.py

预期：
    - 第 1 次请求: cache_hit=False
    - 第 2 次相同请求: cache_hit=True
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

BASE_URL = "http://127.0.0.1:8000"
QUESTION = "急性心肌梗死的典型症状有哪些？"
USER_ID = "cache-test-user"
TIMEOUT = 120


def pretty(label: str, payload: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def test_chat_endpoint() -> None:
    payload = {
        "question": QUESTION,
        "user_id": USER_ID,
    }

    print("\n[1/2] 第一次调用 /chat")
    t1 = time.perf_counter()
    r1 = requests.post(f"{BASE_URL}/chat", json=payload, timeout=TIMEOUT)
    d1 = r1.json()
    elapsed1 = time.perf_counter() - t1
    pretty("/chat first response", d1)
    print(f"elapsed: {elapsed1:.2f}s")

    print("\n[2/2] 第二次调用 /chat（相同问题）")
    t2 = time.perf_counter()
    r2 = requests.post(f"{BASE_URL}/chat", json=payload, timeout=TIMEOUT)
    d2 = r2.json()
    elapsed2 = time.perf_counter() - t2
    pretty("/chat second response", d2)
    print(f"elapsed: {elapsed2:.2f}s")

    print("\n/chat 验证结果：")
    print(f"  first cache_hit:  {d1.get('cache_hit')}")
    print(f"  second cache_hit: {d2.get('cache_hit')}")
    print(f"  second hit_from:  {d2.get('cache_hit_from')}")



def test_conversation_endpoint() -> None:
    print("\n[conversation] 创建会话")
    create_payload = {
        "user_id": USER_ID,
        "title": "缓存验证会话",
    }
    create_resp = requests.post(
        f"{BASE_URL}/conversation/create",
        json=create_payload,
        timeout=TIMEOUT,
    )
    create_data = create_resp.json()
    pretty("session created", create_data)

    session_id = create_data["session_id"]

    msg_payload = {
        "content": QUESTION,
    }

    print("\n[1/2] 第一次调用 /conversation/{id}/message")
    t1 = time.perf_counter()
    r1 = requests.post(
        f"{BASE_URL}/conversation/{session_id}/message",
        json=msg_payload,
        timeout=TIMEOUT,
    )
    d1 = r1.json()
    elapsed1 = time.perf_counter() - t1
    pretty("conversation first response", d1)
    print(f"elapsed: {elapsed1:.2f}s")

    print("\n[2/2] 第二次调用 /conversation/{id}/message（相同问题）")
    t2 = time.perf_counter()
    r2 = requests.post(
        f"{BASE_URL}/conversation/{session_id}/message",
        json=msg_payload,
        timeout=TIMEOUT,
    )
    d2 = r2.json()
    elapsed2 = time.perf_counter() - t2
    pretty("conversation second response", d2)
    print(f"elapsed: {elapsed2:.2f}s")

    print("\n/conversation 验证结果：")
    print(f"  first cache_hit:  {d1.get('cache_hit')}")
    print(f"  second cache_hit: {d2.get('cache_hit')}")
    print(f"  second hit_from:  {d2.get('cache_hit_from')}")


if __name__ == "__main__":
    print("开始手动缓存验证")
    print(f"BASE_URL = {BASE_URL}")
    print(f"QUESTION = {QUESTION}")

    try:
        test_chat_endpoint()
    except Exception as exc:
        print(f"/chat 验证失败: {exc}")

    try:
        test_conversation_endpoint()
    except Exception as exc:
        print(f"/conversation 验证失败: {exc}")
