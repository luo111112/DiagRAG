"""pytest 配置与共享 fixtures。"""

import pytest

from src.config_loader import get_milvus_config


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "milvus: tests that require a running Milvus service",
    )
    config.addinivalue_line(
        "markers",
        "embedding: tests that require a valid DashScope API key",
    )


@pytest.fixture(scope="session")
def milvus_config() -> dict:
    """加载 Milvus 配置（从 config.yml）。"""
    return get_milvus_config()


@pytest.fixture(scope="session")
def milvus_available(milvus_config: dict) -> bool:
    """检查 Milvus 服务是否可用。不可用时返回 False，测试将被跳过。"""
    try:
        from pymilvus import connections
        alias = "test_milvus_check"
        connections.connect(
            alias=alias,
            host=milvus_config["host"],
            port=str(milvus_config["port"]),
        )
        connected = connections.has_connection(alias)
        connections.disconnect(alias=alias)
        return connected
    except Exception:
        return False


@pytest.fixture(scope="session")
def embedding_api_key() -> str | None:
    """返回 DashScope API key（来自环境变量或配置），若无则返回 None。"""
    import os

    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        try:
            from src.config_loader import get_embedding_config

            cfg = get_embedding_config()
            key = cfg.get("dashscope_api_key", "")
        except Exception:
            pass
    return key if key else None
