import os
import re
import yaml


class ConfigError(Exception):
    """Raised when config loading or key access fails."""
    pass


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR} / ${VAR:-default} placeholders with real env values."""
    def replacer(match: re.Match) -> str:
        expr = match.group(1)
        if ":-" in expr:
            var_name, default = expr.split(":-", 1)
            raw = os.environ.get(var_name.strip())
            return raw if raw else default.strip()
        raw = os.environ.get(expr.strip())
        return raw if raw else match.group(0)
    return re.sub(r"\$\{([^}]+)\}", replacer, value)


def _walk_and_substitute(obj):
    """Recursively substitute env vars in dict/list/str structures."""
    if isinstance(obj, dict):
        return {k: _walk_and_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_substitute(item) for item in obj]
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    return obj


def load_config(config_path: str | None = None) -> dict:
    """Load config.yml and inject environment variables into it.

    Raises:
        ConfigError: If the config file does not exist or cannot be read.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config.yml"
        )
    if not os.path.isfile(config_path):
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        with open(config_path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as e:
        raise ConfigError(f"Failed to parse config file '{config_path}': {e}") from e
    if raw_config is None:
        raise ConfigError(f"Config file is empty: {config_path}")
    return _walk_and_substitute(raw_config)


def get_milvus_config() -> dict:
    """Return the milvus section of the config.

    Raises:
        ConfigError: If the 'milvus' key is missing from the config.
    """
    config = load_config()
    if "milvus" not in config:
        raise ConfigError("Missing required config key: 'milvus'")
    return config["milvus"]


def get_embedding_config() -> dict:
    """Return the embedding section of the config.

    Raises:
        ConfigError: If the 'embedding' key is missing from the config.
    """
    config = load_config()
    if "embedding" not in config:
        raise ConfigError("Missing required config key: 'embedding'")
    return config["embedding"]


def get_llm_config() -> dict:
    """Return the llm section of the config.

    Raises:
        ConfigError: If the 'llm' key is missing from the config.
    """
    config = load_config()
    if "llm" not in config:
        raise ConfigError("Missing required config key: 'llm'")
    return config["llm"]


def get_retrieval_config() -> dict:
    """Return the retrieval section of the config.

    Raises:
        ConfigError: If the 'retrieval' key is missing from the config.
    """
    config = load_config()
    if "retrieval" not in config:
        raise ConfigError("Missing required config key: 'retrieval'")
    return config["retrieval"]


def get_reranker_config() -> dict:
    """Return the reranker section of the config.

    Falls back to defaults for keys not explicitly set in config.yml.
    Raises:
        ConfigError: If the 'retrieval' key is missing from the config.
    """
    retrieval = get_retrieval_config()
    defaults = {
        "enable_rerank": False,
        "rerank_top_k": 3,
        "rerank_mode": "score",
        "enable_bm25_blend": False,
        "rerank_fusion": "rrf",
        "max_docs_per_call": 10,
    }
    result = {**defaults}
    for key in defaults:
        if key in retrieval:
            result[key] = retrieval[key]
    return result


def get_preprocessor_config() -> dict:
    """Return the query preprocessor section of the config.

    Falls back to defaults for keys not explicitly set in config.yml.
    """
    retrieval = get_retrieval_config()
    defaults = {
        "enable_rewrite": False,
        "enable_expand": False,
        "max_variants": 2,
        "max_expand_terms": 4,
    }
    result = {**defaults}
    pp = retrieval.get("preprocessor", {})
    if isinstance(pp, dict):
        for key in defaults:
            if key in pp:
                result[key] = pp[key]
    return result


# ---------------------------------------------------------------------------
# 元数据过滤器配置
# ---------------------------------------------------------------------------

def get_metadata_filter_config() -> dict:
    """返回配置文件中 metadata_filter 部分的配置。

    配置中未显式设置的键会回退到默认值。
    """
    retrieval = get_retrieval_config()
    defaults = {
        "enabled": False,
        "mode": "whitelist",
        "whitelist": {},
        "blacklist": {},
    }
    result = {**defaults}
    mf = retrieval.get("metadata_filter", {})
    if isinstance(mf, dict):
        for key in defaults:
            if key in mf:
                result[key] = mf[key]
    return result


# ---------------------------------------------------------------------------
# 对话历史（长中短期记忆）配置
# ---------------------------------------------------------------------------

def get_conversation_config() -> dict:
    """返回配置文件中 conversation 部分的配置。

    配置中未显式设置的键会回退到默认值。
    """
    config = load_config()
    defaults = {
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": "",
        },
        "mysql": {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "",
            "database": "diagrag",
            "pool_size": 10,
            "pool_recycle": 3600,
        },
        "kafka": {
            "bootstrap_servers": "localhost:9092",
            "topic": "rag-conversation-events",
            "consumer_group": "rag-conversation-consumer",
            "acks": "all",
            "retries": 3,
        },
        "redis_cache_turns": 10,
        "summary_interval_turns": 5,
        "redis_ttl_days": 7,
        "mysql_retention_days": 30,
        "summary_llm_model": "qwen-plus",
        "summary_llm_temperature": 0.1,
        "summary_max_tokens": 500,
    }
    raw = config.get("conversation", {})
    return _deep_merge(defaults, raw)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, overriding existing keys."""
    result = {**base}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
