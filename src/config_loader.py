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
