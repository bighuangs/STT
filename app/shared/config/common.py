"""旧配置读取函数兼容层；环境读取已统一收口到 AppSettings。"""
from app.shared.config.settings_config import get_settings


def env_str(name: str, default: str = "") -> str:
    return str(get_settings().legacy_env(name, default))


def env_bool(name: str, default: bool = False) -> bool:
    value = get_settings().legacy_env(name, default)
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float = 0.0) -> float:
    try:
        return float(get_settings().legacy_env(name, default))
    except (TypeError, ValueError):
        return default
