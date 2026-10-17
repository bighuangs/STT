"""MinerU 旧导入路径适配器。"""
from app.shared.config.settings_config import MinerUSettings as MinerUConfig, get_settings

mineru_config = get_settings().mineru

__all__ = ["MinerUConfig", "mineru_config"]
