"""DashScope MCP 旧导入路径适配器。"""
from app.shared.config.settings_config import MCPSettings as McpConfig, get_settings

mcp_config = get_settings().mcp

__all__ = ["McpConfig", "mcp_config"]
