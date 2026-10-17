"""连接器基础设施适配器。

默认只提供无网络的 deterministic fake 与内存控制库，生产连接器需由部署显式注册并
开启外部连接器开关。
"""

from app.infra.connectors.deterministic import DeterministicFakeConnector
from app.infra.connectors.in_memory import (
    InMemoryConnectorSyncRepository,
    InMemoryDocumentVersionBoundary,
)

__all__ = [
    "DeterministicFakeConnector",
    "InMemoryConnectorSyncRepository",
    "InMemoryDocumentVersionBoundary",
]
