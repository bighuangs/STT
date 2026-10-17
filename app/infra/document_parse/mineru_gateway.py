"""
保留 MinerU 外部能力的统一出口，便于后续继续收口公共交互逻辑。
"""

from app.infra.config.providers import infra_config

class MinerUGateway:
    @property
    def base_url(self) -> str:
        """
         获取 MinerU 服务基础地址。

         Returns:
           str: MinerU 接口基础 URL。
        """
        return infra_config.mineru.base_url

    @property
    def api_key(self) -> str:
        """
         获取 MinerU 服务 API Token。

         Returns:
            str: MinerU 调用所需的 Token。
        """
        return infra_config.mineru.api_key

mineru_gateway = MinerUGateway()