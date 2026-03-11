"""LangChain 风格的 MCP 工具定义

将 MCP 工具包装成 LangChain Tool，支持 bind_tools() 绑定
"""

from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from app.mcp_tools.brave_search import get_brave_search
from app.core.logger import get_logger

logger = get_logger(service="lc_tools")


class BraveSearchInput(BaseModel):
    """Brave 搜索工具输入参数"""
    query: str = Field(description="搜索关键词，要查找的内容")
    count: int = Field(default=5, description="返回结果数量，默认5条，最大20条")
    freshness: str = Field(default="pw", description="结果新鲜度: pd(过去一天), pw(过去一周), pm(过去一月), py(过去一年)")


class BraveSearchTool(BaseTool):
    """Brave 实时搜索工具

    当用户询问需要实时信息的问题时使用，如：
    - 最新产品价格
    - 最近新闻事件
    - 当前市场动态
    - 时效性信息
    """

    name: str = "brave_search"
    description: str = """使用 Brave 搜索引擎进行实时网络搜索。

适用场景：
- 查询最新产品价格、促销信息
- 获取最近的新闻或行业动态
- 需要实时、时效性信息的问题
- 超出知识库范围的外部信息

注意：搜索会增加响应时间，只在必要时使用。"""
    args_schema: Type[BaseModel] = BraveSearchInput

    async def _arun(self, query: str, count: int = 5, freshness: str = "pw") -> str:
        """异步执行搜索"""
        logger.info(f"Tool brave_search invoked with query: {query}")

        brave = get_brave_search()
        result = await brave.search(
            query=query,
            count=min(count, 20),
            freshness=freshness,
            mkt="zh-CN"
        )

        # 格式化结果为文本
        return brave.format_results_for_prompt(result)

    def _run(self, query: str, count: int = 5, freshness: str = "pw") -> str:
        """同步执行（不推荐，使用异步版本）"""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._arun(query, count, freshness)
        )


# 工具实例
brave_search_tool = BraveSearchTool()

# 工具列表（用于 bind_tools）
mcp_tools = [brave_search_tool]
