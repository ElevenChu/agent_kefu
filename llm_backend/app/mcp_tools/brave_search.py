"""Brave Search MCP 工具实现

基于 MCP 协议的 Brave 搜索引擎工具，支持实时网络搜索。
"""

import os
from typing import List, Dict, Any, Optional
import aiohttp
from app.core.logger import get_logger

logger = get_logger(service="brave_search_mcp")


class BraveSearchMCP:
    """Brave Search MCP 客户端

    提供实时网络搜索能力，支持获取最新的产品信息、价格、新闻等。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.search.brave.com/api/v1",
        timeout: int = 10,
    ):
        """初始化 Brave Search MCP 客户端

        Args:
            api_key: Brave API Key，默认从环境变量 BRAVE_API_KEY 获取
            base_url: Brave API 基础 URL
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            logger.warning("BRAVE_API_KEY not found, Brave Search will be disabled")

        self.base_url = base_url
        self.timeout = timeout

    async def search(
        self,
        query: str,
        count: int = 5,
        offset: int = 0,
        mkt: str = "zh-CN",
        safesearch: str = "moderate",
        freshness: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行搜索查询

        Args:
            query: 搜索关键词
            count: 返回结果数量（最大 20）
            offset: 结果偏移量（用于分页）
            mkt: 市场区域代码，如 zh-CN, en-US
            safesearch: 安全搜索级别：off, moderate, strict
            freshness: 结果新鲜度：pd（过去一天）、pw（过去一周）、pm（过去一月）、py（过去一年）

        Returns:
            包含搜索结果的字典，格式：
            {
                "results": [
                    {
                        "title": "结果标题",
                        "url": "结果链接",
                        "description": "结果描述",
                        "published": "发布时间"
                    }
                ],
                "query": "原始查询"
            }
        """
        if not self.api_key:
            logger.error("Brave Search API key not configured")
            return {"results": [], "query": query, "error": "API key not configured"}

        url = f"{self.base_url}/search"
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        params = {
            "q": query,
            "count": min(count, 20),  # Brave API 最大返回 20 条
            "offset": offset,
            "mkt": mkt,
            "safesearch": safesearch,
        }

        if freshness:
            params["freshness"] = freshness

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, params=params, timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = self._parse_results(data)
                        logger.info(
                            f"Brave Search success: '{query[:30]}...' returned {len(results)} results"
                        )
                        return {
                            "results": results,
                            "query": query,
                        }
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Brave Search failed: {response.status} - {error_text}"
                        )
                        return {
                            "results": [],
                            "query": query,
                            "error": f"API error: {response.status}",
                        }

        except aiohttp.ClientTimeout:
            logger.error(f"Brave Search timeout for query: {query[:30]}...")
            return {"results": [], "query": query, "error": "Request timeout"}

        except Exception as e:
            logger.error(f"Brave Search error: {str(e)}")
            return {"results": [], "query": query, "error": str(e)}

    def _parse_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析 Brave API 返回的结果"""
        results = []

        # 解析网页结果
        web_results = data.get("web", {}).get("results", [])
        for item in web_results:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "published": item.get("age", ""),
                }
            )

        return results

    def format_results_for_prompt(self, search_result: Dict[str, Any]) -> str:
        """将搜索结果格式化为 LLM 可用的上下文

        Args:
            search_result: search() 方法返回的结果

        Returns:
            格式化后的文本，可直接作为 LLM 的上下文
        """
        if search_result.get("error"):
            return f"<!-- 搜索失败: {search_result['error']} -->"

        results = search_result.get("results", [])
        if not results:
            return "<!-- 未找到相关搜索结果 -->"

        formatted = []
        formatted.append(f"<搜索结果: '{search_result.get('query', '')}'>")
        formatted.append("")

        for i, result in enumerate(results, 1):
            title = result.get("title", "")
            url = result.get("url", "")
            desc = result.get("description", "")
            published = result.get("published", "")

            formatted.append(f"[{i}] {title}")
            if published:
                formatted.append(f"发布时间: {published}")
            formatted.append(f"链接: {url}")
            formatted.append(f"摘要: {desc}")
            formatted.append("")

        formatted.append("</搜索结果>")
        return "\n".join(formatted)


# 全局单例实例
_brave_search: Optional[BraveSearchMCP] = None


def get_brave_search() -> BraveSearchMCP:
    """获取 Brave Search MCP 单例实例"""
    global _brave_search
    if _brave_search is None:
        _brave_search = BraveSearchMCP()
    return _brave_search
