from typing import Any, Callable, Coroutine, Dict, List
import asyncio
import os
from pathlib import Path
from pydantic import BaseModel, Field

# 导入GraphRAG相关模块
import app.graphrag.graphrag.api as api
from app.graphrag.graphrag.config.load_config import load_config
from app.graphrag.graphrag.callbacks.noop_query_callbacks import NoopQueryCallbacks
from app.graphrag.graphrag.utils.storage import load_table_from_storage
from app.graphrag.graphrag.storage.file_pipeline_storage import FilePipelineStorage
from app.lg_agent.kg_sub_graph.kg_neo4j_conn import get_neo4j_graph
from app.core.logger import get_logger
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from app.core.config import settings, ServiceType
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.northwind_retriever import NorthwindCypherRetriever
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.cypher_tools.utils import (
    create_text2cypher_generation_node,
    create_text2cypher_validation_node,
    create_text2cypher_execution_node,
    run_react_cypher_loop,
)



# 获取日志记录器
logger = get_logger(service="cypher_tools")


class CypherQueryInputState(BaseModel):
    task: str
    query: str
    steps: List[str]


class CypherQueryOutputState(BaseModel):
    task: str
    query: str
    errors: List[str]
    records: Dict[str, Any]
    steps: List[str]



def create_cypher_query_node(
    use_react: bool = True,
    max_attempts: int = 3,
) -> Callable[
    [CypherQueryInputState],
    Coroutine[Any, Any, Dict[str, List[CypherQueryOutputState] | List[str]]],
]:
    """
    创建 Text2Cypher 查询节点，用于LangGraph工作流。

    Parameters
    ----------
    use_react : bool, optional
        是否使用ReAct模式，默认为True
    max_attempts : int, optional
        ReAct模式下的最大尝试次数，默认为3

    返回
    -------
    Callable[[CypherQueryInputState], Dict[str, List[CypherQueryOutputState] | List[str]]]
        名为`cypher_query`的LangGraph节点。
    """

    async def cypher_query(
        state: Dict[str, Any],
    ) -> Dict[str, List[CypherQueryOutputState] | List[str]]:
        """
        执行Text2Cypher查询并返回结果。
        使用ReAct模式：Thought -> Action -> Observation -> 循环
        """
        print("==========================================进入cypher_query (ReAct模式)===================")
        errors = list()
        # 获取查询文本
        query = state.get("task", "")
        if not query:
            errors.append("未提供查询文本")
            return {
                "cyphers": [
                    CypherQueryOutputState(
                        **{
                            "task": "",
                            "query": "",
                            "statement": "",
                            "parameters": "",
                            "errors": errors,
                            "records": {"result": []},
                            "steps": ["error_no_task"],
                        }
                    )
                ],
                "steps": ["error_no_task"],
            }

        # 1. 根据.env文件中AGENT_SERVICE的设置，选择使用DeepSeek或Ollama启动的模型服务
        if settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
            model = ChatDeepSeek(
                api_key=settings.DEEPSEEK_API_KEY,
                model_name=settings.DEEPSEEK_MODEL,
                temperature=0.7,
                tags=["react_cypher"]
            )
            logger.info(f"[ReAct] 使用DeepSeek模型: {settings.DEEPSEEK_MODEL}")
        else:
            model = ChatOllama(
                model=settings.OLLAMA_AGENT_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.7,
                tags=["react_cypher"]
            )
            logger.info(f"[ReAct] 使用Ollama模型: {settings.OLLAMA_AGENT_MODEL}")

        # 2. 获取Neo4j图数据库连接
        try:
            neo4j_graph = get_neo4j_graph()
            logger.info("[ReAct] 成功获取Neo4j连接")
        except Exception as e:
            logger.error(f"[ReAct] 获取Neo4j连接失败: {e}")
            errors.append(f"数据库连接失败: {str(e)}")
            return {
                "cyphers": [
                    CypherQueryOutputState(
                        **{
                            "task": query,
                            "query": query,
                            "statement": "",
                            "parameters": "",
                            "errors": errors,
                            "records": {"result": []},
                            "steps": ["error_db_connection"],
                        }
                    )
                ],
                "steps": ["error_db_connection"],
            }

        # 3. 创建自定义检索器实例
        cypher_retriever = NorthwindCypherRetriever()

        # 4. 使用ReAct循环执行查询
        if use_react:
            logger.info(f"[ReAct] 开始处理查询: {query[:50]}...")
            react_result = await run_react_cypher_loop(
                state=state,
                llm=model,
                graph=neo4j_graph,
                cypher_example_retriever=cypher_retriever,
                max_attempts=max_attempts,
            )

            # 封装ReAct结果
            cypher_data = react_result.get("cyphers", [{}])[0]
            return {
                "cyphers": [
                    CypherQueryOutputState(
                        **{
                            "task": query,
                            "query": query,
                            "statement": cypher_data.get("statement", ""),
                            "parameters": "",
                            "errors": cypher_data.get("errors", []),
                            "records": {"result": cypher_data.get("records", [])},
                            "steps": react_result.get("steps", []),
                        }
                    )
                ],
                "steps": react_result.get("steps", []),
            }
        else:
            # 传统线性模式（保留作为fallback）
            logger.info("[ReAct] 使用传统线性模式")
            cypher_generation = create_text2cypher_generation_node(
                llm=model, graph=neo4j_graph, cypher_example_retriever=cypher_retriever
            )
            cypher_result = await cypher_generation(state)

            validate_cypher = create_text2cypher_validation_node(
                llm=model,
                graph=neo4j_graph,
                llm_validation=True,
                cypher_statement=cypher_result
            )
            execute_info = await validate_cypher(state=state)

            execute_cypher = create_text2cypher_execution_node(
                graph=neo4j_graph, cypher=execute_info
            )
            final_result = await execute_cypher(state)

            return {
                "cyphers": [
                    CypherQueryOutputState(
                        **{
                            "task": query,
                            "query": query,
                            "statement": "",
                            "parameters": "",
                            "errors": errors,
                            "records": {"result": final_result["cyphers"][0]["records"]} if final_result.get("cyphers") and len(final_result["cyphers"]) > 0 else {"result": []},
                            "steps": ["execute_cypher_query"],
                        }
                    )
                ],
                "steps": ["execute_cypher_query"],
            }

    return cypher_query

