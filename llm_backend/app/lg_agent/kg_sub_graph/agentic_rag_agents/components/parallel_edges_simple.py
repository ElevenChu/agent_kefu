"""
并行边定义（修正版 - 真正并行）

关键修改：
- map_planner_to_direct_parallel：直接路由到具体工具（跳过 tool_selection，真正并行）
- 使用 asyncio.gather 让 LLM 调用并发执行
"""
import asyncio
from typing import List
from langgraph.types import Send
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.state import OverallState
from app.core.logger import get_logger
from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticToolsParser
from typing import Any, Dict
from langchain_core.runnables.base import Runnable

# 定义工具选择提示词
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.tool_selection.prompts import create_tool_selection_prompt_template

tool_selection_prompt = create_tool_selection_prompt_template()
logger = get_logger(service="parallel_edges")


async def map_planner_to_direct_parallel(
    llm: BaseChatModel,
    tool_schemas: List[type[BaseModel]],
    state: OverallState
) -> List[Send]:
    """
    将 Planner 分解的任务直接映射到具体工具节点

    关键：
    - 不经过 tool_selection 节点
    - 使用 asyncio.gather 并发执行 LLM 调用
    - 直接 Send 到 cypher_query、predefined_cypher 或 customer_tools
    - LangGraph 会并发执行这些 Send
    """
    tool_selection_chain: Runnable[Dict[str, Any], Any] = (
        tool_selection_prompt
        | llm.bind_tools(tools=tool_schemas)
        | PydanticToolsParser(tools=tool_schemas, first_tool_only=True)
    )

    tasks = state.get("tasks", [])

    if not tasks:
        logger.warning("No tasks to execute")
        return []

    logger.info(f"Preparing to route {len(tasks)} tasks in parallel")

    # 1. 并发收集所有工具选择结果
    llm_calls = [
        tool_selection_chain.ainvoke({"question": task.get("question", "")})
        for task in tasks
    ]

    # 并发执行所有 LLM 调用
    tool_outputs = await asyncio.gather(*llm_calls)
    logger.info(f"Completed {len(tool_outputs)} parallel LLM calls")

    # 2. 统一构建 Send 列表
    sends = []
    for i, (task, tool_output) in enumerate(zip(tasks, tool_outputs)):
        task_question = task.get("question", "").lower()
        task_parent = task.get("parent_task", "")

        send_state = {
            "task": task_question,
            "parent_task": task_parent,
            "task_index": i,
            "steps": ["direct_parallel"],
        }

        tool_name: str = tool_output.model_json_schema().get("title", "")

        # 根据工具名称路由到不同节点
        if tool_name == "predefined_cypher":
            logger.info(f"Task {i}: Routing to predefined_cypher")
            sends.append(
                Send(
                    "predefined_cypher",
                    send_state
                )
            )
        elif tool_name == "cypher_query":
            logger.info(f"Task {i}: Routing to cypher_query")
            sends.append(
                Send(
                    "cypher_query",
                    send_state
                )
            )
        else:
            # 其他查询 → customer_tools
            logger.info(f"Task {i}: Routing to customer_tools")
            sends.append(
                Send(
                    "customer_tools",
                    send_state
                )
            )

    logger.info(f"Created {len(sends)} Send objects for parallel execution")
    return sends
