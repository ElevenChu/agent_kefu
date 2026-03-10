"""
并行边定义（修正版 - 真正并行）

关键修改：
- map_planner_to_direct_parallel：直接路由到具体工具（跳过 tool_selection，真正并行）
- map_planner_to_selection_parallel：通过 tool_selection 路由（可能串行）
"""
from typing import List
from langgraph.types import Send
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.state import OverallState
from app.core.logger import get_logger
from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticToolsParser
logger = get_logger(service="parallel_edges")
from typing import Any, Callable, Coroutine, Dict, List, Literal, Set
# 定义工具选择提示词
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.tool_selection.prompts import create_tool_selection_prompt_template
tool_selection_prompt = create_tool_selection_prompt_template()
from langchain_core.runnables.base import Runnable
async def map_planner_to_direct_parallel(llm: BaseChatModel,
    tool_schemas: List[type[BaseModel]],
    state: OverallState) -> List[Send]:
    """
    将 Planner 分解的任务直接映射到具体工具节点

    关键：
    - 不经过 tool_selection 节点
    - 直接 Send 到 cypher_query、predefined_cypher 或 customer_tools
    - LangGraph 会并发执行这些 Send

    这实现了真正的并行执行。
    """
    tool_selection_chain: Runnable[Dict[str, Any], Any] = (
        tool_selection_prompt
        | llm.bind_tools(tools=tool_schemas)
        | PydanticToolsParser(tools=tool_schemas, first_tool_only=True)
    )
    tasks = state.get("tasks", [])

    if not tasks:

        logger.warning("No tasks to execute")

            
     # 为每个任务创建 Send 到具体工具节点
        sends = []
        for i, task in enumerate(tasks):
                task_question = task.get("question", "").lower()
                task_parent = task.get("parent_task", "")

                send_state = {
                    "task": task_question,
                    "parent_task": task_parent,
                    "task_index": i,
                    "steps": ["direct_parallel"],
                }
                tool_selection_output: BaseModel = await tool_selection_chain.ainvoke(
                {"question": task_question}
                    )
                tool_name: str = tool_selection_output.model_json_schema().get("title", "")

                # 根据任务内容智能路由
                if tool_name == "predefined_cypher":
            
                     sends.append(
                        Send(
                         "predefined_cypher",
                          send_state
                        )
                        )
                elif tool_name == "cypher_query":
            
                      sends.append(
                        Send(
                     "cypher_query",
                        send_state
                    )
                     )
                else:
                  # 其他查询 → customer_tools
            
                  sends.append(
                      Send(
                    "customer_tools",
                    send_state
                )
            )

    return sends
