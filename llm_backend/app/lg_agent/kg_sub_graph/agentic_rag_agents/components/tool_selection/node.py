"""
A tool_selection node must
* take a single task at a time
* retrieve a list of available tools
    * text2cypher
    * custom pre-written cypher executors
        * these can be numerous and may be retrieved in the same fashion as CypherQuery node contents
    * unstructured text search (sim search)
* decide the appropriate tool for the task
* generate and validate parameters for the selected tool
* send the validated parameters to the appropriate tool node
"""

from typing import Any, Callable, Coroutine, Dict, List, Literal, Set
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.runnables.base import Runnable
from langgraph.types import Command, Send
from pydantic import BaseModel


from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.state import ToolSelectionInputState
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.tool_selection.prompts import create_tool_selection_prompt_template

# 定义工具选择提示词
tool_selection_prompt = create_tool_selection_prompt_template()


# 声明式的使用可配置模型：https://python.langchain.com/docs/how_to/chat_models_universal_init/#using-a-configurable-model-declaratively
def create_tool_selection_node(
    llm: BaseChatModel,
    tool_schemas: List[type[BaseModel]],
    default_to_text2cypher: bool = True,
) -> Callable[[ToolSelectionInputState], Coroutine[Any, Any, Command[Any]]]:
   

    # 构建工具选择链，由大模型根据传递过来的 Task，在预定义的工具列表中选择一个工具。
    tool_selection_chain: Runnable[Dict[str, Any], Any] = (
        tool_selection_prompt
        | llm.bind_tools(tools=tool_schemas)
        | PydanticToolsParser(tools=tool_schemas, first_tool_only=True)
    )

    # 从传入的tool_schemas列表中，获取每个工具的title属性，创建出一个工具名称集合。
    predefined_cypher_tools: Set[str] = {
        t.model_json_schema().get("title", "") for t in tool_schemas
    }


    # async def tool_selection(
    #     state: ToolSelectionInputState,
    # ) -> Command[Literal["text2cypher", "predefined_cypher", "customer_tools"]]:
    async def tool_selection(
        state: ToolSelectionInputState,
    ) -> Dict[str: Any]:
        question = state.get("question", "")
        
        # 为每个任务生成唯一标识，便于日志追踪
        task_id = id(state)
        
        try:
            # 调用工具选择链
            tool_selection_output: BaseModel = await tool_selection_chain.ainvoke(
                {"question": question}
            )

            if tool_selection_output is not None:
                tool_name: str = tool_selection_output.model_json_schema().get("title", "")
                
                # 根据工具名称返回不同的执行路径
                if tool_name == "predefined_cypher":
                    return {
                        "next_node": "predefined_cypher",
                        "task": question,
                        "query_name": tool_name,
                        "query_parameters": tool_selection_output.model_dump(),
                        "steps": ["tool_selection"],
                    }
                elif tool_name == "cypher_query":
                    return {
                        "next_node": "cypher_query",
                        "task": question,
                        "query_name": tool_name,
                        "query_parameters": tool_selection_output.model_dump(),
                        "steps": ["tool_selection"],
                    }
                else:
                    return {
                        "next_node": "customer_tools",
                        "task": question,
                        "query_name": tool_name,
                        "query_parameters": tool_selection_output.model_dump(),
                        "steps": ["tool_selection"],
                    }

            elif default_to_text2cypher:
                return {
                    "next_node": "cypher_query",
                    "task": question,
                    "query_name": "cypher_query",
                    "query_parameters": {},
                    "steps": ["tool_selection"],
                }

            else:
                return {
                    "next_node": "error",
                    "task": question,
                    "error": f"Unable to assign tool to question: `{question}`",
                    "steps": ["tool_selection"],
                }

        except Exception as e:
            return {
                "next_node": "error",
                "task": question,
                "error": str(e),
                "steps": ["tool_selection"],
            }

    return tool_selection
