"""
多工具工作流（并行版本 - 保持原有三种工具）

这个版本：
1. 保持原有的三种工具节点结构
"""
from typing import Dict, List, Optional, Literal

from langchain_core.language_models import BaseChatModel
from langchain_neo4j import Neo4jGraph
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from pydantic import BaseModel

# 导入输入输出状态定义
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.state import (
    InputState,
    OutputState,
    OverallState,
)

# 导入 guardrails 逻辑
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.guardrails.node import create_guardrails_node

# 导入分解节点
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.planner import create_planner_node

# 导入工具选择节点
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.tool_selection import create_tool_selection_node

# 导入 text2cypher 节点
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.cypher_tools import create_cypher_query_node

# 导入 Cypher 示例检索器基类
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.base import BaseCypherExampleRetriever

# 导入预定义 Cypher 节点
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.predefined_cypher import create_predefined_cypher_node

# 导入自定义工具函数节点
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.customer_tools import create_graphrag_query_node

# 导入错误处理
from ...components.errors import create_error_tool_selection_node

# 导入最终答案节点
from ...components.final_answer import create_final_answer_node

# 导入汇总节点
from ...components.summarize import create_summarization_node

# 导入并行边定义
from ..parallel_edges_simple import (
    map_planner_to_direct_parallel,
)

# 导入 guardrails 边（从 parallel_edges_simple 移到这里，或使用原来的）
from ...edges import guardrails_conditional_edge

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class AgentState(InputState):
    """The router's classification of user's query."""
    steps: list[str] = field(default_factory=list)
    question: str = field(default_factory=str)
    answer: str = field(default_factory=str)


def create_multi_tool_workflow_parallel_simple(
    llm: BaseChatModel,
    graph: Neo4jGraph,
    tool_schemas: List[type[BaseModel]],
    predefined_cypher_dict: Dict[str, str],
    cypher_example_retriever: BaseCypherExampleRetriever,
    scope_description: Optional[str] = None,
    llm_cypher_validation: bool = True,
    max_attempts: int = 3,
    attempt_cypher_execution_on_final_attempt: bool = False,
    default_to_text2cypher: bool = True,
) -> CompiledStateGraph:
    """
    Create a multi tool Agent workflow with parallel execution support.

    Parameters
    ----------
    routing_mode : Literal["direct", "selection"], optional
        Routing mode:
        - "direct": Use direct routing to tool nodes (TRUE parallel, recommended)
        - "selection": Route via tool_selection node (may be serial)

    Returns
    -------
    CompiledStateGraph
        The workflow.
    """
    # 1. 创建 guardrails 节点
    guardrails = create_guardrails_node(
        llm=llm, graph=graph, scope_description=scope_description
    )

    # 2. 创建 planner 节点
    planner = create_planner_node(llm=llm)

    # 3. 创建三种工具节点（保持原有结构）
    cypher_query = create_cypher_query_node()
    predefined_cypher = create_predefined_cypher_node(
        graph=graph, predefined_cypher_dict=predefined_cypher_dict
    )
    customer_tools = create_graphrag_query_node()

  

    # 5. 创建错误处理节点
    error_tool_selection = create_error_tool_selection_node()

    # 6. 创建汇总节点
    summarize = create_summarization_node(llm=llm)

    # 7. 创建最终答案节点
    final_answer = create_final_answer_node()

    # 创建状态图
    main_graph_builder = StateGraph(OverallState, input=InputState, output=OutputState)

    # ====== 添加所有节点（保持原有结构）======
    main_graph_builder.add_node(guardrails)
    main_graph_builder.add_node(planner)
    main_graph_builder.add_node(cypher_query)        # 工具 1
    main_graph_builder.add_node(predefined_cypher)    # 工具 2
    main_graph_builder.add_node("customer_tools", customer_tools)  # 工具 3
    main_graph_builder.add_node(summarize)
    main_graph_builder.add_node(error_tool_selection)
    main_graph_builder.add_node(final_answer)

    # ====== 添加边 ======
    main_graph_builder.add_edge(START, "guardrails")

    # Guardrails 路由
    main_graph_builder.add_conditional_edges(
        "guardrails",
        guardrails_conditional_edge,
    )

    # ====== 根据 routing_mode 选择边函数 ======
    
    # 直接路由到工具节点（真正并行）
    main_graph_builder.add_conditional_edges(
            "planner",
            map_planner_to_direct_parallel,
            ["predefined_cypher", "cypher_query", "customer_tools"],
    )

    # ====== 工具执行到汇总（保持原有）======
    main_graph_builder.add_edge("cypher_query", "summarize")
    main_graph_builder.add_edge("predefined_cypher", "summarize")
    main_graph_builder.add_edge("customer_tools", "summarize")
    main_graph_builder.add_edge("error_tool_selection", "summarize")

    # ====== 最终输出（保持原有）======
    main_graph_builder.add_edge("summarize", "final_answer")
    main_graph_builder.add_edge("final_answer", END)

    return main_graph_builder.compile()
