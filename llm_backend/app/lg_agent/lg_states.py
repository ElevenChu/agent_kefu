from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict, List, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from uuid import uuid4


class Router(TypedDict):
    """Classify user query."""
    logic: str
    type: Literal["general-query", "additional-query", "graphrag-query", "image-query", "file-query"]
    question: str = field(default_factory=str)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, '1' or '0'"
    )


class QueryRewriteResult(BaseModel):
    """查询改写结果"""
    original_query: str = Field(description="原始查询")
    rewritten_query: str = Field(description="改写后的查询")
    rewrite_type: Literal["none", "coreference", "ellipsis", "clarification"] = Field(
        description="改写类型: none(无需改写), coreference(指代消解), ellipsis(省略补全), clarification(意图澄清)"
    )
    confidence: float = Field(description="改写置信度 0-1", ge=0, le=1)
    reasoning: str = Field(description="改写理由")

# @dataclass(kw_only=True)： 强制要求数据类中的所有字段必须以关键字参数的形式提供。即不能以位置参数的方式传递。
@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. 
    """

    messages: Annotated[list[AnyMessage], add_messages]
    
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    
        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )
    

    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.
    

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""
    

# @dataclass(kw_only=True)： 强制要求数据类中的所有字段必须以关键字参数的形式提供。即不能以位置参数的方式传递。
@dataclass(kw_only=True)
class AgentState(InputState):
    """State of the retrieval graph / agent."""
    router: Router = field(default_factory=lambda: Router(type="general-query", logic=""))
    """The router's classification of the user's query."""
    steps: list[str] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""
    question: str = field(default_factory=str)
    answer: str = field(default_factory=str)
    hallucination: GradeHallucinations = field(default_factory=lambda: GradeHallucinations(binary_score="0"))

    # 查询改写相关
    rewrite_result: Optional[QueryRewriteResult] = None
    """查询改写结果"""
    trace_id: str = field(default_factory=lambda: str(uuid4())[:8])
    """追踪ID，用于日志关联"""
    rewrite_latency_ms: Optional[float] = None
    """查询改写耗时(毫秒)"""
