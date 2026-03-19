from typing import Any, Callable, Coroutine, Dict
import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.cypher_tools.prompts import (
    create_text2cypher_generation_prompt_template,
    create_text2cypher_validation_prompt_template,
    create_text2cypher_correction_prompt_template,
    create_react_thought_prompt_template,
    create_react_cypher_generation_prompt_template,
    create_react_observation_prompt_template,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.base import BaseCypherExampleRetriever
from typing_extensions import TypedDict
from typing import Annotated, Any, Dict, List, Optional, Callable, Coroutine
from operator import add
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
import regex as re
from langchain_core.runnables.base import Runnable
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from neo4j.exceptions import CypherSyntaxError
from langchain_ollama import ChatOllama
from app.core.config import settings

# 设置Neo4j驱动的日志级别为ERROR，禁止WARNING消息
logging.getLogger("neo4j").setLevel(logging.ERROR)
# 禁用langchain_neo4j相关日志
logging.getLogger("langchain_neo4j").setLevel(logging.ERROR)
# 禁用驱动相关日志
logging.getLogger("neo4j.io").setLevel(logging.ERROR)
logging.getLogger("neo4j.bolt").setLevel(logging.ERROR)

# 创建模块logger
logger = logging.getLogger(__name__)
logging.getLogger("neo4j.bolt").setLevel(logging.ERROR)


class CypherInputState(TypedDict):
    task: Annotated[list, add]

class CypherState(TypedDict):
    task: Annotated[list, add]
    statement: str
    parameters: Optional[Dict[str, Any]]
    errors: List[str]
    records: List[Dict[str, Any]]
    next_action_cypher: str
    attempts: int
    steps: Annotated[List[str], add]

class CypherOutputState(TypedDict):
    task: Annotated[list, add]
    statement: str
    parameters: Optional[Dict[str, Any]]
    errors: List[str]
    records: List[Dict[str, Any]]
    steps: List[str]


# ==================== ReAct 状态定义 ====================

class ReActCypherState(TypedDict):
    """ReAct模式下的Cypher查询状态"""
    task: str                          # 原始任务/问题
    thought: str                       # 当前思考
    cypher_statement: str              # 生成的Cypher
    execution_result: List[Dict[str, Any]]  # 执行结果
    errors: List[str]                  # 错误信息
    attempts: int                      # 尝试次数
    is_complete: bool                  # 是否完成
    should_retry: bool                 # 是否重试
    retry_reason: str                  # 重试原因
    steps: List[str]                   # 执行步骤


class ReActThoughtOutput(BaseModel):
    """ReAct思考阶段的结构化输出"""
    thought: str = Field(description="对当前情况的分析思考")
    action: str = Field(description="下一步行动: generate(生成)/correct(修正)/end(结束)")
    reasoning: str = Field(description="选择该行动的理由")


class ReActObservationOutput(BaseModel):
    """ReAct观察阶段的结构化输出"""
    is_satisfactory: bool = Field(description="结果是否满足需求")
    analysis: str = Field(description="结果分析")
    suggestion: str = Field(description="改进建议(如不满意)")

class Property(BaseModel):
    """
    Represents a filter condition based on a specific node property in a graph in a Cypher statement.
    """

    node_label: str = Field(
        description="The label of the node to which this property belongs."
    )
    property_key: str = Field(description="The key of the property being filtered.")
    property_value: str = Field(
        description="The value that the property is being matched against.",
        coerce_numbers_to_str=True,
    )

class ValidateCypherOutput(BaseModel):
    """
    Represents the validation result of a Cypher query's output,
    including any errors and applied filters.
    """

    errors: Optional[List[str]] = Field(
        description="A list of syntax or semantical errors in the Cypher statement. Always explain the discrepancy between schema and Cypher statement"
    )
    filters: Optional[List[Property]] = Field(
        description="A list of property-based filters applied in the Cypher statement."
    )

# 定义text2cypher generation prompt
generation_prompt = create_text2cypher_generation_prompt_template()

# 定义text2cypher validation prompt
validation_prompt_template = create_text2cypher_validation_prompt_template()

# 定义text2cypher correction prompt
correction_cypher_prompt = create_text2cypher_correction_prompt_template()


def validate_cypher_query_syntax(graph: Neo4jGraph, cypher_statement: str) -> List[str]:
    """
    Validate the Cypher statement syntax by running an EXPLAIN query.

    Parameters
    ----------
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    cypher_statement : str
        The Cypher statement to validate.

    Returns
    -------
    List[str]
        If the statement contains invalid syntax, return an error message in a list
    """
    errors = list()
    try:
        # 使用 EXPLAIN 查询来验证Cypher语句的语法，仅仅查看语法是否正确，而不实际执行查询
        graph.query(f"EXPLAIN {cypher_statement}")
    except CypherSyntaxError as e:
        errors.append(str(e.message))
    return errors


def correct_cypher_query_relationship_direction(
    graph: Neo4jGraph, cypher_statement: str
) -> str:
    """
    Correct Relationship directions in the Cypher statement with LangChain's `CypherQueryCorrector`.

    Parameters
    ----------
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    cypher_statement : str
        The Cypher statement to validate.

    Returns
    -------
    str
        The Cypher statement with corrected Relationship directions.
    """
    # 从数据库中提取关系的结构性信息
    corrector_schema = [
        Schema(el["start"], el["type"], el["end"])
        for el in graph.structured_schema.get("relationships", list())
    ]

    # 使用langchain_neo4j 的CypherQueryCorrector 来校验Cypher语句的语法
    # 比如 ：MATCH (a:Person)-[r:FRIENDS_WITH]->(b:Person) ，如果r:FRIENDS_WITH 是反向的，则会被纠正为：MATCH (a:Person)-[r:FRIENDS_WITH]->(b:Person)
    cypher_query_corrector = CypherQueryCorrector(corrector_schema)

    corrected_cypher: str = cypher_query_corrector(cypher_statement)

    return corrected_cypher


def get_cypher_query_node_graph_schema() -> str:
    # 以 "- CypherQuery" 开始的整个段落，直到 "Relationship properties" 或 "- " 为止
    return r"^(- \*\*CypherQuery\*\*[\s\S]+?)(^Relationship properties|- \*)"

def retrieve_and_parse_schema_from_graph_for_prompts(graph: Neo4jGraph) -> str:
    
    """
    关键点：
    schema 指的是 Neo4j 数据库的结构描述，包括：
    - 节点类型：如 Product, Category, Supplier 等
    - 节点属性：如 ProductName, UnitPrice, CategoryName 等
    - 关系类型：如 BELONGS_TO, SUPPLIED_BY, CONTAINS 等
    - 关系属性：关系上可能的属性（如有）

    提取出来的Schema 大致如下：
    Node properties:
        - **Product**: ProductID, ProductName, UnitPrice, UnitsInStock...
        - **Category**: CategoryID, CategoryName, Description...

    Relationship properties:
        - **BELONGS_TO**: 
        - **SUPPLIED_BY**: 
    
    必要性：
    1. 动态适应数据库变化：如果数据库结构变化（新增节点类型、关系或属性），系统无需修改代码即可适应
    2. 提高查询准确性：通过向大语言模型提供准确的数据库结构，大大降低生成错误查询的可能性
    3. 促进零样本学习：即使没有特定领域的示例，模型也能根据提供的结构信息生成符合语法的查询
    """
    
    schema: str = graph.get_schema

    # 过滤掉对用户查询不相关的内部结构信息
    if "CypherQuery" in schema:
        schema = re.sub(  
            get_cypher_query_node_graph_schema(), r"\2", schema, flags=re.MULTILINE
        )
    
    # 在这里添加一行：将所有花括号替换为方括号，避免模板变量冲突
    # 因为 Schema 中包含 { } ，会与 ChatPromptTemplate 模版中的 input_variables 
    schema = schema.replace("{", "[").replace("}", "]")
    
    return schema


async def validate_cypher_query_with_llm(
    validate_cypher_chain: Runnable[Dict[str, Any], Any],
    question: str,
    graph: Neo4jGraph,
    cypher_statement: str,
) -> Dict[str, List[str]]:
    """
    Validate the Cypher statement with an LLM.
    Use declared LLM to find Node and Property pairs to validate.
    Validate Node and Property pairs against the Neo4j graph.

    Parameters
    ----------
    validate_cypher_chain : RunnableSerializable
        The LangChain LLM to perform processing.
    question : str
        The question associated with the Cypher statement.
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    cypher_statement : str
        The Cypher statement to validate.

    Returns
    -------
    Dict[str, List[str]]
        A Python dictionary with keys `errors` and `mapping_errors`, each with a list of found errors.
    """

    errors: List[str] = []
    mapping_errors: List[str] = []


    # 使用大模型验证Cypher语句的语法， 通过 Pydantic 结构化输出
    llm_output: ValidateCypherOutput = await validate_cypher_chain.ainvoke(
        {
            "question": question,
            "schema": retrieve_and_parse_schema_from_graph_for_prompts(graph),
            "cypher": cypher_statement,
        }
    )

    # 如果 Pydantic 结构化输出中包含 errors，则将 errors 添加到 errors 列表中
    if llm_output.errors:
        errors.extend(llm_output.errors)
    # 如果 Pydantic 结构化输出中包含 filters，则遍历每个过滤器。
    if llm_output.filters:
        for filter in llm_output.filters:
            # 仅对字符串类型的属性进行映射检查。通过检查 graph.structured_schema 中的节点属性，判断属性类型是否为字符串。
            if (
                not [
                    prop
                    for prop in graph.structured_schema["node_props"][filter.node_label]
                    if prop["property"] == filter.property_key
                ][0]["type"]
                == "STRING"
            ):
                continue

            # 对于每个过滤器，构建一个 Cypher 查询，检查数据库中是否存在具有指定属性值的节点。
            mapping = graph.query(
                f"MATCH (n:{filter.node_label}) WHERE toLower(n.`{filter.property_key}`) = toLower($value) RETURN 'yes' LIMIT 1",
                {"value": filter.property_value},
            )
            if not mapping:
                mapping_error = f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                mapping_errors.append(mapping_error)
    return {"errors": errors, "mapping_errors": mapping_errors}


def validate_cypher_query_with_schema(
    graph: Neo4jGraph, cypher_statement: str
) -> List[str]:
    """
    Validate the provided Cypher statement using the schema retrieved from the graph.
    This will ensure the existance of names nodes, relationships and properties.
    This will validate property values with enums and number ranges, if available.
    This method does not use an LLM.

    Parameters
    ----------
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    cypher_statement : str
        The Cypher to be validated.

    Returns
    -------
    List[str]
        A list of any found errors.
    """
    from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.models import (
    CypherValidationTask,
    Neo4jStructuredSchema,
    Neo4jStructuredSchemaPropertyNumber,
)
    from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.validators import (
    extract_entities_for_validation,
    update_task_list_with_property_type,
    _validate_node_property_names_with_enum,
    _validate_node_property_values_with_enum,
    _validate_node_property_values_with_range,
    _validate_relationship_property_names_with_enum,
    _validate_relationship_property_values_with_enum,
    _validate_relationship_property_values_with_range,
    )

    schema: Neo4jStructuredSchema = Neo4jStructuredSchema.model_validate(
        graph.get_structured_schema
    )
    nodes_and_rels = extract_entities_for_validation(cypher_statement=cypher_statement)

    node_tasks = update_task_list_with_property_type(
        nodes_and_rels.get("nodes", list()), schema, "node"
    )
    rel_tasks = update_task_list_with_property_type(
        nodes_and_rels.get("relationships", list()), schema, "rel"
    )

    errors: List[str] = list()

    node_prop_name_enum_tasks = node_tasks
    node_prop_val_enum_tasks = [n for n in node_tasks if n.property_type == "STRING"]
    node_prop_val_range_tasks = [
        n
        for n in node_tasks
        if (n.property_type == "INTEGER" or n.property_type == "FLOAT")
    ]

    rel_prop_name_enum_tasks = rel_tasks
    rel_prop_val_enum_tasks = [n for n in rel_tasks if n.property_type == "STRING"]
    rel_prop_val_range_tasks = [
        n
        for n in rel_tasks
        if (n.property_type == "INTEGER" or n.property_type == "FLOAT")
    ]

    errors.extend(
        _validate_node_property_names_with_enum(schema, node_prop_name_enum_tasks)
    )
    errors.extend(
        _validate_node_property_values_with_enum(schema, node_prop_val_enum_tasks)
    )
    errors.extend(
        _validate_node_property_values_with_range(schema, node_prop_val_range_tasks)
    )

    errors.extend(
        _validate_relationship_property_names_with_enum(
            schema, rel_prop_name_enum_tasks
        )
    )
    errors.extend(
        _validate_relationship_property_values_with_enum(
            schema, rel_prop_val_enum_tasks
        )
    )
    errors.extend(
        _validate_relationship_property_values_with_range(
            schema, rel_prop_val_range_tasks
        )
    )

    return errors


def validate_no_writes_in_cypher_query(cypher_statement: str) -> List[str]:
    """
    Validate whether the provided Cypher contains any write clauses.

    Parameters
    ----------
    cypher_statement : str
        The Cypher statement to validate.

    Returns
    -------
    List[str]
        A list of any found errors.
    """
    errors: List[str] = list()

    # 限制不允许使用写操作
    WRITE_CLAUSES = {
    "CREATE",
    "DELETE",
    "DETACH DELETE",
    "SET",
    "REMOVE",
    "FOREACH",
    "MERGE",
    }

    for wc in WRITE_CLAUSES:
        if wc in cypher_statement.upper():
            errors.append(f"Cypher contains write clause: {wc}")

    return errors


def create_text2cypher_generation_node(
    llm: BaseChatModel,
    graph: Neo4jGraph,
    cypher_example_retriever: BaseCypherExampleRetriever,
) -> str:
    
    text2cypher_chain = generation_prompt | llm | StrOutputParser()

    async def generate_cypher(state: CypherInputState) -> Dict[str, Any]:
        """
        Generates a cypher statement based on the provided schema and user input
        """
        task = state.get("task", "")
        # 获取针对当前任务的cypher示例, 选择 k 个
        examples: str = cypher_example_retriever.get_examples(
            **{"query": task[0] if isinstance(task, list) else task, "k": 3}
        )
        generated_cypher = await text2cypher_chain.ainvoke(
            {
                "question": state.get("task", ""),
                "fewshot_examples": examples,
                "schema": graph.schema,
            }
        )
        return generated_cypher

    return generate_cypher

def create_text2cypher_validation_node(
    graph: Neo4jGraph,
    llm: Optional[BaseChatModel] = None,
    llm_validation: bool = True,
    cypher_statement: str = None,
) -> Callable[[CypherState], Coroutine[Any, Any, dict[str, Any]]]:
    """
    Create a Text2Cypher query validation node for a LangGraph workflow.

    Parameters
    ----------
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    llm : Optional[BaseChatModel], optional
        The LLM to use for processing if LLM validation is desired. By default None
    llm_validation : bool, optional
        Whether to perform LLM validation with the provided LLM, by default True
    Returns
    -------
    Callable[[CypherState], CypherState]
        The LangGraph node.
    """
    # 如果传递了 LLM， 则会借助大模型进行Cypher 校验：针对语法格式的
    if llm is not None and llm_validation:
        validate_cypher_chain = validation_prompt_template | llm.with_structured_output(
            ValidateCypherOutput
        )

    async def validate_cypher(state: CypherState) -> Dict[str, Any]:
        """
        Validates the Cypher statements and maps any property values to the database.
        """

        errors = []
        mapping_errors = []

        # 1. 语法校验：检查Cypher查询的语法是否正确，例如括号匹配、关键字使用等。
        syntax_error = validate_cypher_query_syntax(
            graph=graph, cypher_statement=cypher_statement
        )
        errors.extend(syntax_error)

        # 检查Cypher查询中是否包含写操作(如CREATE、DELETE、SET等)，防止大模型意外修改数据库,
        write_errors = validate_no_writes_in_cypher_query(cypher_statement=cypher_statement)
        errors.extend(write_errors)

        # Neo4j的关系是有方向性的。这一步会检查关系方向是否正确，如果不正确，会尝试自动修复。这对提高查询成功率很重要。
        corrected_cypher = correct_cypher_query_relationship_direction(
            graph=graph, cypher_statement=cypher_statement
        )

        # 如果启用了大模型验证，会使用语言模型检查Cypher查询的更高级错误，
        # 例如语义上是否符合用户问题、属性映射是否正确等。这是一种更智能的验证方式。
        if llm is not None and llm_validation:
            llm_errors = await validate_cypher_query_with_llm(
                validate_cypher_chain=validate_cypher_chain,
                question=state.get("task", ""),
                graph=graph,
                cypher_statement=cypher_statement,
            )
            errors.extend(llm_errors.get("errors", []))
            mapping_errors.extend(llm_errors.get("mapping_errors", []))

        # 如果禁用大模型验证，会使用更严格的模式检查Cypher查询，确保所有节点和关系都存在，并且属性值符合类型限制。
        if not llm_validation:
            cypher_errors = validate_cypher_query_with_schema(
                graph=graph, cypher_statement=cypher_statement
            )
            errors.extend(cypher_errors)

        # 区分真正的语法错误和数据不存在的情况
        # Map：mapping_errors: ['Missing value mapping for Order on property orderId with value 12345', 'Missing value mapping for Product on property ProductName with value 小米音箱']
        # Map 会表明你的Cypher查询语法是正确的，但查询中使用的具体值在数据库中不存在。这是数据不存在的问题，而不是查询语法的问题。
        if errors:  # 真正的语法错误
            correct_cypher_chain = correction_cypher_prompt | llm | StrOutputParser()
            corrected_cypher_update = correct_cypher_chain.ainvoke(
                {
                    "question": state.get("task"),
                    "errors": errors, 
                    "cypher": cypher_statement,
                    "schema": graph.schema,
                }
            )
            corrected_cypher = corrected_cypher_update
            next_action = "execute_cypher" 

        elif mapping_errors:  # 数据映射错误
            # TODO：1. 可以直接结束查询，告诉用户数据库中不存在 2. 可以再次引导用户提问，确认信息， 3. 也可以针对历史对话重新生成Cypher，再次尝试
            next_action = "execute_cypher"  # 或 "__end__"
        else:  # 没有错误
            next_action = "execute_cypher"

        # # 如果有错误且未达到最大尝试次数，转到"correct_cypher"节点尝试修复错误
        # if (errors or mapping_errors) and GENERATION_ATTEMPT < max_attempts:
        #     next_action = "correct_cypher"
        # # 如果未达到最大尝试次数，转到"execute_cypher"节点执行Cypher查询
        # elif GENERATION_ATTEMPT < max_attempts:
        #     next_action = "execute_cypher"
        # elif (
        #     GENERATION_ATTEMPT == max_attempts
        #     and attempt_cypher_execution_on_final_attempt
        # ):
        #     next_action = "execute_cypher"
        # else:
        #     next_action = "__end__"

        return {
            "next_action_cypher": next_action,
            "statement": corrected_cypher,
            "errors": errors,
            "steps": ["validate_cypher"],
        }

    return validate_cypher

def create_text2cypher_execution_node(
    graph: Neo4jGraph,
    cypher: str
) -> Callable[
    [CypherState], Coroutine[Any, Any, Dict[str, List[CypherOutputState] | List[str]]]
]:
    """
    Create a Text2Cypher execution node for a LangGraph workflow.

    Parameters
    ----------
    graph : Neo4jGraph
        The Neo4j graph wrapper. 

    Returns
    -------
    Callable[[CypherState], Dict[str, List[CypherOutputState] | List[str]]]
        The LangGraph node.
    """

    async def execute_cypher(
        state: CypherState,
    ) -> Dict[str, List[CypherOutputState] | List[str]]:
        """
        Executes the given Cypher statement.
        """
        
        # 清理cypher语句中的换行符
        cypher_statement = cypher["statement"].replace("\n", " ").strip()
        records = graph.query(cypher_statement)
        steps = state.get("steps", list())
        steps.append("execute_cypher")
        
        NO_CYPHER_RESULTS = [{"error": "在数据库中找不到任何相关信息。"}]
        
        return {
            "cyphers": [
                CypherOutputState(
                    **{
                        "task": state.get("task", []),
                        "statement": cypher_statement,
                        "parameters": None,
                        "errors": cypher["errors"],
                        "records": records if records !=[] else NO_CYPHER_RESULTS, 
                        "steps": steps,
                    }
                )
            ],
            "steps": ["text2cypher"],
        }

    return execute_cypher


# ==================== ReAct 核心函数 ====================

# 定义ReAct prompt
react_thought_prompt = create_react_thought_prompt_template()
react_generation_prompt = create_react_cypher_generation_prompt_template()
react_observation_prompt = create_react_observation_prompt_template()


def create_react_thought_node(
    llm: BaseChatModel,
    graph: Neo4jGraph,
) -> Callable[[ReActCypherState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    创建ReAct思考节点。

    Parameters
    ----------
    llm : BaseChatModel
        大语言模型
    graph : Neo4jGraph
        Neo4j图数据库连接

    Returns
    -------
    Callable[[ReActCypherState], Dict[str, Any]]
        思考节点函数
    """
    thought_chain = react_thought_prompt | llm.with_structured_output(ReActThoughtOutput)

    async def generate_thought(state: ReActCypherState) -> Dict[str, Any]:
        """生成思考过程"""
        task = state.get("task", "")
        attempts = state.get("attempts", 0)
        errors = state.get("errors", [])

        # 构建上下文信息
        context_parts = []
        if attempts > 0:
            context_parts.append(f"之前生成的Cypher: {state.get('cypher_statement', '')}")
            context_parts.append(f"执行结果: {state.get('execution_result', [])}")
            if errors:
                context_parts.append(f"错误信息: {errors}")
        context = "\n\n".join(context_parts) if context_parts else "首次尝试，无历史信息"

        try:
            result: ReActThoughtOutput = await thought_chain.ainvoke({
                "question": task,
                "schema": retrieve_and_parse_schema_from_graph_for_prompts(graph),
                "attempt": attempts + 1,
                "context": context,
            })

            return {
                "thought": result.thought,
                "should_retry": result.action in ["generate", "correct"],
                "retry_reason": result.reasoning,
                "steps": state.get("steps", []) + [f"thought_{attempts + 1}"],
            }
        except Exception as e:
            # 降级处理：主模型失败时切换 Ollama 本地模型重试一次
            logger.warning(f"[ReAct Thought] 主模型调用失败: {e}，尝试 Ollama 降级")
            try:
                fallback_llm = ChatOllama(
                    model=settings.OLLAMA_AGENT_MODEL,
                    base_url=settings.OLLAMA_BASE_URL,
                    temperature=0.7,
                )
                fallback_chain = react_thought_prompt | fallback_llm.with_structured_output(ReActThoughtOutput)
                result: ReActThoughtOutput = await fallback_chain.ainvoke({
                    "question": task,
                    "schema": retrieve_and_parse_schema_from_graph_for_prompts(graph),
                    "attempt": attempts + 1,
                    "context": context,
                })
                logger.info("[ReAct Thought] Ollama 降级成功")
                return {
                    "thought": result.thought,
                    "should_retry": result.action in ["generate", "correct"],
                    "retry_reason": result.reasoning,
                    "steps": state.get("steps", []) + [f"thought_{attempts + 1}_ollama_fallback"],
                }
            except Exception as fallback_e:
                logger.error(f"[ReAct Thought] Ollama 降级也失败: {fallback_e}，走兜底")
                return {
                    "thought": "生成思考时出错，继续生成Cypher",
                    "should_retry": True,
                    "retry_reason": "主模型和降级模型均失败",
                    "steps": state.get("steps", []) + [f"thought_{attempts + 1}_fallback"],
                }

    return generate_thought


def create_react_cypher_generation_node(
    llm: BaseChatModel,
    graph: Neo4jGraph,
    cypher_example_retriever: BaseCypherExampleRetriever,
) -> Callable[[ReActCypherState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    创建ReAct风格的Cypher生成节点。

    Parameters
    ----------
    llm : BaseChatModel
        大语言模型
    graph : Neo4jGraph
        Neo4j图数据库连接
    cypher_example_retriever : BaseCypherExampleRetriever
        Cypher示例检索器

    Returns
    -------
    Callable[[ReActCypherState], Dict[str, Any]]
        生成节点函数
    """
    generation_chain = react_generation_prompt | llm | StrOutputParser()

    async def generate_cypher(state: ReActCypherState) -> Dict[str, Any]:
        """生成Cypher查询"""
        task = state.get("task", "")
        thought = state.get("thought", "")
        attempts = state.get("attempts", 0)
        errors = state.get("errors", [])

        # 获取示例
        examples: str = cypher_example_retriever.get_examples(
            **{"query": task, "k": 3}
        )

        # 构建错误上下文
        error_context = ""
        if errors and attempts > 0:
            error_context = f"""之前的错误:
{chr(10).join(errors)}

请务必修正上述错误，重新生成正确的Cypher查询。"""

        try:
            cypher_statement = await generation_chain.ainvoke({
                "question": task,
                "schema": retrieve_and_parse_schema_from_graph_for_prompts(graph),
                "fewshot_examples": examples,
                "thought": thought,
                "error_context": error_context,
            })

            # 清理生成的Cypher
            cypher_statement = cypher_statement.strip()
            if cypher_statement.startswith("```"):
                cypher_statement = cypher_statement.replace("```cypher", "").replace("```", "").strip()

            return {
                "cypher_statement": cypher_statement,
                "attempts": attempts + 1,
                "steps": state.get("steps", []) + [f"generate_{attempts + 1}"],
            }
        except Exception as e:
            return {
                "cypher_statement": "",
                "errors": errors + [f"生成Cypher时出错: {str(e)}"],
                "attempts": attempts + 1,
                "steps": state.get("steps", []) + [f"generate_{attempts + 1}_error"],
            }

    return generate_cypher


def create_react_observation_node(
    llm: BaseChatModel,
    graph: Neo4jGraph,
) -> Callable[[ReActCypherState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    创建ReAct观察节点（执行 + 分析）。

    Observation阶段包含两个子步骤：
    1. 执行Cypher查询（Action的执行结果）
    2. 分析执行结果，决定是否继续

    Parameters
    ----------
    llm : BaseChatModel
        大语言模型
    graph : Neo4jGraph
        Neo4j图数据库连接

    Returns
    -------
    Callable[[ReActCypherState], Dict[str, Any]]
        观察节点函数
    """
    observation_chain = react_observation_prompt | llm.with_structured_output(ReActObservationOutput)

    async def observe(state: ReActCypherState) -> Dict[str, Any]:
        """
        执行Cypher并观察分析结果。

        返回包含：
        - execution_result: 执行结果
        - errors: 错误信息
        - is_complete: 是否完成
        - should_retry: 是否重试
        - retry_reason: 重试原因
        """
        task = state.get("task", "")
        cypher_statement = state.get("cypher_statement", "").strip()
        attempts = state.get("attempts", 0)
        steps = state.get("steps", [])

        # ========== Step 1: 执行Cypher ==========
        execution_result = []
        errors = []

        if not cypher_statement:
            errors.append("Cypher语句为空")
            steps.append(f"observe_{attempts}_empty_cypher")
        else:
            # 清理Cypher语句
            cypher_statement = cypher_statement.replace("\n", " ").strip()

            # 检查写操作
            write_errors = validate_no_writes_in_cypher_query(cypher_statement)
            if write_errors:
                errors.extend(write_errors)
                steps.append(f"observe_{attempts}_write_error")
            else:
                try:
                    # 执行查询
                    records = graph.query(cypher_statement)

                    # 处理空结果
                    if not records:
                        records = [{"error": "在数据库中找不到任何相关信息。"}]

                    execution_result = records
                    steps.append(f"observe_{attempts}_executed")
                    logger.info(f"[ReAct Observation] 执行成功: {len(records)} 条记录")

                except CypherSyntaxError as e:
                    errors.append(f"Cypher语法错误: {str(e.message)}")
                    steps.append(f"observe_{attempts}_syntax_error")
                    logger.warning(f"[ReAct Observation] 语法错误: {e.message}")
                except Exception as e:
                    errors.append(f"执行异常: {str(e)}")
                    steps.append(f"observe_{attempts}_exception")
                    logger.error(f"[ReAct Observation] 执行异常: {e}")

        # ========== Step 2: 分析结果 ==========

        # 如果有执行错误，直接判定需要重试
        if errors:
            return {
                "execution_result": execution_result,
                "errors": errors,
                "is_complete": False,
                "should_retry": True,
                "retry_reason": f"执行错误: {errors[-1]}",
                "steps": steps,
            }

        # 如果结果为空（标记为error），可能需要重试
        if not execution_result or (len(execution_result) == 1 and "error" in execution_result[0]):
            return {
                "execution_result": execution_result,
                "errors": errors,
                "is_complete": False,
                "should_retry": attempts < 3,
                "retry_reason": "查询结果为空，可能需要调整查询条件" if attempts < 3 else "已达到最大重试次数",
                "steps": steps,
            }

        # 使用LLM分析结果是否满足需求
        try:
            result_text = str(execution_result[:5])  # 只取前5条避免过长
        except:
            result_text = str(execution_result)

        try:
            observation: ReActObservationOutput = await observation_chain.ainvoke({
                "question": task,
                "cypher": cypher_statement,
                "result": result_text,
                "errors": "无错误",
            })

            logger.info(f"[ReAct Observation] LLM分析: is_satisfactory={observation.is_satisfactory}")

            return {
                "execution_result": execution_result,
                "errors": errors,
                "is_complete": observation.is_satisfactory,
                "should_retry": not observation.is_satisfactory and attempts < 3,
                "retry_reason": observation.suggestion if not observation.is_satisfactory else "",
                "steps": steps + [f"observe_{attempts}_analyzed"],
            }
        except Exception as e:
            # 降级处理：假设完成
            logger.warning(f"[ReAct Observation] LLM分析失败，降级处理: {e}")
            return {
                "execution_result": execution_result,
                "errors": errors,
                "is_complete": True,
                "should_retry": False,
                "retry_reason": f"结果分析出错: {str(e)}，假设结果可用",
                "steps": steps + [f"observe_{attempts}_fallback"],
            }

    return observe


async def run_react_cypher_loop(
    state: Dict[str, Any],
    llm: BaseChatModel,
    graph: Neo4jGraph,
    cypher_example_retriever: BaseCypherExampleRetriever,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """
    运行ReAct循环，直到完成或达到最大尝试次数。

    Parameters
    ----------
    state : Dict[str, Any]
        初始状态
    llm : BaseChatModel
        大语言模型
    graph : Neo4jGraph
        Neo4j图数据库连接
    cypher_example_retriever : BaseCypherExampleRetriever
        Cypher示例检索器
    max_attempts : int
        最大尝试次数

    Returns
    -------
    Dict[str, Any]
        最终结果
    """
    # 初始化ReAct状态
    react_state: ReActCypherState = {
        "task": state.get("task", ""),
        "thought": "",
        "cypher_statement": "",
        "execution_result": [],
        "errors": [],
        "attempts": 0,
        "is_complete": False,
        "should_retry": True,
        "retry_reason": "",
        "steps": [],
    }

    # 创建节点
    thought_node = create_react_thought_node(llm, graph)
    generation_node = create_react_cypher_generation_node(llm, graph, cypher_example_retriever)
    observation_node = create_react_observation_node(llm, graph)

    logger.info(f"[ReAct] 开始处理任务: {react_state['task'][:50]}...")

    while react_state["attempts"] < max_attempts and not react_state["is_complete"]:
        current_attempt = react_state["attempts"] + 1
        logger.info(f"[ReAct] 第 {current_attempt} 次尝试")

        # ========== 1. Thought: 思考 ==========
        logger.info(f"[ReAct] Step 1/3: 思考...")
        thought_result = await thought_node(react_state)
        react_state["thought"] = thought_result.get("thought", "")
        react_state["should_retry"] = thought_result.get("should_retry", True)
        react_state["steps"] = thought_result.get("steps", [])
        logger.info(f"[ReAct] 思考: {react_state['thought'][:100]}...")

        if not react_state["should_retry"]:
            logger.info("[ReAct] 思考决定结束")
            break

        # ========== 2. Action: 生成Cypher ==========
        logger.info(f"[ReAct] Step 2/3: 生成Cypher...")
        generation_result = await generation_node(react_state)
        react_state["cypher_statement"] = generation_result.get("cypher_statement", "")
        react_state["attempts"] = generation_result.get("attempts", react_state["attempts"])
        react_state["errors"] = generation_result.get("errors", [])
        react_state["steps"] = generation_result.get("steps", react_state["steps"])
        logger.info(f"[ReAct] 生成Cypher: {react_state['cypher_statement'][:100]}...")

        if react_state["errors"] and not react_state["cypher_statement"]:
            logger.warning(f"[ReAct] 生成失败: {react_state['errors']}")
            continue

        # ========== 3. Observation: 执行 + 分析（合并节点）==========
        logger.info(f"[ReAct] Step 3/3: 观察（执行+分析）...")
        observation_result = await observation_node(react_state)
        react_state["execution_result"] = observation_result.get("execution_result", [])
        react_state["errors"] = observation_result.get("errors", [])
        react_state["is_complete"] = observation_result.get("is_complete", False)
        react_state["should_retry"] = observation_result.get("should_retry", False)
        react_state["retry_reason"] = observation_result.get("retry_reason", "")
        react_state["steps"] = observation_result.get("steps", react_state["steps"])
        logger.info(f"[ReAct] 执行结果: {len(react_state['execution_result'])} 条记录")
        logger.info(f"[ReAct] 观察结论: complete={react_state['is_complete']}, retry={react_state['should_retry']}")

        if react_state["is_complete"]:
            logger.info("[ReAct] 任务完成")
            break

    # 返回最终结果
    if react_state["attempts"] >= max_attempts and not react_state["is_complete"]:
        logger.warning("[ReAct] 达到最大尝试次数，返回当前结果")
        react_state["errors"].append(f"达到最大尝试次数({max_attempts})，可能无法获得理想结果")

    return {
        "cyphers": [
            {
                "task": react_state["task"],
                "statement": react_state["cypher_statement"],
                "errors": react_state["errors"],
                "records": react_state["execution_result"],
                "steps": react_state["steps"],
            }
        ],
        "steps": react_state["steps"] + ["react_complete"],
        "attempts": react_state["attempts"],
        "is_complete": react_state["is_complete"],
    }
