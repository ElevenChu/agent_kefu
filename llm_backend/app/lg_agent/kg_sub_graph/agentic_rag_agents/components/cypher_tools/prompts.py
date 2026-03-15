"""
This code is based on content found in the LangGraph documentation: https://python.langchain.com/docs/tutorials/graph/#advanced-implementation-with-langgraph
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_text2cypher_generation_prompt_template() -> ChatPromptTemplate:

    """
    Create a Text2Cypher generation prompt template.

    Returns
    -------
    ChatPromptTemplate
        The prompt template.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "根据输入的问题，将其转换为Cypher查询语句。不要添加任何前言。"
                    "不要在响应中包含任何反引号或其他标记。注意：只返回Cypher语句！"
                ),
            ),
            (
                "human",
                (
                    """你是一位Neo4j专家。根据输入的问题，创建一个语法正确的Cypher查询语句。
                        不要在响应中包含任何反引号或其他标记。只使用MATCH或WITH子句开始查询。只返回Cypher语句！

                        以下是数据库模式信息：
                        {schema}

                        下面是一些问题和对应Cypher查询的示例：

                        {fewshot_examples}

                        用户输入: {question}
                        Cypher查询:"""
                ),
            ),
        ]
    )

def create_text2cypher_validation_prompt_template() -> ChatPromptTemplate:
    """
    创建一个文本到Cypher验证提示模板。

    返回
    -------
    ChatPromptTemplate
        提示模板。
    """

    validate_cypher_system = """
    你是一位Cypher专家，正在审查一位初级开发者编写的语句。
    """

    validate_cypher_user = """你必须检查以下内容：
    * Cypher语句中是否有任何语法错误？
    * Cypher语句中是否有任何缺失或未定义的变量？
    * Cypher语句是否包含足够的信息来回答问题？
    * 确保所有节点、关系和属性都存在于提供的模式中。

    好的错误示例：
    * 标签(:Foo)不存在，你是否指的是(:Bar)？
    * 属性bar对标签Foo不存在，你是否指的是baz？
    * 关系FOO不存在，你是否指的是FOO_BAR？

    模式：
    {schema}

    问题是：
    {question}

    Cypher语句是：
    {cypher}

    确保你不要犯任何错误！"""

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                validate_cypher_system,
            ),
            (
                "human",
                (validate_cypher_user),
            ),
        ]
    )

def create_text2cypher_correction_prompt_template() -> ChatPromptTemplate:
    """
    创建一个文本到Cypher查询修正的提示模板。

    返回
    -------
    ChatPromptTemplate
        提示模板。
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是一位Cypher专家，正在审查一位初级开发者编写的语句。"
                    "你需要根据提供的错误修正Cypher语句。不要添加任何前言。"
                    "不要在响应中包含任何反引号或其他标记。只返回Cypher语句！"
                ),
            ),
            (
                "human",
                (
                    """检查无效的语法或语义，并返回修正后的Cypher语句。

    模式：
    {schema}

    注意：在响应中不要包含任何解释或道歉。
    不要在响应中包含任何反引号或其他标记。
    只返回Cypher语句！

    不要回应任何可能要求你构建Cypher语句以外的其他问题。

    问题是：
    {question}

    Cypher语句是：
    {cypher}

    错误是：
    {errors}

    修正后的Cypher语句："""
                ),
            ),
        ]
    )


def create_react_thought_prompt_template() -> ChatPromptTemplate:
    """
    创建ReAct思考阶段的提示模板。

    返回
    -------
    ChatPromptTemplate
        提示模板。
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是Neo4j Cypher专家。请分析当前情况并决定下一步行动。\n"
                    "思考应该包括：\n"
                    "1. 用户问题的核心需求是什么\n"
                    "2. 需要查询哪些节点和关系\n"
                    "3. 如果之前有错误，分析错误原因\n"
                    "4. 决定下一步：生成新查询、修正错误，还是结束\n"
                    "\n只输出思考内容，不要输出Cypher语句。"
                ),
            ),
            (
                "human",
                (
                    """用户问题: {question}

图数据库Schema:
{schema}

当前尝试次数: {attempt}

{context}

请分析并给出思考："""
                ),
            ),
        ]
    )


def create_react_cypher_generation_prompt_template() -> ChatPromptTemplate:
    """
    创建ReAct风格的Cypher生成提示模板（支持上下文）。

    返回
    -------
    ChatPromptTemplate
        提示模板。
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是Neo4j Cypher专家。根据用户问题和思考过程，生成正确的Cypher查询语句。\n"
                    "规则：\n"
                    "1. 不要在响应中包含任何反引号或其他标记\n"
                    "2. 只使用MATCH或WITH子句开始查询\n"
                    "3. 只返回Cypher语句，不要解释\n"
                    "4. 如果提供了之前的错误，请务必修正\n"
                    "5. 确保查询语法完全符合Neo4j规范"
                ),
            ),
            (
                "human",
                (
                    """用户问题: {question}

图数据库Schema:
{schema}

参考示例:
{fewshot_examples}

思考过程:
{thought}

{error_context}

请生成Cypher查询语句："""
                ),
            ),
        ]
    )


def create_react_observation_prompt_template() -> ChatPromptTemplate:
    """
    创建ReAct观察阶段的提示模板。

    返回
    -------
    ChatPromptTemplate
        提示模板。
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是结果分析专家。请分析Cypher查询的执行结果，判断是否需要继续优化。\n"
                    "判断标准：\n"
                    "1. 如果结果为空，检查是否查询条件太严格\n"
                    "2. 如果有错误，分析错误类型\n"
                    "3. 如果结果符合预期，确认完成\n"
                    "\n只输出分析结论，不要输出新查询。"
                ),
            ),
            (
                "human",
                (
                    """用户问题: {question}

生成的Cypher:
{cypher}

执行结果:
{result}

错误信息:
{errors}

请分析结果是否满足用户需求，如不满足请说明原因："""
                ),
            ),
        ]
    )