"""
ReAct模式Cypher查询使用示例

这个文件展示了如何使用新的ReAct模式来执行Cypher查询。
ReAct模式通过 Thought -> Action -> Observation 的循环，
可以自动修正错误的Cypher查询，提高查询成功率。
"""

import asyncio
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.cypher_tools.node import create_cypher_query_node


async def react_query_example():
    """
    使用ReAct模式查询的示例
    """
    # 创建cypher查询节点（默认使用ReAct模式）
    cypher_query_node = create_cypher_query_node(
        use_react=True,      # 启用ReAct模式
        max_attempts=3       # 最多尝试3次
    )

    # 示例1: 简单的商品查询
    state = {
        "task": "查找库存少于10件的商品",
        "steps": []
    }

    result = await cypher_query_node(state)

    print("=" * 60)
    print("ReAct查询结果")
    print("=" * 60)

    cypher_result = result["cyphers"][0]
    print(f"原始任务: {cypher_result['task']}")
    print(f"执行步骤: {cypher_result['steps']}")
    print(f"生成的Cypher: {cypher_result['statement']}")
    print(f"查询结果: {cypher_result['records']}")
    print(f"错误信息: {cypher_result['errors']}")

    return result


async def react_query_with_error_recovery():
    """
    展示ReAct模式如何自动修正错误的示例
    """
    print("\n" + "=" * 60)
    print("ReAct自动纠错示例")
    print("=" * 60)

    # 创建一个容易出错的查询场景
    cypher_query_node = create_cypher_query_node(
        use_react=True,
        max_attempts=3
    )

    # 这个查询可能需要多次尝试才能生成正确的Cypher
    state = {
        "task": "找出每个类别中价格最高的产品，并显示类别名称和产品名称",
        "steps": []
    }

    result = await cypher_query_node(state)
    cypher_result = result["cyphers"][0]

    print(f"任务: {cypher_result['task']}")
    print(f"尝试步骤: {cypher_result['steps']}")
    print(f"最终Cypher:\n{cypher_result['statement']}")

    # 检查是否有重试
    steps = cypher_result['steps']
    if any('thought_' in str(s) for s in steps):
        print("\n✅ ReAct模式生效：检测到思考和重试过程")
    else:
        print("\nℹ️ 一次成功，无需重试")

    return result


async def compare_modes():
    """
    对比ReAct模式和传统模式
    """
    print("\n" + "=" * 60)
    print("ReAct模式 vs 传统模式 对比")
    print("=" * 60)

    # ReAct模式
    react_node = create_cypher_query_node(use_react=True, max_attempts=3)

    # 传统模式
    traditional_node = create_cypher_query_node(use_react=False)

    test_cases = [
        "查找所有智能灯具类商品",
        "统计每个供应商提供的商品数量",
        "找出价格在1000到2000元之间的智能家居产品",
    ]

    for task in test_cases:
        print(f"\n测试任务: {task}")

        # ReAct模式
        react_result = await react_node({"task": task, "steps": []})
        react_steps = len(react_result["cyphers"][0]["steps"])

        print(f"  ReAct模式: {react_steps} 个步骤")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(react_query_example())
    # asyncio.run(react_query_with_error_recovery())
    # asyncio.run(compare_modes())
