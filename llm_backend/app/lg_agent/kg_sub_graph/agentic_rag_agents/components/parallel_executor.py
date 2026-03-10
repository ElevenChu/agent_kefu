"""
并行执行器节点，用于并发执行多个任务

这个节点接收任务列表，并并发地调用 tool_selection 节点
"""
import asyncio
from typing import Any, Callable, Coroutine, Dict, List
from app.core.logger import get_logger

logger = get_logger(service="parallel_executor")

async def parallel_executor_node(
        state: Dict[str: Any],
        tool_selection_func: Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]],
        max_concurrent: int = 3
) ->Dict[str: Any]:
    tasks = state.get("tasks",[])
    if not tasks:
        logger.warning("No tasks to execute")
        return {"cyphers": [], "steps": ["parallel_executor: no tasks"]}
    # 使用信号量控制并发数
    semphore = asyncio.Semaphore(max_concurrent)

    async def execute_single_task(task: Dict[str: Any], task_index: int) -> Dict[str: Any]:
         
         
         """执行单个任务"""
         task_question = task.get("question", "")
         task_parent = task.get("parent_task", "")
         start_time = asyncio.get_event_loop().time()

         try:
              async with semphore:
                # 调用 tool_selection 逻辑
                task_state = {
                    "question": task_question,
                    "parent_task": task_parent,
                }
                
                result = await tool_selection_func(task_state)
                
                execution_time = asyncio.get_event_loop().time() - start_time
                logger.info(f"[Task {task_index}] Completed in {execution_time:.2f}s")
                
                return {
                    "task_index": task_index,
                    "question": task_question,
                    "result": result,
                    "success": True,
                    "error": None,
                }
                
         except Exception as e:
            
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"[Task {task_index}] Failed after {execution_time:.2f}s: {str(e)}")
            
            return {
                "task_index": task_index,
                "question": task_question,
                "result": None,
                "success": False,
                "error": str(e),
            }
          # 并发执行所有任务
    task_coroutines = [
        execute_single_task(task, i)
        for i, task in enumerate(tasks)
    ]
    
    # 使用 asyncio.gather 并发执行，return_exceptions=True 确保一个失败不影响其他
    results = await asyncio.gather(
        *task_coroutines,
        return_exceptions=True
    )
    
    # 统计执行结果
    success_count = sum(1 for r in results if r.get("success"))
    failure_count = len(results) - success_count
    
    logger.info(f"Parallel execution completed: {success_count} success, {failure_count} failures")
    
    # 汇总结果到状态
    cyphers = []
    for result in results:
        if result.get("success") and result.get("result"):
            cyphers.extend(result["result"].get("cyphers", []))
    
    return {
        "cyphers": cyphers,
        "steps": [
            f"parallel_executor: executed {len(tasks)} tasks, {success_count} success"
        ]
    }
def create_parallel_executor(
    tool_selection_func: Callable,
    max_concurrent: int = 3,
) -> Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    创建并行执行器节点
    
    Parameters
    ----------
    tool_selection_func : Callable
        工具选择函数
    max_concurrent : int
        最大并发数
    
    Returns
    -------
    Callable
        可用于 LangGraph 的节点函数
    """
    
    async def parallel_executor(state: Dict[str, Any]) -> Dict[str, Any]:
        return await parallel_executor_node(
            state=state,
            tool_selection_func=tool_selection_func,
            max_concurrent=max_concurrent,
        )
    
    return parallel_executor

         