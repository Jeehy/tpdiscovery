# tpdiscovery/state.py
from typing import TypedDict, List, Dict, Any, Annotated
import operator

class AgentState(TypedDict):
    """
    Agent 协作共享状态
    """
    # 原始输入
    user_input: str
    
    # 规划结果 (Planner -> Executor)
    # 结构: [{"skill": "path_a", "args": {...}}, ...]
    plan: List[Dict[str, Any]]
    
    # 执行记录 (Executor -> Synthesizer)
    # 使用 operator.add 实现增量更新，避免覆盖
    execution_trace: Annotated[List[Dict[str, Any]], operator.add]
    
    # 最终产出 (Synthesizer -> Output)
    final_candidates: List[Dict[str, Any]]