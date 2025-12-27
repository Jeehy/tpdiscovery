# graph/workflow.py
"""
LangGraph 工作流定义
使用标准的 StateGraph 构建方式
"""
from typing import Literal

from langgraph.graph import StateGraph, START, END

from graph.state import GraphState, create_initial_state
from graph.nodes import (
    planner_node,
    omics_node,
    kg_node,
    opentargets_node,
    merge_node,
    literature_node,
    report_node
)


def should_run_paths(state: GraphState) -> Literal["run_paths", "skip_to_merge"]:
    """
    条件路由：判断是否需要执行数据收集路径
    """
    steps = state.get("plan_steps", [])
    
    has_data_steps = any(
        s.get("skill") in ["run_omics_path", "run_kg_path", "run_validation"]
        for s in steps
    )
    
    return "run_paths" if has_data_steps else "skip_to_merge"


def create_workflow() -> StateGraph:
    """
    创建 LangGraph 工作流
    
    流程图:
    
        START
          │
          ▼
      ┌─────────┐
      │ Planner │
      └────┬────┘
           │
           ▼ (条件路由)
      ┌────┴────┐
      │         │
      ▼         ▼
    ┌─────┐  ┌─────┐  ┌──────────────┐
    │Omics│  │ KG  │  │ OpenTargets  │  (并行)
    └──┬──┘  └──┬──┘  └──────┬───────┘
       │        │            │
       └────────┼────────────┘
                │
                ▼
          ┌───────────┐
          │   Merge   │
          └─────┬─────┘
                │
                ▼
          ┌───────────┐
          │ Literature│
          └─────┬─────┘
                │
                ▼
          ┌───────────┐
          │  Report   │
          └─────┬─────┘
                │
                ▼
               END
    """
    # 创建图
    workflow = StateGraph(GraphState)
    
    # === 添加节点 ===
    workflow.add_node("planner", planner_node)
    workflow.add_node("omics", omics_node)
    workflow.add_node("kg", kg_node)
    workflow.add_node("opentargets", opentargets_node)
    workflow.add_node("merge", merge_node)
    workflow.add_node("literature", literature_node)
    workflow.add_node("report", report_node)
    
    # === 添加边 ===
    
    # 入口 -> Planner
    workflow.add_edge(START, "planner")
    
    # Planner -> 并行执行三个数据收集节点
    # 使用 fan-out 模式
    workflow.add_edge("planner", "omics")
    workflow.add_edge("planner", "kg")
    workflow.add_edge("planner", "opentargets")
    
    # 三个节点汇聚到 Merge (fan-in)
    workflow.add_edge("omics", "merge")
    workflow.add_edge("kg", "merge")
    workflow.add_edge("opentargets", "merge")
    
    # Merge -> Literature -> Report -> END
    workflow.add_edge("merge", "literature")
    workflow.add_edge("literature", "report")
    workflow.add_edge("report", END)
    
    return workflow


def compile_graph():
    """
    编译工作流
    
    可选添加:
    - checkpointer: 支持断点恢复
    - interrupt_before/after: 支持人工审核
    """
    workflow = create_workflow()
    
    # 基础编译
    app = workflow.compile()
    
    # 如需添加断点恢复功能，可使用 MemorySaver:
    # from langgraph.checkpoint.memory import MemorySaver
    # checkpointer = MemorySaver()
    # app = workflow.compile(checkpointer=checkpointer)
    
    return app


# === 便捷入口 ===
def run_discovery(user_input: str, disease: str = "liver cancer"):
    """
    运行发现/验证流程
    """
    app = compile_graph()
    initial_state = create_initial_state(user_input, disease)
    
    # 执行并返回最终状态
    final_state = app.invoke(initial_state)
    
    return final_state
