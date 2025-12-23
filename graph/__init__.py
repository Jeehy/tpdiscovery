# graph/__init__.py
"""
LangGraph 风格的工作流模块

使用方式:
    from graph import run_discovery, compile_graph
    
    # 方式1: 快捷调用
    result = run_discovery("发现肝癌新靶点")
    
    # 方式2: 手动控制
    app = compile_graph()
    state = app.invoke({"user_input": "发现肝癌新靶点", "disease": "liver cancer"})
"""

from graph.state import GraphState, create_initial_state
from graph.workflow import create_workflow, compile_graph, run_discovery

__all__ = [
    "GraphState",
    "create_initial_state", 
    "create_workflow",
    "compile_graph",
    "run_discovery"
]
