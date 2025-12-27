# graph/nodes.py
"""
LangGraph 节点定义
每个节点是纯函数: 接收 state -> 返回 partial update

使用单例模式避免重复初始化工具类
"""
from typing import Dict, Any
from graph.state import GraphState
from planner import PlannerAgent
from explorers import ResearchExplorers
from integrator import ValidationAgent
from tools.omics.opentargets_tool import OpenTargetsTool
from tools.literature.literature_tool import LiteratureTool


# ============================
# 单例工具实例 (避免重复初始化)
# ============================
_explorer: ResearchExplorers = None
_lit_tool: LiteratureTool = None
_ot_tool: OpenTargetsTool = None
_validator: ValidationAgent = None


def get_explorer() -> ResearchExplorers:
    """获取或创建 ResearchExplorers 单例"""
    global _explorer
    if _explorer is None:
        print("🔧 [Init] 初始化数据探索器...")
        _explorer = ResearchExplorers()
    return _explorer


def get_lit_tool() -> LiteratureTool:
    """获取或创建 LiteratureTool 单例"""
    global _lit_tool
    if _lit_tool is None:
        _lit_tool = LiteratureTool()
    return _lit_tool


def get_ot_tool() -> OpenTargetsTool:
    """获取或创建 OpenTargetsTool 单例"""
    global _ot_tool
    if _ot_tool is None:
        _ot_tool = OpenTargetsTool()
    return _ot_tool


def get_validator() -> ValidationAgent:
    """获取或创建 ValidationAgent 单例"""
    global _validator
    if _validator is None:
        _validator = ValidationAgent()
    return _validator


# ============================
# Planner Node
# ============================
def planner_node(state: GraphState) -> Dict[str, Any]:
    """
    规划节点：解析用户意图，生成执行计划
    """
    planner = PlannerAgent()
    steps = planner.plan(state["user_input"])
    
    return {"plan_steps": steps}


# ============================
# Executor Nodes (并行执行)
# ============================
def omics_node(state: GraphState) -> Dict[str, Any]:
    """
    组学路径节点 (Path A: Bottom-Up)
    """
    # 检查计划中是否包含此步骤
    has_omics_step = any(
        s.get("skill") in ["run_omics_path", "run_validation"] 
        for s in state.get("plan_steps", [])
    )
    
    if not has_omics_step:
        return {"omics_results": {}}
    
    explorer = get_explorer()
    
    # 获取参数
    step = next(
        (s for s in state["plan_steps"] if s["skill"] in ["run_omics_path", "run_validation"]),
        {"args": {}}
    )
    args = step.get("args", {})
    
    try:
        if step["skill"] == "run_validation":
            genes = args.get("genes", [])
            if isinstance(genes, str):
                genes = [genes]
            results = explorer.run_validation_deep_diven(
                target_list=genes,
                disease=state.get("disease", "liver cancer")
            )
        else:
            results = explorer.run_path_omics_driven(
                threshold=args.get("threshold", 6.0),
                disease=state.get("disease", "liver cancer")
            )
        return {"omics_results": results}
    except Exception as e:
        return {"omics_results": {}, "errors": [f"Omics error: {str(e)}"]}


def kg_node(state: GraphState) -> Dict[str, Any]:
    """
    知识图谱路径节点 (Path B: Top-Down)
    """
    has_kg_step = any(
        s.get("skill") == "run_kg_path" 
        for s in state.get("plan_steps", [])
    )
    
    if not has_kg_step:
        return {"kg_results": {}}
    
    explorer = get_explorer()
    
    try:
        results = explorer.run_path_kg_driven(
            disease=state.get("disease", "liver cancer")
        )
        return {"kg_results": results}
    except Exception as e:
        return {"kg_results": {}, "errors": [f"KG error: {str(e)}"]}


def opentargets_node(state: GraphState) -> Dict[str, Any]:
    """
    OpenTargets 外部验证节点
    
    注意：此节点仅在 Planner 显式请求 check_external 且提供有效基因列表时执行
    大部分情况下 OpenTargets 验证已在 Path A/B 内部完成
    """
    has_ot_step = any(
        s.get("skill") == "check_external" 
        for s in state.get("plan_steps", [])
    )
    
    if not has_ot_step:
        return {"opentargets_results": {}}
    
    step = next(
        (s for s in state["plan_steps"] if s["skill"] == "check_external"),
        {"args": {}}
    )
    args = step.get("args", {})
    
    # 获取基因列表
    genes = args.get("genes", [])
    
    # 处理 "<auto>" 占位符 - 这种情况下跳过，等待 merge 后再验证
    if not genes or genes == "<auto>" or (isinstance(genes, list) and "<auto>" in genes):
        print("   ⏳ [OpenTargets] 等待候选池确定后再验证...")
        return {"opentargets_results": {}}
    
    # 确保是列表
    if isinstance(genes, str):
        genes = [genes]
    
    ot_tool = get_ot_tool()
    
    try:
        result = ot_tool.run({"genes": genes, "topic": state.get("disease", "liver cancer")})
        ot_scores = {}
        if "results" in result:
            for r in result["results"]:
                ot_scores[r["symbol"]] = r["score"]
        return {"opentargets_results": ot_scores}
    except Exception as e:
        return {"opentargets_results": {}, "errors": [f"OpenTargets error: {str(e)}"]}


# ============================
# Merge Node
# ============================
def merge_node(state: GraphState) -> Dict[str, Any]:
    """
    合并节点：整合两条路径结果，生成候选池
    """
    print("🔀 [Merge] 合并两条路径结果...")
    
    validator = get_validator()
    
    # 合并并排名
    ranked = validator.validate_and_rank(
        top_down_results=state.get("kg_results", {}),
        bottom_up_results=state.get("omics_results", {}),
        disease=state.get("disease", "liver cancer")
    )
    
    # 提取 Top 20 基因
    top_20 = ranked[:20]
    top_genes = [c["Gene"] for c in top_20]
    
    # 构建候选字典
    merged = {c["Gene"]: c for c in ranked}
    
    print(f"   筛选 Top {len(top_genes)} 基因: {top_genes[:5]}...")
    
    return {
        "merged_candidates": merged,
        "top_candidates": top_genes
    }


# ============================
# Literature Node
# ============================
def literature_node(state: GraphState) -> Dict[str, Any]:
    """
    文献验证节点：对 Top N 候选进行文献检索
    """
    top_genes = state.get("top_candidates", [])
    if not top_genes:
        return {"literature_results": {}}
    
    print(f"📖 [Literature] 验证 {len(top_genes)} 个候选基因...")
    
    lit_tool = get_lit_tool()
    
    try:
        results = lit_tool.run_batch_verification(
            gene_list=top_genes,
            disease=state.get("disease", "liver cancer"),
            mode=state.get("mode", "discovery")
        )
        return {"literature_results": results}
    except Exception as e:
        return {"literature_results": {}, "errors": [f"Literature error: {str(e)}"]}


# ============================
# Report Node
# ============================
def report_node(state: GraphState) -> Dict[str, Any]:
    """
    报告生成节点：整合所有证据，生成最终报告
    """
    print("📝 [Report] 生成最终报告...")
    
    merged = state.get("merged_candidates", {})
    top_genes = state.get("top_candidates", [])
    lit_results = state.get("literature_results", {})
    ot_scores = state.get("opentargets_results", {})
    
    # 获取 Top N 的候选对象
    top_candidates = [merged[g] for g in top_genes if g in merged]
    
    # 附加文献证据
    validator = get_validator()
    validator.attach_literature_evidence(top_candidates, lit_results)
    
    # 附加 OpenTargets 分数
    for cand in top_candidates:
        gene = cand["Gene"]
        if gene in ot_scores:
            cand["Score"] += 1.0
            cand["External_DB_Score"] = ot_scores[gene]
    
    # 重新排序
    top_candidates.sort(key=lambda x: x["Score"], reverse=True)
    
    return {"final_report": top_candidates}
