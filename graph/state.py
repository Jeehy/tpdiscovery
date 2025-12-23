# graph/state.py
"""
LangGraph 标准状态定义
使用 TypedDict + Annotated + reducer 实现状态累积
"""
from typing import TypedDict, List, Dict, Any, Annotated, Literal, Optional
from typing_extensions import Required
import operator


def merge_dict(left: Dict, right: Dict) -> Dict:
    """字典合并 reducer"""
    if not left:
        return right
    if not right:
        return left
    merged = left.copy()
    merged.update(right)
    return merged


class OmicsData(TypedDict, total=False):
    """组学数据结构"""
    log2fc: float
    padj: float
    spearman_r: float
    p_correlation: float
    drug_source: str
    ai_summary: str


class CandidateEvidence(TypedDict, total=False):
    """候选靶点证据结构"""
    gene: str
    sources: List[str]
    scores: Dict[str, float]
    kg_hypothesis: str
    kg_raw_facts: List[str]
    omics_data: Optional[OmicsData]
    literature_evidence: Optional[Dict]
    opentargets_score: float


class GraphState(TypedDict, total=False):
    """
    LangGraph 状态定义
    
    使用 Annotated 定义 reducer:
    - operator.add: 列表累积
    - merge_dict: 字典合并
    """
    # === 输入 ===
    user_input: Required[str]
    disease: str
    mode: Literal["discovery", "validation"]
    
    # === Planner 输出 ===
    plan_steps: List[Dict[str, Any]]
    
    # === 路径执行结果 (使用 reducer 累积) ===
    omics_results: Annotated[Dict[str, Any], merge_dict]
    kg_results: Annotated[Dict[str, Any], merge_dict]
    opentargets_results: Annotated[Dict[str, float], merge_dict]
    
    # === 合并后的候选池 ===
    merged_candidates: Dict[str, CandidateEvidence]
    top_candidates: List[str]  # Top N 基因名
    
    # === 文献验证结果 ===
    literature_results: Annotated[Dict[str, Any], merge_dict]
    
    # === 最终输出 ===
    final_report: List[Dict[str, Any]]
    
    # === 错误追踪 ===
    errors: Annotated[List[str], operator.add]


# === 状态初始化辅助函数 ===
def create_initial_state(user_input: str, disease: str = "liver cancer") -> GraphState:
    """创建初始状态"""
    mode = "validation" if "验证" in user_input else "discovery"
    return {
        "user_input": user_input,
        "disease": disease,
        "mode": mode,
        "plan_steps": [],
        "omics_results": {},
        "kg_results": {},
        "opentargets_results": {},
        "merged_candidates": {},
        "top_candidates": [],
        "literature_results": {},
        "final_report": [],
        "errors": []
    }
