# tpdiscovery/agent_graph.py
from langgraph.graph import StateGraph, END
from state import AgentState
from planner import PlannerAgent
from bridge import ResearchBridge
from integrator import ValidationAgent 

class DiscoveryGraph:
    def __init__(self):
        self.planner = PlannerAgent()
        self.bridge = ResearchBridge()
        self.validator = ValidationAgent()
        self.graph = self.build_graph()

    def build_graph(self):
        workflow = StateGraph(AgentState)
        
        # 定义节点
        workflow.add_node("Planner", self.node_plan)
        workflow.add_node("Executor", self.node_execute)
        workflow.add_node("Synthesizer", self.node_synthesize)
        
        # 定义流程
        workflow.set_entry_point("Planner")
        workflow.add_edge("Planner", "Executor")
        workflow.add_edge("Executor", "Synthesizer")
        workflow.add_edge("Synthesizer", END)
        
        return workflow.compile()

    # --- Node 实现 ---

    def node_plan(self, state: AgentState):
        steps = self.planner.plan(state["user_input"])
        return {"plan": steps, "execution_trace": []}

    def node_execute(self, state: AgentState):
        """
        执行数据收集步骤（不包括文献检索，文献检索移到 Synthesizer 中）
        """
        steps = state["plan"]
        trace = []
        
        print(f"⚙️ [Executor] 开始执行 {len(steps)} 个步骤...")
        
        for step in steps:
            skill = step["skill"]
            args = step["args"].copy()
            
            # 跳过文献检索，稍后在 Synthesizer 中处理
            if skill == "check_literature":
                trace.append({"skill": skill, "args": args, "deferred": True})
                print(f"   ⏳ 延迟执行: {skill} (等待合并筛选后)")
                continue
            
            # 调用 Bridge 执行
            result = self.bridge.call_skill(skill, args)
            
            if result["status"] == "success":
                data = result["data"]
                trace.append({"skill": skill, "data": data})
        
        return {"execution_trace": trace}

    def node_synthesize(self, state: AgentState):
        """
        整合两条路径结果 -> 筛选 Top 20 -> 文献检索 -> 最终排名
        """
        print("📝 [Synthesizer] 整合结果...")
        trace = state["execution_trace"]
        
        # 1. 分类结果
        res_a = {}  # Omics (Bottom-Up)
        res_b = {}  # KG (Top-Down)
        ot_data = {}
        lit_args = None  # 保存文献检索参数
        
        for item in trace:
            skill = item["skill"]
            
            if item.get("deferred"):
                # 保存延迟执行的参数
                if skill == "check_literature":
                    lit_args = item.get("args", {})
                continue
                
            data = item["data"]
            
            if skill == "run_omics_path": 
                res_a = data
            elif skill == "run_kg_path": 
                res_b = data
            elif skill == "run_validation": 
                res_a.update(data)
            elif skill == "check_external": 
                if "results" in data:
                    for r in data["results"]:
                        ot_data[r["symbol"]] = r["score"]

        # 2. 合并两条路径并打分
        print("🔀 [Synthesizer] 合并两条路径结果...")
        final_candidates = self.validator.validate_and_rank(res_b, res_a)
        
        # 3. 筛选 Top 20 进行文献检索
        top_20 = final_candidates[:20]
        top_20_genes = [c["Gene"] for c in top_20]
        print(f"🎯 [Synthesizer] 筛选 Top {len(top_20_genes)} 基因进行文献验证: {top_20_genes[:5]}...")
        
        # 4. 执行文献检索
        if top_20_genes:
            lit_result = self.bridge.call_skill("check_literature", {
                "genes": top_20_genes,
                "mode": "discovery"  # 默认发现模式
            })
            
            if lit_result["status"] == "success":
                lit_data = lit_result["data"]
                self.validator.attach_literature_evidence(top_20, lit_data)
        
        # 5. 附加 OpenTargets 证据
        for cand in top_20:
            g = cand["Gene"]
            if g in ot_data:
                cand["Score"] += 1.0
                cand["External_DB_Score"] = ot_data[g]
        
        # 6. 重新排序
        top_20.sort(key=lambda x: x["Score"], reverse=True)
                
        return {"final_candidates": top_20}