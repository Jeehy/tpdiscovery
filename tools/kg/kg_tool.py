from py2neo import Graph
import os
import sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompts import KG_DISCOVERY_ANALYSIS, KG_VALIDATION_ANALYSIS
from deepseek_api import call_llm

class KGTool:
    """
    知识图谱工具 (Logic Fixed Version)
    
    1. Validation 模式下不再过滤已知基因，确保能查到 TP53 等核心基因的证据。
    2. Discovery 模式增加 Hub Gene 黑名单 (如 UBC)，减少噪音。
    """

    def __init__(self):
        self.uri = "bolt://neo4j.het.io:7687"
        self.user = "neo4j"
        self.password = "neo4j"
        self.graph = None
        # 定义通用噪音基因 (泛素、核糖体蛋白等高连接度但低特异性的基因)
        self.BLACKLIST = {'UBC', 'UBB', 'RPS27A', 'UBA52'} 
        self._connect()
    
    def _connect(self):
        try:
            self.graph = Graph(self.uri, auth=(self.user, self.password))
            print("KGTool: 已连接到 Hetionet")
        except Exception as e:
            print(f"KGTool: 连接失败 - {e}")

    # ============================================================
    #  底层通用检索 (逻辑已修复)
    # ============================================================

    def _query_ppi(self, disease, candidate_genes=None, exclude_known=True, limit=20):
        """
        exclude_known: Discovery模式为True(找新的), Validation模式为False(查旧的)
        """
        cypher = """
        MATCH (d:Disease)-[:ASSOCIATES_DaG]-(seed:Gene)-[:INTERACTS_GiG]-(candidate:Gene)
        WHERE toLower(d.name) = toLower($disease)
        """
        
        # 修复逻辑：只有在显式要求排除已知基因时(Discovery)，才加这个过滤条件
        if exclude_known:
            cypher += " AND NOT (d)-[:ASSOCIATES_DaG]-(candidate) "
        
        params = {"disease": disease, "limit": limit}

        if candidate_genes:
            cypher += " AND candidate.name IN $genes "
            params["genes"] = candidate_genes
            params["limit"] = 1000 
        
        cypher += """
        RETURN candidate.name AS gene, 
               count(DISTINCT seed) AS count, 
               collect(DISTINCT seed.name)[0..5] AS evidence
        ORDER BY count DESC LIMIT $limit
        """
        return self.graph.run(cypher, **params).data()

    def _query_pathway(self, disease, candidate_genes=None, exclude_known=True, limit=20):
        cypher = """
        MATCH (d:Disease)-[:ASSOCIATES_DaG]-(seed:Gene)-[:PARTICIPATES_GpPW]->(p:Pathway)<-[:PARTICIPATES_GpPW]-(candidate:Gene)
        WHERE toLower(d.name) = toLower($disease)
        """
        
        if exclude_known:
            cypher += " AND NOT (d)-[:ASSOCIATES_DaG]-(candidate) "
            
        params = {"disease": disease, "limit": limit}

        if candidate_genes:
            cypher += " AND candidate.name IN $genes "
            params["genes"] = candidate_genes
            params["limit"] = 1000

        cypher += """
        RETURN candidate.name AS gene, 
               count(DISTINCT p) AS count, 
               collect(DISTINCT p.name)[0..3] AS evidence
        ORDER BY count DESC LIMIT $limit
        """
        return self.graph.run(cypher, **params).data()


    # === 新增辅助函数：严格检查是否为已知靶点 ===
    def _check_is_known_batch(self, disease, gene_list):
        if not gene_list: return set()
        cypher = """
        MATCH (d:Disease)-[:ASSOCIATES_DaG]-(g:Gene)
        WHERE toLower(d.name) = toLower($disease) AND g.name IN $genes
        RETURN g.name as gene
        """
        res = self.graph.run(cypher, disease=disease, genes=gene_list).data()
        return {r['gene'] for r in res}

    # ==========================================
    #  业务逻辑：Discovery (返回结构化列表)
    # ==========================================
    def _run_discovery_struct(self, disease):
        print(f"KGTool [Discovery]: Mining for {disease}...")
        ppi = self._query_ppi(disease, exclude_known=True) # 这里已经排除了已知
        pw = self._query_pathway(disease, exclude_known=True)
        
        candidates = {}
        # 保存更详细的 Raw Facts 用于后续展示
        raw_facts_map = {} 

        for r in ppi:
            if r['gene'] in self.BLACKLIST: continue
            fact = f"Network: Interacts with {r['count']} known genes (e.g., {','.join(r['evidence'])})."
            candidates.setdefault(r['gene'], []).append(fact)
            raw_facts_map.setdefault(r['gene'], []).append(fact)

        for r in pw:
            if r['gene'] in self.BLACKLIST: continue
            fact = f"Mechanism: In {r['count']} pathways (e.g., {','.join(r['evidence'])})."
            candidates.setdefault(r['gene'], []).append(fact)
            raw_facts_map.setdefault(r['gene'], []).append(fact)
        
        top_list = sorted(candidates.items(), key=lambda x: len(x[1]), reverse=True)[:15]
        facts_text = "\n".join([f"- {g}: {' '.join(ev)}" for g, ev in top_list])

        sys_prompt = "你是资深生物信息学家，请严格输出JSON格式。"
        user_prompt = KG_DISCOVERY_ANALYSIS.format(disease=disease, facts_text=facts_text)
        
        llm_result = call_llm(user_prompt, system_prompt=sys_prompt, json_mode=True, temperature=0.2, parse_json=True)
        
        if llm_result:
            return {
                "mode": "discovery",
                "status": "success",
                "target_list_for_omics": llm_result.get("omics_targets", []), 
                "evidence_details": llm_result.get("evidence_map", {}),
                "kg_scores": llm_result.get("kg_scores", {}),  # ✅ 新增: LLM 评分
                "raw_facts_map": raw_facts_map # ✅ 传递原始事实
            }
        return {"error": "LLM failed"}
        
    # ==========================================
    #  业务逻辑：Validation (返回结构化解释)
    # ==========================================
    def _run_validation_struct(self, disease, gene_list):
        print(f"KGTool [Validation]: Analyzing {gene_list}...")
        
        # 1. 检查已知状态
        known_set = self._check_is_known_batch(disease, gene_list)
        
        # 2. 获取证据
        ppi = self._query_ppi(disease, candidate_genes=gene_list, exclude_known=False)
        pw = self._query_pathway(disease, candidate_genes=gene_list, exclude_known=False)
        
        evidence_map = {g: [] for g in gene_list}
        for r in ppi: evidence_map[r['gene']].append(f"PPI: Interacts with {r['evidence']}.")
        for r in pw: evidence_map[r['gene']].append(f"Pathway: {r['evidence']}.")
        
        # 3. 🛠️ 关键修复：构建 raw_facts_map 并生成 Prompt 文本
        raw_facts_map = {}
        facts_text_list = []
        
        for g, evs in evidence_map.items():
            # 标记状态
            status = "Known" if g in known_set else "Novel"
            # 拼接该基因的所有证据
            evidence_str = " ".join(evs)
            
            # 存入 raw_facts_map (用于最终报告展示)
            # 注意：ValidationAgent 期望的是一个 list of strings
            raw_facts_map[g] = [f"Status: {status}", evidence_str] if evidence_str else [f"Status: {status}. No direct KG evidence."]
            
            # 添加到 Prompt 文本
            facts_text_list.append(f"- {g} ({status}): {evidence_str}")

        facts_text = "\n".join(facts_text_list)

        sys_prompt = "你是资深生物信息学家，请严格输出JSON格式。"
        user_prompt = KG_VALIDATION_ANALYSIS.format(disease=disease, facts_text=facts_text)

        llm_result = call_llm(user_prompt, system_prompt=sys_prompt, json_mode=True, temperature=0.2, parse_json=True)
        
        if llm_result:
            return {
                "mode": "validation",
                "status": "success",
                "analysis_results": llm_result.get("gene_hypotheses", {}),
                "kg_scores": llm_result.get("kg_scores", {}),  # ✅ 新增: LLM 评分
                "known_status_map": {g: (g in known_set) for g in gene_list}, # 返回每个基因是否已知
                "raw_facts_map": raw_facts_map # ✅ 新增：必须返回这个，Validator 才能拿到数据
            }
        return {"error": "LLM failed"}

    def run(self, context=None):
        context = context or {}
        mode = context.get("mode", "discovery")
        disease = context.get("disease", "liver cancer")
        
        if mode == "discovery": return self._run_discovery_struct(disease)
        elif mode == "validation": return self._run_validation_struct(disease, context.get("gene_list", []))
        return {"error": "Unknown mode"}

# --- 测试代码 ---
if __name__ == "__main__":
    tool = KGTool()
    
    # 场景 1: 发现模式 -> 直接拿 List 给 Omics
    print("\n--------- Discovery Mode ---------")
    disc_res = tool.run({"mode": "discovery", "disease": "liver cancer"})
    
    if "error" not in disc_res:
        # 模拟传给 Omics
        omics_input = disc_res['target_list_for_omics']
        print(f"传给 Omics 的列表: {omics_input}") 
        print(f"第一名理由: {disc_res['evidence_details'].get(omics_input[0])}")

    # 场景 2: 验证模式 -> 直接拿 Dict 做展示
    print("\n--------- Validation Mode ---------")
    val_res = tool.run({"mode": "validation", "disease": "liver cancer", "gene_list": ["STAMBP", "TP53"]})
    
    if "error" not in val_res:
        print(json.dumps(val_res['analysis_results'], indent=2))
        print(val_res.get("raw_facts_map")) # 检查是否有数据