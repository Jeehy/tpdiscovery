import json
from tools.kg.kgtool import KGTool
from tools.omics.omicstool import OmicsDataRetriever

class ResearchExplorers:
    def __init__(self):
        self.kg = KGTool()
        self.omics = OmicsDataRetriever()

    # ======================================================
    # 🛤️ 路径 A (Discovery): Omics -> KG
    # 逻辑：先看数据谁在变，再看图谱里它是不是潜力股（排除已知）
    # ======================================================
    def run_path_omics_driven(self, threshold=6.0, disease="liver cancer"):
        print(f"\n🔬 [Path A: Omics-Driven] 启动: 实验数据 -> 潜在图谱关联...")
        
        # 1. 获取 Omics 高分基因
        top_genes_map = self.omics.get_top_genes(limit=30, threshold=threshold)
        gene_list = list(top_genes_map.keys())
        print(f"   -> Omics 初步筛选出 {len(gene_list)} 个高分基因 (Top 30)")
        
        # 2. KG 探索 (使用 Validation 模式获取详细连接证据)
        kg_res = self.kg.run({"mode": "validation", "gene_list": gene_list, "disease": disease})
        
        # 提取数据
        kg_narratives = kg_res.get('analysis_results', {})
        known_map = kg_res.get('known_status_map', {})
        # 🛠️ [关键修复] 获取 KGTool 返回的原始事实字典
        kg_raw_map = kg_res.get('raw_facts_map', {}) 

        # [追踪打印]
        print(f"   [DEBUG Explorer] KGTool 返回了 {len(kg_raw_map)} 条原始证据。")
        if "LAMA1" in gene_list:
            if "LAMA1" in kg_raw_map:
                print(f"   [DEBUG Explorer] ✅ LAMA1 数据已成功提取: {str(kg_raw_map['LAMA1'])[:50]}...")
            else:
                print(f"   [DEBUG Explorer] ❌ LAMA1 在基因列表中，但 KGTool 没返回它的 raw_facts！")
                print(f"   [DEBUG Explorer] KGTool Keys: {list(kg_raw_map.keys())[:5]}")
        # -------------------------------
        results = {}
        for gene in gene_list:
            is_known = known_map.get(gene, False)
            
            # 🚨 核心过滤：Discovery 模式剔除已知靶点
            if is_known: 
                continue 
            # 确保即使 KG 没返回，也给个空列表，防止 KeyError
            raw_facts = kg_raw_map.get(gene, [])
            if not raw_facts:
                 # 再次尝试匹配（处理潜在的大小写问题）
                 # 有时候 gene 是 'LAMA1' 但 map 里是 'Lama1'
                 for k, v in kg_raw_map.items():
                     if k.upper() == gene.upper():
                         raw_facts = v
                         break
            results[gene] = {
                "strategy": "Path A (Omics-First)",
                "omics_signal": top_genes_map.get(gene),
                "kg_narrative": kg_narratives.get(gene, "Potential novel link identified via Omics..."),
                # 🛠️ [关键修复] 将原始事实传递给下游
                "kg_raw": raw_facts,
                "is_known": False
            }
        
        print(f"   -> 剔除已知靶点后，Path A 保留 {len(results)} 个新颖候选")
        return results

    # ======================================================
    # 🛤️ 路径 B (Discovery): KG -> Omics
    # 逻辑：先看图谱谁有理论潜力（排除已知），再看数据里它变没变
    # ======================================================
    def run_path_kg_driven(self, disease="liver cancer"):
        print(f"\n🔭 [Path B: KG-Driven] 启动: 理论挖掘 -> 实验数据回填...")
        
        # 1. KG 纯探索 (内部已自动剔除 Known Targets)
        kg_res = self.kg.run({"mode": "discovery", "disease": disease})
        candidate_list = kg_res.get('target_list_for_omics', [])
        evidence_map = kg_res.get('evidence_details', {})
        # 🛠️ [关键修复] 获取原始事实
        raw_facts_map = kg_res.get('raw_facts_map', {})
        
        print(f"   -> KG 挖掘出 {len(candidate_list)} 个理论潜力基因")
        
        # 2. Omics 验证 (补充数据表现)
        # 注意：这里会查表，如果没有数据，会返回 found_in_omics=False
        omics_data_map = self.omics.check_gene_list(candidate_list)
        
        results = {}
        for gene in candidate_list:
            om_info = omics_data_map.get(gene, {})
            
            results[gene] = {
                "strategy": "Path B (KG-First)",
                "kg_narrative": evidence_map.get(gene),
                # 🛠️ [关键修复] 传递原始事实
                "kg_raw": raw_facts_map.get(gene, []),
                "omics_signal": om_info, 
                "is_known": False 
            }
            
        print(f"   -> 经 Omics 对齐，Path B 输出 {len(results)} 个候选")
        return results

    # ======================================================
    # 🎯 验证路径 (Validation): Target -> All Sources
    # 逻辑：针对特定名单，全量提取所有证据 (不剔除已知)
    # ======================================================
    def run_validation_deep_diven(self, target_list: list, disease="liver cancer"):
        print(f"\n🛡️ [Validation Path] 启动: 针对 {target_list} 进行全维取证...")
        
        # 1. 获取 Omics 证据
        omics_data = self.omics.check_gene_list(target_list)
        
        # 2. 获取 KG 证据
        kg_res = self.kg.run({"mode": "validation", "gene_list": target_list, "disease": disease})
        kg_evidence = kg_res.get('analysis_results', {})
        known_map = kg_res.get('known_status_map', {})
        # 🛠️ [关键修复] 获取原始事实
        kg_raw_facts = kg_res.get('raw_facts_map', {})
        
        results = {}
        for gene in target_list:
            results[gene] = {
                "strategy": "Validation (Targeted)",
                "omics_signal": omics_data.get(gene),
                "kg_narrative": kg_evidence.get(gene),
                # 🛠️ [关键修复] 传递原始事实
                "kg_raw": kg_raw_facts.get(gene, []),
                "is_known": known_map.get(gene, False)
            }
            
        return results