import logging

class ValidationAgent:
    """
    验证智能体 (Validation Agent) - 追踪探针版
    """
    def __init__(self):
        self.logger = logging.getLogger("ValidationAgent")

    def validate_and_rank(self, top_down_results, bottom_up_results, disease="liver cancer"):
        print(f"\n🛡️ [Validation Agent] 启动验证程序 (Disease: {disease})...")
        
        candidates = {}
        rejected_known = []
        
        # --- 内部辅助：注册候选基因 ---
        def register(gene, source, data):
            # 1. 核心过滤：剔除已知靶点
            if data.get('is_known', False):
                if gene not in rejected_known: rejected_known.append(gene)
                return

            if gene not in candidates:
                candidates[gene] = {
                    "gene": gene,
                    "evidence_sources": set(),
                    "scores": {"omics": 0, "kg": 0, "opentargets": 0},
                    "raw_evidence_vault": {
                        "kg_raw_facts": [],
                        "omics_full_summary": "",
                        "lit_raw_abstracts": [],
                        "ot_summary": ""
                    },
                    "evidence_chain": {"kg_hypothesis": "N/A", "omics_data": None}
                }
            
            entry = candidates[gene]
            entry['evidence_sources'].add(source)

            # 2. 提取 KG 证据
            if 'kg_narrative' in data and data['kg_narrative']:
                if entry['evidence_chain']['kg_hypothesis'] == "N/A" or len(data['kg_narrative']) > len(entry['evidence_chain']['kg_hypothesis']):
                    entry['evidence_chain']['kg_hypothesis'] = data['kg_narrative']
            
            # 提取 KG 原始事实 (兼容多种键名 + 暴力存储)
            # 优先找 kg_raw (Explorer 传过来的), 其次找 raw_facts_map (KGTool 原生的)
            raw_facts = data.get('kg_raw') or data.get('raw_facts_map')
            if raw_facts:
                if isinstance(raw_facts, str):
                    raw_facts = [raw_facts]
                elif not isinstance(raw_facts, list):
                    raw_facts = []
                if entry['raw_evidence_vault']['kg_raw_facts']:
                     for fact in raw_facts:
                         if fact not in entry['raw_evidence_vault']['kg_raw_facts']:
                             entry['raw_evidence_vault']['kg_raw_facts'].append(fact)
                else:
                    entry['raw_evidence_vault']['kg_raw_facts'] = list(raw_facts)

            # 3. 提取 Omics 证据
            if 'omics_signal' in data and data['omics_signal']:
                om_data = data['omics_signal']
                if om_data.get('omics_score', 0) > entry['scores']['omics']:
                    entry['scores']['omics'] = om_data.get('omics_score', 0)
                
                if not entry['evidence_chain']['omics_data']:
                    entry['evidence_chain']['omics_data'] = {
                        "log2fc": om_data.get('log2fc', "N/A"),
                        "p_value": om_data.get('p_value', "N/A")
                    }
                if 'ai_summary' in om_data:
                    entry['raw_evidence_vault']['omics_full_summary'] = om_data['ai_summary']

        # 4. 提取 OpenTargets 证据
            if 'opentargets_data' in data:
                ot_data = data['opentargets_data']
                ot_score = ot_data.get('score', 0.0)
                entry['scores']['opentargets'] = ot_score
                if ot_score > 0:
                    entry['raw_evidence_vault']['ot_summary'] = f"OpenTargets Score: {ot_score} (Ranked in Top Lists)"

        # 5. 执行数据摄入
        for g, d in top_down_results.items(): 
            register(g, "Top-Down (KG-Driven)", d)
            
        for g, d in bottom_up_results.items(): 
            register(g, "Bottom-Up (Omics-Driven)", d)

        print(f"   [DEBUG] 合并后候选池大小: {len(candidates)} 个")

        # 5. 评级逻辑
        ranked_results = []
        for gene, info in candidates.items():
            sources = info['evidence_sources']
            base_score = info['scores']['omics']
            
            if "Top-Down (KG-Driven)" in sources and "Bottom-Up (Omics-Driven)" in sources:
                tier = "Tier 1: Consensus (双重验证)"
                final_score = base_score + 5.0
            elif "Bottom-Up (Omics-Driven)" in sources:
                tier = "Tier 2: Data-Driven (新颖发现)"
                final_score = base_score + 2.0
            else:
                tier = "Tier 3: Theory-Only (理论预测)"
                final_score = 1.0
            
            # 如果是发现模式，OT 分数低其实是好事（说明新颖），但如果是 0 分可能意味着完全没研究。
            # 这里简单加分：只要有 OT 数据，说明有一定外部可信度，加 0.5 分作为奖励
            if info['scores']['opentargets'] > 0:
                final_score += 0.5
            info['evidence_sources'] = list(info['evidence_sources'])
            
            ranked_results.append({
                "Gene": gene,
                "Tier": tier,
                "Score": round(final_score, 2),
                "Omics_Log2FC": info['evidence_chain']['omics_data'].get('log2fc') if info['evidence_chain']['omics_data'] else "N/A",
                "KG_Hypothesis": info['evidence_chain']['kg_hypothesis'],
                "Raw_Evidence": info['raw_evidence_vault'],
                "_raw_data": info 
            })

        ranked_results.sort(key=lambda x: x['Score'], reverse=True)
        return ranked_results

    def attach_literature_evidence(self, ranked_list, lit_results):
        for item in ranked_list:
            gene = item['Gene']
            if gene in lit_results:
                lit_data = lit_results[gene]
                
                support = str(lit_data.get('lit_support_level', '')).lower()
                if 'high' in support or 'strong' in support:
                    item['Score'] += 3.0
                elif 'medium' in support:
                    item['Score'] += 1.0
                
                item['Literature_Evidence'] = {
                    "support": lit_data.get('lit_support_level'),
                    "conclusion": lit_data.get('lit_conclusion'),
                    "citations": lit_data.get('key_citations', [])
                }
                
                if 'raw_evidence_snippets' in lit_data:
                    item['Raw_Evidence']['lit_raw_abstracts'] = lit_data['raw_evidence_snippets']
                    
        ranked_list.sort(key=lambda x: x['Score'], reverse=True)
        return ranked_list