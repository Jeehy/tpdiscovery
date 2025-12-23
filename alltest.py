import pandas as pd
import json, traceback
from datetime import datetime
from explorers import ResearchExplorers
from integrator import ValidationAgent
from tools.literature.literature_agent import LiteratureAgent
from playbook import Playbook

# ==========================================
# 任务控制台 (Mission Control)
# ==========================================
DISEASE_NAME = "liver cancer"
PROJECT_MODE = "discovery"
# 如果是 validation 模式，请在这里填入你想验证的基因
TARGETS_TO_VALIDATE = ["TP53", "EGFR", "HBEGF"] 

def main():
    print(f"🚀 启动 AI 科研助理 | 模式: {PROJECT_MODE.upper()} | 疾病: {DISEASE_NAME}\n")
    
    playbook = Playbook()
    print(f"📚 [Playbook] 已加载历史策略库，当前包含 {len(playbook.strategies)} 条经验。")
    explorers = ResearchExplorers()
    validator = ValidationAgent()
    lit_agent = LiteratureAgent()
    
    final_candidates = []
    step_trace = []

    try:
        # ==========================================
        # 🔄 分支 1: Discovery 模式 (双路并行)
        # ==========================================
        if PROJECT_MODE == "discovery":
            print(">>> [Phase 1] 执行双路探索工作流...")
            # --- Path A: 数据驱动 (Omics -> KG) ---
            step_trace.append("run_omics_driven")
            res_path_a = explorers.run_path_omics_driven(threshold=6.0, disease=DISEASE_NAME)
            
            # --- Path B: 理论驱动 (KG -> Omics) ---
            step_trace.append("run_kg_driven")
            res_path_b = explorers.run_path_kg_driven(disease=DISEASE_NAME)
            
            # --- 合并与初步评级 ---
            print("\n>>> [Phase 2] 整合双路结果...")
            # Validator 会处理合并、去重、打分
            # 注意：Validator 内部的 is_known 过滤在这里依然有效，作为双重保险
            final_candidates = validator.validate_and_rank(res_path_b, res_path_a, DISEASE_NAME)
        
        # ==========================================
        # 🎯 分支 2: Validation 模式 (定点清除)
        # ==========================================
        elif PROJECT_MODE == "validation":
            print(f">>> [Phase 1] 执行靶点验证工作流 (Targets: {len(TARGETS_TO_VALIDATE)})...")
            step_trace.append(f"validation_deep_dive_{len(TARGETS_TO_VALIDATE)}_genes")

            res_validation = explorers.run_validation_deep_dive(TARGETS_TO_VALIDATE, disease=DISEASE_NAME)
            
            final_candidates = []
            for gene, info in res_validation.items():
                ot_score = info['opentargets_data'].get('score', 0.0)
                final_candidates.append({
                    "Gene": gene,
                    "Tier": "Target Validation", # 固定 Tier
                    "Score": 10.0 + ot_score, # 固定高分 + OT 分数
                    "Omics_Log2FC": info['omics_signal'].get('log2fc'),
                    "KG_Hypothesis": info['kg_narrative'],
                    "Raw_Evidence": {
                        "kg_raw_facts": info.get('kg_raw', []),
                        "ot_summary": f"OpenTargets Score: {ot_score}"
                    },
                    "_raw_data": info
                })
                
        # ==========================================
        # 📚 通用步骤: 文献核查 (Mode 透传)
        # ==========================================
        if not final_candidates:
            print("❌ 未发现任何候选基因，流程结束。")
            playbook.add_strategy({
                "task": f"{PROJECT_MODE} {DISEASE_NAME}",
                "status": "failure",
                "steps_summary": step_trace,
                "conclusion": "No candidates found"
            })
            return

        print(f"\n>>> [Phase 3] 启动文献核查 (Mode: {PROJECT_MODE})...")
        step_trace.append("literature_verification")
        # 提取基因列表
        targets_list = [item['Gene'] for item in final_candidates]
        
        # 执行检索 (传入全局 PROJECT_MODE)
        # Discovery -> 查泛癌/旁证
        # Validation -> 查肝癌/铁证
        lit_results = lit_agent.run_batch_verification(
            gene_list=targets_list, 
            disease=DISEASE_NAME, 
            mode=PROJECT_MODE 
        )

        # ==========================================
        # 📝 Phase 4: 报告与记忆
        # ==========================================
        print("\n>>> [Phase 4] 生成最终报告...")
        
        # 挂载文献结果
        final_report = validator.attach_literature_evidence(final_candidates, lit_results)
        
        # 保存
        file_suffix = f"{PROJECT_MODE}_{len(final_report)}genes"
        with open(f"Final_Report_{file_suffix}.json", "w", encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
            
        # Save Excel
        flat_data = []
        for item in final_report:
            raw_kg = "; ".join(item['Raw_Evidence'].get('kg_raw_facts', []))
            raw_lit = "\n---\n".join([f"[{s.get('citation','')}] {s.get('abstract','')[:200]}..." for s in item['Raw_Evidence'].get('lit_raw_abstracts', [])])
            raw_ot = item['Raw_Evidence'].get('ot_summary', 'N/A')

            flat = {
                "Gene": item['Gene'],
                "Mode": PROJECT_MODE,
                "Rank_Score": item['Score'],
                "Tier": item['Tier'],
                "Omics_Log2FC": item['Omics_Log2FC'],
                "OpenTargets_Summary": raw_ot,
                "KG_Hypothesis": item['KG_Hypothesis'],
                "Lit_Conclusion": item.get('Literature_Evidence', {}).get('conclusion', 'N/A'),
                "Raw_KG_Facts": raw_kg[:5000], # 防止 Excel 溢出
                "Raw_Lit_Abstracts": raw_lit[:5000] 
            }
            flat_data.append(flat)
        
        pd.DataFrame(flat_data).to_excel(f"Final_Report_{file_suffix}.xlsx", index=False)
        print(f"✅ 任务完成！文件已保存: Final_Report_{file_suffix}.xlsx")

        # 成功记录到 Playbook
        top_genes = [f['Gene'] for f in final_report[:5]]
        playbook.add_strategy({
            "task": f"{PROJECT_MODE} {DISEASE_NAME}",
            "status": "success",
            "steps_summary": step_trace,
            "conclusion": f"Found {len(final_report)} candidates. Top: {top_genes}",
            "timestamp": datetime.now().isoformat()
        })
        print("📚 [Playbook] 本次运行策略已归档保存。")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        traceback.print_exc()
        playbook.add_strategy({
            "task": f"{PROJECT_MODE} {DISEASE_NAME}",
            "status": "error",
            "steps_summary": step_trace,
            "conclusion": str(e)
        })

if __name__ == "__main__":
    main()