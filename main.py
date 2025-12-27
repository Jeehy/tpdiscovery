# main.py
"""
TwoPathDiscovery 主入口 (LangGraph 版本)
"""
import sys
import os
import json

import pandas as pd

from graph import run_discovery


def save_reports(candidates, task_name="discovery"):
    """保存报告"""
    if not candidates:
        return
    
    output_dir = "D:/Bit/TwoPathDiscovery/result"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存 JSON
    json_filename = os.path.join(output_dir, f"Final_Report_{task_name}.json")
    try:
        with open(json_filename, "w", encoding='utf-8') as f:
            json.dump(candidates, f, indent=2, ensure_ascii=False)
        print(f"📄 [Report] JSON 报告已保存: {json_filename}")
    except Exception as e:
        print(f"⚠️ 保存 JSON 失败: {e}")
    
    # 2. 保存 Excel
    xlsx_filename = os.path.join(output_dir, f"Final_Report_{task_name}.xlsx")
    flat_data = []
    
    for item in candidates:
        raw_data = item.get('_raw_data', {})
        raw_evidence_vault = raw_data.get('raw_evidence_vault', {})
        evidence_chain = raw_data.get('evidence_chain', {})
        omics_data = evidence_chain.get('omics_data', {}) or {}
        lit_evidence = raw_data.get('literature_evidence', {})
        
        kg_facts = raw_evidence_vault.get('kg_raw_facts', [])
        raw_kg = str(kg_facts) if kg_facts else ""
        
        lit_abstracts = raw_evidence_vault.get('lit_raw_abstracts', [])
        raw_lit = ""
        if isinstance(lit_abstracts, list):
            raw_lit = "\n---\n".join([
                f"[{s.get('citation','?')}] {s.get('abstract','?')[:200]}..."
                for s in lit_abstracts if isinstance(s, dict)
            ])
        
        flat = {
            "Gene": item.get('Gene'),
            "Rank_Score": item.get('Score'),
            "Tier": item.get('Tier'),
            "External_DB_Score": raw_data.get('scores', {}).get('opentargets', 'N/A'),
            "Omics_Log2FC": omics_data.get('log2fc', 'N/A'),
            "Omics_Padj": omics_data.get('padj', 'N/A'),
            "Omics_Spearman_R": omics_data.get('spearman_r', 'N/A'),
            "Omics_P_Correlation": omics_data.get('p_correlation', 'N/A'),
            "Omics_Drug": omics_data.get('drug_source', 'N/A'),
            "KG_Hypothesis": item.get('KG_Hypothesis'),
            "Lit_Conclusion": lit_evidence.get('conclusion', 'N/A'),
            "Raw_KG_Facts": raw_kg[:5000],
            "Raw_Lit_Abstracts": raw_lit[:5000]
        }
        flat_data.append(flat)
    
    try:
        pd.DataFrame(flat_data).to_excel(xlsx_filename, index=False)
        print(f"📊 [Report] Excel 报告已保存: {xlsx_filename}")
    except Exception as e:
        print(f"⚠️ 保存 Excel 失败: {e}")


def main():
    print("🚀 启动 LangGraph 科研 Agent...")
    
    # 获取输入
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = input("请输入任务 (如: 发现肝癌新靶点): ")
    
    if not task:
        return
    
    # 运行图
    try:
        final_state = run_discovery(task)
        candidates = final_state.get("final_report", [])
        errors = final_state.get("errors", [])
        
        # 显示错误
        if errors:
            print("\n⚠️ 执行过程中的错误:")
            for e in errors:
                print(f"  - {e}")
        
        print("\n=== 🎯 最终推荐 ===")
        if not candidates:
            print("❌ 未发现候选靶点。")
        else:
            # 终端展示 Top 5
            for i, c in enumerate(candidates[:5]):
                print(f"{i+1}. {c['Gene']} | {c['Tier']} | Score: {c['Score']}")
                if "External_DB_Score" in c:
                    print(f"   [外部验证] OpenTargets Score: {c['External_DB_Score']}")
            
            # 保存报告
            mode_name = "verification" if "验证" in task else "discovery"
            save_reports(candidates, task_name=mode_name)
    
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
