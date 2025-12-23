import pandas as pd
import os
import glob
import re

# ================= 配置区域 =================
STEP2_DIR = "D:/Bit/tools/data/IC50_correlation"
STEP4_DIR = "D:/Bit/tools/data/System_Input_Features"
OUTPUT_DIR = "D:/Bit/tools/data/LLM_Input_Ready"
# ===========================================

def ensure_clean_symbol(row):
    """确保获取清洗后的基因名"""
    if pd.notnull(row.get('clean_symbol')) and str(row['clean_symbol']).strip() != '':
        return str(row['clean_symbol']).strip().upper()
    s = str(row['gene_id']).strip()
    s = re.sub(r'^(RNA|GENE)[-_]', '', s, flags=re.IGNORECASE)
    if '(' in s: s = s.split('(')[0]
    return s.strip().upper()

def generate_llm_profiles_deep_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    feature_files = glob.glob(os.path.join(STEP4_DIR, "*_System_Features.csv"))
    if not feature_files:
        print("❌ 未找到 Step 4 文件。")
        return

    print(f"🚀 开始构建 LLM 深度分析指令 (Deep Insight Prompt)...\n")

    for feat_file in feature_files:
        filename = os.path.basename(feat_file)
        drug_name = filename.split('_')[0]
        print(f"🔹 处理药物: {drug_name}")
        
        # 1. 读取 & 清洗 Step 4
        df_step4 = pd.read_csv(feat_file)
        df_step4['merge_key'] = df_step4.apply(ensure_clean_symbol, axis=1)
        cols_to_drop = ['Spearman_R', 'P_Correlation', 'Log2FC_DEA', 'Padj_DEA']
        df_step4 = df_step4.drop(columns=[c for c in cols_to_drop if c in df_step4.columns])

        # 2. 读取 & 清洗 Step 2
        step2_file = os.path.join(STEP2_DIR, f"{drug_name}_Step2_System_Full.csv")
        if not os.path.exists(step2_file):
            step2_file = os.path.join(STEP2_DIR, f"{drug_name}_Step2_Correlated.csv")
            
        if not os.path.exists(step2_file): continue
            
        df_step2 = pd.read_csv(step2_file)
        df_step2['merge_key'] = df_step2.apply(ensure_clean_symbol, axis=1)
        
        # 3. 合并
        right_cols = ['merge_key', 'Spearman_R', 'P_Correlation', 'Log2FC_DEA', 'Padj_DEA']
        right_cols = [c for c in right_cols if c in df_step2.columns]
        df_step2_clean = df_step2[right_cols].drop_duplicates(subset=['merge_key'])
        df_merged = pd.merge(df_step4, df_step2_clean, on='merge_key', how='left')

        # 4. 【核心修改】构建结构化 Prompt
        prompts = []
        for _, row in df_merged.iterrows():
            gene = row['merge_key']
            
            # 数值提取
            fc = row.get('Log2FC_DEA', 0.0)
            if pd.isna(fc): fc = 0.0
            padj = row.get('Padj_DEA', 1.0)
            if pd.isna(padj): padj = 1.0
            r_val = row.get('Spearman_R', 0.0)
            if pd.isna(r_val): r_val = 0.0
            p_corr = row.get('P_Correlation', 1.0)
            
            pathways = str(row.get('Enriched_Pathways', ''))
            if pathways == 'nan' or not pathways.strip():
                path_desc = "None"
            else:
                # 简化通路描述，只取前8个，去掉KEGG:前缀，让LLM读得更顺
                p_list = [p.split(':')[-1] for p in pathways.split(';')]
                path_desc = ', '.join(p_list[:8])

            # ---------------------------------------------------------
            # 这里是让 LLM 变聪明的关键指令
            # ---------------------------------------------------------
            prompt = (
                f"You are a Senior Oncologist and Bioinformatician specializing in Liver Cancer.\n"
                f"Please evaluate the gene '{gene}' as a potential biomarker for {drug_name} resistance.\n\n"
                
                f"[Omics Data Profile]\n"
                f"1. **Differential Expression**: Log2FC = {fc:.2f}, Padj = {padj:.2e}\n"
                f"   (Note: Log2FC < 0 implies higher expression in the Resistant group; Log2FC > 0 implies Sensitive group.)\n"
                f"2. **Drug Response Correlation**: Spearman R = {r_val:.2f}, P-value = {p_corr:.2e}\n"
                f"   (Note: Positive R implies that higher expression correlates with higher IC50/Resistance.)\n"
                f"3. **Pathway Context**: Enriched in: {path_desc}\n\n"
                
                f"[Analysis Task]\n"
                f"Please provide a structured report covering the following dimensions:\n"
                f"1. **Data Consistency Check**: Do the Log2FC and Correlation values align logically? (e.g., Does Negative FC align with Positive R?)\n"
                f"2. **Mechanism Hypothesis**: Based on the gene's function and the enriched pathways, hypothesize HOW it might cause resistance (e.g., via efflux pumps, anti-apoptosis, EMT, or angiogenesis bypass).\n"
                f"3. **Clinical Relevance**: Is this gene a known target or marker in HCC or other cancers?\n"
                f"4. **Final Verdict**: Give a Resistance Driver Score (0-10) and a one-sentence conclusion."
            )
            # ---------------------------------------------------------
            
            prompts.append(prompt)

        df_merged['LLM_Prompt'] = prompts
        
        # 5. 保存
        save_path = os.path.join(OUTPUT_DIR, f"{drug_name}_LLM_Input_Deep.csv")
        final_cols = ['merge_key', 'Pathway_Score', 'Spearman_R', 'Log2FC_DEA', 'LLM_Prompt']
        save_cols = [c for c in final_cols if c in df_merged.columns]
        
        df_merged[save_cols].to_csv(save_path, index=False)
        print(f"   ✅ 已生成深度分析 Prompt: {save_path}")
        
        # 打印一个 HBEGF 的例子看看效果
        if 'HBEGF' in df_merged['merge_key'].values:
            sample = df_merged[df_merged['merge_key']=='HBEGF'].iloc[0]['LLM_Prompt']
            print(f"\n--- HBEGF Prompt 预览 ---\n{sample}\n-------------------------\n")

    print("🎉 Step 5 (Deep) 完成！")

if __name__ == "__main__":
    generate_llm_profiles_deep_analysis()