import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import glob
import re  # 引入正则库用于清洗名字

# ================= 配置区域 =================
# 1. 原始大文件路径
MAIN_DATA_PATH = "D:/Bit/tools/data/最终三表合一数据.csv"

# 2. Step 1 (差异分析) 结果所在的文件夹
STEP1_DIR = "D:/Bit/tools/data/deseq2_results" 

# 3. 本次 (Step 2) 结果输出文件夹
OUTPUT_DIR = "D:/Bit/tools/data/IC50_correlation"

# 4. 筛选阈值 (仅用于生成给人类看的精简报表和绘图，全量数据会被完整保留)
HUMAN_VIEW_CORR_THRESHOLD = 0.3 
# ===========================================

def clean_gene_symbol(gene_str):
    """
    清洗基因名：去除 RNA- 或 GENE_ 前缀，去除括号
    """
    s = str(gene_str).strip()
    # 替换 RNA- 或 GENE_ (不区分大小写)
    s = re.sub(r'^(RNA|GENE)[-_]', '', s, flags=re.IGNORECASE)
    # 去除括号内容
    if '(' in s:
        s = s.split('(')[0]
    return s.strip().upper()

def run_step2_correlation_final():
    # 设置绘图风格
    plt.style.use('default') 
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 1. 读取原始大表 ---
    print("🚀 Step 2 (Final): 正在读取原始数据...")
    if not os.path.exists(MAIN_DATA_PATH):
        print(f"❌ 找不到原始数据文件: {MAIN_DATA_PATH}")
        return

    try:
        df_main = pd.read_csv(MAIN_DATA_PATH, encoding='gb18030')
    except:
        try:
            df_main = pd.read_csv(MAIN_DATA_PATH, encoding='utf-8')
        except:
            print("❌ 无法读取数据文件，请检查编码格式。")
            return
    
    # 清理列名
    clean_cols = {c: c.replace(' (μM)', '').strip() for c in df_main.columns}
    df_main = df_main.rename(columns=clean_cols)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. 遍历 Step 1 的结果 ---
    if not os.path.exists(STEP1_DIR):
        print(f"❌ 未找到 Step 1 目录: {STEP1_DIR}")
        return

    drug_folders = [f for f in os.listdir(STEP1_DIR) if os.path.isdir(os.path.join(STEP1_DIR, f))]
    
    if not drug_folders:
        print(f"⚠️ {STEP1_DIR} 为空，请先运行 Step 1。")
        return

    print(f"📂 发现 {len(drug_folders)} 个药物文件夹，开始全量计算与清洗...\n")

    for drug_name in drug_folders:
        drug_path = os.path.join(STEP1_DIR, drug_name)
        
        # 优先读取全量显著基因
        target_file = "All_Significant_DEGs.csv"
        deg_file = os.path.join(drug_path, target_file)
        
        if not os.path.exists(deg_file):
            target_file = "DEGs_hybrid_final.csv"
            deg_file = os.path.join(drug_path, target_file)
            if not os.path.exists(deg_file):
                print(f"⚠️ 跳过 {drug_name}: 无输入文件。")
                continue
        
        print(f"🔹 正在分析: {drug_name}")
        
        deg_df = pd.read_csv(deg_file)
        if deg_df.empty: continue
            
        # 寻找 IC50 列
        ic50_keyword = f"Organoid-{drug_name}-IC50"
        ic50_cols = [c for c in df_main.columns if ic50_keyword in c]
        if not ic50_cols:
            print(f"   ❌ 未找到 {drug_name} IC50 数据，跳过。")
            continue
        ic50_col = ic50_cols[0]
        
        # 寻找分组列
        sens_keyword = f"Organoid-{drug_name}-Sensitive"
        sens_cols = [c for c in df_main.columns if sens_keyword in c]
        sens_col = sens_cols[0] if sens_cols else None

        # 提取有效数据
        valid_df = df_main.dropna(subset=[ic50_col]).copy()
        if len(valid_df) < 5:
            print(f"   ⚠️ 有效样本过少，跳过。")
            continue
        
        # --- 3. 批量计算相关性 (全量计算，不预先过滤) ---
        results = []
        
        for _, row in deg_df.iterrows():
            gene_id = row['gene_id'] # 原始ID
            
            # 【清洗步骤】 生成干净的 gene symbol
            clean_symbol = clean_gene_symbol(gene_id)
            
            if gene_id not in df_main.columns:
                continue
            
            # 获取差异分析数据
            log2fc = row.get('log2FoldChange', np.nan)
            padj = row.get('padj', np.nan)

            # 计算相关性
            expr_vals = np.log2(valid_df[gene_id] + 1)
            ic50_vals = valid_df[ic50_col]
            
            corr, p_corr = stats.spearmanr(expr_vals, ic50_vals)
            
            # 处理 NaN
            if np.isnan(corr): corr = 0
            if np.isnan(p_corr): p_corr = 1
            
            # 存入结果
            results.append({
                'gene_id': gene_id,          # 原始ID (用于索引)
                'clean_symbol': clean_symbol,# 清洗ID (用于展示/LLM)
                'Spearman_R': corr,
                'P_Correlation': p_corr,
                'Log2FC_DEA': log2fc,
                'Padj_DEA': padj
            })
        
        if not results: continue
            
        res_df = pd.DataFrame(results)
        
        # --- 4. 保存两份文件 ---
        
        # 文件 A: 全量数据 (System Full Data) -> 包含所有相关性低的数据，给 LLM 用
        system_save_path = os.path.join(OUTPUT_DIR, f"{drug_name}_Step2_System_Full.csv")
        res_df.to_csv(system_save_path, index=False)
        
        # 文件 B: 精简筛选数据 (Human View) -> 给热图和人眼检查用
        filtered_df = res_df[
            (res_df['Spearman_R'].abs() > HUMAN_VIEW_CORR_THRESHOLD) & 
            (res_df['P_Correlation'] < 0.05)
        ].copy()
        
        # 兼容旧文件名
        filtered_save_path = os.path.join(OUTPUT_DIR, f"{drug_name}_Step2_Correlated.csv")
        filtered_df.to_csv(filtered_save_path, index=False)
        
        print(f"   ✅ [系统用] 全量数据保存: {len(res_df)} 个基因")
        print(f"   ✅ [筛选后] 高相关基因: {len(filtered_df)} 个基因")
        
        # --- 5. 绘图 (Top 6) ---
        if not filtered_df.empty:
            # 按 R 绝对值排序
            plot_df = filtered_df.assign(abs_R=filtered_df['Spearman_R'].abs()).sort_values('abs_R', ascending=False).head(6)
            plot_top_genes(plot_df, valid_df, ic50_col, sens_col, drug_name)

    print("\n🎉 Step 2 (Final) 全部完成！现在数据已清洗且完整。")

def plot_top_genes(top_genes_df, valid_df, ic50_col, sens_col, drug_name):
    """绘制 Top 基因散点图，标题使用清洗后的基因名"""
    num_plots = len(top_genes_df)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    if num_plots == 1: axes = [axes]
    else: axes = axes.flatten()
    
    for i, (_, g_row) in enumerate(top_genes_df.iterrows()):
        gene = g_row['gene_id']
        clean_name = g_row['clean_symbol'] # 使用清洗后的名字
        r_val = g_row['Spearman_R']
        p_corr = g_row['P_Correlation']
        fc_val = g_row['Log2FC_DEA']
        padj_val = g_row['Padj_DEA']
        
        ax = axes[i]
        
        x = np.log2(valid_df[gene] + 1)
        y = valid_df[ic50_col]
        hue_data = valid_df[sens_col] if sens_col else None
        
        palette = None
        if hue_data is not None:
            unique_groups = hue_data.unique()
            palette = {}
            for g in unique_groups:
                if str(g).lower() in ['yes', 'sensitive']: palette[g] = '#E64B35'
                elif str(g).lower() in ['no', 'resistant']: palette[g] = '#4DBBD5'
                else: palette[g] = 'gray'
        
        sns.scatterplot(x=x, y=y, hue=hue_data, palette=palette, s=80, alpha=0.8, edgecolor='w', ax=ax)
        sns.regplot(x=x, y=y, scatter=False, color='#555555', line_kws={'linestyle':'--'}, ax=ax)
        
        # 构建标题
        dea_info = ""
        if pd.notnull(fc_val) and pd.notnull(padj_val):
            dea_info = f"\nLog2FC={fc_val:.2f}, Padj={padj_val:.1e}"
            
        title_str = f"{clean_name}\nR={r_val:.2f} (p={p_corr:.1e}){dea_info}"
        
        ax.set_title(title_str, fontsize=11, fontweight='bold')
        ax.set_xlabel("Log2 Expression")
        ax.set_ylabel("IC50")
        
        if i == 0 and sens_col: ax.legend(loc='best', fontsize=9)
        elif sens_col: 
            if ax.get_legend(): ax.get_legend().remove()
    
    for j in range(i+1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{drug_name}_Correlation_TopGenes.png"), dpi=300)
    plt.close()
    print(f"   🖼️ 散点图已更新: {drug_name}_Correlation_TopGenes.png")

if __name__ == "__main__":
    run_step2_correlation_final()