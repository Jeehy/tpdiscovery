import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import glob

# ================= 配置区域 =================
# 1. 原始大文件路径
MAIN_DATA_PATH = "D:/Bit/tools/data/最终三表合一数据.csv"

# 2. Step 2 (IC50相关性) 结果所在的文件夹
STEP2_DIR = "D:/Bit/tools/data/IC50_correlation"

# 3. 本次 (Step 3) 结果输出文件夹
OUTPUT_DIR = "D:/Bit/tools/data/Final_Heatmaps"

# 4. 丰度筛选阈值 (平均 Count > 此值才保留)
# 建议：原始Count数据设为 10-20；如果已经是TPM/FPKM可设为 1-5
EXPR_THRESHOLD = 10 
# ===========================================

def run_step3_heatmap():
    # 设置绘图风格
    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 1. 读取原始大表 ---
    print("🚀 Step 3: 正在读取原始数据...")
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

    # --- 2. 寻找 Step 2 的结果文件 ---
    if not os.path.exists(STEP2_DIR):
        print(f"❌ 未找到 Step 2 结果目录: {STEP2_DIR}")
        return

    # 查找所有 *_Step2_Correlated.csv 文件
    candidate_files = glob.glob(os.path.join(STEP2_DIR, "*_Step2_Correlated.csv"))
    
    if not candidate_files:
        print(f"⚠️ {STEP2_DIR} 中没有找到 CSV 文件，请检查 Step 2 是否成功运行。")
        return

    print(f"📂 发现 {len(candidate_files)} 个药物的候选基因表，开始处理...\n")

    for file_path in candidate_files:
        filename = os.path.basename(file_path)
        # 提取药物名 (文件名格式: DrugName_Step2_Correlated.csv)
        drug_name = filename.split('_')[0]
        
        print(f"🔹 正在处理: {drug_name}")
        
        # 读取 Step 2 筛选出的基因
        try:
            cand_df = pd.read_csv(file_path)
            if cand_df.empty:
                print("   ⚠️ 基因表为空，跳过。")
                continue
            candidate_genes = cand_df['gene_id'].astype(str).tolist()
        except Exception as e:
            print(f"   ❌ 读取失败: {e}")
            continue

        # 确保基因在原始大表中存在
        valid_genes = [g for g in candidate_genes if g in df_main.columns]
        if not valid_genes:
            print("   ⚠️ 有效基因数为0，跳过。")
            continue

        # --- 3. 提取数据与丰度过滤 ---
        
        # 获取分组列
        sens_col = f"Organoid-{drug_name}-Sensitive"
        if sens_col not in df_main.columns:
            print(f"   ❌ 未找到分组列 {sens_col}，跳过。")
            continue

        # 提取有分组信息的样本
        sub_df = df_main.dropna(subset=[sens_col]).copy()
        
        # 计算平均表达量
        expr_data = sub_df[valid_genes]
        mean_expr = expr_data.mean(axis=0)
        
        # 过滤低表达基因
        high_expr_genes = mean_expr[mean_expr > EXPR_THRESHOLD].index.tolist()
        
        print(f"   📊 初始基因: {len(valid_genes)} -> 丰度过滤后: {len(high_expr_genes)} (Mean > {EXPR_THRESHOLD})")
        
        if len(high_expr_genes) < 2:
            print("   ⚠️ 剩余基因过少 (<2)，无法绘图。")
            continue

        # --- 4. 保存最终列表 ---
        final_df = cand_df[cand_df['gene_id'].isin(high_expr_genes)].copy()
        
        # 添加一列 Mean_Expression 供参考
        final_df['Mean_Expr'] = final_df['gene_id'].map(mean_expr)
        
        save_path = os.path.join(OUTPUT_DIR, f"{drug_name}_Final_Targets.csv")
        final_df.to_csv(save_path, index=False)
        print(f"   ✅ 最终靶点列表已保存: {save_path}")

        # --- 5. 绘制热图 ---
        # 如果基因太多(>50)，为了图好看，只画方差最大的50个
        # 但 CSV 列表里是全的
        plot_genes = high_expr_genes
        if len(plot_genes) > 50:
            variances = expr_data[plot_genes].var().sort_values(ascending=False)
            plot_genes = variances.head(50).index.tolist()
            print(f"   🖼️ 基因较多，热图仅展示方差最大的 Top 50。")

        draw_heatmap(sub_df, sens_col, plot_genes, drug_name)
        print("   ✅ 完成。\n")

    print(f"🎉 Step 3 全部完成！最终结果请查看: {OUTPUT_DIR}")

def draw_heatmap(df, group_col, genes, title):
    """绘制标准化聚类热图"""
    # 准备数据
    plot_data = df[genes].copy()
    
    # Log2 转换 (伪计数+1)
    plot_data = np.log2(plot_data + 1)
    
    # 转置: 行=基因, 列=样本
    data_t = plot_data.T 
    
    # 准备分组颜色条
    groups = df[group_col]
    # 定义颜色: Sensitive=红, Resistant=蓝
    lut = {'Yes': '#E64B35', 'No': '#4DBBD5'} 
    # 处理可能的大小写不一致
    group_map = {g: lut.get(str(g).capitalize(), lut.get(g, 'gray')) for g in groups.unique()}
    
    # 手动修正常见的 Yes/No 匹配
    for g in groups.unique():
        g_str = str(g).lower()
        if g_str in ['yes', 'sensitive']: group_map[g] = '#E64B35'
        elif g_str in ['no', 'resistant']: group_map[g] = '#4DBBD5'
        
    col_colors = groups.map(group_map)
    
    try:
        # z_score=0 表示对行(基因)进行标准化(Z-score)，这会让差异更明显
        g = sns.clustermap(data_t, 
                           col_colors=col_colors, 
                           z_score=0,             
                           cmap="vlag", # 红蓝配色 (Blue=Low, Red=High)
                           center=0, 
                           figsize=(12, 12) if len(genes) > 30 else (10, 8),
                           dendrogram_ratio=(.15, .15),
                           cbar_pos=(.02, .8, .03, .15)) # 图例位置
        
        # 添加分组图例
        handles = [mpatches.Patch(facecolor=color, label=label) for label, color in group_map.items()]
        # 将图例放在合适的位置
        plt.legend(handles=handles, title='Group', loc='upper right', 
                   bbox_to_anchor=(0.98, 0.98), bbox_transform=g.fig.transFigure)
        
        g.fig.suptitle(f"{title} Final Targets Expression", fontsize=16, y=1.02)
        
        # 保存
        save_file = os.path.join(OUTPUT_DIR, f"{title}_Heatmap.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   🖼️ 热图已保存: {save_file}")
        
    except Exception as e:
        print(f"   ⚠️ 绘图失败: {e}")

if __name__ == "__main__":
    run_step3_heatmap()