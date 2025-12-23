import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback

# 尝试导入 pydeseq2
try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
except ImportError:
    print("❌ 请先安装 pydeseq2: pip install pydeseq2")
    exit()

try:
    from adjustText import adjust_text
except ImportError:
    print("⚠️ 未安装 adjustText，图片标签可能重叠 (pip install adjustText)")
    adjust_text = None

# ==========================================
# 1. 分组工具类 (BioDataGroupTool)
# ==========================================
class BioDataGroupTool:
    def __init__(self, df):
        self.df = df
        # 自动识别 RNA 列 (以 RNA 或 GENE_ 开头)
        self.rna_cols = [c for c in df.columns if ('RNA' in c or c.startswith('GENE_'))]
        # 排除非表达量的元数据列
        exclude = ['Date', 'ID', 'Batch', 'CNA', 'Mutation']
        self.rna_cols = [c for c in self.rna_cols if not any(k in c for k in exclude)]

    def get_groups(self, rule_type, param=None):
        rule_type = rule_type.lower()
        if rule_type in ['organoid_drug', '类器官药物']:
            if not param:
                raise ValueError("请提供药物名称")
            col_name = f"Organoid-{param}-Sensitive"
            
            if col_name not in self.df.columns:
                print(f"❌ 列不存在: {col_name}")
                return None, None, None, None
            
            # 提取 Sensitive (Yes) 和 Resistant (No)
            df_sens = self.df[self.df[col_name] == 'Yes'].copy()
            df_res = self.df[self.df[col_name] == 'No'].copy()
            
            print(f"   └─ 分组依据: {col_name}")
            print(f"   └─ Sensitive (Yes): {len(df_sens)} | Resistant (No): {len(df_res)}")
            return df_sens, df_res, "Sensitive", "Resistant"
        else:
            print("⚠️ 目前脚本仅演示 [类器官药物] 分组")
            return None, None, None, None

# ==========================================
# 2. 差异分析核心函数 (修复版)
# ==========================================
def run_deseq2_analysis(tool, drug_name, output_base="results_deseq2"):
    print(f"\n🚀 === 正在分析药物: {drug_name} ===")
    
    # 1. 获取分组数据
    df_sens, df_res, _, _ = tool.get_groups("类器官药物", drug_name)
    
    # 检查样本量
    if df_sens is None or len(df_sens) < 2 or len(df_res) < 2:
        print(f"⚠️ 样本不足，跳过 {drug_name}")
        return

    # 2. 数据准备
    df_combined = pd.concat([df_sens, df_res])
    
    # 提取表达矩阵 (Samples x Genes)
    counts_df = df_combined[tool.rna_cols].fillna(0).round().astype(int)
    
    # 过滤低表达基因
    counts_df = counts_df.loc[:, (counts_df > 0).sum(axis=0) >= 2]
    
    # 构建 Metadata
    metadata = pd.DataFrame({
        'Condition': ['Sensitive'] * len(df_sens) + ['Resistant'] * len(df_res)
    }, index=counts_df.index)
    
    # 3. 运行 DESeq2
    print("⏳ 初始化 DeseqDataSet...")
    try:
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design_factors="Condition", # 旧版写法
            n_cpus=8
        )
    except TypeError:
        # 新版写法
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design="~Condition", 
            n_cpus=8
        )
    
    print("⏳ 运行 DESeq2 分析...")
    dds.deseq2()
    
    print("📊 进行 Wald Test (Sensitive vs Resistant)...")
    stat_res = DeseqStats(dds, contrast=("Condition", "Sensitive", "Resistant"))
    stat_res.run_wald_test()
    
    # 尝试调用 summary，有时这能帮助初始化结果
    try:
        stat_res.summary()
    except Exception:
        pass

    # --- 修复点：更稳健的结果提取 ---
    res = None
    
    # 1. 尝试标准属性 results_df (新版)
    if hasattr(stat_res, "results_df") and stat_res.results_df is not None:
        res = stat_res.results_df.copy()
        
    # 2. 尝试旧版属性 result_df
    elif hasattr(stat_res, "result_df") and stat_res.result_df is not None:
        res = stat_res.result_df.copy()
        
    # 3. 尝试手动构建 (Fallback)
    else:
        print("⚠️ 未找到 results_df，尝试手动构建结果表...")
        try:
            # 提取 pvalue 和 padj
            pvals = stat_res.p_values
            padj = stat_res.padj
            
            # 提取 LFC
            # LFC 可能存储在 stat_res.LFC (DataFrame) 或 dds.varm['LFC']
            if hasattr(stat_res, "LFC") and stat_res.LFC is not None:
                if isinstance(stat_res.LFC, pd.DataFrame):
                    # 通常取最后一列作为当前 contrast 的 LFC
                    lfc = stat_res.LFC.iloc[:, -1]
                else:
                    lfc = stat_res.LFC
            elif hasattr(dds, "varm") and "LFC" in dds.varm:
                 # 尝试从 dds 取 (可能对应 Condition_Sensitive_vs_Resistant)
                 lfc_df = dds.varm["LFC"]
                 lfc = lfc_df.iloc[:, -1] # 盲猜最后一列
            else:
                 print("❌ 无法找到 Log2FoldChange 数据")
                 lfc = np.nan

            res = pd.DataFrame({
                "log2FoldChange": lfc,
                "pvalue": pvals,
                "padj": padj
            }, index=counts_df.columns)
            
        except Exception as e:
            print(f"❌ 手动构建失败: {e}")
            print(f"🔍 可用属性: {dir(stat_res)}")
            return

    if res is None:
        print("❌ 提取结果失败，跳过该药物。")
        return

    # 整理结果表 (确保有 gene_id 列)
    if "gene_id" not in res.columns:
        res = res.reset_index().rename(columns={"index": "gene_id"})

    res = res.sort_values("padj").dropna()
    
    # 4. 筛选重点基因
    fc_cutoff = 2.0
    padj_cutoff = 0.05

    print(f"🔍 筛选中 (FC > {fc_cutoff}, padj < {padj_cutoff})...")
    mask_wide = (res["padj"] < padj_cutoff) & (res["log2FoldChange"].abs() > fc_cutoff)
    top_20_padj = res.nsmallest(20, "padj")
    mask_top = res["gene_id"].isin(top_20_padj["gene_id"])
    
    sig_res = res[mask_wide | mask_top].drop_duplicates("gene_id")
    print(f"✅ 筛选后剩余重点基因数: {sig_res.shape[0]}")
    
    # 5. 保存结果
    save_dir = os.path.join(output_base, drug_name)
    os.makedirs(save_dir, exist_ok=True)
    
    sig_res.to_csv(os.path.join(save_dir, "DEGs_hybrid_final.csv"), index=False)
    # 只要 Padj < 0.05 就保留，不做差异倍数(FC)限制
    all_sig = res[res['padj'] < 0.05].copy()
    all_sig_path = os.path.join(save_dir, "All_Significant_DEGs.csv")
    all_sig.to_csv(all_sig_path, index=False)
    print(f"   💾 已额外保存全量显著基因 ({len(all_sig)}个): {all_sig_path}")
    # 6. 绘图
    plot_volcano(res, sig_res, drug_name, fc_cutoff, padj_cutoff, save_dir)

# ==========================================
# 3. 绘图函数
# ==========================================
def plot_volcano(res, sig_res, title_suffix, fc_cutoff, padj_cutoff, save_dir):
    plt.figure(figsize=(10, 8))
    
    plt.scatter(res["log2FoldChange"], -np.log10(res["padj"]), 
                s=10, alpha=0.3, color="lightgray", label="Insignificant")
    
    up_genes = sig_res[sig_res['log2FoldChange'] > 0]
    down_genes = sig_res[sig_res['log2FoldChange'] < 0]
    
    plt.scatter(up_genes["log2FoldChange"], -np.log10(up_genes["padj"]),
                s=35, color="#E64B35", alpha=0.8, label=f"Sensitive High ({len(up_genes)})")
    
    plt.scatter(down_genes["log2FoldChange"], -np.log10(down_genes["padj"]),
                s=35, color="#4DBBD5", alpha=0.8, label=f"Resistant High ({len(down_genes)})")
    
    plt.axhline(-np.log10(padj_cutoff), color="gray", ls="--", lw=1)
    plt.axvline(fc_cutoff, color="gray", ls="--", lw=1)
    plt.axvline(-fc_cutoff, color="gray", ls="--", lw=1)
    
    plt.xlabel("log2(Fold Change)", fontsize=12)
    plt.ylabel("-log10(Adjusted P-value)", fontsize=12)
    plt.title(f"Volcano Plot: {title_suffix}", fontsize=14)
    
    labels_to_plot = pd.concat([
        sig_res.nlargest(8, "log2FoldChange"),  
        sig_res.nsmallest(8, "log2FoldChange"), 
        sig_res.nsmallest(10, "padj")           
    ]).drop_duplicates("gene_id")
    
    texts = []
    for _, row in labels_to_plot.iterrows():
        clean_name = str(row['gene_id']).replace('RNA-', '').replace('GENE_', '')
        texts.append(plt.text(
            row["log2FoldChange"], 
            -np.log10(row["padj"]), 
            clean_name,
            fontsize=9, fontweight='bold'
        ))
    
    if adjust_text:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "volcano_plot.png"), dpi=300)
    plt.close()
    print(f"✅ 图片已保存: {save_dir}/volcano_plot.png")

# ==========================================
# 4. 主入口
# ==========================================
if __name__ == "__main__":
    input_file = "D:/Bit/tools/data/最终三表合一数据.csv"
    output_dir = "D:/Bit/tools/data/deseq2_results"
    
    if os.path.exists(input_file):
        print(f"📄 读取文件: {input_file}")
        
        try:
            df = pd.read_csv(input_file, encoding='gb18030')
        except UnicodeDecodeError:
            print("⚠️ gb18030 解码失败，尝试 gbk...")
            try:
                df = pd.read_csv(input_file, encoding='gbk')
            except Exception:
                df = pd.read_csv(input_file, encoding='utf-8')

        tool = BioDataGroupTool(df)
        drugs = ['Lenvatinib', 'Sorafenib', 'Regorafenib', 'Apatinib','Bevacizumab','Pemigatinib','Ivosidenib']
        
        for drug in drugs:
            try:
                run_deseq2_analysis(tool, drug, output_base=output_dir)
            except Exception as e:
                print(f"❌ 分析 {drug} 时出错: {e}")
                traceback.print_exc()
            
        print("\n🎉 所有分析已完成！")
    else:
        print(f"❌ 找不到文件: {input_file}")