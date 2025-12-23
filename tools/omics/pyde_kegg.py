import pandas as pd
import gseapy as gp
import os
import glob
import json
import re  # 引入正则库

# ================= 配置区域 =================
# 输入：Step 3 的结果
INPUT_DIR = "D:/Bit/tools/data/Final_Heatmaps" 
# 输出：系统特征文件
OUTPUT_DIR = "D:/Bit/tools/data/System_Input_Features"
# 数据库目录
DB_DIR = "D:/Bit/tools/data/databases"

# 本地数据库文件
GMT_FILES = {
    'KEGG': os.path.join(DB_DIR, "KEGG_2021_Human.gmt"),
    'GO_BP': os.path.join(DB_DIR, "GO_Biological_Process_2025.gmt")
}
# ===========================================

def clean_gene_symbol(gene_str):
    """
    【升级版】清洗基因名
    兼容: RNA-TP53, RNA_TP53, GENE_EGFR, GENE-EGFR 等各种格式
    """
    s = str(gene_str).strip()
    
    # 1. 使用正则表达式智能去除前缀
    # ^(RNA|GENE) 表示以RNA或GENE开头
    # [-_] 表示后面跟着 - 或 _
    # flags=re.IGNORECASE 表示不区分大小写
    s = re.sub(r'^(RNA|GENE)[-_]', '', s, flags=re.IGNORECASE)
    
    # 2. 去除括号及内容 (例如 TINAGL1)
    if '(' in s:
        s = s.split('(')[0]
        
    return s.strip().upper()

def run_system_enrichment_offline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查数据库
    for name, path in GMT_FILES.items():
        if not os.path.exists(path):
            print(f"❌ 致命错误: 找不到数据库文件 {path}")
            return

    files = glob.glob(os.path.join(INPUT_DIR, "*_Final_Targets.csv"))
    if not files:
        print(f"❌ 未在 {INPUT_DIR} 找到靶点文件。")
        return

    for file_path in files:
        drug_name = os.path.basename(file_path).split('_')[0]
        print(f"⚙️ [离线模式] 正在生成特征: {drug_name}...")
        
        # 1. 读取并【清洗】基因
        try:
            df = pd.read_csv(file_path)
            # 生成清洗后的 clean_symbol 列
            df['clean_symbol'] = df['gene_id'].apply(clean_gene_symbol)
            
            # 提取基因列表
            gene_list = df['clean_symbol'].unique().tolist()
            
            # 打印前几个检查 (这次应该没有 RNA_ 了)
            print(f"   🔍 清洗后基因示例: {gene_list[:5]}")
            
        except Exception as e:
            print(f"   ⚠️ 读取失败: {e}")
            continue
            
        if len(gene_list) < 3: # 稍微放宽限制
            print(f"   ⚠️ 基因过少，跳过。")
            continue

        # 2. 运行离线富集
        all_sig_paths = []
        
        for db_name, gmt_path in GMT_FILES.items():
            try:
                enr = gp.enrichr(gene_list=gene_list,
                                 gene_sets=gmt_path,
                                 background=None, 
                                 outdir=None,
                                 no_plot=True,
                                 verbose=False)
                
                res = enr.results
                if res.empty: continue
                    
                # 筛选 P < 0.05
                sig = res[res['Adjusted P-value'] < 0.05].copy()
                if not sig.empty:
                    sig['Source'] = db_name
                    all_sig_paths.append(sig)
                
            except Exception as e:
                pass

        # 如果没有通路
        if not all_sig_paths:
            print(f"   ⚠️ {drug_name} 未发现显著富集通路。")
            # 存个空特征文件防止系统报错
            df['Enriched_Pathways'] = ""
            df['Pathway_Score'] = 0
            df.to_csv(os.path.join(OUTPUT_DIR, f"{drug_name}_System_Features.csv"), index=False)
            continue
            
        # 3. 特征工程转化
        combined_paths = pd.concat(all_sig_paths)
        print(f"   ✅ 成功发现 {len(combined_paths)} 条显著通路！")

        symbol_to_pathway = {g: [] for g in gene_list}
        
        for _, row in combined_paths.iterrows():
            pathway_name = row['Term']
            source = row['Source']
            full_path_tag = f"{source}:{pathway_name}"
            
            # gseapy 返回的 Genes 是清洗后的 symbol
            genes_in_path = str(row['Genes']).split(';')
            
            for gene in genes_in_path:
                gene = gene.strip().upper()
                if gene in symbol_to_pathway:
                    symbol_to_pathway[gene].append(full_path_tag)
        
        # 4. 映射回原始表
        df['Enriched_Pathways'] = df['clean_symbol'].map(
            lambda x: ';'.join(symbol_to_pathway.get(x, []))
        )
        
        df['Pathway_Score'] = df['clean_symbol'].map(
            lambda x: len(symbol_to_pathway.get(x, []))
        )
        
        # 5. 保存
        save_path = os.path.join(OUTPUT_DIR, f"{drug_name}_System_Features.csv")
        df.to_csv(save_path, index=False)
        
        # JSON Map
        json_path = os.path.join(OUTPUT_DIR, f"{drug_name}_KG_Map.json")
        with open(json_path, 'w') as f:
            json.dump(symbol_to_pathway, f, indent=4)
            
        # 打印 Top 1
        top_row = df.sort_values('Pathway_Score', ascending=False).iloc[0]
        print(f"      🌟 核心基因: {top_row['clean_symbol']} -> 命中 {top_row['Pathway_Score']} 条通路")

    print("\n🎉 修复版特征工程完成！")

if __name__ == "__main__":
    run_system_enrichment_offline()