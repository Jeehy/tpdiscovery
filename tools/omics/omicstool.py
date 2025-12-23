import pandas as pd
import os
import glob
class OmicsDataRetriever:
    """
    组学数据检索器 (Omics Data Retriever)
    
    职责:
    1. 加载并合并所有药物处理后的组学分析报告 (*_Final_Report.csv)。
    2. 为 Bottom-Up 路径提供高分基因列表。
    3. 为 Top-Down 路径提供特定基因的查表服务。
    """
    
    def __init__(self, data_dir="D:/Bit/TwoPathDiscovery/data/Final_LLM_Results"):
        self.data_dir = data_dir
        # 初始化时加载所有数据到内存，避免重复读取 IO
        self.combined_df = self._load_all_data()

    def _load_all_data(self):
        """
        扫描目录下所有 _Final_Report.csv 文件，合并为一个大 DataFrame
        """
        if not os.path.exists(self.data_dir):
            print(f"⚠️ OmicsDataRetriever 警告: 目录不存在 {self.data_dir}")
            return pd.DataFrame()

        all_files = glob.glob(os.path.join(self.data_dir, "*_Final_Report.csv"))
        if not all_files:
            print(f"⚠️ OmicsDataRetriever 警告: 在 {self.data_dir} 未找到报告文件")
            return pd.DataFrame()

        df_list = []
        print(f"📂 OmicsReader: 正在加载 {len(all_files)} 个组学报告文件...")
        
        for f in all_files:
            try:
                temp_df = pd.read_csv(f)
                # 从文件名提取药物名称 (假设文件名格式为 "DrugName_Final_Report.csv")
                file_name = os.path.basename(f)
                drug_name = file_name.split('_')[0]
                temp_df['Source_Drug'] = drug_name
                
                # 确保关键列存在，防止报错
                required_cols = ['merge_key', 'AI_Score', 'Log2FC_DEA']
                if all(col in temp_df.columns for col in required_cols):
                    df_list.append(temp_df)
                else:
                    print(f"  ⚠️ 跳过文件 {file_name}: 缺少关键列 {required_cols}")
            except Exception as e:
                print(f"  ❌ 加载失败 {f}: {e}")
        
        if df_list:
            full_df = pd.concat(df_list, ignore_index=True)
            # 将基因名转为大写，方便后续匹配
            full_df['merge_key'] = full_df['merge_key'].astype(str).str.upper()
            print(f"✅ OmicsReader: 数据加载完毕，共 {len(full_df)} 条记录。")
            return full_df
        
        return pd.DataFrame()

    # ============================================================
    #  功能 A: 获取高分基因 (Bottom-Up 路径起点)
    # ============================================================
    def get_top_genes(self, limit=10, threshold=6.0):
        """
        从所有数据中筛选 AI_Score >= threshold 的基因，并按分数排序。
        返回字典: {GeneName: {details...}}
        """
        if self.combined_df.empty:
            return {}
        
        # 1. 筛选
        mask = self.combined_df['AI_Score'] >= threshold
        df_high = self.combined_df[mask].copy()
        
        if df_high.empty:
            return {}

        # 2. 排序 (分数降序 -> Log2FC绝对值降序)
        df_sorted = df_high.sort_values(by=['AI_Score', 'Log2FC_DEA'], ascending=[False, False])
        
        # 3. 去重 (如果一个基因在多个药物里都出现，保留分数最高的那个)
        df_unique = df_sorted.drop_duplicates(subset=['merge_key'], keep='first')
        
        # 4. 截取前 N 个
        top_df = df_unique.head(limit)
        
        # 5. 格式化输出
        result = {}
        for _, row in top_df.iterrows():
            gene = row['merge_key']
            # 提取 LLM 评价的一小段作为摘要
            summary = str(row.get('LLM_Response', ''))[:150].replace('\n', ' ') + "..."
            
            result[gene] = {
                "omics_score": float(row['AI_Score']),
                "log2fc": float(row['Log2FC_DEA']),
                "drug_source": row['Source_Drug'],
                "ai_summary": summary,
                "found_in_omics": True
            }
            
        return result

    # ============================================================
    #  功能 B: 查表验证 (Top-Down 路径终点)
    # ============================================================
    def check_gene_list(self, gene_list):
        """
        接收外部传入的基因列表 (来自 KG)，查询它们在组学数据中的表现。
        返回字典: {GeneName: {details...}}
        """
        result = {}
        if not gene_list:
            return result

        # 预处理：转大写
        query_genes = [str(g).upper() for g in gene_list]
        
        if self.combined_df.empty:
            # 如果没数据，全返回 Not Found
            for g in query_genes:
                result[g] = {"found_in_omics": False, "omics_score": 0}
            return result

        for gene in query_genes:
            # 查找匹配行
            matches = self.combined_df[self.combined_df['merge_key'] == gene]
            
            if not matches.empty:
                # 如果有多个匹配，取分数最高的
                best_match = matches.loc[matches['AI_Score'].idxmax()]
                
                result[gene] = {
                    "found_in_omics": True,
                    "omics_score": float(best_match['AI_Score']),
                    "log2fc": float(best_match['Log2FC_DEA']),
                    "drug_source": best_match['Source_Drug'],
                    "ai_summary": str(best_match.get('LLM_Response', ''))[:100] + "..."
                }
            else:
                result[gene] = {
                    "found_in_omics": False,
                    "omics_score": 0,
                    "comment": "Not detected or filtered out in DEA analysis"
                }
                
        return result

# --- 单元测试 ---
if __name__ == "__main__":
    # 确保路径存在，否则创建一个假的测试文件
    test_dir = "D:/Bit/tools/data/Final_LLM_Results"
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建一个模拟 CSV 用于测试 (如果你还没有运行之前的步骤)
    mock_csv = os.path.join(test_dir, "TestDrug_Final_Report.csv")
    if not os.path.exists(mock_csv):
        print("⚠️ 创建模拟测试数据...")
        data = {
            "merge_key": ["EGFR", "PDE4D", "TP53", "MYC"],
            "AI_Score": [8.5, 7.5, 9.0, 4.0],
            "Log2FC_DEA": [2.1, -1.9, 3.5, 0.5],
            "LLM_Response": ["EGFR is significant...", "PDE4D acts via cAMP...", "TP53 driver...", "Low confidence..."]
        }
        pd.DataFrame(data).to_csv(mock_csv, index=False)

    # 1. 初始化
    retriever = OmicsDataRetriever(data_dir=test_dir)
    
    # 2. 测试获取高分基因 (Bottom-Up)
    print("\n--- Test 1: Get Top Genes (Score >= 7.0) ---")
    top_genes = retriever.get_top_genes(limit=5, threshold=7.0)
    for g, info in top_genes.items():
        print(f"🧬 {g}: Score={info['omics_score']}, LogFC={info['log2fc']}")
        
    # 3. 测试查表 (Top-Down)
    print("\n--- Test 2: Check Specific List ---")
    query = ["EGFR", "TP53"]
    checks = retriever.check_gene_list(query)
    for g, info in checks.items():
        found = "✅ Found" if info['found_in_omics'] else "❌ Not Found"
        print(f"🔍 {g}: {found}")