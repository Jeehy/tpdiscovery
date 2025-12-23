"""tools/pubmed_tool.py - 在线 PubMed 检索工具 (增强版)"""

import logging
import re
import time
from typing import List, Dict
from Bio import Entrez

# 配置邮箱 (NCBI 要求)
Entrez.email = "826329938@qq.com" 

logger = logging.getLogger(__name__)

class PubMedTool:
    def __init__(self, email: str = None):
        if email:
            self.email = email
            Entrez.email = email

    def search(self, query: str, max_results: int = 3, retries: int = 3) -> List[Dict]:
        """
        使用 Biopython 查询 PubMed 并解析详细元数据 (作者、年份、期刊)
        """
        # logger.info(f"🔍 [PubMed] Searching: {query}")
        
        for attempt in range(retries):
            try:
                # Step 1: ESearch 获取 ID
                handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
                record = Entrez.read(handle)
                handle.close()
                id_list = record["IdList"]
                
                if not id_list:
                    return []

                # Step 2: EFetch 获取 MEDLINE 格式 (包含丰富元数据)
                handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
                records = handle.read().split("\n\n")
                handle.close()

                results = []
                for rec in records:
                    if not rec.strip(): continue
                    
                    # === 正则解析元数据 ===
                    # 标题
                    ti_match = re.search(r"TI\s+-\s+(.*?)\n[A-Z]", rec, re.DOTALL)
                    title = ti_match.group(1).replace("\n      ", " ") if ti_match else "Unknown Title"
                    
                    # 摘要
                    ab_match = re.search(r"AB\s+-\s+(.*?)\n[A-Z]", rec, re.DOTALL)
                    abstract = ab_match.group(1).replace("\n      ", " ") if ab_match else ""
                    
                    # 年份 (DP - Date of Publication)
                    dp_match = re.search(r"DP\s+-\s+(\d{4})", rec)
                    year = dp_match.group(1) if dp_match else "n.d."
                    
                    # 作者 (AU - Author) - 取第一个作者做引用
                    au_match = re.search(r"AU\s+-\s+(.*?)\n", rec)
                    author = au_match.group(1) if au_match else "Unknown"
                    
                    # 期刊 (TA - Journal Title Abbreviation)
                    ta_match = re.search(r"TA\s+-\s+(.*?)\n", rec)
                    journal = ta_match.group(1) if ta_match else "Journal"

                    # 构建引用字符串 (用于报告展示)
                    citation_str = f"{author} et al., {year}, {journal}"

                    if abstract:
                        results.append({
                            "content": abstract,
                            "source_metadata": {
                                "paper_title": title,
                                "section": "Abstract",
                                "filename": "PubMed Online",
                                "year": year,
                                "citation_str": citation_str, # 关键新增字段
                                "pmid": re.search(r"PMID- (\d+)", rec).group(1) if re.search(r"PMID- (\d+)", rec) else ""
                            },
                            "scores": {"final": 0.95}, # 在线结果默认高置信度
                            "source_type": "Online"
                        })
                return results

            except Exception as e:
                logger.warning(f"PubMed connection error (Attempt {attempt+1}/{retries}): {e}")
                time.sleep(2) # 失败等待
        
        return []

if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)
    tool = PubMedTool()
    res = tool.search("STAMBP liver cancer", max_results=2)
    for item in res:
        print(f"📄 {item['source_metadata']['citation_str']}")
        print(f"   {item['source_metadata']['paper_title']}")