import time
import logging
import re
import sys
import urllib.error
from Bio import Entrez
from pymongo import MongoClient
from tqdm import tqdm

# === 🛠️ 彻底修复 Windows 编码崩溃问题 ===
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception as e:
        pass

# === ⚙️ 配置区域 ===
MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME = "bio"
COLLECTION_NAME = "DMLLM"

Entrez.email = "826329938@qq.com" 
ENTREZ_API_KEY = None  

# 日志配置
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dmllm_collection.log", encoding='utf-8'), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MassCollector:
    def __init__(self):
        try:
            self.client = MongoClient(MONGO_HOST, MONGO_PORT)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            
            # 创建索引
            self.collection.create_index("pmid", unique=True)
            self.collection.create_index("source_tag")
            self.collection.create_index("processed")
            
            logger.info(f"✅ 已连接 MongoDB: {DB_NAME}.{COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"❌ 数据库连接失败: {e}")
            raise

    def run_strategy(self, search_queries: dict):
        total_strategies = len(search_queries)
        # 按年份排序执行，体验更好
        sorted_keys = sorted(search_queries.keys(), reverse=True)
        
        for idx, name in enumerate(sorted_keys, 1):
            query = search_queries[name]
            logger.info(f"\n🚀 [任务 {idx}/{total_strategies}] 启动: {name}")
            # logger.info(f"   Query: {query[:100]}...") # 日志太长可取消注释
            self._download_by_query(query, tag=name)

    def _get_search_session(self, query):
        """执行搜索并返回 Session 信息"""
        try:
            handle = Entrez.esearch(
                db="pubmed", term=query, usehistory="y", api_key=ENTREZ_API_KEY
            )
            res = Entrez.read(handle)
            handle.close()
            return {
                "count": int(res["Count"]),
                "webenv": res["WebEnv"],
                "query_key": res["QueryKey"]
            }
        except Exception as e:
            logger.error(f"   ⚠️ Search Session 获取失败: {e}")
            return None

    def _download_by_query(self, query, tag, batch_size=200):
        # 1. 初始搜索
        session = self._get_search_session(query)
        if not session: return

        count = session['count']
        webenv = session['webenv']
        query_key = session['query_key']

        if count == 0:
            logger.warning(f"   ⚠️ 未找到文献 (Count=0)，跳过。")
            return

        # 2. 断点续传计算
        existing_count = self.collection.count_documents({"source_tag": tag})
        start_index = 0
        if existing_count > 0:
            # 回退一个 batch 防止漏数据
            start_index = max(0, (existing_count // batch_size) * batch_size - batch_size)
            logger.info(f"   🔄 [断点续传] 库中已有 {existing_count} 条，从索引 {start_index}/{count} 继续...")
        else:
            logger.info(f"   🔎 命中 {count} 篇，准备下载...")

        # 3. 分批下载循环
        pbar = tqdm(total=count, initial=start_index, desc=f"📥 {tag}")
        
        current_start = start_index
        while current_start < count:
            success = self._fetch_and_save_with_refresh(
                current_start, batch_size, webenv, query_key, tag, query
            )
            
            if success:
                updated_count = min(batch_size, count - current_start)
                pbar.update(updated_count)
                current_start += batch_size
                time.sleep(0.3 if ENTREZ_API_KEY else 0.5)
            else:
                logger.warning(f"   🔄 Batch {current_start} 失败，刷新 Session 重试...")
                new_session = self._get_search_session(query)
                if new_session:
                    webenv = new_session['webenv']
                    query_key = new_session['query_key']
                    time.sleep(2)
                else:
                    logger.error("   ❌ Session 刷新失败，停止当前策略。")
                    break
        pbar.close()

    def _fetch_and_save_with_refresh(self, start, batch_size, webenv, query_key, tag, query):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                handle = Entrez.efetch(
                    db="pubmed", 
                    retstart=start, 
                    retmax=batch_size,
                    webenv=webenv, 
                    query_key=query_key,
                    rettype="medline", 
                    retmode="text", 
                    api_key=ENTREZ_API_KEY
                )
                data = handle.read()
                handle.close()
                
                if not data: return True 

                self._parse_and_insert(data, tag)
                return True

            except urllib.error.HTTPError as e:
                if e.code == 400:
                    logger.warning(f"   ⚠️ HTTP 400 (Bad Request). Session 失效。")
                    return False # 返回 False 请求外层刷新
                time.sleep(3 * (attempt + 1))
            except Exception as e:
                time.sleep(3 * (attempt + 1))
        
        return False

    def _parse_and_insert(self, raw_text, tag):
        records = raw_text.split("\n\n")
        docs = []
        for rec in records:
            if not rec.strip(): continue
            
            pmid_match = re.search(r"PMID- (\d+)", rec)
            if not pmid_match: continue
            pmid = pmid_match.group(1)
            
            if self.collection.count_documents({"pmid": pmid}, limit=1):
                continue

            ab_match = re.search(r"AB\s+-\s+(.*?)\n[A-Z]", rec, re.DOTALL)
            abstract = ab_match.group(1).replace("\n      ", " ") if ab_match else ""
            if len(abstract) < 50: continue 

            ti_match = re.search(r"TI\s+-\s+(.*?)\n[A-Z]", rec, re.DOTALL)
            title = ti_match.group(1).replace("\n      ", " ") if ti_match else "Unknown Title"

            year = self._extract_regex(r"DP\s+-\s+(\d{4})", rec)
            journal = self._extract_regex(r"TA\s+-\s+(.*?)\n", rec)
            author = self._extract_regex(r"AU\s+-\s+(.*?)\n", rec)

            doc = {
                "pmid": pmid,
                "paper_title": title,
                "text": abstract,
                "section": "Abstract",
                "source_tag": tag,
                "source_filename": "Local_DMLLM",
                "processed": False,
                "metadata": {
                    "year": year,
                    "journal": journal,
                    "author": author,
                    "citation": f"{author} et al., {year}, {journal}"
                },
                "crawled_at": time.time()
            }
            docs.append(doc)

        if docs:
            try:
                self.collection.insert_many(docs, ordered=False)
            except Exception:
                pass 

    def _extract_regex(self, pattern, text):
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).replace("\n      ", " ") if m else "Unknown"

# ==========================================
# 🎯 主程序：全策略生成逻辑
# ==========================================
if __name__ == "__main__":
    collector = MassCollector()
    
    # 1. 定义你的核心检索逻辑 (Base Queries)
    # 这些是你要找的所有方向，我们保持它们的完整性
    BASE_QUERIES = {
        # [Validation] 肝癌深度验证
        "Validation_Liver": '("Carcinoma, Hepatocellular"[MeSH] OR "Liver Neoplasms"[MeSH]) AND (Review[pt] OR Clinical Trial[pt])',
        
        # [Discovery] 泛癌种机制 (Review)
        "Discovery_PanCancer": '(Neoplasms[MeSH] AND (Signaling Pathways OR Molecular Mechanisms)) AND Review[pt]',
        
        # [Discovery] 耐药性与靶向 (顶刊)
        "Discovery_Drug_Resistance": '("Drug Resistance, Neoplasm"[MeSH] OR "Molecular Targeted Therapy") AND (Inhibitor OR Antagonist) AND ("Nature"[Journal] OR "Cell"[Journal] OR "Science"[Journal] OR "Cancer Cell"[Journal] OR "Hepatology"[Journal])',
        
        # [Discovery] 新兴热点 (铁死亡等)
        "Discovery_Emerging_Topics": '(Ferroptosis OR Pyroptosis OR "Immune Checkpoint" OR "Metabolic Reprogramming" OR "Liquid Biopsy") AND Neoplasms'
    }

    FINAL_STRATEGIES = {}
    
    # 2. 自动分年处理 (2015 - 2026)
    # 为什么所有策略都要分年？因为 "耐药性" 或 "泛癌" 的总数也极可能超过 1万条。
    # 分年是避免 HTTP 400 错误最保险的方法。
    START_YEAR = 2015
    END_YEAR = 2026 
    SPLIT_YEAR_THRESHOLD = 2020 # 从2020年开始拆分上半年和下半年
    
    print("📋 正在生成策略 (2020年后自动启用半年切分模式)...")
    
    print("📋 正在生成全维度分年检索策略...")
    
    for base_name, base_query in BASE_QUERIES.items():
        for year in range(START_YEAR, END_YEAR + 1):
            
            if year < SPLIT_YEAR_THRESHOLD:
                # === 模式 A: 整年 (适合老数据) ===
                key = f"{base_name}_{year}"
                time_filter = f' AND "{year}/01/01"[Date - Publication] : "{year}/12/31"[Date - Publication]'
                FINAL_STRATEGIES[key] = base_query + time_filter
                
            else:
                # === 模式 B: 半年拆分 (适合新数据，避开 10k 限制) ===
                # 上半年 (Part A)
                key_a = f"{base_name}_{year}_PartA" # Jan - Jun
                time_filter_a = f' AND "{year}/01/01"[Date - Publication] : "{year}/06/30"[Date - Publication]'
                FINAL_STRATEGIES[key_a] = base_query + time_filter_a
                
                # 下半年 (Part B)
                key_b = f"{base_name}_{year}_PartB" # Jul - Dec
                time_filter_b = f' AND "{year}/07/01"[Date - Publication] : "{year}/12/31"[Date - Publication]'
                FINAL_STRATEGIES[key_b] = base_query + time_filter_b

    print(f"🚀 策略生成完毕！")
    print(f"   - 总任务数: {len(FINAL_STRATEGIES)}")
    print(f"   - 说明: {SPLIT_YEAR_THRESHOLD}年及以后已拆分为 PartA/PartB 以确保小于 10000 条。")
    print("🔥 开始执行 (支持断点续传)...")
    
    collector.run_strategy(FINAL_STRATEGIES)
    
    print("\n✅ 所有采集任务执行完毕！")