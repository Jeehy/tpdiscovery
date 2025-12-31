import logging
import re
import time
import numpy as np
import faiss
from typing import List, Dict, Tuple
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class LiteratureRetriever:
    """
    文献检索工具 (纯净本地版)
    
    职责：
    1. 管理本地 MongoDB (DMLLM_EMBEDDING) 和 FAISS 索引资源
    2. 生成 Discovery/Validation 模式的查询策略
    3. 执行 Hybrid Search (Vector + Keyword + Year Weighting)
    4. 返回带有完整溯源信息的文献列表
    """
    
    def __init__(self, host: str = "localhost", 
                 port: int = 27017, 
                 db_name: str = "bio", 
                 collection_name: str = "DMLLM_EMBEDDING"):
        self.host = host
        self.port = port
        self.db_name = db_name
        self.collection_name = collection_name
        
        # 资源占位
        self.model = None
        self.index = None
        self.doc_ids = []
        self.client = None
        self.collection = None

    def _connect_db(self):
        """连接数据库 (Lazy Load)"""
        if self.client: return
        try:
            self.client = MongoClient(host=self.host, port=self.port, serverSelectionTimeoutMS=2000)
            self.collection = self.client[self.db_name][self.collection_name]
            logger.info(f"✅ [Retriever] 已连接数据库: {self.db_name}.{self.collection_name}")
        except Exception as e:
            logger.error(f"DB Connection Error: {e}")

    def _ensure_resources(self):
        """加载模型与构建索引"""
        if self.model and self.index: return
        
        self._connect_db()
        
        try:
            # 1. 加载模型
            if not self.model:
                logger.info("⏳ 正在加载 Embedding 模型...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 2. 构建索引 (只加载向量和ID，不加载文本以节省内存)
            if self.collection is not None and self.index is None:
                logger.info("⏳ 正在构建本地 FAISS 索引 (这可能需要几秒钟)...")
                start_time = time.time()
                
                # 只查 vector 和 _id
                cursor = self.collection.find(
                    {"vector": {"$exists": True}}, 
                    {"vector": 1, "_id": 1} 
                )
                
                vectors = []
                self.doc_ids = [] # 用于从 FAISS index 映射回 MongoDB _id
                
                for doc in cursor:
                    vec = doc.get('vector')
                    if vec and len(vec) == 384: # 确保维度正确
                        vectors.append(vec)
                        self.doc_ids.append(doc['_id'])
                
                if vectors:
                    # 转为 float32 矩阵
                    vectors_np = np.array(vectors).astype('float32')
                    
                    # 归一化 (让内积等价于余弦相似度)
                    faiss.normalize_L2(vectors_np)
                    
                    # 创建 FAISS 索引 (Inner Product)
                    dimension = vectors_np.shape[1]
                    self.index = faiss.IndexFlatIP(dimension)
                    self.index.add(vectors_np)
                    
                    elapsed = time.time() - start_time
                    logger.info(f"✅ FAISS 索引构建完成！包含 {self.index.ntotal} 条向量，耗时 {elapsed:.2f}s")
                else:
                    logger.warning("⚠️ 数据库中没有找到有效的向量数据！")
                    
        except Exception as e:
            logger.warning(f"Local Resources Load Failed: {e}")

    def _calculate_keyword_score(self, query: str, text: str) -> float:
        """关键词重合度打分"""
        if not query or not text: return 0.0
        q_terms = set(re.findall(r'\w+', query.lower()))
        t_terms = set(re.findall(r'\w+', text.lower()))
        if not q_terms: return 0.0
        return len(q_terms.intersection(t_terms)) / len(q_terms)

    # === 核心检索逻辑 (FAISS + MongoDB回查 + 混合打分) ===
    def _search_local(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        执行步骤：
        1. Query -> Vector
        2. FAISS 检索 -> Top K IDs
        3. MongoDB 批量查详情 (Text, Metadata)
        4. 混合打分 (Vector + Keyword + Year)
        """
        if not self.index or not self.model: return []
        
        try:
            # 1. 向量编码
            q_vec = self.model.encode([query])
            q_vec = np.array(q_vec).astype('float32')
            faiss.normalize_L2(q_vec) # 归一化查询向量
            
            # 2. FAISS 搜索 (多取一点做重排)
            D, I = self.index.search(q_vec, top_k * 2)
            
            # 获取命中的 MongoDB ID 和 向量分数
            hit_ids = []
            vec_scores = {}
            
            for rank, idx in enumerate(I[0]):
                if idx == -1: continue
                mongo_id = self.doc_ids[idx]
                hit_ids.append(mongo_id)
                vec_scores[mongo_id] = float(D[0][rank]) # 记录 FAISS 分数
            
            if not hit_ids: return []

            # 3. 回查 MongoDB 获取文本详情
            cursor = self.collection.find(
                {"_id": {"$in": hit_ids}},
                {"text": 1, "paper_title": 1, "metadata": 1, "pmid": 1}
            )
            
            results = []
            for doc in cursor:
                doc_id = doc['_id']
                text = doc.get('text', '')
                title = doc.get('paper_title', 'Unknown')
                
                # --- 混合打分逻辑 ---
                vec_score = vec_scores.get(doc_id, 0.0)
                kw_score = self._calculate_keyword_score(query, text)
                
                # 权重: 向量 70%, 关键词 30%
                hybrid_score = (0.7 * vec_score) + (0.3 * kw_score)
                
                # 章节/年份加权
                multiplier = 1.0
                metadata = doc.get('metadata', {})
                year = str(metadata.get('year', ''))
                
                # 给最近 3-4 年的文献加分
                if year in ['2023', '2024', '2025', '2026']:
                    multiplier = 1.1
                
                final_score = round(hybrid_score * multiplier, 4)
                
                # 组装证据对象 (带原文链接)
                pmid = doc.get('pmid', 'N/A')
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != "N/A" else ""
                
                author = metadata.get('author', 'Unknown')
                journal = metadata.get('journal', 'Journal')
                citation = f"{author} et al., {journal} ({year})"

                results.append({
                    "content": text,
                    "score": final_score,
                    "source": "Local_DB",
                    "search_aspect": "unknown", # 在上层赋值
                    
                    # 提供完整的元数据给 Agent
                    "source_metadata": {
                        "title": title,
                        "pmid": pmid,
                        "url": url,
                        "citation": citation,
                        "year": year
                    },
                    # Markdown 引用格式
                    "reference": f"[{citation}]({url}) - {title}"
                })
            
            # 按最终分数排序
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Local search error: {e}")
            return []

    def _generate_queries(self, gene: str, disease: str, mode: str) -> List[Tuple[str, str]]:
        queries = []
        
        if mode == "discovery":
            # === 探索模式 (Discovery) ===
            # 策略：不搜肝癌，只搜泛癌种、机制、药物
            queries.append(("pan_cancer", f"{gene} AND (Cancer OR Tumor) AND Review"))
            queries.append(("drug_target", f"{gene} AND (Inhibitor OR Resistance)"))
            queries.append(("mechanism", f"{gene} AND (Signaling Pathways OR Mechanism)"))
        else:
            # === 验证模式 (Validation) ===
            # 策略：强制绑定肝癌关键词
            # 1. 直接关联
            queries.append(("direct_link", f"{gene} AND {disease}"))
            # 2. 临床预后
            queries.append(("clinical", f"{gene} AND Prognosis AND {disease}"))
            # 3. 特定耐药/机制 (Specific Mechanism) - 已补回
            queries.append(("mechanism", f"{gene} AND ({disease} OR HCC) AND (Resistance OR Metastasis)"))
            
        return queries

    def get_evidence(self, gene: str, disease: str = "liver cancer", mode: str = "discovery") -> List[Dict]:
        """
        获取证据的主入口
        """
        self._ensure_resources()
        queries = self._generate_queries(gene, disease, mode)
        
        all_results = []
        seen_pmids = set()
        
        for aspect, q_str in queries:
            logger.info(f"🔍 [Local Search] Aspect [{aspect}]: {q_str}")
            
            # 本地检索 Top 5
            results = self._search_local(q_str, top_k=5)
            
            for res in results:
                pmid = res['source_metadata'].get('pmid') 
                if pmid and pmid not in seen_pmids:
                    res['search_aspect'] = aspect # 标记是哪种角度搜出来的
                    all_results.append(res)
                    seen_pmids.add(pmid)
        
        # 再次按分数排序
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"✅ 共检索到 {len(all_results)} 条本地证据")
        return all_results
    
# === 自测代码 ===
if __name__ == "__main__":
    retriever = LiteratureRetriever()
    
    # 测试 Validation 模式，检查是否会执行 3 个策略 (含 specific mechanism)
    test_gene = "TP53"
    print(f"\n🚀 测试检索: {test_gene} (Validation Mode)...")
    
    results = retriever.get_evidence(test_gene, mode="validation")
    
    print(f"\n✅ 找到 {len(results)} 条证据：")
    for i, res in enumerate(results[:3]):
        print(f"\n--- [Result {i+1}] (Aspect: {res.get('search_aspect')}) ---")
        print(f"📄 {res['reference']}")
        print(f"📝 {res['content'][:100]}...")