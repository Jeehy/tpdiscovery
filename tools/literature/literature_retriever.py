import logging
import re
import numpy as np
import faiss
from typing import List, Dict, Tuple
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tools.literature.pubmed_tool import PubMedTool  # 复用已改好的 PubMedTool

logger = logging.getLogger(__name__)

class LiteratureRetriever:
    """
    文献检索工具 (纯净版)
    
    职责：
    1. 管理本地 MongoDB 和 FAISS 索引资源
    2. 生成 Discovery/Validation 模式的查询策略
    3. 执行 Hybrid Search (Local Vector + Online PubMed)
    4. 返回原始文献列表 (无 LLM 介入)
    """
    
    def __init__(self, host: str = "localhost", port: int = 27017, 
                 db_name: str = "bio", collection_name: str = "evidence_chunks"):
        self.host = host
        self.port = port
        self.db_name = db_name
        self.collection_name = collection_name
        
        # 资源占位
        self.model = None
        self.index = None
        self.doc_map = []
        self.client = None
        self.collection = None
        
        self.pubmed = PubMedTool()

    def _connect_db(self):
        """连接数据库 (Lazy Load)"""
        if self.client: return
        try:
            self.client = MongoClient(host=self.host, port=self.port, serverSelectionTimeoutMS=2000)
            self.collection = self.client[self.db_name][self.collection_name]
        except Exception as e:
            logger.error(f"DB Connection Error: {e}")

    def _ensure_resources(self):
        """加载模型与构建索引 (保留原逻辑)"""
        if self.model: return
        self._connect_db()
        
        try:
            # 1. 加载模型
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 2. 构建索引
            if self.collection is not None:
                cursor = self.collection.find(
                    {"vector": {"$exists": True}}, 
                    {"vector": 1, "text": 1, "section": 1, "paper_title": 1, "source_filename": 1}
                )
                vectors = []
                self.doc_map = []
                for doc in cursor:
                    vec = doc.get('vector')
                    if vec:
                        vectors.append(np.array(vec, dtype='float32'))
                        self.doc_map.append({
                            'text': doc.get('text', ''),
                            'section': doc.get('section', 'Unknown'),
                            'title': doc.get('paper_title', 'Unknown'),
                            'source': doc.get('source_filename', 'Local')
                        })
                if vectors:
                    self.index = faiss.IndexFlatIP(vectors[0].shape[0])
                    self.index.add(np.array(vectors))
                    logger.info(f"Local Index built with {len(vectors)} docs.")
        except Exception as e:
            logger.warning(f"Local Resources Load Failed: {e}")

    def _calculate_keyword_score(self, query: str, text: str) -> float:
        """关键词重合度打分 (保留原逻辑)"""
        if not query or not text: return 0.0
        q_terms = set(re.findall(r'\w+', query.lower()))
        t_terms = set(re.findall(r'\w+', text.lower()))
        if not q_terms: return 0.0
        return len(q_terms.intersection(t_terms)) / len(q_terms)

    def _search_local(self, query: str, top_k: int = 2) -> List[Dict]:
        if not self.index or not self.model: return []
        try:
            q_vec = self.model.encode([query])
            # 多取一点数据 (top_k * 2) 用于重排
            D, I = self.index.search(np.array(q_vec, dtype='float32'), top_k * 2)
            
            results = []
            for rank, idx in enumerate(I[0]):
                if idx == -1: continue
                doc = self.doc_map[idx]
                
                # --- 您的原始打分逻辑 ---
                # 1. 原始向量分
                vec_score = float(D[0][rank])
                # 2. 关键词分
                kw_score = self._calculate_keyword_score(query, doc['text'])
                # 3. 混合打分
                hybrid_score = (0.7 * vec_score) + (0.3 * kw_score)
                
                # 4. 章节加权
                section = str(doc.get('section', 'Unknown')).lower()
                multiplier = 1.0
                if any(x in section for x in ['result', 'discussion', 'conclusion']):
                    multiplier = 1.2
                elif 'abstract' in section:
                    multiplier = 1.1
                
                final_score = round(hybrid_score * multiplier, 4)
                # -----------------------

                results.append({
                    "content": doc['text'],
                    # 统一使用 'metadata' 格式以适配 Agent
                    "metadata": {
                        "title": doc['title'], 
                        "citation": f"Local: {doc['source']}",
                        "section": doc.get('section', 'Unknown')
                    },
                    "score": final_score, # 使用加权后的分数
                    "source": "Local"
                })
            
            # 按最终加权分数排序
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Local search error: {e}")
            return []

    def _generate_queries(self, gene: str, disease: str, mode: str) -> List[Tuple[str, str]]:
        queries = []
        
        if mode == "discovery":
            # === 探索模式 (Discovery) ===
            # 假设：该基因在肝癌中是未知的。
            # 策略：不搜肝癌，只搜泛癌种、机制、药物。
            
            # 1. 泛癌种关联 (Pan-Cancer)
            # 意图：寻找它在其他癌症（肺癌、乳腺癌等）中的致癌证据
            queries.append(("pan_cancer", f"{gene}[Title] AND (Cancer OR Tumor OR Carcinoma)"))
            
            # 2. 药物靶点潜力 (Druggability)
            # 意图：寻找是否有现成的抑制剂或耐药机制
            queries.append(("drug_target", f"{gene}[Title/Abstract] AND (Inhibitor OR Drug OR Resistance)"))
            
            # 3. 核心机制 (Mechanism)
            # 意图：寻找它参与的通用信号通路 (e.g., Wnt, PI3K)
            queries.append(("mechanism", f"{gene} signaling pathway function"))

        else:
            # === 验证模式 (Validation) ===
            # 假设：该基因与肝癌有强关联，需要确认。
            # 策略：强制绑定肝癌关键词。
            
            # 1. 直接关联 (Direct Link)
            queries.append(("direct_link", f"{gene}[Title] AND ({disease} OR Hepatocellular Carcinoma OR HCC)"))
            
            # 2. 临床预后 (Clinical)
            queries.append(("clinical", f"{gene} AND ({disease} OR HCC) AND (Prognosis OR Survival OR Patient)"))
            
            # 3. 特定耐药/机制 (Specific Mechanism)
            queries.append(("mechanism", f"{gene} AND ({disease} OR HCC) AND (Resistance OR Metastasis)"))
            
        return queries

    def get_evidence(self, gene: str, disease: str = "liver cancer", mode: str = "discovery") -> List[Dict]:
        self._ensure_resources()
        queries = self._generate_queries(gene, disease, mode)
        # print(f"    🔍 [Retriever] {mode.upper()} Search for {gene} ({len(queries)} queries)...")
        
        combined_results = []
        seen_hashes = set()
        
        for aspect, q_str in queries:
            k_online = 3 if mode == "discovery" else 2
            k_local = 2
            
            # 1. 在线检索
            online_raw = self.pubmed.search(q_str, max_results=k_online)
            
            # 2. 本地检索 (带加权)
            local_res = self._search_local(q_str, top_k=k_local)
            
            # 3. 格式化在线结果 (标准化清洗)
            formatted_online = []
            for item in online_raw:
                raw_meta = item.get('source_metadata', {})
                formatted_online.append({
                    "content": item['content'],
                    "metadata": {
                        "title": raw_meta.get('paper_title', 'Unknown Title'), 
                        "citation": raw_meta.get('citation_str', 'PubMed')
                    },
                    "score": 0.9, 
                    "source": "Online"
                })
            
            # 4. 合并去重
            for item in formatted_online + local_res:
                h = hash(item['content'][:100])
                if h not in seen_hashes:
                    item['aspect'] = aspect 
                    combined_results.append(item)
                    seen_hashes.add(h)
        
        # 最终再次排序 (确保加权后的本地结果能和在线结果正确竞争)
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return combined_results