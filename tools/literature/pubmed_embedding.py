import time
import logging
import sys
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === ⚙️ 配置区域 ===
MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME = "bio"

# 1. 输入：原始采集数据 (只读)
SOURCE_COLLECTION = "DMLLM"  

# 2. 输出：带向量的成品数据 (写入)
TARGET_COLLECTION = "DMLLM_EMBEDDING" 

# 3. 模型：必须与 retriever.py 保持一致
MODEL_NAME = 'all-MiniLM-L6-v2'        
BATCH_SIZE = 64  # 根据显存/内存调整 (CPU建议32-64, GPU可以128-256)

# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def run_vectorization():
    # --- 1. 连接数据库 ---
    try:
        client = MongoClient(MONGO_HOST, MONGO_PORT)
        db = client[DB_NAME]
        source_col = db[SOURCE_COLLECTION]
        target_col = db[TARGET_COLLECTION]
        
        # 为目标集合创建索引 (加速检索和去重)
        target_col.create_index("pmid", unique=True)
        target_col.create_index("source_tag")
        
        logger.info(f"✅ 数据库连接成功")
        logger.info(f"   📂 源数据 (Raw): {SOURCE_COLLECTION}")
        logger.info(f"   📂 目标库 (Vec): {TARGET_COLLECTION}")
        
    except Exception as e:
        logger.error(f"❌ 数据库连接失败: {e}")
        return

    # --- 2. 检查断点 (Smart Resume) ---
    logger.info("🔍 正在检查增量状态...")
    # 获取目标库里已存在的 PMID，放入内存 Set 中
    existing_cursor = target_col.find({}, {"pmid": 1})
    existing_pmids = set(doc['pmid'] for doc in existing_cursor if 'pmid' in doc)
    logger.info(f"   📊 目标库已包含 {len(existing_pmids)} 条数据 (将自动跳过)。")

    # --- 3. 统计任务量 ---
    total_source = source_col.count_documents({})
    logger.info(f"   📊 源数据共有 {total_source} 条。")
    
    if len(existing_pmids) >= total_source:
        logger.info("🎉 所有数据均已向量化，无需操作！")
        return

    # --- 4. 加载模型 ---
    logger.info(f"⏳ 正在加载模型 {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        device = model.device
        logger.info(f"✅ 模型加载成功 (运行设备: {device})")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        return

    # --- 5. 批量处理主循环 ---
    # 使用 tqdm 显示进度条，初始位置设为已完成的数量
    pbar = tqdm(total=total_source, initial=len(existing_pmids), desc="Vectorizing", unit="docs")
    
    # 游标遍历源数据
    source_cursor = source_col.find({}, batch_size=500)
    
    batch_docs = []
    
    for doc in source_cursor:
        pmid = doc.get('pmid')
        
        # [关键] 如果目标库有了，直接跳过 (断点续传核心)
        if pmid in existing_pmids:
            continue
            
        batch_docs.append(doc)
        
        # 攒够一个 Batch 就处理
        if len(batch_docs) >= BATCH_SIZE:
            _process_and_insert_batch(batch_docs, model, target_col)
            pbar.update(len(batch_docs))
            batch_docs = [] # 清空缓存

    # 处理剩余的尾巴
    if batch_docs:
        _process_and_insert_batch(batch_docs, model, target_col)
        pbar.update(len(batch_docs))

    pbar.close()
    logger.info("✅ 向量化任务全部完成！Agent 现在可以使用这些数据了。")

def _process_and_insert_batch(docs, model, target_col):
    """
    处理逻辑：
    1. 提取文本
    2. 计算向量
    3. 复制原始元数据 + 插入向量字段
    4. 写入新表
    """
    if not docs: return

    texts = []
    valid_docs = []
    
    # 1. 提取有效文本
    for d in docs:
        text_content = d.get('text', '')
        # 简单清洗：去除太短的无效文本
        if text_content and len(text_content.strip()) > 5:
            texts.append(text_content)
            valid_docs.append(d)
    
    if not texts: return

    try:
        # 2. 计算向量 (Embedding)
        # show_progress_bar=False 避免和外层 tqdm 冲突
        embeddings = model.encode(texts, batch_size=len(texts), show_progress_bar=False)
        
        # 3. 组装新文档
        docs_to_insert = []
        for original_doc, vec in zip(valid_docs, embeddings):
            # [核心步骤] 复制原始对象，确保保留 PMID, Title, Journal 等信息
            new_doc = original_doc.copy()
            
            # 删除原有的 _id，让 MongoDB 在新集合里生成新的，避免主键冲突
            if '_id' in new_doc:
                del new_doc['_id']
            
            # 注入向量字段
            new_doc['vector'] = vec.tolist()
            new_doc['vectorized_at'] = time.time()
            
            docs_to_insert.append(new_doc)
        
        # 4. 批量写入
        if docs_to_insert:
            # ordered=False 允许部分成功 (即使某条因意外重复报错，其他也能插进去)
            target_col.insert_many(docs_to_insert, ordered=False)
            
    except Exception as e:
        # 忽略 Duplicate Key Error (E11000)，只打印其他错误
        if "E11000" not in str(e):
            logger.error(f"⚠️ Batch Insert Error: {e}")

if __name__ == "__main__":
    run_vectorization()