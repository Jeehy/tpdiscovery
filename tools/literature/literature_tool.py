import json
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.literature.retriever import LiteratureRetriever
from prompts import LITERATURE_DISCOVERY_ANALYSIS, LITERATURE_VALIDATION_ANALYSIS
from deepseek_api import call_llm


class LiteratureTool:
    """
    文献验证智能体 (Literature Agent)
    职责：
    1. 接收基因验证任务 (必须显式指定 mode)
    2. 调度 Retriever 获取原始数据
    3. 构建 Prompt 并调用 LLM 进行验证
    """
    
    def __init__(self):
        self.retriever = LiteratureRetriever()

    def verify_target(self, gene: str, disease: str, mode: str):
        """
        核心验证逻辑
        :param mode: "discovery" | "validation" (由上游强制指定，不再猜测)
        """
        # 1. 直接调用工具获取数据
        # 工具层会根据 mode 自动选择是查泛癌(Discovery)还是查直接关联(Validation)
        raw_docs = self.retriever.get_evidence(gene, disease, mode)
        
        if not raw_docs:
            return {
                "support_level": "No Evidence",
                "conclusion": f"No relevant literature found in {mode} mode.",
                "citations": []
            }

        # 2. 数据预处理 (Context Preparation)
        top_docs = raw_docs[:5]
        context_str = "\n".join([
            f"[{i+1}] Title: {d['source_metadata'].get('title', 'Unknown')}\n"  
            f"    Aspect: {d.get('search_aspect', 'general')}\n"             
            f"    Content: {d['content'][:500]}..." 
            for i, d in enumerate(top_docs)
        ])

        # 3. 构建 Prompt (根据 mode 选择完全不同的阅读策略)
        sys_prompt = "你是资深生物医学文献分析师，请严格输出JSON格式。"
        
        if mode == "discovery":
            # === Discovery Prompt: 寻找旁证 ===
            user_prompt = LITERATURE_DISCOVERY_ANALYSIS.format(
                gene=gene, disease=disease, context_str=context_str
            )
        else:
            # === Validation Prompt: 寻找实锤 ===
            user_prompt = LITERATURE_VALIDATION_ANALYSIS.format(
                gene=gene, disease=disease, context_str=context_str
            )

        print(f"  🧠 [LitAgent] Analyzing {gene} ({mode})...")
        try:
            llm_res_str = call_llm(user_prompt, system_prompt=sys_prompt, json_mode=True)
            res_json = json.loads(llm_res_str)
            
            # =========== 🛠️ 关键修改：回填原始证据 ===========
            # 将 Top Docs 的原始文本塞回返回结果中
            # 这样主程序就能拿到原始摘要了，索引号与 LLM 引用对应
            res_json['raw_evidence_snippets'] = [
                {
                    "index": f"[{i+1}]",
                    "title": d['source_metadata'].get('title', 'Unknown'),       
                    "citation": d['source_metadata'].get('citation', 'Unknown'), 
                    "url": d['source_metadata'].get('url', ''),                  
                    "abstract": d['content'],
                    "source": d.get('source', 'Local_DB')
                }
                for i, d in enumerate(top_docs)
            ]
            # ===============================================
            
            return res_json
        except Exception as e:
            print(f"  ⚠️ LLM Error: {e}")
            return {"error": "LLM Analysis Failed"}

    def run_batch_verification(self, gene_list: list, disease: str, mode: str, max_workers: int = 2, max_genes: int = 20, request_delay: float = 1.0):
        """
        批量运行入口
        :param mode: 必须显式传入 "discovery" 或 "validation"
        :param max_workers: 并行线程数 (默认2，避免 PubMed API 限流)
        :param max_genes: 最多验证的基因数量 (默认20)
        :param request_delay: 每次请求间隔秒数 (默认1.0秒，PubMed 限制约3次/秒)
        """
        
        if len(gene_list) > max_genes:
            print(f"⚠️ [LitAgent] 候选池过大 ({len(gene_list)})，只验证前 {max_genes} 个")
            gene_list = gene_list[:max_genes]
        print(f"\n📖 [LitAgent] 并行处理 {len(gene_list)} 个基因 ({max_workers} workers) [{mode.upper()}] mode...")
        results = {}
        
        # 预处理基因名
        genes_to_verify = [
            item['Gene'] if isinstance(item, dict) else item 
            for item in gene_list
        ]
        
        def verify_single(gene):
            return gene, self.verify_target(gene, disease, mode)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, g in enumerate(genes_to_verify):
                futures[executor.submit(verify_single, g)] = g
                # 本地检索几乎不需要 delay，给一点点只是为了日志不刷屏
                if i < len(genes_to_verify) - 1:
                    time.sleep(request_delay)
            
            for future in as_completed(futures):
                try:
                    gene, res = future.result()
                    results[gene] = res
                except Exception as e:
                    gene = futures[future]
                    print(f"  ⚠️ {gene} 验证失败: {e}")
                    results[gene] = {"error": str(e)}
        
        print(f"✅ [LitAgent] 完成 {len(results)} 个基因验证")
        return results