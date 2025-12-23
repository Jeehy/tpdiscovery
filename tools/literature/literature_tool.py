import json
from tools.literature.literature_retriever import LiteratureRetriever
import requests, time, os
from dotenv import load_dotenv

# === 配置 DeepSeek ===
BASE_URL = "https://api.deepseek.com/chat/completions"
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

def call_deepseek(user_prompt: str, system_prompt: str = "You are a helpful assistant.", json_mode: bool = False, timeout: int = 60, retries: int = 3) -> str:
    """
    通用 DeepSeek 调用函数
    :param user_prompt: 用户输入
    :param system_prompt: 系统设定 (角色/任务约束)
    :param json_mode: 是否强制输出 JSON 格式
    :param timeout: 超时时间
    :param retries: 重试次数
    :return: 模型返回的文本内容 (如果是 JSON 模式，通常需要 json.loads 解析)
    """
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "temperature": 0.3 # 保持低温度以获得稳定结果
    }
    
    # 关键修改：支持 JSON Mode
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(retries):
        try:
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=timeout)
            
            if response.status_code != 200:
                print(f"⚠️ [API Error] {response.status_code}: {response.text}")
                if response.status_code >= 500: # 服务端错误可以重试
                    time.sleep(2)
                    continue
                else:
                    return "" # 客户端错误(4xx)直接返回空

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content

        except Exception as e:
            print(f"⚠️ [Network Error] Attempt {attempt+1}/{retries}: {e}")
            time.sleep(2)
            
    return ""

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
            f"[{i+1}] Title: {d['metadata']['title']}\n"
            f"    Aspect: {d.get('aspect', 'general')}\n"
            f"    Content: {d['content'][:500]}..." 
            for i, d in enumerate(top_docs)
        ])

        # 3. 构建 Prompt (根据 mode 选择完全不同的阅读策略)
        sys_prompt = "You are a Senior Bio-curator. Output strictly in JSON."
        
        if mode == "discovery":
            # === Discovery Prompt: 寻找旁证 ===
            user_prompt = f"""
            Target Gene: {gene}
            Context: Potential NOVEL target for {disease}.
            Search Mode: Discovery (Looking for indirect evidence in other cancers/mechanisms).
            
            Literature Evidence:
            {context_str}
            
            Task:
            1. **Translatability**: Is this gene a driver or drug target in OTHER cancers (e.g., Lung, Breast)?
            2. **Mechanism**: Does it regulate a core pathway (e.g., Apoptosis, EMT) that is relevant to {disease}?
            
            Return JSON:
            {{
                "lit_support_level": "Indirect-High (Proven in other cancers)" or "Low",
                "lit_conclusion": "Briefly summarize its potential for repurposing in {disease} based on side evidence.",
                "key_citations": ["Author, Year", ...]
            }}
            """
        else:
            # === Validation Prompt: 寻找实锤 ===
            user_prompt = f"""
            Target Gene: {gene}
            Context: Candidate target for {disease}.
            Search Mode: Validation (Looking for DIRECT evidence in {disease}).
            
            Literature Evidence:
            {context_str}
            
            Task:
            1. **Direct Evidence**: Is there direct mention of {gene} in {disease}?
            2. **Clinical Link**: Is it linked to prognosis, survival, or drug resistance in {disease}?
            
            Return JSON:
            {{
                "lit_support_level": "Strong (Direct Link)" or "Weak",
                "lit_conclusion": "Briefly summarize the direct evidence in {disease}.",
                "key_citations": ["Author, Year", ...]
            }}
            """

        print(f"  🧠 [LitAgent] Analyzing {gene} ({mode})...")
        try:
            llm_res_str = call_deepseek(user_prompt, sys_prompt, json_mode=True)
            res_json = json.loads(llm_res_str)
            
            # =========== 🛠️ 关键修改：回填原始证据 ===========
            # 将 Top Docs 的原始文本塞回返回结果中
            # 这样主程序就能拿到原始摘要了
            res_json['raw_evidence_snippets'] = [
                {
                    "title": d['metadata']['title'],
                    "citation": d['metadata'].get('citation', 'Unknown'),
                    "abstract": d['content'], # 保留完整摘要
                    "source": d.get('source', 'Online')
                }
                for d in top_docs
            ]
            # ===============================================
            
            return res_json
        except Exception as e:
            print(f"  ⚠️ LLM Error: {e}")
            return {"error": "LLM Analysis Failed"}

    def run_batch_verification(self, gene_list: list, disease: str, mode: str):
        """
        批量运行入口
        :param mode: 必须显式传入 "discovery" 或 "validation"
        """
        print(f"\n📖 [LitAgent] Batch processing {len(gene_list)} genes in [{mode.upper()}] mode...")
        results = {}
        
        for item in gene_list:
            # 兼容 item 是字典或字符串的情况
            gene = item['Gene'] if isinstance(item, dict) else item
            
            # 直接使用传入的全局 mode，不再看 Tier
            res = self.verify_target(gene, disease, mode)
            results[gene] = res
            
        return results