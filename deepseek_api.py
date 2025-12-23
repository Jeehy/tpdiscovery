# deepseek_api.py
# DeepSeek 统一调用封装
# - 支持 system_prompt、json_mode、retries 等高级参数
# - 若设置环境变量 DEEPSEEK_API_KEY，则调用真实接口
# - 否则使用本地 deterministic stub 以便离线开发

from dotenv import load_dotenv
import os
import json
import requests
import time

# === 全局配置 (只加载一次) ===
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


def call_llm(
    user_prompt: str,
    system_prompt: str = "你是严谨的科研助理，请尽量只输出 JSON。",
    model_name: str = "deepseek-chat",
    json_mode: bool = False,
    temperature: float = 0.3,
    timeout: int = 60,
    retries: int = 3,
    parse_json: bool = False
):
    """
    统一的 LLM 调用入口
    
    :param user_prompt: 用户输入
    :param system_prompt: 系统角色设定
    :param model_name: 模型名称
    :param json_mode: 是否强制 JSON 输出格式
    :param temperature: 温度参数 (越低越稳定)
    :param timeout: 超时时间
    :param retries: 重试次数
    :param parse_json: 是否自动解析返回的 JSON 为 Python 对象
    :return: 模型返回内容 (str 或 dict，取决于 parse_json)
    """
    if not DEEPSEEK_API_KEY:
        return _model_call_stub(user_prompt)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "temperature": temperature
    }
    
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    
    for attempt in range(retries):
        try:
            response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=timeout)
            
            if response.status_code != 200:
                print(f"⚠️ [API Error] {response.status_code}: {response.text}")
                if response.status_code >= 500:
                    time.sleep(2)
                    continue
                else:
                    return None if parse_json else ""
            
            content = response.json()["choices"][0]["message"]["content"]
            
            if parse_json:
                return json.loads(content)
            return content
            
        except json.JSONDecodeError as e:
            print(f"⚠️ [JSON Parse Error]: {e}")
            return None if parse_json else content
        except Exception as e:
            print(f"⚠️ [Network Error] Attempt {attempt+1}/{retries}: {e}")
            time.sleep(2)
    
    return None if parse_json else ""


def model_call(prompt: str, model_name: str = "deepseek-chat") -> str:
    """
    兼容旧接口：简单调用 (仅 user_prompt)
    """
    return call_llm(user_prompt=prompt, model_name=model_name)


def _model_call_stub(prompt: str) -> str:
    """
    简单 deterministic stub，用于离线开发和测试
    """
    p = prompt.lower()
    if "任务理解" in p or "task understanding" in p:
        return json.dumps({
            "topic": "liver cancer",
            "goal": "target_discovery",
            "known_databases": ["OpenTargets", "PubMed"],
            "suggested_start": ["query_opentargets", "run_omics"],
            "reason": "基于疾病名优先查询已知靶点，再验证组学"
        }, ensure_ascii=False)
    if "路径生成" in p or "path planner" in p or "generate" in p and "paths" in p:
        return json.dumps([
            {"path_id": "p1", "steps": ["query_opentargets", "run_omics", "search_pubmed"], "reason": "先查已知靶点再验证表达"},
            {"path_id": "p2", "steps": ["run_omics", "query_kg", "search_pubmed"], "reason": "从差异表达出发，匹配图谱与文献"}
        ], ensure_ascii=False)
    if "推理综合" in p or "synthesize" in p or "reasoning_chain" in p:
        return json.dumps({
            "reasoning_chain": [
                "OpenTargets 显示 TP53 与肝癌相关",
                "差异分析中 RPS6KA1 上调显著",
                "文献检索显示 TP53 在肝癌中多次被报道"
            ],
            "candidate_targets": ["TP53", "RPS6KA1"],
            "confidence": 0.88,
            "new_queries": ["查询 TP53 在 OpenTargets 中的关联证据"],
            "change_path": False
        }, ensure_ascii=False)
    if "反思" in p or "reflector" in p or "reflect" in p:
        return json.dumps({
            "consensus": ["TP53"],
            "converged": True,
            "suggested_paths": [],
            "new_queries": []
        }, ensure_ascii=False)
    return json.dumps({"note": "stub fallback"}, ensure_ascii=False)
