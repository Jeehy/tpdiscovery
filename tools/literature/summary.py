# deepseek_api.py
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

# 为了兼容旧代码的 summary 函数 (可选)
def summary(prompt: str) -> str:
    return call_deepseek(prompt, system_prompt="你是严谨的科研助理，请用中文简练总结。", json_mode=False)