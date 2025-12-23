import pandas as pd
import os
import glob
import time
import re
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from dotenv import load_dotenv

# === 配置 DeepSeek ===
BASE_URL = "https://api.deepseek.com/chat/completions"
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = "deepseek-chat"

# =========================
# 路径与并发配置
# =========================
INPUT_DIR = "D:/Bit/tools/data/LLM_Input_Ready"
OUTPUT_DIR = "D:/Bit/tools/data/Final_LLM_Results"
MAX_WORKERS = 4  # 测试时并发设小一点，方便观察
TEST_LIMIT = 9999  # ⚠️ 仅处理前 10 个基因


# =========================
# DeepSeek 调用函数（单文件内嵌）
# =========================
def call_deepseek(prompt, timeout=60):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an expert Bioinformatician assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    response = requests.post(
        BASE_URL,
        headers=headers,
        json=payload,
        timeout=timeout
    )

    if response.status_code != 200:
        print(f"\n[DeepSeek API Error] Status: {response.status_code}")
        print(response.text)
        response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


# =========================
# LLM 调用封装（带重试）
# =========================
def call_llm_api(prompt, gene_name, retries=3):
    for i in range(retries):
        try:
            return call_deepseek(prompt)
        except Exception as e:
            if i == retries - 1:
                return f"Error: {str(e)}"
            time.sleep(2)
    return "Error: Timeout"


# =========================
# 从 LLM 输出中提取分数
# =========================
def extract_score_robust(text):
    """
    【核心升级】鲁棒性极强的分数提取函数
    能识别: "Score: 9/10", "8.5/10", "Resistance Driver Score: 7", "**Score**: 9"
    """
    if not isinstance(text, str):
        return 0.0

    # 策略 1 (最准): 寻找 "数字/10" 的格式 (e.g., "8/10", "8.5 / 10")
    # pattern: 数字 + 0个或多个空格 + / + 0个或多个空格 + 10
    match_fraction = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", text)
    if match_fraction:
        score = float(match_fraction.group(1))
        # 防止提取出奇怪的数字 (比如日期 2023/10)
        if 0 <= score <= 10:
            return score

    # 策略 2 (备选): 寻找 "Score: 数字" 或 "Verdict: 数字"
    # pattern: Score/Verdict + 任意非数字字符 + 数字
    match_keyword = re.search(r"(?:Score|Verdict|Rating)[\D]*?(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match_keyword:
        score = float(match_keyword.group(1))
        if 0 <= score <= 10:
            return score
            
    return 0.0


# =========================
# 处理单个 CSV 文件
# =========================
def process_single_file(file_path):
    filename = os.path.basename(file_path)
    drug_name = filename.split('_')[0]
    save_path = os.path.join(OUTPUT_DIR, f"{drug_name}_Final_Report.csv")
    
    print(f"📘 正在处理: {drug_name}")
    
    # 读取输入
    df = pd.read_csv(file_path)
    
    # # 断点续传逻辑
    # if os.path.exists(save_path):
    #     df_existing = pd.read_csv(save_path)
    #     processed_genes = df_existing['merge_key'].tolist()
    #     df_to_process = df[~df['merge_key'].isin(processed_genes)].copy()
    #     if df_to_process.empty:
    #         print(f"   ✅ {drug_name} 已全部完成，跳过。")
    #         return
    #     print(f"   🔄 恢复进度：剩余 {len(df_to_process)} 个基因")
    # else:
    #     df_to_process = df.copy()
    # ⚠️ 关键步骤：只取前 10 个
    df_test = df.head(TEST_LIMIT).copy()
    print(f"   📊 原始数据有 {len(df)} 个基因，本次仅测试前 {len(df_test)} 个: {df_test['merge_key'].tolist()}")
    # 多线程处理
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_gene = {
            executor.submit(call_llm_api, row['LLM_Prompt'], row['merge_key']): row 
            for _, row in df_test.iterrows()
        }
        
        # 进度条
        for future in tqdm(as_completed(future_to_gene), total=len(df_test), desc=f"Analyzing {drug_name}"):
            row = future_to_gene[future]
            try:
                llm_response = future.result()
                score = extract_score_robust(llm_response)
                
                res_row = row.to_dict()
                res_row['LLM_Response'] = llm_response
                res_row['AI_Score'] = score
                
                results.append(res_row)
                
                # 打印简报，方便您实时看效果
                print(f"   ✅ {row['merge_key']}: AI评分 {score}/10")
                
            except Exception as e:
                print(f"❌ {row['merge_key']} 失败: {e}")

    # 保存结果
    res_df = pd.DataFrame(results)
    res_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"   🎉 测试报告已生成: {save_path}\n")

def run_batch_llm_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_files = glob.glob(os.path.join(INPUT_DIR, "*_LLM_Input_Deep.csv")) # 读取Deep版输入
    
    if not input_files:
        print("❌ 未找到输入文件，请检查 Step 5 是否运行成功。")
        return

    print(f"🚀 开始 LLM 小规模测试 (Top {TEST_LIMIT})...\n")
    
    for file_path in input_files:
        process_single_file(file_path)
        
    print(f"\n🏆 测试完成！请去 {OUTPUT_DIR} 查看 CSV 报告。")

if __name__ == "__main__":
    run_batch_llm_analysis()
