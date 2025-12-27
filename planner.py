# tpdiscovery/planner.py
import json
from deepseek_api import model_call
from prompts import PLANNER_TASK

class PlannerAgent:
    def plan(self, user_input: str) -> list:
        print(f"🧠 [Planner] 思考任务: {user_input}")
        
        # 1. 简单规则加速 (可选，为了稳定)
        if "验证" in user_input or "TP53" in user_input.upper():
            # 简单的验证逻辑
            import re
            match = re.search(r"[a-zA-Z0-9]+", user_input.replace("验证", ""))
            target = match.group(0) if match else "TP53"
            return [
                {"skill": "run_validation", "args": {"genes": [target]}},
                {"skill": "check_external", "args": {"genes": [target]}},
                {"skill": "check_literature", "args": {"genes": [target]}}
            ]
            
        # 2. 调用 LLM 进行规划
        try:
            prompt = PLANNER_TASK.format(user_input=user_input)
            response = model_call(prompt)
            # 清洗 markdown 格式
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            data = json.loads(response.strip())
            return data.get("steps", [])
        except Exception as e:
            print(f"⚠️ Planner LLM 出错，使用默认探索流程: {e}")
            # 兜底：默认跑全流程
            return [
                {"skill": "run_omics_path", "args": {}},
                {"skill": "run_kg_path", "args": {}},
                {"skill": "check_literature", "args": {"genes": "<auto>"}}
            ]