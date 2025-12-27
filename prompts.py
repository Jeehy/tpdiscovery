# prompts.py
# 集中管理所有 LLM User Prompt 模板，避免代码冗余
# 注意：System Prompt 保留在各源文件中

# =====================================================
# Planner Agent Prompts
# =====================================================

PLANNER_TASK = """
你是生物医药科研规划师。请将用户意图转化为执行步骤(JSON)。

可用技能(Skill):
1. "run_omics_path": 数据驱动探索(Path A)。参数: threshold(默认6.0)。
2. "run_kg_path": 知识驱动探索(Path B)。
3. "run_validation": 验证特定基因。参数: genes(列表)。
4. "check_literature": 查文献。参数: genes("<auto>"表示自动使用上一步结果)。
5. "check_external": 查OpenTargets库。参数: genes("<auto>")。

用户输入: "{user_input}"

输出格式(JSON):
{{
  "steps": [
    {{"skill": "run_omics_path", "args": {{"threshold": 5.0}}}},
    {{"skill": "run_kg_path", "args": {{}}}},
    {{"skill": "check_literature", "args": {{"genes": "<auto>"}}}} 
  ]
}}
如果用户是验证任务(如"验证TP53")，请只使用 run_validation 和 check_literature。
"""


# =====================================================
# Knowledge Graph (KG) Tool Prompts
# =====================================================

KG_DISCOVERY_ANALYSIS = """
针对疾病 "{disease}" 分析以下候选靶点：
{facts_text}

任务：
1. 筛选出最有潜力的 Top 20 靶点
2. 用中文给出筛选理由

JSON 输出格式：
{{ 
    "omics_targets": ["基因A", "基因B", ...], 
    "evidence_map": {{ 
        "基因A": "详细的中文推理说明...", 
        "基因B": "..." 
    }} 
}}
"""

KG_VALIDATION_ANALYSIS = """
疾病背景：{disease}
证据汇总：
{facts_text}

任务：为每个基因生成科学假设（中文）

JSON 输出格式：
{{ 
    "gene_hypotheses": {{ 
        "基因A": "该基因可能通过...机制参与疾病发生", 
        "基因B": "..." 
    }} 
}}
"""


# =====================================================
# Literature Tool Prompts
# =====================================================

LITERATURE_DISCOVERY_ANALYSIS = """
目标基因：{gene}
研究背景：该基因是 {disease} 的潜在新靶点
检索模式：发现模式（寻找其他癌症/机制中的间接证据）

文献证据：
{context_str}

分析任务：
1. **可迁移性**：该基因在其他癌症（如肺癌、乳腺癌）中是否已被证实为驱动基因或药物靶点？
2. **机制关联**：该基因是否调控与 {disease} 相关的核心通路（如凋亡、EMT、血管生成）？

请用中文返回 JSON：
{{
    "lit_support_level": "间接证据-强（已在其他癌症中验证）" 或 "间接证据-弱",
    "lit_conclusion": "基于旁证，简要总结该基因在{disease}中的转化潜力...",
    "key_citations": ["[1]", "[3]"]  // 引用文献时必须使用方括号索引号如[1],[2]，对应上方文献证据的编号
}}
"""

LITERATURE_VALIDATION_ANALYSIS = """
目标基因：{gene}
研究背景：该基因是 {disease} 的候选靶点
检索模式：验证模式（寻找直接证据）

文献证据：
{context_str}

分析任务：
1. **直接证据**：文献中是否直接提及 {gene} 与 {disease} 的关联？
2. **临床相关性**：该基因是否与 {disease} 的预后、生存期或耐药性相关？

请用中文返回 JSON：
{{
    "lit_support_level": "直接证据-强" 或 "直接证据-弱",
    "lit_conclusion": "简要总结该基因在{disease}中的直接证据...",
    "key_citations": ["[1]", "[2]"]  // 引用文献时必须使用方括号索引号如[1],[2]，对应上方文献证据的编号
}}
"""


# =====================================================
# 报告中文总结 Prompts
# =====================================================

REPORT_SUMMARY_BATCH = """
你是资深生物信息学家，请为以下候选靶点生成中文摘要。

候选靶点数据：
{candidates_json}

任务：
为每个基因生成简洁的中文总结（每个50-100字），包括：
1. omics_summary_cn: 基于组学数据的中文解读（差异表达、药物响应等）
2. literature_summary_cn: 基于文献证据的中文总结（研究现状、临床潜力）

JSON 输出格式：
{{
    "基因A": {{
        "omics_summary_cn": "该基因在XX药物处理后显著上调(Log2FC=X.X)，提示...",
        "literature_summary_cn": "文献表明该基因在多种癌症中..."
    }},
    "基因B": {{ ... }}
}}
"""