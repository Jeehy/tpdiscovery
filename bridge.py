# tpdiscovery/bridge.py
import logging
from explorers import ResearchExplorers 
from integrator import ValidationAgent
from tools.literature.literature_tool import LiteratureTool
# 假设你已按之前的步骤添加了 OpenTargetsTool，如果没有请注释掉
from tools.omics.opentargets_tool import OpenTargetsTool 

class ResearchBridge:
    def __init__(self):
        self.logger = logging.getLogger("ResearchBridge")
        # 实例化原有工具类
        self.explorer = ResearchExplorers()
        self.validator = ValidationAgent()
        self.lit_agent = LiteratureTool()
        self.opentargets = OpenTargetsTool()
        
        # === 技能注册表 ===
        # 将 "Agent指令" 映射到 "具体函数"
        self.skill_map = {
            "run_omics_path": self._run_path_a,        # 对应 Path A
            "run_kg_path": self._run_path_b,           # 对应 Path B
            "run_validation": self._run_validation,    # 对应 验证模式
            "check_literature": self._run_lit,         # 对应 文献检索
            "check_external": self._run_opentargets    # 对应 OpenTargets
        }

    def call_skill(self, skill_name: str, args: dict):
        """Executor 调用的统一接口"""
        func = self.skill_map.get(skill_name)
        if not func:
            return {"status": "error", "message": f"Skill {skill_name} not found"}
        
        print(f"   🔧 [Bridge] 执行技能: {skill_name} | 参数: {list(args.keys())}")
        try:
            result = func(args)
            # 包装返回，标记来源，方便 Synthesizer 识别
            return {"status": "success", "data": result, "source": skill_name}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _run_path_a(self, args):
        # 调用 explorers.py 的 run_path_omics_driven
        return self.explorer.run_path_omics_driven(
            threshold=args.get("threshold", 6.0),
            disease=args.get("disease", "liver cancer")
        )

    def _run_path_b(self, args):
        # 调用 explorers.py 的 run_path_kg_driven
        return self.explorer.run_path_kg_driven(
            disease=args.get("disease", "liver cancer")
        )

    def _run_validation(self, args):
        # 兼容参数差异
        genes = args.get("genes", [])
        if isinstance(genes, str): genes = [genes]
        return self.explorer.run_validation_deep_diven(
            target_list=genes, 
            disease=args.get("disease", "liver cancer")
        )

    def _run_lit(self, args):
        genes = args.get("genes", [])
        if not genes: return {}
        return self.lit_agent.run_batch_verification(
            gene_list=genes, 
            disease=args.get("disease", "liver cancer"),
            mode=args.get("mode", "auto")
        )

    def _run_opentargets(self, args):
        return self.opentargets.run(args)