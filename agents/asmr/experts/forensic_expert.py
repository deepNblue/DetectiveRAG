"""
法医视角专家Agent
从物证和死因角度分析案件
v2: 记忆增强 — 利用积累的推理技能和历史经验
v3: 多轮推理 — 初步分析→自我审视→深入调查→最终结论
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent
from agents.memory.memory_mixin import MemoryEnhancedMixin
from agents.asmr.multi_round_mixin import MultiRoundMixin


class ForensicExpert(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """法医专家 — 从物证科学和死因分析角度给出专业判断 (多轮推理版)"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-ForensicExpert", config=config, llm_client=llm_client)
        self._init_memory(self.name)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_processing(input_data)
        sk = input_data.get("structured_knowledge", {})
        search = input_data.get("search_results", {})
        suspects = input_data.get("suspects", [])

        suspect_names = [s.get("name", "?") if isinstance(s, dict) else s for s in suspects]
        evidence = sk.get("evidence", {}).get("data", {})
        timeline = sk.get("timeline", {}).get("data", {})

        motive_summary = json.dumps(
            search.get("motive", {}).get("data", {}).get("ranking", []), ensure_ascii=False)
        opportunity_summary = json.dumps(
            search.get("opportunity", {}).get("data", {}).get("ranking", []), ensure_ascii=False)

        case_type = input_data.get("case_type", "")
        memory_ctx = self.retrieve_memory_context(case_type=case_type, case_data=input_data)

        # ===== Round 1 Prompt (原始专业分析) =====
        initial_prompt = f"""你是一位资深法医专家，正在参与一起案件的调查。请从法医学和物证科学的角度给出你的专业判断。

嫌疑人: {', '.join(suspect_names)}

物证分析:
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:1000]}

数字证据:
{json.dumps(evidence.get('digital_evidence', []), ensure_ascii=False, indent=2)[:600]}

时间线关键节点:
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

其他分析员结论参考:
- 动机排名: {motive_summary}
- 机会排名: {opportunity_summary}
{memory_ctx.get('context_text', '')}
请严格以JSON格式返回你的法医分析结论:
{{
    "cause_of_death": "死因分析",
    "method": "作案手法(基于物证推断)",
    "culprit": "你认为的真凶",
    "confidence": 0.0-1.0,
    "reasoning": "你的推理过程(从物证到结论)",
    "key_evidence": ["支撑结论的关键物证"],
    "unresolved_questions": ["物证中仍无法解释的问题"],
    "alternative_theory": "是否有其他可能的解释"
}}

要求:
1. 一切以物证为基础，不做人身推测
2. 如果物证不足以得出结论，降低confidence
3. 必须考虑alternative theory
4. 只返回JSON

⚠️ 姓名规范要求:
- culprit 字段只写一个人名，不要写多人
- 如果该人物有头衔（医生、博士、教授等），去掉头衔只保留姓名
- 如果人物全名包含间隔点（如"格里姆斯比·罗伊洛特"），保持全名完整，不要拆成两个人
- 同一人的不同称呼视为同一人（如"罗伊洛特"和"罗伊洛特医生"是同一个人）
"""

        # ===== 多轮推理 =====
        parsed = self.multi_round_reasoning(
            initial_prompt=initial_prompt,
            context=input_data,
            expert_role="法医专家",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning = "未知", 0.2, "物证分析失败"
            total_rounds = 0

        self.logger.info(f"法医分析完成(多轮×{total_rounds}): 真凶={culprit}, 置信度={confidence}")

        # 🧠 记忆存储
        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="forensic",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["物证分析", "法医推理", "多轮推理"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "forensic",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
        })
