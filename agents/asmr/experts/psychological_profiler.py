"""
心理画像专家Agent
从行为模式和心理动机角度分析案件
v2: 记忆增强 — 利用积累的推理技能和历史经验
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent
from agents.memory.memory_mixin import MemoryEnhancedMixin
from agents.asmr.multi_round_mixin import MultiRoundMixin


class PsychologicalProfiler(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """心理画像专家 — 从犯罪心理学角度给出行为模式分析 (记忆增强版)"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-PsychologicalProfiler", config=config, llm_client=llm_client)
        self._init_memory(self.name)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_processing(input_data)
        sk = input_data.get("structured_knowledge", {})
        search = input_data.get("search_results", {})
        suspects = input_data.get("suspects", [])

        suspect_names = [s.get("name", "?") if isinstance(s, dict) else s for s in suspects]

        person_rels = sk.get("person_relation", {}).get("data", {})
        evidence = sk.get("evidence", {}).get("data", {})
        motive_data = search.get("motive", {}).get("data", {}).get("motive_analysis", [])

        # 🧠 记忆增强
        case_type = input_data.get("case_type", "")
        memory_ctx = self.retrieve_memory_context(case_type=case_type, case_data=input_data)

        initial_prompt = f"""你是一位犯罪心理画像专家，擅长从行为模式推断犯罪者特征。请从犯罪心理学角度分析此案。

嫌疑人: {', '.join(suspect_names)}

人物性格特征:
{json.dumps(person_rels.get('persons', []), ensure_ascii=False, indent=2)[:800]}

人物关系(关注情感和冲突):
{json.dumps(person_rels.get('relations', []), ensure_ascii=False, indent=2)[:600]}

证言分析(关注言行不一致):
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:600]}

动机分析:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:500]}
{memory_ctx.get('context_text', '')}
请严格以JSON格式返回你的心理画像分析结论:
{{
    "culprit": "你认为的真凶",
    "confidence": 0.0-1.0,
    "psychological_profile": {{
        "personality_type": "犯罪者人格类型",
        "stress_response": "在压力下的反应模式",
        "decision_pattern": "决策风格(冲动型/预谋型)"
    }},
    "behavioral_analysis": [
        {{
            "suspect": "嫌疑人",
            "behavioral_signals": ["可疑行为信号"],
            "deception_indicators": ["说谎/隐瞒的迹象"],
            "stress_markers": "压力表现"
        }}
    ],
    "reasoning": "你的心理分析推理过程",
    "key_insight": "从心理学角度最关键的发现"
}}

要求:
1. 基于行为证据进行心理推断
2. 分析犯罪是冲动型还是预谋型
3. 关注证言中的心理破绽
4. 只返回JSON

⚠️ 姓名规范要求:
- culprit 字段只写一个人名，不要写多人
- 如果该人物有头衔（医生、博士、教授等），去掉头衔只保留姓名
- 如果人物全名包含间隔点（如"格里姆斯比·罗伊洛特"），保持全名完整，不要拆成两个人
- 同一人的不同称呼视为同一人（如"罗伊洛特"和"罗伊洛特医生"是同一个人）
"""

        # 🔄 多轮推理
        parsed = self.multi_round_reasoning(
            initial_prompt=initial_prompt,
            context=input_data,
            expert_role="犯罪心理画像师",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "心理分析失败", 0

        self.logger.info(f"心理画像完成: 真凶={culprit}, 置信度={confidence}")

        # 🧠 记忆存储
        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="psychological",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["心理画像", "行为分析"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "psychological_profiling",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
        })
