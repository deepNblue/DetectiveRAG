"""
刑侦视角专家Agent
从作案过程和证据链角度分析案件
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


class CriminalExpert(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """刑侦专家 — 从作案过程重建和证据链角度给出专业判断 (记忆增强版)"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-CriminalExpert", config=config, llm_client=llm_client)
        self._init_memory(self.name)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_processing(input_data)
        sk = input_data.get("structured_knowledge", {})
        search = input_data.get("search_results", {})
        suspects = input_data.get("suspects", [])

        suspect_names = [s.get("name", "?") if isinstance(s, dict) else s for s in suspects]

        timeline = sk.get("timeline", {}).get("data", {})
        person_rels = sk.get("person_relation", {}).get("data", {})
        evidence = sk.get("evidence", {}).get("data", {})

        motive_data = search.get("motive", {}).get("data", {}).get("motive_analysis", [])
        opportunity_data = search.get("opportunity", {}).get("data", {}).get("opportunity_analysis", [])
        temporal_data = search.get("temporal", {}).get("data", {}).get("temporal_contradictions", [])

        # 🧠 记忆增强: 获取相关的技能和历史经验
        case_type = input_data.get("case_type", "")
        memory_ctx = self.retrieve_memory_context(case_type=case_type, case_data=input_data)

        initial_prompt = f"""你是一位资深刑侦专家，擅长犯罪现场重建和证据链分析。

核心原则:
- 基于现有证据进行犯罪重建并锁定嫌疑人
- 当多条证据（动机、机会、能力、异常行为）共同指向同一人时，应果断给出结论
- 刑侦分析包含合理推断，不要求每个环节都完美无缺
- 如果证据链主体指向明确，不应因个别环节不足而回避判断

嫌疑人: {', '.join(suspect_names)}

人物关系:
{json.dumps(person_rels.get('relations', []), ensure_ascii=False, indent=2)[:600]}

时间线:
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

物证:
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:600]}

证言:
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:500]}

物证关联:
{json.dumps(evidence.get('connections', []), ensure_ascii=False, indent=2)[:500]}

时间矛盾:
{json.dumps(temporal_data, ensure_ascii=False, indent=2)[:500]}

动机分析:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:500]}

机会分析:
{json.dumps(opportunity_data, ensure_ascii=False, indent=2)[:500]}
{memory_ctx.get('context_text', '')}
请严格以JSON格式返回你的刑侦分析结论:
{{
    "culprit": "根据证据链指向的真凶",
    "confidence": 0.0-1.0,
    "crime_reconstruction": {{
        "preparation": "案前准备阶段分析",
        "execution": "作案过程重建",
        "cover_up": "掩饰和反侦查行为分析",
        "post_crime": "案后异常行为分析"
    }},
    "evidence_chain": ["完整证据链条"],
    "reasoning": "你的推理过程",
    "key_breakthrough": "案件突破口",
    "unresolved_questions": ["待解答的问题"]
}}

要求:
1. 重建完整犯罪过程（事前准备-实施-掩饰-案后行为）
2. 证据链必须逻辑自洽，允许合理推断填补证据之间的空白
3. 充分考虑作案人的反侦查意识和行为
4. 综合评估各嫌疑人的动机强度、作案机会、行为异常程度
5. 只返回JSON

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
            expert_role="刑侦专家",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "刑侦分析失败", 0

        self.logger.info(f"刑侦分析完成(多轮×{total_rounds}): 真凶={culprit}, 置信度={confidence}")

        # 🧠 记忆存储
        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="criminal",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["犯罪重建", "证据链分析"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "criminal_investigation",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
        })
