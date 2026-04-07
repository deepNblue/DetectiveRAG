"""
夏洛克·福尔摩斯推理专家 (Sherlock Holmes Analyst)
基于阿瑟·柯南·道尔笔下福尔摩斯的经典推理方法论

核心方法论提炼自原著作品:
  - 《血字的研究》: 演绎推理的基本方法，从结果反推原因
  - 《波希米亚丑闻》: 观察微小细节推断全局信息
  - 《银色马》: "排除一切不可能，剩下的无论多么不可思议，必是真相"
  - 《四签名》: 逆向推理链条，从已知事实逐步回溯
  - 《最后一案》: 对手心理预判和行为预测
  - 《布鲁斯-帕廷顿计划》: 从细节重建完整事件链

专业能力:
  1. 演绎推理 (Deductive Reasoning): 从已知事实推导必然结论
  2. 微观观察 (Keen Observation): 发现他人忽略的细节线索
  3. 排除法 (Elimination): 系统排除不可能的假设
  4. 逆向推理 (Abductive Reasoning): 从结果推导最可能的原因
  5. 行为预判 (Behavioral Prediction): 基于人物特征预测其行为
  6. 信息关联 (Information Synthesis): 将看似无关的线索建立联系

v1: 记忆增强版
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent
from agents.memory.memory_mixin import MemoryEnhancedMixin
from agents.asmr.multi_round_mixin import MultiRoundMixin


class SherlockAnalyst(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """福尔摩斯推理专家 — 基于演绎推理和微观观察的经典侦探方法论"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-SherlockAnalyst", config=config, llm_client=llm_client)
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

        case_type = input_data.get("case_type", "")
        memory_ctx = self.retrieve_memory_context(case_type=case_type, case_data=input_data)

        initial_prompt = f"""你是基于夏洛克·福尔摩斯方法论的经典侦探推理专家。你的分析风格基于柯南·道尔笔下福尔摩斯的六大核心推理方法:

**核心方法论:**
1. **演绎推理** — 从已确认的事实出发，通过逻辑链条推导必然结论。每一步推理都必须有确凿依据。
2. **微观观察** — 对案件中的每一个细节保持高度敏感。注意时间线中的微小矛盾、物证中的异常特征、证言中的不一致之处。往往是微不足道的细节暴露了真相。
3. **排除法** — "排除一切不可能之后，剩下的无论多么不可思议，必定是真相。"系统性地评估每个嫌疑人，排除不具备作案条件的人。
4. **逆向推理** — 从已知的结果（被害人死亡、现场状态）出发，逆向推导最可能的事件序列和因果关系。
5. **行为预判** — 理解犯罪者的心理和习惯，预判其在犯罪前、中、后各阶段的行为模式。预谋犯罪者往往在案前有准备行为，案后有掩饰行为。
6. **信息关联** — 将看似无关的线索串联起来，发现隐藏的关联。一条搜索记录、一个时间差、一笔转账都可能是拼图的关键。

嫌疑人: {', '.join(suspect_names)}

人物关系:
{json.dumps(person_rels.get('relations', []), ensure_ascii=False, indent=2)[:600]}

时间线:
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

物证:
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:600]}

证言:
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:500]}

时间矛盾:
{json.dumps(temporal_data, ensure_ascii=False, indent=2)[:500]}

动机分析:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:500]}

机会分析:
{json.dumps(opportunity_data, ensure_ascii=False, indent=2)[:500]}
{memory_ctx.get('context_text', '')}
请严格以JSON格式返回你的推理结论:
{{
    "culprit": "真凶",
    "confidence": 0.0-1.0,
    "deductive_chain": {{
        "step1_facts": "已确认的客观事实",
        "step2_elimination": "排除不可能的嫌疑人及其理由",
        "step3_remaining": "剩余嫌疑人及各自的可能性",
        "step4_conclusion": "最终推论及逻辑依据"
    }},
    "micro_observations": ["发现的关键细节及其意义"],
    "abductive_reasoning": "从结果逆推最可能的事件序列",
    "behavioral_prediction": "犯罪者各阶段行为模式分析",
    "hidden_connections": "发现的隐藏关联线索",
    "reasoning": "完整的演绎推理过程",
    "key_insight": "最关键的推理发现"
}}

要求:
1. 推理每一步都要有事实依据，不凭空假设
2. 注意时间线中的微小矛盾和行为异常
3. 系统排除不具备条件的嫌疑人
4. 发现看似无关线索之间的隐藏关联
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
            expert_role="夏洛克·福尔摩斯",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "推理分析失败", 0

        self.logger.info(f"福尔摩斯推理完成: 真凶={culprit}, 置信度={confidence}")

        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="sherlock",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["演绎推理", "微观观察", "排除法", "逆向推理"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "sherlock_analysis",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
        })
