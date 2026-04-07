"""
陪审员Agent (Juror) — 审判层普通判断者
代表"常识理性"视角，用普通人直觉审视证据

对应真实角色:
  - 人民陪审员: 以常识和理性判断案件
  - 普通市民: 不受专业训练影响，用朴素正义感判断

核心理念: "常识判断" — 用普通人的逻辑和直觉审视案件
不受法医/技侦等专业术语影响，看重"谁的行为最不合理"

v1: 审判层 — 记忆增强版
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent
from agents.memory.memory_mixin import MemoryEnhancedMixin
from agents.asmr.multi_round_mixin import MultiRoundMixin


class Juror(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """
    陪审员 — 审判层普通判断者

    专长:
    - 常识判断: 用普通人的逻辑推理谁最可疑
    - 行为合理性: 判断每个人的行为是否符合常理
    - 朴素正义感: 不受专业分析影响，凭直觉判断
    - 发现"不对劲": 关注那些"说不通"的地方
    
    设计目的:
    - 平衡专业分析的过度复杂化
    - 提供"普通人的声音"
    - 有时常识判断比复杂推理更准确
    """

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-Juror", config=config, llm_client=llm_client)
        self._init_memory(self.name)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_processing(input_data)
        sk = input_data.get("structured_knowledge", {})
        search = input_data.get("search_results", {})
        suspects = input_data.get("suspects", [])
        investigation_results = input_data.get("investigation_results", [])

        suspect_names = [s.get("name", "?") if isinstance(s, dict) else s for s in suspects]

        timeline = sk.get("timeline", {}).get("data", {})
        person_rels = sk.get("person_relation", {}).get("data", {})
        evidence = sk.get("evidence", {}).get("data", {})

        motive_data = search.get("motive", {}).get("data", {}).get("motive_analysis", [])
        opportunity_data = search.get("opportunity", {}).get("data", {}).get("opportunity_analysis", [])

        # 调查层结论汇总
        investigation_summary = self._summarize_investigations(investigation_results)

        # 🧠 记忆增强
        case_type = input_data.get("case_type", "")
        memory_ctx = self.retrieve_memory_context(case_type=case_type, case_data=input_data)

        initial_prompt = f"""你是一位普通公民，被选为人民陪审员参与此案的审理。

你不是法医、不是侦探、不是律师。你只是一个有常识、有判断力的普通人。
你用朴素的逻辑和直觉来分析: "如果这件事发生在我身边，谁最可疑？"

你可以参考调查层（法医、名侦探等）的专业分析，但你要用**自己的常识**来判断他们的结论是否合理。
专业分析有时会"过度解读"或"忽视常理"，你的职责就是用常识来检验。

嫌疑人: {', '.join(suspect_names)}

=== 调查层专业分析（参考，但用你的常识检验）===
{investigation_summary}

=== 案件事实 ===
时间线:
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

人物关系:
{json.dumps(person_rels.get('relations', []), ensure_ascii=False, indent=2)[:500]}

证言:
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:600]}

物证:
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:500]}

动机:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:400]}

作案机会:
{json.dumps(opportunity_data, ensure_ascii=False, indent=2)[:400]}
{memory_ctx.get('context_text', '')}
请以JSON格式返回你作为陪审员的判断:
{{
    "culprit": "你认为是凶手的人",
    "confidence": 0.0-1.0,
    "common_sense_analysis": {{
        "most_suspicious_behavior": "谁的行为最不符合常理",
        "motive_check": "谁有最合理的杀人动机（用常识判断）",
        "opportunity_check": "谁最有机会作案（用常识判断）",
        "something_doesnt_add_up": "案件中什么地方"说不通""
    }},
    "professional_opinion_check": {{
        "agree_with_experts": true/false,
        "reason": "你是否同意调查层多数专家的结论，为什么",
        "expert_blindspot": "专家们可能忽视了什么常理问题"
    }},
    "gut_feeling": "你的直觉告诉你谁是凶手，为什么",
    "reasoning": "你的推理过程（用通俗语言）",
    "key_judgment": "你最关键的判断"
}}

⚠️ 陪审员原则:
1. **常识判断** — 用普通人的逻辑，不要被专业术语迷惑
2. **独立思考** — 可以同意也可以反对专家意见
3. **关注行为** — 谁在案发前后的行为最不合理？
4. **朴素正义** — 谁的犯罪最"说得通"（动机+行为一致性）
5. 只返回JSON

⚠️ 姓名规范: culprit只写一个人名，去掉头衔，保持全名完整
"""

        # 🔄 多轮推理
        parsed = self.multi_round_reasoning(
            initial_prompt=initial_prompt,
            context=input_data,
            expert_role="陪审员",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "陪审员判断失败", 0

        self.logger.info(f"陪审员判断完成: 认为={culprit}, 置信度={confidence}")

        # 🧠 记忆存储
        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="juror",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["常识判断", "朴素正义", "独立思考"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "juror",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
        })

    def _summarize_investigations(self, investigation_results: List[Dict]) -> str:
        """汇总调查层各专家的结论"""
        if not investigation_results:
            return "（无调查层结论）"
        
        lines = []
        for result in investigation_results:
            data = result.get("data", result)
            perspective = data.get("perspective", "未知专家")
            culprit = data.get("culprit", "未知")
            confidence = data.get("confidence", 0)
            
            lines.append(f"【{perspective}】→ 认为凶手是 {culprit} (置信度={confidence:.1%})")
        
        return "\n".join(lines)
