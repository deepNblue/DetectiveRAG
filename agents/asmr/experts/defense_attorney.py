"""
辩护律师Agent (Defense Attorney) — 审判层辩方角色
从无罪推定出发，为嫌疑人辩护、质疑控方证据链

对应真实角色:
  - 刑事辩护律师: 为嫌疑人辩护，寻找合理怀疑
  - 无罪推定的守护者

核心理念: "无罪推定" — 在排除合理怀疑之前，视所有嫌疑人为无辜
与ProsecutionReviewer形成控辩对抗，Judge做中立裁判

v2: 审判层重写 — 接收调查层结论，从辩方角度独立判断
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent
from agents.memory.memory_mixin import MemoryEnhancedMixin
from agents.asmr.multi_round_mixin import MultiRoundMixin


class DefenseAttorney(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """
    辩护律师 — 审判层辩方角色

    专长:
    - 无罪推定思维: 除非证据排除合理怀疑，否则视嫌疑人为无罪
    - 质疑证据链: 找出控方证据链的每一个薄弱环节
    - 构建合理怀疑: 为每位嫌疑人找到"不是他做的"的理由
    - 对抗确认偏差: 主动寻找"不支持主流假设"的证据
    - 替被忽视的嫌疑人辩护: 重点关注被多数人忽略的嫌疑人

    v2变化:
    - 从调查层移到审判层
    - 能看到调查层(法医、名侦探等)的分析结论
    - 专注于"从辩方角度质疑调查层结论"
    """

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-DefenseAttorney", config=config, llm_client=llm_client)
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
        temporal_data = search.get("temporal", {}).get("data", {}).get("temporal_contradictions", [])
        contradiction_data = search.get("contradiction", {}).get("data", {})

        # 调查层结论汇总
        investigation_summary = self._summarize_investigations(investigation_results)

        # 🧠 记忆增强
        case_type = input_data.get("case_type", "")
        memory_ctx = self.retrieve_memory_context(case_type=case_type, case_data=input_data)

        initial_prompt = f"""你是一位经验丰富的刑事辩护律师，你的职责是**无罪推定**和**寻找合理怀疑**。

你面前有:
1. 调查层（法医、技侦、名侦探等）的分析结论 ← 你需要质疑这些结论
2. 原始案件证据材料 ← 你需要寻找其中对嫌疑人有利的线索

你的立场是**无罪推定** — 质疑调查层的结论，寻找合理怀疑。

⚠️ 关键策略:
- 如果调查层多数人指向A，你要特别检查: "会不会大家都犯了同样的错误？"
- 仔细审查被多数人忽略的嫌疑人（排名第2、第3的），他们可能才是真凶
- 质疑证据的可靠性: 证言可能有偏见，物证可能有其他解释

嫌疑人: {', '.join(suspect_names)}

=== 调查层分析汇总（你需要质疑这些结论）===
{investigation_summary}

=== 原始证据材料 ===
人物关系:
{json.dumps(person_rels.get('relations', []), ensure_ascii=False, indent=2)[:600]}

时间线:
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

时间矛盾:
{json.dumps(temporal_data, ensure_ascii=False, indent=2)[:500]}

物证关联:
{json.dumps(evidence.get('connections', []), ensure_ascii=False, indent=2)[:500]}

证言:
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:600]}

动机排名:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:500]}

作案机会:
{json.dumps(opportunity_data, ensure_ascii=False, indent=2)[:500]}

矛盾/异常:
{json.dumps(contradiction_data, ensure_ascii=False, indent=2)[:400]}
{memory_ctx.get('context_text', '')}
请以JSON格式返回你的辩护分析:
{{
    "culprit": "基于合理怀疑排除后的判断：如果调查层多数指向A但你有合理怀疑指向B，就写B。如果对所有人都有合理怀疑，写'无法确定'",
    "confidence": "你的确信程度(0-1)：如果你对多数派结论只有弱支持，给低置信度(0.2-0.4)；如果你找到了推翻多数派的有力证据，给中等置信度(0.5-0.7)；如果你无法确定，给0.1-0.3",
    "defense_for_each": [
        {{
            "suspect": "嫌疑人",
            "innocence_arguments": ["为其辩护的无罪理由"],
            "alibi_reliability": "不在场证明的可靠程度(高/中/低)",
            "evidence_weaknesses": ["指向该嫌疑人的证据的薄弱点"],
            "reasonable_doubt_score": 0.0-1.0
        }}
    ],
    "investigation_critique": {{
        "majority_blindspot": "调查层多数意见可能忽视了什么",
        "confirmation_bias": "调查层可能存在的确认偏差",
        "overlooked_suspect": "被调查层忽略的嫌疑人及理由"
    }},
    "alternative_theory": "不同于调查层多数意见的另一种真相解释",
    "reasoning": "你的辩护推理过程",
    "key_defense": "你最重要的辩护论点"
}}

⚠️ 辩护原则:
1. **无罪推定** — 除非证据排除合理怀疑，否则视嫌疑人为无罪
2. **质疑调查层** — 调查层的结论不是真理，可能存在群体思维
3. **关注被忽略者** — 排名第2、第3的嫌疑人可能被冤枉，也可能才是真凶
4. **对抗确认偏差** — 主动寻找"不支持调查层主流结论"的证据
5. **保护无辜者** — 宁可放过，不可冤枉
6. ⚠️ **关键：你的culprit不一定要跟调查层多数一致！** 如果少数专家的意见更有说服力，就站在少数派这边
7. ⚠️ **置信度要真实反映你的不确定程度** — 不要因为"大家都选A"就给高置信度
8. 只返回JSON

⚠️ 姓名规范: culprit只写一个人名，去掉头衔，保持全名完整
"""

        # 🔄 多轮推理
        parsed = self.multi_round_reasoning(
            initial_prompt=initial_prompt,
            context=input_data,
            expert_role="辩护律师",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "辩护分析失败", 0

        self.logger.info(f"辩护分析完成: 最可能凶手={culprit}, 置信度={confidence}")

        # 🧠 记忆存储
        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="defense",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["无罪推定", "合理怀疑", "对抗确认偏差"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "defense_attorney",
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
            reasoning = data.get("reasoning", "")
            
            short_reasoning = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
            lines.append(f"【{perspective}】→ 真凶={culprit} (置信度={confidence:.1%})\n  {short_reasoning}")
        
        return "\n".join(lines)
