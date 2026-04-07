"""
检察官Agent (Prosecution Reviewer) — 审判层控方角色
从控方角度审查证据、提出指控、反驳辩方论点

对应真实角色:
  - 公诉检察官: 审查起诉、证据链完整性评估
  - 控方律师: 提出有罪论据，力争闭合证据链

核心理念: "有罪推定倾向" — 积极寻找指向嫌疑人的证据，构建完整指控链
与DefenseAttorney形成控辩对抗，Judge做中立裁判

v2: 审判层重写 — 接收调查层结论，从控方角度独立判断
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent
from agents.memory.memory_mixin import MemoryEnhancedMixin
from agents.asmr.multi_round_mixin import MultiRoundMixin


class ProsecutionReviewer(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """
    检察官 — 审判层控方角色

    专长:
    - 有罪推定思维: 积极构建有罪证据链
    - 证据链闭合: 动机→准备→实施→掩饰→事后行为
    - 反驳辩护: 预判并反驳辩方论点
    - 法律适用分析: 罪名认定、构成要件齐备性

    v2变化:
    - 从调查层移到审判层
    - 能看到调查层(法医、名侦探等)的分析结论
    - 专注于"从控方角度论证谁是凶手"
    """

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-ProsecutionReviewer", config=config, llm_client=llm_client)
        self._init_memory(self.name)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_processing(input_data)
        sk = input_data.get("structured_knowledge", {})
        search = input_data.get("search_results", {})
        suspects = input_data.get("suspects", [])
        investigation_results = input_data.get("investigation_results", [])

        suspect_names = [s.get("name", "?") if isinstance(s, dict) else s for s in suspects]

        evidence = sk.get("evidence", {}).get("data", {})
        timeline = sk.get("timeline", {}).get("data", {})
        person_rels = sk.get("person_relation", {}).get("data", {})

        motive_data = search.get("motive", {}).get("data", {}).get("motive_analysis", [])
        opportunity_data = search.get("opportunity", {}).get("data", {}).get("opportunity_analysis", [])
        capability_data = search.get("capability", {}).get("data", {}).get("ranking", [])
        contradiction_data = search.get("contradiction", {}).get("data", {})

        # 调查层结论汇总
        investigation_summary = self._summarize_investigations(investigation_results)

        # 🧠 记忆增强
        case_type = input_data.get("case_type", "")
        memory_ctx = self.retrieve_memory_context(case_type=case_type, case_data=input_data)

        initial_prompt = f"""你是一位资深刑事检察官，你的职责是**代表控方提出指控**。

你面前有:
1. 调查层（法医、技侦、名侦探等）的分析结论 ← 这是侦查机关提交的证据
2. 原始案件证据材料

你的立场是**有罪推定** — 积极寻找和构建指向嫌疑人的证据链。
你需要选择最有力的指控对象，构建完整的控诉论证。

嫌疑人: {', '.join(suspect_names)}

=== 调查层分析汇总 ===
{investigation_summary}

=== 原始证据材料 ===
物证:
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:800]}

数字证据:
{json.dumps(evidence.get('digital_evidence', []), ensure_ascii=False, indent=2)[:600]}

证言:
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:800]}

物证关联:
{json.dumps(evidence.get('connections', []), ensure_ascii=False, indent=2)[:500]}

时间线:
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

人物关系:
{json.dumps(person_rels.get('relations', []), ensure_ascii=False, indent=2)[:500]}

动机分析:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:500]}

作案机会:
{json.dumps(opportunity_data, ensure_ascii=False, indent=2)[:500]}

作案能力:
{json.dumps(capability_data, ensure_ascii=False, indent=2)[:400]}

矛盾/异常:
{json.dumps(contradiction_data, ensure_ascii=False, indent=2)[:400]}
{memory_ctx.get('context_text', '')}
请以JSON格式返回你的控方起诉意见:
{{
    "culprit": "你指控的嫌疑人",
    "confidence": 0.0-1.0,
    "prosecution_case": {{
        "primary_suspect": "首要指控对象",
        "charge": "建议罪名",
        "case_strength": "案件证据强度(强/中/弱)",
        "conviction_probability": "定罪概率评估(高/中/低)"
    }},
    "evidence_chain": {{
        "motive_evidence": "动机证据论证",
        "opportunity_evidence": "作案机会证据论证",
        "capability_evidence": "作案能力证据论证",
        "behavioral_evidence": "行为异常证据论证",
        "chain_completeness": "证据链完整性评估(完整/基本完整/有缺口)"
    }},
    "rebuttal_preparation": {{
        "anticipated_defenses": ["辩方可能提出的抗辩"],
        "counter_arguments": ["控方的反驳论据"]
    }},
    "reasoning": "你的控方论证推理过程（必须引用具体证据）",
    "key_judgment": "最关键的控方论点"
}}

⚠️ 控方原则:
1. **有罪推定** — 积极寻找指向嫌疑人的证据
2. **证据链闭合** — 动机+机会+能力+行为异常形成完整链条
3. **利用调查层结论** — 调查层的分析是你的有力证据
4. **预判辩护** — 提前准备反驳辩方论点
5. 但如果证据确实不足以定罪，也应诚实说明
6. 只返回JSON

⚠️ 姓名规范: culprit只写一个人名，去掉头衔，保持全名完整
"""

        # 🔄 多轮推理
        parsed = self.multi_round_reasoning(
            initial_prompt=initial_prompt,
            context=input_data,
            expert_role="检察官",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "控方审查失败", 0

        self.logger.info(f"检察官起诉意见: 指控={culprit}, 置信度={confidence}")

        # 🧠 记忆存储
        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="prosecution",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["有罪推定", "证据链闭合", "控方起诉"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "prosecution_review",
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
