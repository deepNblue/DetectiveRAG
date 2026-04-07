"""
法官Agent (Judge) — 审判层核心角色
中立裁判者，综合控辩双方意见，做出公正判断

对应真实角色:
  - 合议庭法官: 综合控辩双方证据和论点，做出裁决
  - 审判长: 主持审判程序，确保程序公正

核心理念: "中立、公正、以事实为依据、以法律为准绳"
与检察官(有罪推定)、辩护律师(无罪推定)形成三角审判结构

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


class Judge(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """
    法官 — 中立裁判者

    专长:
    - 中立裁判: 不偏不倚地审视控辩双方论点
    - 证据裁判: 以证据为基础，评估各方论点的说服力
    - 心证形成: 根据全案证据形成内心确信
    - 程序公正: 确保分析过程符合逻辑和法律规范
    - 自由心证: 在控辩对抗中独立形成判断
    
    与Adjudicator的区别:
    - Adjudicator是Stage 4的最终裁判，综合投票+推理树+审判团
    - Judge是Stage 3.2审判层的投票成员，只看调查层提供的证据材料
    """

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-Judge", config=config, llm_client=llm_client)
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
        capability_data = search.get("capability", {}).get("data", {}).get("ranking", [])
        contradiction_data = search.get("contradiction", {}).get("data", {})

        # 调查层结论汇总
        investigation_summary = self._summarize_investigations(investigation_results)

        # 🧠 记忆增强
        case_type = input_data.get("case_type", "")
        memory_ctx = self.retrieve_memory_context(case_type=case_type, case_data=input_data)

        initial_prompt = f"""你是一位经验丰富的刑事审判法官，你的职责是**中立裁判**。

你面前有以下材料:
1. 调查层（法医、技侦、名侦探等）的分析结论
2. 控方（检察官）的起诉意见
3. 辩方（辩护律师）的辩护意见
4. 原始案件证据材料

你的角色是**独立裁判者**，既不受控方影响，也不偏向辩方。
你需要独立审查所有材料，形成自己的心证。

⚠️ 特别注意:
- 统计调查层专家的意见分布 — 谁被多数人支持？谁是少数派？
- **少数派不一定错** — 如果少数派给出了具体证据链而多数派只是在"感觉像"，少数派可能更可信
- 如果某嫌疑人被5+个专家支持但证据链薄弱，而另一个嫌疑人只有3个专家支持但证据链扎实，后者更可信
- **证据链 > 专家数量**

嫌疑人: {', '.join(suspect_names)}

=== 调查层分析汇总 ===
{investigation_summary}

=== 原始证据材料 ===
物证:
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:600]}

时间线:
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

人物关系:
{json.dumps(person_rels.get('relations', []), ensure_ascii=False, indent=2)[:500]}

动机分析:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:400]}

作案机会:
{json.dumps(opportunity_data, ensure_ascii=False, indent=2)[:400]}

作案能力:
{json.dumps(capability_data, ensure_ascii=False, indent=2)[:300]}

矛盾/异常:
{json.dumps(contradiction_data, ensure_ascii=False, indent=2)[:300]}
{memory_ctx.get('context_text', '')}
请以JSON格式返回你的裁判意见:
{{
    "culprit": "你认定为凶手的人",
    "confidence": 0.0-1.0,
    "verdict": "有罪/无罪/证据不足",
    "reasoning": "你的裁判推理过程（必须引用具体证据）",
    "evidence_evaluation": {{
        "strongest_evidence": "指向真凶的最有力证据",
        "weakest_link": "证据链中最薄弱的环节",
        "key_contradiction": "关键矛盾点"
    }},
    "investigation_agreement": {{
        "agree_with_majority": true/false,
        "reason": "是否同意调查层多数意见及理由"
    }},
    "certainty_level": "确信/基本确信/倾向于/无法确定",
    "key_judgment": "你最关键的裁判判断"
}}

⚠️ 裁判原则:
1. **中立公正** — 不预设立场，以证据说话
2. **独立心证** — 可以同意也可以反对调查层多数意见
3. **排除合理怀疑** — 只有在排除合理怀疑后才认定某人有罪
4. **综合判断** — 综合调查层、控方、辩方的意见
5. **重视矛盾** — 如果证据存在重大矛盾，应降低置信度
6. **不要盲目跟从多数** — 真理有时在少数人手中
7. 只返回JSON

⚠️ 姓名规范: culprit只写一个人名，去掉头衔，保持全名完整
"""

        # 🔄 多轮推理
        parsed = self.multi_round_reasoning(
            initial_prompt=initial_prompt,
            context=input_data,
            expert_role="法官",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "法官裁判失败", 0

        self.logger.info(f"法官裁判完成: 认定={culprit}, 置信度={confidence}")

        # 🧠 记忆存储
        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="judge",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["中立裁判", "排除合理怀疑", "独立心证"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "judge",
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
            
            # 截取reasoning前200字
            short_reasoning = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
            lines.append(f"【{perspective}】→ 真凶={culprit} (置信度={confidence:.1%})\n  {short_reasoning}")
        
        return "\n".join(lines)
