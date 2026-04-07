"""
赫尔克里·波洛心理侦探专家 (Hercule Poirot Psychological Detective)
基于阿加莎·克里斯蒂笔下波洛的心理推理方法论

核心方法论提炼自原著作品:
  - 《东方快车谋杀案》(1934): 心理画像与一致性分析，从证言矛盾中锁定真相
  - 《罗杰疑案》(1926): 叙事可靠性分析，识别叙述者和证人的欺骗
  - 《无人生还》(1939): 心理压力测试法，观察嫌疑人面对信息时的反应
  - 《ABC谋杀案》(1936): 模式识别与伪装动机分析，识破连环案中的真实目标
  - 《尼罗河上的惨案》(1937): 情感关系分析与三角关系动机推理
  - 《帷幕》(1975): 犯罪心理深层分析，理解完美犯罪者的心理结构

专业能力:
  1. 心理一致性分析 (Psychological Consistency): 分析嫌疑人行为的内在一致性
  2. 谎言识别 (Lie Detection): 从证言中识别矛盾、遗漏和过度解释
  3. 情感动机推理 (Emotional Motive Analysis): 理解爱、恨、嫉妒、贪婪等情感驱动
  4. 信息不对称分析 (Information Asymmetry): 利用嫌疑人知道/不知道的信息判断
  5. 模式识别 (Pattern Recognition): 识别犯罪行为中的心理模式
  6. 人性洞察 (Human Nature Insight): 理解人性弱点与犯罪冲动的关系

核心理念: "灰色脑细胞" — 用心理学和人性的理解来解读案件

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


class PoirotAnalyst(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """波洛心理侦探专家 — 基于心理画像与人性洞察的推理方法论"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-PoirotAnalyst", config=config, llm_client=llm_client)
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

        initial_prompt = f"""你是基于赫尔克里·波洛方法论的心理侦探专家。你的分析风格基于阿加莎·克里斯蒂笔下波洛的推理哲学:

**核心方法论:**
1. **心理一致性分析** — 每个人都有其行为模式和心理逻辑。审查每个嫌疑人的言行是否与其性格、处境和已知行为模式一致。说谎者往往在细节上露出破绽——过于流畅的叙述反而是可疑的。
2. **谎言识别** — 仔细审查证言中的矛盾、遗漏和过度解释。注意: (a)不应知道却知道的信息 (b)应该知道却声称不知道的信息 (c)证言之间的细微不一致 (d)过度解释简单事实的倾向。
3. **情感动机推理** — 犯罪的根源往往在于人性的基本情感: 贪婪、嫉妒、恐惧、仇恨、爱情。理解嫌疑人之间的情感关系网络，找到最强情感驱动。表面动机可能是伪装，深层情感才是真相。
4. **信息不对称分析** — 分析每个人在案发前后"应该知道什么"和"实际知道了什么"。犯罪者往往在不经意间展现出只有作案者才知道的信息。无辜者的反应是真实的困惑，而有罪者的反应是精心设计的表演。
5. **模式识别** — 寻找行为中的规律性。犯罪者的准备行为（提前了解、获取工具、制造不在场证明）往往遵循可识别的模式。识别这种行为模式可以揭示预谋程度。
6. **人性洞察** — 理解人性在极端情况下的表现。无辜者的典型反应（恐惧、困惑、配合）与有罪者的典型反应（过度镇定、主动引导、急于解释）有本质区别。每个人都有"面具"，案件就是揭下面具的过程。

嫌疑人: {', '.join(suspect_names)}

人物关系(关注情感关系):
{json.dumps(person_rels.get('relations', []), ensure_ascii=False, indent=2)[:600]}

人物信息(关注性格特征):
{json.dumps(person_rels.get('persons', []), ensure_ascii=False, indent=2)[:600]}

时间线:
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

证言(仔细审查每一句):
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:800]}

物证:
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:600]}

时间矛盾:
{json.dumps(temporal_data, ensure_ascii=False, indent=2)[:500]}

动机分析:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:500]}
{memory_ctx.get('context_text', '')}
请严格以JSON格式返回你的心理侦探分析结论:
{{
    "culprit": "真凶",
    "confidence": 0.0-1.0,
    "psychological_consistency": {{
        "suspect_profiles": [
            {{"suspect": "嫌疑人", "behavioral_pattern": "行为模式分析", "consistency_check": "言行一致性评估", "deception_indicators": "欺骗指标"}}
        ]
    }},
    "lie_detection": {{
        "testimony_contradictions": ["证言矛盾点"],
        "omissions": ["可疑的遗漏"],
        "over_explanations": ["过度解释之处"],
        "information_leaks": ["不当信息泄露（不应知道却知道）"]
    }},
    "emotional_motives": {{
        "primary_emotion": "核心情感驱动",
        "relationship_dynamics": "关系动态分析",
        "hidden_tensions": "隐藏的紧张关系"
    }},
    "information_asymmetry": {{
        "who_knew_what": ["各嫌疑人的信息掌握分析"],
        "key_slip": "关键的信息泄露"
    }},
    "human_nature_insight": "基于人性理解的案件洞察",
    "reasoning": "完整的心理推理过程",
    "key_psychological_finding": "最关键的心理发现"
}}

要求:
1. 从人性角度理解每个嫌疑人的行为动机
2. 仔细审查证言的一致性，发现矛盾和遗漏
3. 分析情感关系网络中的紧张和矛盾
4. 注意谁"知道得太多"或"解释得太多"
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
            expert_role="赫尔克里·波洛",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "心理分析失败", 0

        self.logger.info(f"波洛心理分析完成(多轮×{total_rounds}): 真凶={culprit}, 置信度={confidence}")

        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="poirot",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["心理一致性", "谎言识别", "情感动机", "人性洞察"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "poirot_analysis",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
        })
