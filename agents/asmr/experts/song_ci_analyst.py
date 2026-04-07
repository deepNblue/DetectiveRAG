"""
宋慈法医鉴识专家 (Song Ci Forensic Analyst)
基于宋慈《洗冤集录》(1247) 的传统法医鉴识方法论

核心方法论提炼自《洗冤集录》原著:
  - 卷一《检覆总说》: 检验基本原则 — "遇有死人，多是立足不定，须是仔细检验"
  - 卷二《验尸》: 尸体检验系统方法 — 外部检验、内部检验、四季变化考量
  - 卷三/四《疑难杂说》: 疑难案件的鉴别方法 — 自杀vs他杀、中毒vs疾病鉴别
  - 卷五《验毒》: 毒物检测方法 — 银针验毒法、物证观察法
  - 卷六《验伤》: 伤痕检验 — 生前伤vs死后伤鉴别、凶器推断

专业能力:
  1. 系统尸检方法论: 由外至内、由表及里的检验体系
  2. 死因鉴别: 自然死亡/中毒/窒息/外伤/疾病的鉴别诊断
  3. 生前伤与死后伤鉴别: 伤痕的颜色、形态、组织反应差异
  4. 毒物分析: 古代毒物检测原理的现代延伸
  5. 疑难案件鉴别: 自杀vs他杀、意外vs谋杀的关键区分
  6. 证据保全意识: 重视第一现场的保护和原始证据的保全

核心理念: "洗冤泽物" — 洗刷冤屈，明辨是非，让死者说话

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


class SongCiAnalyst(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """宋慈法医鉴识专家 — 基于洗冤集录的传统法医鉴识方法论"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-SongCiAnalyst", config=config, llm_client=llm_client)
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

        initial_prompt = f"""你是基于宋慈《洗冤集录》方法论的传统法医鉴识专家。你的分析风格基于这部1247年问世的世界上第一部系统的法医学著作:

**核心方法论:**
1. **系统尸检法** — 由外至内、由表及里地进行系统检验。首先观察体表整体状态（姿势、衣着、表情），再逐步检查头部、躯干、四肢的每一处细节，最后检验内部器官。不放过任何异常。
2. **死因鉴别** — 综合分析多种死因的可能性。仔细区分: 自然疾病死亡 vs 投毒致死、意外窒息 vs 故意勒杀、自杀 vs 他杀伪装自杀。关注每一种死因的典型特征和不一致之处。
3. **生前伤与死后伤鉴别** — 伤痕的颜色、形态、组织反应是关键判据。生前伤有生活反应（出血、炎症、凝血），死后伤则无。判断伤痕是生前造成还是死后伪造。
4. **毒物分析** — 从胃内容物、血液、器官组织中分析毒物证据。关注中毒的典型症状与自然疾病的区别。中毒者往往有特殊的体表征象和脏器变化。
5. **疑难案件鉴别** — "仔细检验，勿致遗漏"是核心原则。面对看似简单的案件也要保持审慎，面对复杂的案件更要条分缕析。关键是找到"不合理之处"——看似自然死亡中隐藏的谋杀证据。
6. **证据保全** — 重视原始现场和原始证据的保全。案后是否有清理痕迹？尸体是否被移动？现场是否被人为改变？这些都是重要的分析线索。

嫌疑人: {', '.join(suspect_names)}

时间线:
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

物证(仔细检验每一项):
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:800]}

证言:
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:500]}

时间矛盾:
{json.dumps(temporal_data, ensure_ascii=False, indent=2)[:400]}

动机分析:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:400]}
{memory_ctx.get('context_text', '')}
请严格以JSON格式返回你的法医鉴识分析结论:
{{
    "culprit": "真凶",
    "confidence": 0.0-1.0,
    "systematic_examination": {{
        "external_examination": "体表检验发现（伤痕、痕迹、异常体征）",
        "internal_examination": "内部检验推断（毒物、病理变化）",
        "clothing_examination": "衣着检验（破损、附着物、穿着异常）",
        "scene_examination": "现场检验（位置、姿势、周围物品状态）"
    }},
    "cause_of_death_analysis": {{
        "primary_cause": "主要死因判断",
        "differential_diagnosis": [
            {{"cause": "可能死因", "supporting": "支持证据", "contradicting": "矛盾点", "likelihood": "可能性(高/中/低)"}}
        ],
        "final_determination": "最终死因判定及依据"
    }},
    "injury_analysis": {{
        "ante_mortem_injuries": ["生前伤分析"],
        "post_mortem_changes": ["死后变化分析"],
        "weapon_inference": "推断的作案工具/手段"
    }},
    "suspicious_findings": ["发现的可疑之处及其意义"],
    "evidence_preservation_issues": ["证据保全问题（是否有人为干扰痕迹）"],
    "reasoning": "完整的法医鉴识推理过程",
    "key_forensic_finding": "最关键的法医发现"
}}

要求:
1. 系统性地检验每一项物证，不遗漏细节
2. 仔细区分各种死因的可能性，给出判据
3. 识别伪装（他杀伪装自杀/意外/疾病）
4. 关注证据是否被人为干扰或破坏
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
            expert_role="宋慈",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "鉴识分析失败", 0

        self.logger.info(f"宋慈鉴识分析完成(多轮×{total_rounds}): 真凶={culprit}, 置信度={confidence}")

        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="song_ci",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["系统尸检", "死因鉴别", "毒物分析", "疑难鉴别"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "song_ci_analysis",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
        })
