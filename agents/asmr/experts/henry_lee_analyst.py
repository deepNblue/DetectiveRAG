"""
李昌钰博士鉴识科学专家 (Dr. Henry Lee Forensic Analyst)
基于李昌钰博士真实案件的鉴识科学方法论

核心方法论提炼自真实项目与著作:
  - 《重审证据》(Physical Evidence, 1982): 物证鉴识的基本原则
  - O.J.辛普森案(1995): DNA证据链审查、现场保全评估、证据污染识别
  - JonBenét Ramsey案(1996): 犯罪现场重建、痕迹物证分析
  - Scott Peterson案(2002): 河流漂流物证重建、时间线物证分析
  - 《重建犯罪现场》: 现场重建七步法
  - 《著名案件重新调查》: 冷案重启的物证重新审视方法

专业能力:
  1. 犯罪现场重建 (Crime Scene Reconstruction): 七步重建法
  2. 微量物证分析 (Trace Evidence Analysis): 纤维、毛发、体液、土壤
  3. 血迹形态分析 (Bloodstain Pattern Analysis): 从血迹形态推断事件
  4. 证据链完整性评估 (Chain of Custody Review): 证据可信度审查
  5. 现场保全评估 (Crime Scene Integrity): 识别证据污染和破坏
  6. 科学推理 (Scientific Reasoning): 基于物证的概率推理

核心理念: "让物证说话" — 物证不会说谎，关键在于正确解读

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


class HenryLeeAnalyst(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """李昌钰鉴识科学专家 — 基于真实鉴识科学方法论的物证分析与现场重建"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-HenryLeeAnalyst", config=config, llm_client=llm_client)
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

        initial_prompt = f"""你是基于李昌钰博士鉴识科学方法论的物证分析专家。你的分析风格基于李昌钰博士在全球8000多起案件中积累的鉴识科学经验:

**核心方法论:**
1. **犯罪现场重建七步法** — (1)收集所有物证 (2)建立时间线 (3)确定死因 (4)重建事件序列 (5)排除不可能的情况 (6)验证假设 (7)形成最终结论。严格按步骤推进，每步有物证支撑。
2. **微量物证分析** — 重视最细微的物证。毒物残留、纤维转移、微量体液、土壤附着物等都可能成为破案关键。"没有完美的犯罪，只有不充分的调查。"
3. **血迹/痕迹形态分析** — 从痕迹的形状、分布、方向推断动作和事件序列。每一处痕迹都是事件的记录。
4. **证据链完整性评估** — 审查证据从发现到呈堂的每个环节。识别证据是否被污染、破坏或遗漏。证据链断裂意味着该证据的证明力下降。
5. **现场保全评估** — 评估犯罪现场是否被妥善保全。识别案后干扰行为（如清理、伪装、移动尸体），区分原始现场与伪装现场。
6. **科学概率推理** — 基于物证进行概率分析，而非绝对判断。将多种物证的指向概率进行综合评估，得出整体结论。

嫌疑人: {', '.join(suspect_names)}

人物关系:
{json.dumps(person_rels.get('relations', []), ensure_ascii=False, indent=2)[:600]}

时间线:
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

物证:
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:800]}

证言:
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:500]}

时间矛盾:
{json.dumps(temporal_data, ensure_ascii=False, indent=2)[:500]}

动机分析:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:400]}
{memory_ctx.get('context_text', '')}
请严格以JSON格式返回你的鉴识科学分析结论:
{{
    "culprit": "真凶",
    "confidence": 0.0-1.0,
    "scene_reconstruction": {{
        "step1_evidence_inventory": "物证清单与分类",
        "step2_timeline": "基于物证的时间线重建",
        "step3_cause_analysis": "物证反映的事件因果",
        "step4_sequence": "事件序列重建",
        "step5_elimination": "排除的不可能情况",
        "step6_hypothesis_check": "假设验证",
        "step7_conclusion": "最终结论"
    }},
    "trace_evidence": {{
        "key_traces": ["关键微量物证及其意义"],
        "transfer_analysis": "物质转移分析（嫌疑人是否与现场有物理接触证据）",
        "contamination_check": "证据污染可能性评估"
    }},
    "evidence_chain_review": {{
        "integrity": "证据链完整性评估",
        "gaps": ["证据链缺口"],
        "strong_links": ["最可靠的物证环节"]
    }},
    "scientific_probability": {{
        "suspect_probability": [
            {{"suspect": "嫌疑人", "probability_assessment": "物证指向概率评估"}}
        ],
        "overall_reasoning": "综合概率推理"
    }},
    "reasoning": "完整的鉴识科学推理过程",
    "key_physical_finding": "最关键的物证发现"
}}

要求:
1. 分析必须基于物证，每项推论都要对应具体物证
2. 重视微量物证和容易忽略的细节
3. 评估证据链的完整性，指出断裂和缺口
4. 区分原始现场与可能的伪装
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
            expert_role="李昌钰博士",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "鉴识分析失败", 0

        self.logger.info(f"李昌钰鉴识分析完成(多轮×{total_rounds}): 真凶={culprit}, 置信度={confidence}")

        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="henry_lee",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["现场重建", "微量物证", "证据链评估", "科学概率推理"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "henry_lee_analysis",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
        })
