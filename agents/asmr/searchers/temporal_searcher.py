"""
时间推理搜索Agent (ASMR特色)
专注发现时间线中的矛盾、异常和不可解释的空隙
ASMR在LongMemEval中temporal reasoning达76.69%准确率
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent


class TemporalSearcher(BaseAgent):
    """时间推理搜索Agent — 分析时间矛盾，这是ASMR区别于传统RAG的核心能力"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="ASMR-TemporalSearcher", config=config, llm_client=llm_client)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        专门分析时间线中的矛盾和异常

        Args:
            input_data: {
                "structured_knowledge": dict,
                "suspects": list,
                "case_text": str
            }
        """
        self.log_processing(input_data)
        sk = input_data.get("structured_knowledge", {})
        suspects = input_data.get("suspects", [])
        case_text = input_data.get("case_text", "")

        suspect_names = [s.get("name", "?") if isinstance(s, dict) else s for s in suspects]

        timeline = sk.get("timeline", {}).get("data", {})
        evidence = sk.get("evidence", {}).get("data", {})

        context = f"""=== 完整时间线 ===
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)}

=== 时间空白 ===
{json.dumps(timeline.get('gaps', []), ensure_ascii=False, indent=2)}

=== 时间异常 ===
{json.dumps(timeline.get('anomalies', []), ensure_ascii=False, indent=2)}

=== 数字证据(带时间戳) ===
{json.dumps(evidence.get('digital_evidence', []), ensure_ascii=False, indent=2)[:800]}"""

        prompt = f"""你是一个专注时间推理的刑侦专家。这是ASMR系统的核心能力——发现时间线中传统向量检索无法发现的矛盾。

嫌疑人: {', '.join(suspect_names)}

案件时间知识:
{context}

原始案件文本(补充参考):
{case_text[:1500]}

请严格以JSON格式返回:
{{
    "temporal_contradictions": [
        {{
            "description": "时间矛盾描述",
            "person_involved": "涉及的人",
            "claimed_time": "声称的时间",
            "actual_time": "实际时间(根据证据)",
            "evidence": "矛盾的证据依据",
            "significance": "高/中/低",
            "implication": "这个矛盾意味着什么"
        }}
    ],
    "unexplained_windows": [
        {{
            "time_range": "时间段",
            "who_was_unaccounted": ["时间未被记录的人"],
            "what_could_have_happened": "可能发生了什么",
            "relevance_to_crime": "与犯罪的关联度"
        }}
    ],
    "timeline_reconstruction": {{
        "most_likely_sequence": ["按时间排列的最可能事件序列"],
        "confidence": 0.0-1.0,
        "key_assumptions": ["关键假设"]
    }},
    "alibi_breakdown": [
        {{
            "suspect": "嫌疑人",
            "alibi": "不在场声称",
            "verifiable": true/false,
            "breaks_at": "不在场证明在哪里断裂",
            "significance": "高/中/低"
        }}
    ],
    "key_insight": "时间分析中最关键的发现"
}}

要求:
1. 深度分析时间矛盾，不能只看表面时间
2. 重建最可能的时间线序列
3. 逐个检验嫌疑人的不在场证明
4. 特别关注证据之间的时间不一致性
5. 只返回JSON"""

        response = self.call_llm(prompt, temperature=0.3)
        parsed = self.extract_json_from_response(response)

        if parsed and isinstance(parsed, dict):
            contradictions = parsed.get("temporal_contradictions", [])
            windows = parsed.get("unexplained_windows", [])
            reconstruction = parsed.get("timeline_reconstruction", {})
            alibi_breakdown = parsed.get("alibi_breakdown", [])
            insight = parsed.get("key_insight", "")
        else:
            contradictions, windows, reconstruction, alibi_breakdown, insight = [], [], {}, [], ""

        self.logger.info(f"时间推理完成: {len(contradictions)}个矛盾, {len(windows)}个未解释窗口")

        return self.format_output({
            "temporal_contradictions": contradictions,
            "unexplained_windows": windows,
            "timeline_reconstruction": reconstruction,
            "alibi_breakdown": alibi_breakdown,
            "key_insight": insight,
        })
