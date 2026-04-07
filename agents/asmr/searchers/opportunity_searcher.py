"""
机会搜索Agent
从结构化知识中主动推理搜索作案机会
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent


class OpportunitySearcher(BaseAgent):
    """机会搜索Agent — 分析谁有作案的时空条件"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="ASMR-OpportunitySearcher", config=config, llm_client=llm_client)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于摄取的结构化知识，分析每个嫌疑人的作案机会

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

        context = f"""=== 时间线(含空白和异常) ===
事件: {json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:800]}
时间空白: {json.dumps(timeline.get('gaps', []), ensure_ascii=False, indent=2)[:500]}
时间异常: {json.dumps(timeline.get('anomalies', []), ensure_ascii=False, indent=2)[:500]}

=== 数字证据(监控/门禁/通信) ===
{json.dumps(evidence.get('digital_evidence', []), ensure_ascii=False, indent=2)[:800]}"""

        prompt = f"""你是一个专注犯罪时空分析的刑侦专家。根据以下案件结构化知识，分析每个嫌疑人是否有作案的时空机会。

嫌疑人: {', '.join(suspect_names)}

案件知识:
{context}

原始案件文本(补充参考):
{case_text[:1500]}

请严格以JSON格式返回:
{{
    "opportunity_analysis": [
        {{
            "suspect": "嫌疑人姓名",
            "at_scene": true/false,
            "time_window": "可能的作案时间窗口",
            "location_access": "对案发地点的接触程度",
            "opportunity_score": 0.0-1.0,
            "alibi": "不在场证明(如有)",
            "alibi_verification": "不在场证明的可验证性(已验证/未验证/已推翻)",
            "key_evidence": ["支持/否定机会的关键证据"],
            "spatial_analysis": "空间活动轨迹分析"
        }}
    ],
    "ranking": ["按机会从大到小排列嫌疑人"],
    "key_insight": "关于作案机会的最关键发现"
}}

要求:
1. 严格依据时间线和空间证据分析
2. 特别关注时间空白段(gaps)中谁可能在场
3. 不在场证明必须标明验证状态
4. 只返回JSON"""

        response = self.call_llm(prompt, temperature=0.3)
        parsed = self.extract_json_from_response(response)

        if parsed and isinstance(parsed, dict):
            analysis = parsed.get("opportunity_analysis", [])
            ranking = parsed.get("ranking", [])
            insight = parsed.get("key_insight", "")
        else:
            analysis, ranking, insight = [], [], ""

        self.logger.info(f"机会搜索完成: {len(analysis)}人分析")

        return self.format_output({
            "opportunity_analysis": analysis,
            "ranking": ranking,
            "key_insight": insight,
        })
