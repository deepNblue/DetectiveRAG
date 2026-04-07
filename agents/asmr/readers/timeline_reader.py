"""
时间线Reader Agent
专注从案件原始文本中提取时间序列和事件流
ASMR核心理念: 专职Agent主动提取 >> 被动chunk切分
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent


class TimelineReader(BaseAgent):
    """时间线Reader — 专注提取时间、事件序列、时序关系"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="ASMR-TimelineReader", config=config, llm_client=llm_client)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从案件文本中提取完整时间线

        Args:
            input_data: {"case_text": str, "case_type": str}

        Returns:
            格式化的时间线数据
        """
        self.log_processing(input_data)
        case_text = input_data.get("case_text", "")
        case_type = input_data.get("case_type", "modern")

        if not case_text:
            return self.format_output({"events": [], "gaps": [], "anomalies": []})

        prompt = f"""你是一个专注时间线分析的案件信息提取专家。请从以下{case_type}案件文本中提取所有时间相关信息。

案件文本:
{case_text[:3000]}

请严格以JSON格式返回:
{{
    "events": [
        {{
            "time": "具体时间(尽量精确到分钟)",
            "event": "发生了什么事",
            "actor": "谁做的/涉及的",
            "location": "在哪里",
            "certainty": "确认/推断/未知",
            "evidence_basis": "这个时间点的依据是什么"
        }}
    ],
    "gaps": [
        {{
            "time_range": "时间段",
            "description": "这段时间缺少什么记录",
            "affected_persons": ["涉及的人"]
        }}
    ],
    "anomalies": [
        {{
            "description": "时间上的矛盾或异常",
            "detail": "具体矛盾点",
            "involved_persons": ["涉及的人"],
            "significance": "高/中/低"
        }}
    ]
}}

要求:
1. events按时间排序
2. gaps标注时间记录中的空白
3. anomalies重点关注时间矛盾（ASMR的temporal reasoning核心）
4. 只返回JSON，不要其他文字"""

        response = self.call_llm(prompt, temperature=0.2)
        parsed = self.extract_json_from_response(response)

        if parsed and isinstance(parsed, dict):
            events = parsed.get("events", [])
            gaps = parsed.get("gaps", [])
            anomalies = parsed.get("anomalies", [])
        else:
            events, gaps, anomalies = [], [], []

        self.logger.info(f"时间线提取完成: {len(events)}个事件, {len(gaps)}个空白, {len(anomalies)}个异常")

        return self.format_output({
            "events": events,
            "gaps": gaps,
            "anomalies": anomalies,
            "event_count": len(events),
            "gap_count": len(gaps),
            "anomaly_count": len(anomalies),
        })
