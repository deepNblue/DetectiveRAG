"""
物证Reader Agent
专注从案件原始文本中提取物证细节和证据保管链
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent


class EvidenceReader(BaseAgent):
    """物证Reader — 专注提取物证细节、保管链、潜在关联"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="ASMR-EvidenceReader", config=config, llm_client=llm_client)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从案件文本中提取物证详细信息

        Args:
            input_data: {"case_text": str, "evidence": list}

        Returns:
            格式化的物证数据
        """
        self.log_processing(input_data)
        case_text = input_data.get("case_text", "")
        evidence_list = input_data.get("evidence", [])

        # 构建已知证据摘要
        evidence_info = ""
        for e in evidence_list:
            if isinstance(e, dict):
                evidence_info += f"- {e.get('name', e.get('type', '?'))}: {e.get('description', '')}\n"
            elif isinstance(e, str):
                evidence_info += f"- {e}\n"

        if not case_text:
            return self.format_output({"physical_evidence": [], "digital_evidence": [], "testimony": [], "connections": []})

        prompt = f"""你是一个专注物证分析的案件信息提取专家。请从以下案件文本中提取所有物证相关信息。

案件文本:
{case_text[:3000]}

已知证据列表:
{evidence_info if evidence_info else "无"}

请严格以JSON格式返回:
{{
    "physical_evidence": [
        {{
            "name": "物证名称",
            "type": "物证类型(毒物/指纹/DNA/工具/文件等)",
            "location": "发现位置",
            "condition": "状态描述",
            "handler": "谁经手/接触过",
            "chain_of_custody": "保管链(从产生到发现的完整路径)",
            "significance": "高/中/低",
            "links_to_suspect": "与哪个嫌疑人有关联"
        }}
    ],
    "digital_evidence": [
        {{
            "name": "数字证据名称",
            "type": "类型(监控/通话记录/电子锁/转账等)",
            "content": "证据内容",
            "timestamp": "时间戳",
            "reliability": "可靠/可能被篡改/不确定",
            "links_to_suspect": "关联嫌疑人"
        }}
    ],
    "testimony": [
        {{
            "witness": "证人姓名",
            "statement_summary": "证言摘要",
            "credibility": "高/中/低",
            "contradictions": "与其他证言的矛盾点",
            "links_to_suspect": "为谁提供不在场证明或指控谁"
        }}
    ],
    "connections": [
        {{
            "evidence_a": "证据A",
            "evidence_b": "证据B",
            "connection_type": "关联类型(时间/空间/因果关系等)",
            "connection_strength": "强/中/弱",
            "detail": "关联细节"
        }}
    ]
}}

要求:
1. 区分物理证据和数字证据
2. 特别注意保管链(chain of custody)中的断裂
3. 标注证据的可靠性和可能的篡改痕迹
4. 只返回JSON"""

        response = self.call_llm(prompt, temperature=0.2)
        parsed = self.extract_json_from_response(response)

        if parsed and isinstance(parsed, dict):
            physical = parsed.get("physical_evidence", [])
            digital = parsed.get("digital_evidence", [])
            testimony = parsed.get("testimony", [])
            connections = parsed.get("connections", [])
        else:
            physical, digital, testimony, connections = [], [], [], []

        self.logger.info(f"物证提取完成: {len(physical)}物理证据, {len(digital)}数字证据, {len(testimony)}证言")

        return self.format_output({
            "physical_evidence": physical,
            "digital_evidence": digital,
            "testimony": testimony,
            "connections": connections,
            "physical_count": len(physical),
            "digital_count": len(digital),
            "testimony_count": len(testimony),
            "connection_count": len(connections),
        })
