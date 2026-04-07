"""
人物关系Reader Agent
专注从案件原始文本中提取人际网络和关系图
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent


class PersonRelationReader(BaseAgent):
    """人物关系Reader — 专注提取人际网络、情感关系、利益关系"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="ASMR-PersonRelationReader", config=config, llm_client=llm_client)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从案件文本中提取完整人物关系网络

        Args:
            input_data: {"case_text": str, "suspects": list}

        Returns:
            格式化的人物关系数据
        """
        self.log_processing(input_data)
        case_text = input_data.get("case_text", "")
        suspects = input_data.get("suspects", [])

        # 构建嫌疑人信息摘要
        suspect_info = ""
        for s in suspects:
            if isinstance(s, dict):
                suspect_info += f"- {s.get('name', '?')}: {s.get('background', s.get('description', ''))}\n"

        if not case_text:
            return self.format_output({"persons": [], "relations": [], "network_summary": ""})

        prompt = f"""你是一个专注人物关系分析的案件信息提取专家。请从以下案件文本中提取所有人物关系信息。

案件文本:
{case_text[:3000]}

已知嫌疑人信息:
{suspect_info if suspect_info else "无"}

请严格以JSON格式返回:
{{
    "persons": [
        {{
            "name": "人物姓名",
            "role": "在案件中的角色(死者/嫌疑人/证人/其他)",
            "background": "简要背景",
            "key_traits": ["关键性格/行为特征"],
            "secrets": ["可能隐藏的秘密"]
        }}
    ],
    "relations": [
        {{
            "person_a": "人物A",
            "person_b": "人物B",
            "relation_type": "关系类型(亲属/同事/朋友/利益冲突/债务/情感等)",
            "sentiment": "正面/中性/负面/复杂",
            "tension_level": "高/中/低",
            "detail": "关系细节",
            "evidence": "关系依据"
        }}
    ],
    "network_summary": {{
        "most_central": "关系网络中最核心的人物",
        "most_isolated": "最孤立的人物",
        "key_conflicts": ["核心矛盾"],
        "hidden_alliances": ["可能的隐秘联盟"]
    }}
}}

要求:
1. 提取所有提到的人物，不仅限于嫌疑人
2. 关系要区分表面关系和深层关系
3. 特别关注利益冲突和情感纠葛
4. 只返回JSON"""

        response = self.call_llm(prompt, temperature=0.2)
        parsed = self.extract_json_from_response(response)

        if parsed and isinstance(parsed, dict):
            persons = parsed.get("persons", [])
            relations = parsed.get("relations", [])
            network_summary = parsed.get("network_summary", {})
        else:
            persons, relations, network_summary = [], [], {}

        self.logger.info(f"人物关系提取完成: {len(persons)}人, {len(relations)}条关系")

        return self.format_output({
            "persons": persons,
            "relations": relations,
            "network_summary": network_summary,
            "person_count": len(persons),
            "relation_count": len(relations),
        })
