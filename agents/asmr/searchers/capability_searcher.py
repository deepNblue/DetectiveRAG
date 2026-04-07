"""
能力搜索Agent
从结构化知识中主动推理搜索实施犯罪的能力
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent


class CapabilitySearcher(BaseAgent):
    """能力搜索Agent — 分析谁有实施犯罪的知识、技能和资源"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="ASMR-CapabilitySearcher", config=config, llm_client=llm_client)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析每个嫌疑人实施犯罪的能力

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

        evidence = sk.get("evidence", {}).get("data", {})
        person_rels = sk.get("person_relation", {}).get("data", {})

        context = f"""=== 物证分析(关注作案手段) ===
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:800]}

=== 人物背景(关注技能和资源) ===
{json.dumps(person_rels.get('persons', []), ensure_ascii=False, indent=2)[:800]}"""

        prompt = f"""你是一个专注犯罪能力分析的刑侦专家。根据以下案件结构化知识，分析每个嫌疑人是否有能力实施犯罪。

嫌疑人: {', '.join(suspect_names)}

案件知识:
{context}

原始案件文本(补充参考):
{case_text[:1500]}

请严格以JSON格式返回:
{{
    "capability_analysis": [
        {{
            "suspect": "嫌疑人姓名",
            "technical_capability": {{
                "score": 0.0-1.0,
                "detail": "是否具备实施犯罪的技术/技能"
            }},
            "resource_access": {{
                "score": 0.0-1.0,
                "detail": "是否能获取作案工具/材料(如毒物、钥匙等)"
            }},
            "knowledge_access": {{
                "score": 0.0-1.0,
                "detail": "是否了解相关专业知识(如密室机制、毒物特性等)"
            }},
            "overall_capability": 0.0-1.0,
            "key_capabilities": ["关键能力1", "关键能力2"],
            "capability_gaps": ["缺失的能力，需要同伙或特殊途径补充"]
        }}
    ],
    "ranking": ["按能力从高到低排列嫌疑人"],
    "key_insight": "关于作案能力的最关键发现"
}}

要求:
1. 从技术能力、资源获取、专业知识三个维度分析
2. 考虑是否需要同伙协助
3. 关注嫌疑人职业/背景与犯罪手法的匹配度
4. 只返回JSON"""

        response = self.call_llm(prompt, temperature=0.3)
        parsed = self.extract_json_from_response(response)

        if parsed and isinstance(parsed, dict):
            analysis = parsed.get("capability_analysis", [])
            ranking = parsed.get("ranking", [])
            insight = parsed.get("key_insight", "")
        else:
            analysis, ranking, insight = [], [], ""

        self.logger.info(f"能力搜索完成: {len(analysis)}人分析")

        return self.format_output({
            "capability_analysis": analysis,
            "ranking": ranking,
            "key_insight": insight,
        })
