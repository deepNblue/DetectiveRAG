"""
嫌疑人分析Agent
负责分析嫌疑人的动机、机会、能力，进行画像
"""

import json
from typing import Dict, List, Any
from loguru import logger

from .base_agent import BaseAgent


class SuspectAnalyzerAgent(BaseAgent):
    """嫌疑人分析Agent - MOT(Motive-Opportunity-Threat)分析"""
    
    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="嫌疑人分析Agent", config=config, llm_client=llm_client)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析嫌疑人
        
        Args:
            input_data: {
                "suspect_info": dict or str,
                "case_clues": dict,
                "evidence": list
            }
        """
        self.log_processing(input_data)
        
        suspect_info = input_data.get("suspect_info", {})
        case_clues = input_data.get("case_clues", {})
        evidence = input_data.get("evidence", [])
        
        if isinstance(suspect_info, str):
            suspect_desc = suspect_info
        elif isinstance(suspect_info, dict):
            suspect_desc = json.dumps(suspect_info, ensure_ascii=False)
        else:
            suspect_desc = str(suspect_info)
        
        # 用LLM分析
        prompt = f"""请对以下嫌疑人进行犯罪画像分析。

嫌疑人信息: {suspect_desc}

相关线索: {json.dumps(case_clues, ensure_ascii=False) if isinstance(case_clues, dict) else str(case_clues)[:500]}

证据: {json.dumps(evidence, ensure_ascii=False) if isinstance(evidence, list) else str(evidence)[:500]}

请以JSON格式返回分析:
{{
    "name": "嫌疑人名称",
    "motive": {{
        "type": "经济/情感/复仇/其他",
        "description": "动机描述",
        "score": 0.0-1.0
    }},
    "opportunity": {{
        "description": "机会描述",
        "score": 0.0-1.0
    }},
    "capability": {{
        "description": "能力描述", 
        "score": 0.0-1.0
    }},
    "overall_suspicion": 0.0-1.0,
    "key_evidence": ["关键证据1", "关键证据2"]
}}

只返回JSON。"""
        
        response = self.call_llm(prompt, temperature=0.4)
        parsed = self.extract_json_from_response(response)
        
        if parsed and isinstance(parsed, dict):
            analysis = parsed
        else:
            analysis = {
                "name": suspect_desc[:30],
                "motive": {"type": "未知", "description": "无法确定", "score": 0.5},
                "opportunity": {"description": "需要更多信息", "score": 0.5},
                "capability": {"description": "需要更多信息", "score": 0.5},
                "overall_suspicion": 0.5,
                "key_evidence": []
            }
        
        return self.format_output({
            "suspect_analysis": analysis,
            "suspicion_level": _classify_suspicion(analysis.get("overall_suspicion", 0.5))
        })


def _classify_suspicion(score: float) -> str:
    """分类嫌疑等级"""
    if score >= 0.8:
        return "高度嫌疑"
    elif score >= 0.6:
        return "中度嫌疑"
    elif score >= 0.4:
        return "低度嫌疑"
    else:
        return "基本排除"
