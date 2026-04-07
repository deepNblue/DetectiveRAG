"""
线索提取Agent
负责从案例文本中提取关键线索、构建时间线
"""

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
from loguru import logger

from .base_agent import BaseAgent


class ClueExtractorAgent(BaseAgent):
    """线索提取Agent - 从案件中提取关键线索和时间线"""
    
    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="线索提取Agent", config=config, llm_client=llm_client)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取线索
        
        Args:
            input_data: {
                "case_text": str,
                "images": list,
                "case_type": str
            }
        """
        self.log_processing(input_data)
        
        case_text = input_data.get("case_text", "")
        case_type = input_data.get("case_type", "modern")
        
        if not case_text:
            return self.format_output({"error": "缺少案件文本", "timeline": [], "key_clues": []})
        
        # 用LLM提取线索
        prompt = f"""请从以下{case_type}案件中提取关键线索。

案件内容:
{case_text[:2000]}

请以JSON格式返回:
{{
    "key_clues": [
        {{"clue": "线索描述", "type": "物理证据/证人证言/数字证据/行为异常", "importance": 1-5}}
    ],
    "timeline": [
        {{"time": "时间点", "event": "事件描述", "involves": "涉及人物"}}
    ],
    "anomalies": ["异常点1", "异常点2"]
}}

只返回JSON，不要其他文字。"""
        
        response = self.call_llm(prompt, temperature=0.3)
        parsed = self.extract_json_from_response(response)
        
        if parsed and isinstance(parsed, dict):
            key_clues = parsed.get("key_clues", [])
            timeline = parsed.get("timeline", [])
            anomalies = parsed.get("anomalies", [])
        else:
            # 基于规则的简单提取
            key_clues = self._rule_based_extract(case_text)
            timeline = []
            anomalies = []
        
        return self.format_output({
            "key_clues": key_clues,
            "timeline": timeline,
            "anomalies": anomalies,
            "clue_count": len(key_clues),
            "case_type": case_type
        })
    
    def _rule_based_extract(self, text: str) -> List[Dict]:
        """基于规则的关键信息提取（LLM不可用时的降级方案）"""
        clues = []
        
        # 提取时间相关
        time_patterns = re.findall(r'(\d{1,2}[月日号]?\s*\d{0,2}[时:]?\d{0,2}分?)', text)
        if time_patterns:
            clues.append({"clue": f"时间线索: {', '.join(time_patterns[:5])}", "type": "时间", "importance": 3})
        
        # 提取金额
        money_patterns = re.findall(r'(\d+万?元|¥[\d,]+)', text)
        if money_patterns:
            clues.append({"clue": f"金额线索: {', '.join(money_patterns[:3])}", "type": "经济", "importance": 4})
        
        # 提取死亡/伤害描述
        if any(w in text for w in ["死", "尸体", "受伤", "中毒"]):
            clues.append({"clue": "案件涉及人身伤害/死亡", "type": "案件性质", "importance": 5})
        
        if not clues:
            clues.append({"clue": "无法自动提取，需人工分析", "type": "系统", "importance": 1})
        
        return clues
