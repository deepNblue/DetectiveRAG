"""
证据关联Agent
负责发现证据之间的关联，构建证据网络
"""

import json
from typing import Dict, List, Any
from loguru import logger

from .base_agent import BaseAgent


class EvidenceConnectorAgent(BaseAgent):
    """证据关联Agent - 发现证据间的隐藏联系"""
    
    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="证据关联Agent", config=config, llm_client=llm_client)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        关联证据
        
        Args:
            input_data: {
                "evidence_list": list,
                "suspects": list,
                "case_clues": dict
            }
        """
        self.log_processing(input_data)
        
        evidence_list = input_data.get("evidence_list", [])
        suspects = input_data.get("suspects", [])
        case_clues = input_data.get("case_clues", {})
        
        evidence_text = json.dumps(evidence_list, ensure_ascii=False) if isinstance(evidence_list, list) else str(evidence_list)
        suspects_text = json.dumps(suspects, ensure_ascii=False) if isinstance(suspects, list) else str(suspects)
        
        prompt = f"""请分析以下证据之间的关联关系。

证据列表:
{evidence_text[:1500]}

嫌疑人:
{suspects_text[:800]}

请以JSON格式返回:
{{
    "connections": [
        {{
            "evidence_a": "证据1描述",
            "evidence_b": "证据2描述", 
            "relation": "关联类型（因果/矛盾/印证/时间链/空间重合）",
            "strength": 0.0-1.0,
            "description": "关联说明"
        }}
    ],
    "contradictions": ["矛盾点1", "矛盾点2"],
    "chains": [
        {{
            "name": "证据链名称",
            "evidence_order": ["证据1", "证据2", "证据3"],
            "conclusion": "链式推理结论"
        }}
    ]
}}

只返回JSON。"""
        
        response = self.call_llm(prompt, temperature=0.3)
        parsed = self.extract_json_from_response(response)
        
        if parsed and isinstance(parsed, dict):
            connections = parsed.get("connections", [])
            contradictions = parsed.get("contradictions", [])
            chains = parsed.get("chains", [])
        else:
            # 降级：简单的共现关联
            connections = self._simple_connections(evidence_list)
            contradictions = []
            chains = []
        
        return self.format_output({
            "connections": connections,
            "contradictions": contradictions,
            "chains": chains,
            "connection_count": len(connections),
            "chain_count": len(chains)
        })
    
    def _simple_connections(self, evidence_list: list) -> list:
        """简单共现关联（降级方案）"""
        connections = []
        if isinstance(evidence_list, list):
            for i, ev_a in enumerate(evidence_list):
                for ev_b in evidence_list[i+1:]:
                    connections.append({
                        "evidence_a": str(ev_a)[:50],
                        "evidence_b": str(ev_b)[:50],
                        "relation": "待分析",
                        "strength": 0.3,
                        "description": "需要进一步分析"
                    })
                    if len(connections) >= 10:
                        return connections
        return connections
