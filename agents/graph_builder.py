"""
知识图谱构建Agent
负责将线索、嫌疑人、证据构建为知识图谱
"""

import json
from typing import Dict, List, Any
from loguru import logger

from .base_agent import BaseAgent


class GraphBuilderAgent(BaseAgent):
    """知识图谱构建Agent - 构建案件关系图谱"""
    
    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="知识图谱构建Agent", config=config, llm_client=llm_client)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建知识图谱
        
        Args:
            input_data: {
                "case_clues": dict,
                "suspects": list,
                "evidence": list,
                "reasoning_chain": list
            }
        """
        self.log_processing(input_data)
        
        case_clues = input_data.get("case_clues", {})
        suspects = input_data.get("suspects", [])
        evidence = input_data.get("evidence", [])
        reasoning_chain = input_data.get("reasoning_chain", [])
        
        clues_text = json.dumps(case_clues, ensure_ascii=False) if isinstance(case_clues, dict) else str(case_clues)
        suspects_text = json.dumps(suspects, ensure_ascii=False) if isinstance(suspects, list) else str(suspects)
        evidence_text = json.dumps(evidence, ensure_ascii=False) if isinstance(evidence, list) else str(evidence)
        
        prompt = (
            "请根据以下案件信息构建知识图谱。\n\n"
            f"线索:\n{clues_text[:1000]}\n\n"
            f"嫌疑人:\n{suspects_text[:800]}\n\n"
            f"证据:\n{evidence_text[:800]}\n\n"
            '请以JSON格式返回图结构:\n'
            '{"nodes":[{"id":"ID","label":"名称","type":"person/evidence/location/time/event","properties":{}}],'
            '"edges":[{"source":"ID","target":"ID","label":"关系","weight":0.8}],'
            '"communities":[{"name":"名称","members":["ID1","ID2"],"significance":"说明"}]}\n'
            "只返回JSON。"
        )
        
        response = self.call_llm(prompt, temperature=0.3)
        parsed = self.extract_json_from_response(response)
        
        if parsed and isinstance(parsed, dict):
            nodes = parsed.get("nodes", [])
            edges = parsed.get("edges", [])
            communities = parsed.get("communities", [])
        else:
            nodes, edges, communities = self._build_basic_graph(suspects, evidence)
        
        return self.format_output({
            "nodes": nodes,
            "edges": edges,
            "communities": communities,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "graph_type": "案件关系图谱"
        })
    
    def _build_basic_graph(self, suspects: list, evidence: list) -> tuple:
        """构建基础图谱（降级方案）"""
        nodes = []
        edges = []
        
        # 添加嫌疑人节点
        for i, s in enumerate(suspects):
            name = s if isinstance(s, str) else s.get("name", f"嫌疑人{i+1}")
            nodes.append({"id": f"suspect_{i}", "label": name, "type": "person", "properties": {}})
        
        # 添加受害者节点
        nodes.append({"id": "victim", "label": "受害者", "type": "person", "properties": {}})
        
        # 添加证据节点
        for i, e in enumerate(evidence[:10]):
            desc = e if isinstance(e, str) else str(e)
            nodes.append({"id": f"evidence_{i}", "label": desc[:20], "type": "evidence", "properties": {"full": desc}})
            # 每个证据关联到受害者
            edges.append({"source": f"evidence_{i}", "target": "victim", "label": "涉及", "weight": 0.5})
        
        # 嫌疑人之间的关系
        for i in range(len(suspects)):
            edges.append({"source": f"suspect_{i}", "target": "victim", "label": "嫌疑人", "weight": 0.7})
        
        return nodes, edges, []
