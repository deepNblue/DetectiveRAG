"""
动机搜索Agent
从结构化知识中主动推理搜索作案动机
ASMR核心理念: Agent主动推理搜索 >> 向量相似度被动检索
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent


class MotiveSearcher(BaseAgent):
    """动机搜索Agent — 寻找谁有作案动机及动机强度"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="ASMR-MotiveSearcher", config=config, llm_client=llm_client)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于摄取的结构化知识，主动推理搜索动机

        Args:
            input_data: {
                "structured_knowledge": dict (来自Readers的输出),
                "suspects": list,
                "case_text": str
            }
        """
        self.log_processing(input_data)
        sk = input_data.get("structured_knowledge", {})
        suspects = input_data.get("suspects", [])
        case_text = input_data.get("case_text", "")

        # 构建嫌疑人列表
        suspect_names = []
        for s in suspects:
            if isinstance(s, dict):
                suspect_names.append(s.get("name", "?"))
            elif isinstance(s, str):
                suspect_names.append(s)

        # 从Readers提取的知识中构建搜索上下文
        person_relations = sk.get("person_relation", {}).get("data", {})
        timeline = sk.get("timeline", {}).get("data", {})
        evidence = sk.get("evidence", {}).get("data", {})

        context = f"""=== 人物关系网络 ===
{json.dumps(person_relations.get('relations', []), ensure_ascii=False, indent=2)[:1000]}

=== 时间线关键事件 ===
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:800]}

=== 关键物证 ===
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:800]}"""

        prompt = f"""你是一个专注犯罪动机分析的刑侦专家。根据以下案件结构化知识，分析每个嫌疑人的作案动机。

嫌疑人: {', '.join(suspect_names)}

案件知识:
{context}

原始案件文本(补充参考):
{case_text[:1500]}

请严格以JSON格式返回:
{{
    "motive_analysis": [
        {{
            "suspect": "嫌疑人姓名",
            "primary_motive": "主要动机(经济/情感/复仇/自卫/灭口/权力等)",
            "motive_strength": 0.0-1.0,
            "motive_detail": "动机详细分析",
            "triggering_event": "什么事件触发了这个动机",
            "evidence_supporting": ["支持动机的证据"],
            "evidence_against": ["削弱动机的证据"],
            "motive_timeline": "动机形成的时间线"
        }}
    ],
    "ranking": ["按动机从强到弱排列嫌疑人"],
    "key_insight": "关于动机的最关键发现"
}}

要求:
1. 每个嫌疑人都要分析，包括动机强度为0的情况
2. 动机强度0.8以上视为强烈动机
3. 特别关注隐藏动机（表面关系下的真实动机）
4. 只返回JSON"""

        response = self.call_llm(prompt, temperature=0.3)
        parsed = self.extract_json_from_response(response)

        if parsed and isinstance(parsed, dict):
            analysis = parsed.get("motive_analysis", [])
            ranking = parsed.get("ranking", [])
            insight = parsed.get("key_insight", "")
        else:
            analysis, ranking, insight = [], [], ""

        self.logger.info(f"动机搜索完成: {len(analysis)}人分析, 最强动机: {ranking[0] if ranking else '?'}")

        return self.format_output({
            "motive_analysis": analysis,
            "ranking": ranking,
            "key_insight": insight,
        })
