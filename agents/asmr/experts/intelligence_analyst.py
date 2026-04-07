"""
情报分析专家Agent (Intelligence Analyst)
从关联分析、情报整合、线索关联、嫌疑人关系网络分析角度分析案件

对应真实角色:
  - 情报分析师: 多源情报整合与研判
  - 关系网络分析师: 社会关系网络挖掘
  - 线索整合员: 多维线索关联与碰撞
  - 全源情报官: 综合研判与态势分析

核心理念: "The truth is in the connections" — 真相隐藏在各种关联之中
从宏观情报视角整合所有线索，发现其他专家可能遗漏的跨维度关联

v1: 记忆增强版
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent
from agents.memory.memory_mixin import MemoryEnhancedMixin
from agents.asmr.multi_round_mixin import MultiRoundMixin


class IntelligenceAnalyst(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """
    情报分析专家 — 从情报整合和关联分析角度研判案件

    专长:
    - 关联分析: 发现看似无关的线索之间的隐含联系
    - 情报整合: 综合所有来源的信息形成全貌
    - 线索关联: 将时间线、物证、证言、动机等多维线索交叉关联
    - 关系网络分析: 绘制嫌疑人社会关系网络，识别关键节点
    - 模式识别: 识别行为模式、时间模式、关联模式
    - 异常发现: 发现信息中的异常和规律性偏差
    - 全源研判: 从宏观视角给出综合情报研判
    """

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-IntelligenceAnalyst", config=config, llm_client=llm_client)
        self._init_memory(self.name)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_processing(input_data)
        sk = input_data.get("structured_knowledge", {})
        search = input_data.get("search_results", {})
        suspects = input_data.get("suspects", [])

        suspect_names = [s.get("name", "?") if isinstance(s, dict) else s for s in suspects]

        person_rels = sk.get("person_relation", {}).get("data", {})
        evidence = sk.get("evidence", {}).get("data", {})
        timeline = sk.get("timeline", {}).get("data", {})

        motive_data = search.get("motive", {}).get("data", {}).get("motive_analysis", [])
        opportunity_data = search.get("opportunity", {}).get("data", {}).get("opportunity_analysis", [])
        capability_data = search.get("capability", {}).get("data", {}).get("ranking", [])
        temporal_data = search.get("temporal", {}).get("data", {}).get("temporal_contradictions", [])
        contradiction_data = search.get("contradiction", {}).get("data", {})

        # 🧠 记忆增强
        case_type = input_data.get("case_type", "")
        memory_ctx = self.retrieve_memory_context(case_type=case_type, case_data=input_data)

        initial_prompt = f"""你是一位资深情报分析专家，擅长多源情报整合和关联分析。请从情报综合研判的角度分析此案。

你的独特价值是"跨维度关联"——发现其他单一视角专家可能遗漏的跨线索关联。

嫌疑人: {', '.join(suspect_names)}

=== 全源情报数据 ===

人物信息:
{json.dumps(person_rels.get('persons', []), ensure_ascii=False, indent=2)[:600]}

人物关系网络(重点: 利益关系、情感关系、冲突关系):
{json.dumps(person_rels.get('relations', []), ensure_ascii=False, indent=2)[:800]}

时间线:
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

物证:
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:600]}

证言:
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:600]}

物证关联:
{json.dumps(evidence.get('connections', []), ensure_ascii=False, indent=2)[:500]}

=== 各维度分析结果 ===

动机分析:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:500]}

作案机会:
{json.dumps(opportunity_data, ensure_ascii=False, indent=2)[:500]}

作案能力:
{json.dumps(capability_data, ensure_ascii=False, indent=2)[:400]}

时间矛盾:
{json.dumps(temporal_data, ensure_ascii=False, indent=2)[:500]}

异常/矛盾数据:
{json.dumps(contradiction_data, ensure_ascii=False, indent=2)[:500]}
{memory_ctx.get('context_text', '')}
请严格以JSON格式返回你的情报综合研判结论:
{{
    "culprit": "你认为的真凶(基于情报综合研判)",
    "confidence": 0.0-1.0,
    "relationship_network": {{
        "key_players": ["案件中的关键人物"],
        "core_relationships": [
            {{
                "from": "人物A",
                "to": "人物B",
                "relationship": "关系类型(利益冲突/暗中合作/权力控制/情感纠葛等)",
                "relevance": "与案件的关联度"
            }}
        ],
        "hidden_connections": ["可能被忽视的隐含关联"],
        "power_dynamics": "权力/利益格局分析"
    }},
    "cross_dimensional_links": [
        {{
            "dimension_a": "线索维度A(如: 时间线中的某事件)",
            "dimension_b": "线索维度B(如: 某物证/某证言)",
            "connection": "发现的关联",
            "significance": "这个关联的重要性"
        }}
    ],
    "intelligence_synthesis": {{
        "overall_assessment": "案件全貌综合研判",
        "key_patterns": ["识别到的行为/时间/关联模式"],
        "anomalies": ["信息中的异常点"],
        "information_gaps": ["情报缺口"],
        "alternative_hypotheses": ["其他可能的假设(不同于主流推论的)"]
    }},
    "suspect_network_analysis": [
        {{
            "suspect": "嫌疑人",
            "network_position": "在关系网络中的位置(核心/边缘/桥梁)",
            "information_advantage": "该嫌疑人掌握的信息优势",
            "opportunity_network": "是否有人脉资源帮助其作案/掩盖",
            "risk_assessment": "该嫌疑人作为真凶的综合风险评估"
        }}
    ],
    "timeline_reconstruction": {{
        "pre_crime_intelligence": "案前情报(预谋迹象/异常准备)",
        "crime_execution": "作案实施的情报重建",
        "post_crime_behavior": "案后行为情报分析",
        "information_flow": "信息传递流向(谁先知道什么、何时知道)"
    }},
    "reasoning": "你的情报综合研判推理过程",
    "key_intelligence_finding": "最关键的情报发现",
    "recommended_follow_up": ["建议进一步调查的情报方向"]
}}

要求:
1. 整合所有来源的信息，绘制完整的关系网络
2. 跨维度关联分析(时间线×物证×证言×动机×关系网络)
3. 识别被忽视的隐含关联和潜在共犯关系
4. 分析各嫌疑人在关系网络中的位置和作用
5. 重建信息传递流向(谁知道什么、什么时候知道的)
6. 提出不同于主流假设的替代假设
7. 只返回JSON

⚠️ 姓名规范要求:
- culprit 字段只写一个人名，不要写多人
- 如果该人物有头衔（医生、博士、教授等），去掉头衔只保留姓名
- 如果人物全名包含间隔点（如"格里姆斯比·罗伊洛特"），保持全名完整，不要拆成两个人
- 同一人的不同称呼视为同一人（如"罗伊洛特"和"罗伊洛特医生"是同一个人）
"""

        # 🔄 多轮推理
        parsed = self.multi_round_reasoning(
            initial_prompt=initial_prompt,
            context=input_data,
            expert_role="情报分析师",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "情报分析失败", 0

        self.logger.info(f"情报研判完成: 真凶={culprit}, 置信度={confidence}")

        # 🧠 记忆存储
        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="intelligence",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["关联分析", "情报整合", "关系网络分析"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "intelligence_analysis",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
        })
