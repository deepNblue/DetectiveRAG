"""
审讯分析专家Agent (Interrogation Analyst)
从供述一致性分析、谎言识别、口供矛盾点分析、审讯策略建议角度分析案件

对应真实角色:
  - 审讯专家: 供述分析与审讯策略
  - 测谎专家: 谎言识别与行为分析
  - 刑事心理学审讯顾问: 供述可靠性评估
  - 口供审查员: 供证矛盾分析

核心理念: "谎言需要记忆，真话自然流露" — 从供述的细节和一致性判断可信度

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


class InterrogationAnalyst(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """
    审讯分析专家 — 从供述一致性和谎言识别角度分析案件

    专长:
    - 供述一致性分析: 比对多次供述/证言的一致性，发现前后矛盾
    - 谎言识别: 通过语言特征、细节描述、情绪反应识别说谎迹象
    - 口供矛盾点分析: 不同嫌疑人的口供对比，找出互相矛盾之处
    - 审讯策略建议: 根据嫌疑人心理特征给出审讯建议
    - 供述可信度评估: 评估各份证言的可信程度
    - 证言 corroborating analysis: 验证证言与物证的吻合程度
    """

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-InterrogationAnalyst", config=config, llm_client=llm_client)
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
        contradiction_data = search.get("contradiction", {}).get("data", {})

        # 🧠 记忆增强
        case_type = input_data.get("case_type", "")
        memory_ctx = self.retrieve_memory_context(case_type=case_type, case_data=input_data)

        initial_prompt = f"""你是一位资深审讯分析专家，擅长供述一致性分析和谎言识别。请从审讯分析的角度给出你的专业判断。

嫌疑人: {', '.join(suspect_names)}

证言记录(重点分析: 细节一致性、情绪表达、回避行为):
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:1000]}

人物信息(关注性格特征和行为模式):
{json.dumps(person_rels.get('persons', []), ensure_ascii=False, indent=2)[:600]}

人物关系(关注各人之间的关系和潜在利益冲突):
{json.dumps(person_rels.get('relations', []), ensure_ascii=False, indent=2)[:600]}

时间线(用于验证证言时间准确性):
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

物证(用于与证言交叉验证):
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:600]}

动机分析:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:500]}

矛盾/异常数据:
{json.dumps(contradiction_data, ensure_ascii=False, indent=2)[:500]}
{memory_ctx.get('context_text', '')}
请严格以JSON格式返回你的审讯分析结论:
{{
    "culprit": "你认为的真凶(基于证言可信度分析)",
    "confidence": 0.0-1.0,
    "testimony_credibility": [
        {{
            "witness_or_suspect": "证人/嫌疑人姓名",
            "credibility_score": 0.0-1.0,
            "consistency_analysis": "供述一致性分析(前后是否有矛盾)",
            "deception_indicators": ["说谎迹象(过度细节/回避关键问题/情绪不匹配等)"],
            "key_contradictions": ["关键矛盾点"],
            "corroboration_with_evidence": "与物证的吻合程度"
        }}
    ],
    "cross_testimony_analysis": {{
        "conflicting_statements": [
            {{
                "person_a": "人物A的陈述",
                "person_b": "人物B的陈述(与A矛盾)",
                "who_is_more_credible": "谁更可信及原因",
                "resolution": "如何解释这个矛盾"
            }}
        ],
        "consistent_points": ["多人证言一致的关键点"],
        "isolated_claims": ["仅一人声称且无旁证的内容"]
    }},
    "interrogation_strategy": {{
        "priority_target": "建议优先深入审讯的对象",
        "approach": "审讯策略(情感突破/证据对质/逻辑陷阱等)",
        "key_questions": ["建议追问的关键问题"],
        "pressure_points": ["该嫌疑人的心理弱点/突破口"]
    }},
    "reasoning": "你的审讯分析推理过程(从证言可信度角度分析谁在说谎谁在说真话)",
    "key_insight": "审讯分析中最关键的发现",
    "most_suspicious_behavior": "最可疑的行为表现"
}}

要求:
1. 深入分析每份证言的细节一致性(时间/地点/人物/细节是否自洽)
2. 识别说谎的语言特征(过度解释/回避/细节过多或过少/情绪不匹配)
3. 对比不同人的证言，找出交叉矛盾和相互印证
4. 将证言与物证/时间线交叉验证
5. 评估谁在说谎、谁在隐瞒、谁在说真话
6. 给出审讯策略建议(如何突破说谎者)
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
            expert_role="审讯分析专家",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "审讯分析失败", 0

        self.logger.info(f"审讯分析完成(多轮×{total_rounds}): 真凶={culprit}, 置信度={confidence}")

        # 🧠 记忆存储
        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="interrogation",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["供述一致性分析", "谎言识别", "证言交叉验证"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "interrogation_analysis",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
        })
