"""
技侦专家Agent (Tech Investigator)
从数字证据、监控、通讯记录、网络行为角度分析案件

对应真实角色:
  - 公安技侦人员: 监控调取、通讯追踪
  - 网安: 网络犯罪侦查、电子证据
  - 数字取证工程师: 数据恢复、电子物证

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


class TechInvestigator(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """
    技侦专家 — 从数字证据和技术侦查角度分析案件
    
    专长:
    - 监控录像时间线分析
    - 通讯记录分析（通话/短信/微信）
    - 网络行为追踪（搜索记录/浏览历史）
    - 电子数据取证（手机/电脑）
    - GPS定位数据分析
    - 数字证据的关联与矛盾发现
    """

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-TechInvestigator", config=config, llm_client=llm_client)
        self._init_memory(self.name)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_processing(input_data)
        sk = input_data.get("structured_knowledge", {})
        search = input_data.get("search_results", {})
        suspects = input_data.get("suspects", [])

        suspect_names = [s.get("name", "?") if isinstance(s, dict) else s for s in suspects]

        timeline = sk.get("timeline", {}).get("data", {})
        evidence = sk.get("evidence", {}).get("data", {})

        temporal_data = search.get("temporal", {}).get("data", {}).get("temporal_contradictions", [])
        motive_data = search.get("motive", {}).get("data", {}).get("motive_analysis", [])

        # 🧠 记忆增强
        case_type = input_data.get("case_type", "")
        memory_ctx = self.retrieve_memory_context(case_type=case_type, case_data=input_data)

        initial_prompt = f"""你是一位资深技侦专家，擅长数字证据分析和电子取证。请从技侦角度分析此案。

嫌疑人: {', '.join(suspect_names)}

时间线(重点关注电子记录的时间戳):
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:800]}

证据中的数字证据(监控/通讯/网络记录/电子数据):
{json.dumps(evidence.get('connections', []), ensure_ascii=False, indent=2)[:800]}

证言记录(关注内容是否与数字证据矛盾):
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:600]}

时间矛盾(数字时间戳的冲突):
{json.dumps(temporal_data, ensure_ascii=False, indent=2)[:500]}

动机分析:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:400]}
{memory_ctx.get('context_text', '')}
请严格以JSON格式返回你的技侦分析结论:
{{
    "culprit": "你认为的真凶",
    "confidence": 0.0-1.0,
    "digital_evidence_analysis": {{
        "surveillance_analysis": "监控录像分析(覆盖/盲区/时间异常)",
        "communication_analysis": "通讯记录分析(通话/消息的异常模式)",
        "network_analysis": "网络行为分析(搜索/浏览/定位数据)",
        "electronic_forensics": "电子取证发现"
    }},
    "tech_contradictions": [
        {{
            "suspect": "嫌疑人",
            "claim": "其声称的内容",
            "digital_truth": "数字证据显示的真相",
            "significance": "矛盾的重要性"
        }}
    ],
    "digital_timeline": ["数字证据重建的关键时间节点"],
    "reasoning": "你的技侦推理过程",
    "key_digital_evidence": "最关键的数字证据"
}}

要求:
1. 重点分析数字证据与证言的矛盾
2. 检查是否有电子数据被删除/篡改的痕迹
3. 利用时间戳建立精确的时间线
4. 关注通讯记录中的异常联系
5. 只返回JSON

⚠️ 姓名规范要求:
- culprit 字段只写一个人名，不要写多人
- 如果该人物有头衔（医生、博士、教授等），去掉头衔只保留姓名
- 同一人的不同称呼视为同一人
"""

        # 🔄 多轮推理
        parsed = self.multi_round_reasoning(
            initial_prompt=initial_prompt,
            context=input_data,
            expert_role="技术调查专家",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "技侦分析失败", 0

        self.logger.info(f"技侦分析完成(多轮×{total_rounds}): 真凶={culprit}, 置信度={confidence}")

        # 🧠 记忆存储
        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="tech",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["数字证据分析", "通讯追踪", "时间戳验证"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "tech_investigation",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
        })
