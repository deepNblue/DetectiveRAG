"""
经侦专家Agent (Financial Investigator)
从资金链追踪、洗钱识别、财产转移、金融异常检测角度分析案件

对应真实角色:
  - 经济犯罪侦查人员: 资金链追踪、金融犯罪分析
  - 反洗钱分析师: 识别异常资金流动
  - 税务稽查员: 财产转移和隐匿追踪
  - 审计师: 财务造假、挪用资金发现

核心理念: "Follow the money" — 跟着钱走，经济动机往往是犯罪的根本驱动力

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


class FinancialInvestigator(MemoryEnhancedMixin, MultiRoundMixin, BaseAgent):
    """
    经侦专家 — 从经济犯罪和资金链追踪角度分析案件

    专长:
    - 资金链追踪: 追踪资金来源、流向和最终去向
    - 洗钱识别: 识别异常资金转移模式（层层转账、跨境转移、空壳公司）
    - 财产转移分析: 分析财产异常变动（保险、遗产、赠与、信托）
    - 金融异常检测: 发现异常交易、大额提现、频繁转账
    - 经济动机追踪: 从经济利益角度分析犯罪动机
    - 财务关联分析: 分析嫌疑人之间的财务往来
    """

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        BaseAgent.__init__(self, name="ASMR-FinancialInvestigator", config=config, llm_client=llm_client)
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
        temporal_data = search.get("temporal", {}).get("data", {}).get("temporal_contradictions", [])

        # 🧠 记忆增强
        case_type = input_data.get("case_type", "")
        memory_ctx = self.retrieve_memory_context(case_type=case_type, case_data=input_data)

        initial_prompt = f"""你是一位资深经济犯罪侦查专家，擅长资金链追踪和金融异常分析。

核心理念: "Follow the money" — 跟着钱走。经济利益是犯罪最常见的驱动力之一，金融证据往往能揭示隐藏的犯罪动机。无论案件类型如何（谋杀、伤害、纵火等），都应从经济利益角度追踪嫌疑人动机。

专业分析方法:
- 追踪异常资金流动（大额转账、频繁小额转账、跨境汇款）
- 分析保险、遗产、债务等经济利益关系
- 识别财产异常变动时间节点与案件时间线的关联
- 评估各嫌疑人的经济利益得失

嫌疑人: {', '.join(suspect_names)}

人物关系(关注经济利益关系):
{json.dumps(person_rels.get('relations', []), ensure_ascii=False, indent=2)[:600]}

人物信息(关注职业、收入、财产状况):
{json.dumps(person_rels.get('persons', []), ensure_ascii=False, indent=2)[:600]}

时间线(关注案发前后的财产变动和金融活动):
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

物证(关注财务文件、银行记录、合同等):
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:800]}

证言(关注涉及金钱的陈述):
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:600]}

动机分析(关注经济动机):
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:500]}

时间矛盾:
{json.dumps(temporal_data, ensure_ascii=False, indent=2)[:400]}
{memory_ctx.get('context_text', '')}
请严格以JSON格式返回你的经侦分析结论:
{{
    "culprit": "根据金融证据判断的真凶",
    "confidence": 0.0-1.0,
    "financial_motivation": {{
        "primary_motive": "主要经济动机",
        "estimated_gain": "预计经济利益规模",
        "financial_pressure": "嫌疑人面临的经济压力",
        "motive_timeline": "经济动机形成的时间线"
    }},
    "fund_tracing": {{
        "suspicious_transactions": [
            {{
                "description": "可疑交易描述",
                "from": "资金来源",
                "to": "资金去向",
                "amount": "金额(如果可知)",
                "timing": "交易时间与案件的时间关系"
            }}
        ],
        "hidden_assets": ["隐藏/转移的资产"],
        "laundering_indicators": ["洗钱迹象"]
    }},
    "property_analysis": {{
        "insurance_analysis": "保险相关分析",
        "inheritance_analysis": "遗产继承分析",
        "asset_transfers": "案发前后财产异常转移"
    }},
    "financial_anomalies": [
        {{
            "suspect": "嫌疑人",
            "anomaly": "金融异常描述",
            "significance": "异常的重要性(高/中/低)"
        }}
    ],
    "reasoning": "你的经侦推理过程",
    "key_finding": "最关键的金融发现",
    "evidence_chain": ["经济动机证据链"]
}}

要求:
1. 从经济利益角度分析各嫌疑人的犯罪动机强度
2. 异常金融行为（如案发前的大额保险、财产转移）应作为重要的动机证据
3. 分析谁从犯罪结果中获得最大经济利益
4. 将金融异常的时间节点与案件时间线交叉验证
5. 只返回JSON

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
            expert_role="经侦分析专家",
            temperature=0.4,
        )

        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            total_rounds = parsed.get("_total_rounds", 1)
        else:
            culprit, confidence, reasoning, total_rounds = "未知", 0.2, "经侦分析失败", 0

        self.logger.info(f"经侦分析完成(多轮×{total_rounds}): 真凶={culprit}, 置信度={confidence}")

        # 🧠 记忆存储
        try:
            case_id = input_data.get("case_id", "")
            if case_id:
                from agents.memory.base_memory import CaseMemory
                mem = CaseMemory(
                    case_id=case_id,
                    expert_type="financial",
                    conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                    case_type=case_type,
                    reasoning_patterns=["资金链追踪", "经济动机分析", "金融异常检测"],
                )
                self._memory_store.add_memory(mem)
        except Exception as e:
            self.logger.debug(f"记忆存储跳过: {e}")

        return self.format_output({
            "perspective": "financial_investigation",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
        })
