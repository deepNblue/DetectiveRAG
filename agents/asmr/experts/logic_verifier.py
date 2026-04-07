"""
逻辑验证专家Agent
检验其他专家结论的逻辑一致性，防止推理谬误
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent


class LogicVerifier(BaseAgent):
    """逻辑验证专家 — 检验其他专家推理的逻辑一致性和可靠性"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="ASMR-LogicVerifier", config=config, llm_client=llm_client)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证其他专家的结论

        Args:
            input_data: {
                "expert_results": list[dict],  # 其他3个专家的结论
                "structured_knowledge": dict,
                "search_results": dict
            }
        """
        self.log_processing(input_data)
        expert_results = input_data.get("expert_results", [])
        sk = input_data.get("structured_knowledge", {})

        # 汇总专家观点
        expert_summary = ""
        for i, er in enumerate(expert_results):
            d = er.get("data", er)
            expert_summary += f"""
专家{i+1} ({d.get('perspective', '?')}):
  - 结论: 真凶是 {d.get('culprit', '?')}
  - 置信度: {d.get('confidence', 0)}
  - 推理: {d.get('reasoning', '?')[:300]}
"""

        prompt = f"""你是一位逻辑审查专家。你的职责是检验以下三位专家的推理是否存在逻辑漏洞。

专家结论汇总:
{expert_summary}

已知物证(用于交叉验证):
{json.dumps(sk.get('evidence', {}).get('data', {}).get('physical_evidence', []), ensure_ascii=False, indent=2)[:600]}

已知时间矛盾:
{json.dumps(sk.get('timeline', {}).get('data', {}).get('anomalies', []), ensure_ascii=False, indent=2)[:400]}

请严格以JSON格式返回你的逻辑验证结果:
{{
    "consensus_suspect": "多数专家认定的嫌疑人(如果一致)",
    "has_consensus": true/false,
    "logic_review": [
        {{
            "expert": "专家编号",
            "perspective": "视角",
            "logic_flaws": ["发现的逻辑漏洞"],
            "unsupported_claims": ["缺乏证据支撑的断言"],
            "strong_points": ["推理的亮点"],
            "adjusted_confidence": 0.0-1.0
        }}
    ],
    "final_verdict": {{
        "culprit": "经过逻辑验证后的最终结论",
        "confidence": 0.0-1.0,
        "confidence_reason": "这个置信度的依据",
        "certainty_level": "确定/很可能/可能/不确定"
    }},
    "reasoning": "你的验证推理过程"
}}

要求:
1. 严格检查每个专家的推理是否存在逻辑跳跃、循环论证、确认偏误等
2. 如有专家做出缺乏证据支撑的断言，降低其adjusted_confidence
3. 最终verdict要综合三位专家中逻辑最可靠的结论
4. 只返回JSON"""

        response = self.call_llm(prompt, temperature=0.3)
        parsed = self.extract_json_from_response(response)

        if parsed and isinstance(parsed, dict):
            verdict = parsed.get("final_verdict", {})
            culprit = verdict.get("culprit", "未知")
            confidence = verdict.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
            has_consensus = parsed.get("has_consensus", False)
        else:
            culprit, confidence, reasoning = "未知", 0.2, "逻辑验证失败"
            verdict = {}
            has_consensus = False

        self.logger.info(f"逻辑验证完成: 真凶={culprit}, 置信度={confidence}, 共识={'是' if has_consensus else '否'}")

        return self.format_output({
            "perspective": "logic_verification",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
            "final_verdict": verdict,
        })
