"""
矛盾搜索Agent
专门寻找嫌疑人供词、行为、动机中的矛盾点和异常模式
借鉴 Supermemory ASMR 的 "主动推理检索" 思路
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent


class ContradictionSearcher(BaseAgent):
    """矛盾搜索Agent — 找出证据链中的矛盾、不一致和异常行为"""

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="ASMR-ContradictionSearcher", config=config, llm_client=llm_client)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        搜索案件中的矛盾和异常

        Args:
            input_data: {
                "structured_knowledge": dict,
                "suspects": list,
                "case_text": str
            }
        """
        self.log_processing(input_data)
        sk = input_data.get("structured_knowledge", {})
        suspects = input_data.get("suspects", [])
        case_text = input_data.get("case_text", "")

        suspect_names = [s.get("name", "?") if isinstance(s, dict) else s for s in suspects]

        timeline = sk.get("timeline", {}).get("data", {})
        evidence = sk.get("evidence", {}).get("data", {})
        person_rels = sk.get("person_relation", {}).get("data", {})

        context = f"""=== 时间线(关注行为异常和时序矛盾) ===
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:800]}

=== 物证(关注与供词的矛盾) ===
{json.dumps(evidence.get('physical_evidence', []), ensure_ascii=False, indent=2)[:800]}

=== 人物关系(关注利益冲突和行为异常) ===
{json.dumps(person_rels.get('persons', []), ensure_ascii=False, indent=2)[:800]}

=== 人物关系(关注矛盾关系) ===
{json.dumps(person_rels.get('relationships', []), ensure_ascii=False, indent=2)[:800]}"""

        prompt = f"""你是一个专注于寻找矛盾和异常的刑侦分析师。根据案件知识，系统性地搜索以下5类矛盾:

嫌疑人: {', '.join(suspect_names)}

案件知识:
{context}

原始案件文本(补充参考):
{case_text[:2000]}

请搜索以下6类矛盾/异常:

**1. 供词矛盾**: 同一人的陈述前后不一致，或与物证冲突
**2. 行为异常**: 不符合常理的行为(如声称关心但带来含毒物品)
**3. 机会隐藏**: 有人声称没有作案时间，但有间接证据表明有机会
**4. 动机掩饰**: 表面动机与实际行为不匹配(如声称关心受害者却受益)
**5. 能力伪装**: 低调隐藏自身能力/资源，使嫌疑降低
**6. 嫁祸替罪**: 有人被证据直接指向，但这些证据可能是他人故意制造的(如盗用账号、在别人物品上留指纹、通过他人设备作案)

⚠️ 特别注意:
- **投毒案关键**: 谁亲手制作/控制了含毒物品，谁嫌疑最大。化学知识只是能力，不是证据。
- **"狗没叫"原则**: 真凶往往是那个看起来"最正常"的人，而不是"最有能力"的人。
- **证据链优先**: 直接物证链(谁→做了什么→导致中毒)比间接推理(谁有动机/能力)更有价值。
- **嫁祸识别**: 如果某人"看起来嫌疑最大"但存在物理上不可能的矛盾(如不在场却有物证)，考虑有人故意嫁祸。
- **多罪犯合谋检测**: 这是极其重要的检查项! 请逐对检查嫌疑人之间是否存在合谋关系:
  - 检查每对嫌疑人(A, B)是否满足: A有动机(经济压力/不明款项) + B有渠道(与对手方的私人关系) → 合谋可完成完整犯罪链
  - 检查是否有嫌疑人单独无法完成全部犯罪环节(如:有经济动机但无法接触对手方，或有渠道但缺少动机)
  - 合谋的强信号: 两人各自的弱项恰好是对方的强项，形成互补
  - 商业间谍/泄密案中特别常见: A窃取数据(有权限/有经济动机) + B传递给对手(有私人关系)
  - 如果发现合谋模式，务必设置 collusion_detected=true 并在 collusion_suspects 和 collusion_evidence 中详细说明

请严格以JSON格式返回:
{{
    "contradictions": [
        {{
            "type": "供词矛盾/行为异常/机会隐藏/动机掩饰/能力伪装/嫁祸替罪",
            "person": "涉及的人",
            "description": "具体矛盾/异常描述",
            "evidence_for": "支持矛盾存在的证据",
            "significance": "high/medium/low",
            "implicates": "这个矛盾暗示谁是真凶"
        }}
    ],
    "anomaly_summary": {{
        "most_suspicious": "最可疑的人(基于矛盾分析)",
        "key_contradiction": "最关键的矛盾点",
        "hidden_culprit_hint": "被忽略但实际最可疑的人"
    }},
    "ranking_by_contradiction": ["按被矛盾指向程度从高到低排列嫌疑人"],
    "key_insight": "关于矛盾分析的最关键发现",
    "framing_detected": false,
    "framing_victim": "被嫁祸的人(如果发现嫁祸模式)",
    "framing_evidence": "为什么认为是嫁祸而非直接证据",
    "collusion_detected": false,
    "collusion_suspects": ["合谋的嫌疑人(如果发现多罪犯合谋模式)"],
    "collusion_evidence": "合谋的证据(互补动机、互补能力、隐秘联系)"
}}

要求:
1. 不要只关注能力最强的人，重点看谁的行为/陈述有矛盾
2. 注意"看起来最清白"的人是否有刻意伪装的痕迹
3. 特别关注那些声称不知情但行为表明知情的人
4. **嫁祸分析**: 检查"嫌疑最大的人"是否被他人栽赃(如账号被盗用、物品被利用)
5. **合谋分析**: 检查是否有多人各提供一部分犯罪要素(动机+渠道、技术+机会)
6. 只返回JSON"""

        response = self.call_llm(prompt, temperature=0.3)
        parsed = self.extract_json_from_response(response)

        if parsed and isinstance(parsed, dict):
            contradictions = parsed.get("contradictions", [])
            anomaly_summary = parsed.get("anomaly_summary", {})
            ranking = parsed.get("ranking_by_contradiction", [])
            insight = parsed.get("key_insight", "")
        else:
            contradictions, anomaly_summary, ranking, insight = [], {}, [], ""

        self.logger.info(f"矛盾搜索完成: 发现{len(contradictions)}个矛盾/异常")

        return self.format_output({
            "contradictions": contradictions,
            "anomaly_summary": anomaly_summary,
            "ranking": ranking,
            "key_insight": insight,
        })
