#!/usr/bin/env python3
"""
证据双面采集器 (Bidirectional Evidence Collector)
借鉴 metareflection/llm-mysteries/graph.py 的 incriminating/exonerating 双面证据收集

核心思路:
  对每个嫌疑人，LLM分别生成:
    1. 有罪证据 (incriminating) — 支持其是真凶的证据
    2. 无罪证据 (exonerating) — 排除其是真凶的证据
  然后LLM判断每条证据是否成立 (Yes/No)
  最终: culprit = incriminating=Yes AND exonerating=No

  这比直接问"谁是凶手"更可靠，因为:
    - 强制LLM为每个嫌疑人找正反两面证据
    - 避免确认偏差 (只看到支持自己结论的证据)
    - 符号化组合: 有罪 AND NOT 无罪 = 真凶

借鉴来源:
  - metareflection/llm-mysteries/graph.py (processCase/processCase2/processCase3)
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from .name_utils import normalize_name, is_valid_suspect, build_name_alias_map


class BidirectionalEvidenceCollector:
    """证据双面采集器 — 对每个嫌疑人收集有罪+无罪证据"""

    INCriminating_PROMPT = """你是一名刑侦证据分析员。请仔细阅读案件，找出指向该嫌疑人有罪的证据。

【案件】
{case_text}

【嫌疑人】{suspect_name}

请列出所有支持"{suspect_name}是真凶"的证据点。每条证据必须来自案件文本。
严格输出JSON:
{{
  "evidence_list": [
    {{
      "claim": "证据描述(50字内)",
      "category": "motive/opportunity/capability/timeline/physical_evidence/testimony",
      "strength": "strong/medium/weak",
      "source": "来自案件文本的具体内容摘要(30字内)"
    }}
  ],
  "summary": "有罪证据总结(一句话)"
}}

如果几乎没有有罪证据，evidence_list可以为空列表。只输出JSON。"""

    EXONERATING_PROMPT = """你是一名刑侦证据分析员。请仔细阅读案件，找出排除该嫌疑人有罪的证据。

【案件】
{case_text}

【嫌疑人】{suspect_name}

请列出所有支持"{suspect_name}不是真凶"的证据点(如不在场证明、缺乏动机、他人有更强嫌疑等)。
严格输出JSON:
{{
  "evidence_list": [
    {{
      "claim": "无罪证据描述(50字内)",
      "category": "alibi/lack_of_motive/lack_of_opportunity/contradiction/other_suspect_stronger",
      "strength": "strong/medium/weak",
      "source": "来自案件文本的具体内容摘要(30字内)"
    }}
  ],
  "summary": "无罪证据总结(一句话)"
}}

如果几乎没有无罪证据，evidence_list可以为空列表。只输出JSON。"""

    VERIFICATION_PROMPT = """判断以下证据是否确实存在于案件文本中。

【案件文本(摘录)】
{case_excerpt}

【待验证证据】
{evidence_claim}

请判断该证据是否在案件文本中有明确支撑。
只输出: "Yes" 或 "No" (不需要解释)"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.logger = logger.bind(module="BidirectionalEvidence")

    def collect(
        self,
        case_text: str,
        suspects: List[str],
        expert_analyses: List[Dict] = None,
        verify: bool = True,
    ) -> Dict[str, Any]:
        """
        对每个嫌疑人收集双向证据
        
        Returns:
            {
                "suspects": {
                    name: {
                        "incriminating": [{claim, category, strength, source, verified}],
                        "exonerating": [{claim, category, strength, source, verified}],
                        "incriminating_score": float,  # 0-1
                        "exonerating_score": float,     # 0-1
                        "net_score": float,             # incriminating - exonerating
                    }
                },
                "ranking": [{"name": str, "net_score": float}],
                "method1_result": str,  # graph.py processCase 风格
                "method2_result": str,  # graph.py processCase2 风格
            }
        """
        results = {}
        
        for suspect in suspects:
            self.logger.info(f"🔍 采集双向证据: {suspect}")
            
            # 1. 收集有罪证据
            incriminating = self._collect_evidence(
                case_text, suspect, "incriminating"
            )
            
            # 2. 收集无罪证据
            exonerating = self._collect_evidence(
                case_text, suspect, "exonerating"
            )
            
            # 3. 可选: 验证证据
            if verify and self.llm_client:
                incriminating = self._verify_evidence_list(case_text, incriminating)
                exonerating = self._verify_evidence_list(case_text, exonerating)
            
            # 4. 计算分数
            inc_score = self._compute_score(incriminating)
            exo_score = self._compute_score(exonerating)
            net_score = inc_score - exo_score
            
            results[suspect] = {
                "incriminating": incriminating,
                "exonerating": exonerating,
                "incriminating_score": round(inc_score, 3),
                "exonerating_score": round(exo_score, 3),
                "net_score": round(net_score, 3),
            }
            
            self.logger.info(f"  📊 {suspect}: 有罪={inc_score:.2f}, 无罪={exo_score:.2f}, 净={net_score:+.2f}")
            self.logger.info(f"    有罪证据: {[e['claim'][:30] for e in incriminating[:3]]}")
            self.logger.info(f"    无罪证据: {[e['claim'][:30] for e in exonerating[:3]]}")
        
        # 5. 排名
        ranking = sorted(
            [{"name": s, "net_score": d["net_score"]} for s, d in results.items()],
            key=lambda x: -x["net_score"]
        )
        
        # 6. Method 1: 布尔判定 (graph.py processCase 风格)
        method1_suspects = []
        for s, d in results.items():
            has_incriminating = d["incriminating_score"] > 0.3
            has_exonerating = d["exonerating_score"] > 0.3
            is_culprit = has_incriminating and not has_exonerating
            method1_suspects.append((s, is_culprit))
        
        culprit_count = sum(1 for _, c in method1_suspects if c)
        if culprit_count == 1:
            method1_result = next(s for s, c in method1_suspects if c)
        elif culprit_count == 0:
            method1_result = "无法确定"
        else:
            # 多个嫌疑人都有罪 → 选net_score最高的
            method1_result = ranking[0]["name"] if ranking else "无法确定"
        
        # 7. Method 2: LLM综合判断 (graph.py processCase2 风格)
        method2_result = self._method2_llm_judge(case_text, results)
        
        self.logger.info(f"📊 Method1(布尔): {method1_result}")
        self.logger.info(f"📊 Method2(LLM): {method2_result}")
        self.logger.info(f"📊 排名: {[(r['name'], r['net_score']) for r in ranking]}")
        
        return {
            "suspects": results,
            "ranking": ranking,
            "method1_result": method1_result,
            "method2_result": method2_result,
        }

    def _collect_evidence(self, case_text: str, suspect: str, direction: str) -> List[Dict]:
        """收集单个方向(有罪/无罪)的证据"""
        prompt = (self.INCriminating_PROMPT if direction == "incriminating" 
                  else self.EXONERATING_PROMPT)
        prompt = prompt.format(
            case_text=case_text[:2000],
            suspect_name=suspect,
        )
        
        response = self._call_llm(prompt, temperature=0.2)
        result = self._extract_json(response)
        
        if result and "evidence_list" in result:
            return [
                {
                    "claim": e.get("claim", ""),
                    "category": e.get("category", "unknown"),
                    "strength": e.get("strength", "weak"),
                    "source": e.get("source", ""),
                    "verified": None,  # 待验证
                }
                for e in result["evidence_list"]
                if e.get("claim")
            ]
        return []

    def _verify_evidence_list(self, case_text: str, evidence_list: List[Dict]) -> List[Dict]:
        """验证每条证据是否在案件文本中有支撑"""
        verified = []
        for e in evidence_list:
            prompt = self.VERIFICATION_PROMPT.format(
                case_excerpt=case_text[:1500],
                evidence_claim=e["claim"],
            )
            response = self._call_llm(prompt, temperature=0.0)
            verified_flag = "yes" in response.strip().lower()[:10]
            e["verified"] = verified_flag
            verified.append(e)
        return verified

    def _compute_score(self, evidence_list: List[Dict]) -> float:
        """计算证据列表的综合得分"""
        if not evidence_list:
            return 0.0
        
        strength_map = {"strong": 1.0, "medium": 0.6, "weak": 0.3}
        total = 0.0
        for e in evidence_list:
            base = strength_map.get(e.get("strength", "weak"), 0.3)
            # 已验证的证据权重更高
            if e.get("verified") is True:
                base *= 1.3
            elif e.get("verified") is False:
                base *= 0.3
            total += base
        
        # 归一化到 0-1
        return min(total / max(len(evidence_list), 1), 1.0)

    def _method2_llm_judge(self, case_text: str, results: Dict) -> str:
        """Method 2: 把所有证据汇总，让LLM综合判断"""
        suspect_names = list(results.keys())
        if not suspect_names:
            return "无法确定"
        
        prompt = f"根据以下证据分析，判断谁是凶手。\n\n"
        for name, data in results.items():
            prompt += f"【{name}】\n"
            prompt += f"有罪证据:\n"
            for e in data["incriminating"][:5]:
                v = f"(已验证)" if e.get("verified") else "(未验证)"
                prompt += f"  - [{e.get('strength','?')}] {e['claim']} {v}\n"
            prompt += f"无罪证据:\n"
            for e in data["exonerating"][:5]:
                v = f"(已验证)" if e.get("verified") else "(未验证)"
                prompt += f"  - [{e.get('strength','?')}] {e['claim']} {v}\n"
            prompt += f"有罪得分={data['incriminating_score']:.2f}, 无罪得分={data['exonerating_score']:.2f}\n\n"
        
        prompt += f"只输出凶手名字，不要解释。可选: {', '.join(suspect_names)}"
        
        response = self._call_llm(prompt, temperature=0.1)
        
        # 匹配嫌疑人名字
        for name in suspect_names:
            if name in response:
                return name
        
        alias_map = build_name_alias_map(suspect_names)
        for alias, canonical in alias_map.items():
            if alias in response:
                return canonical
        
        return "无法确定"

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        if self.llm_client is None:
            return ""
        try:
            return self.llm_client.simple_chat(prompt, temperature=temperature)
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return ""

    def _extract_json(self, text: str) -> Optional[Dict]:
        if not text:
            return None
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass
        depth = 0
        start = -1
        in_string = False
        escape_next = False
        for i, c in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        start = -1
        return None
