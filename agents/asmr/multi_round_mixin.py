"""
多轮推理对话Mixin (Multi-Round Reasoning Mixin) v3
让专家Agent从"一次LLM调用"升级为"多轮深度推理"（最多10轮）。

设计原则:
- 非侵入式: 不要求每个专家重写，通过Mixin自动增强
- 动态判断: 高置信度自动提前终止，避免不必要的LLM调用
- 上下文累积: 每轮对话保留历史，LLM能看到自己之前的推理

核心流程:
  Round 1: 初步分析 — 专家原有prompt，形成假设
  Round 2: 自我审视 — 审查初步结论，找弱点/偏见/盲点
  Round 3: 深入调查 — 聚焦疑点，重新审视被忽略的嫌疑人
  Round 4: 最终整合 — 综合所有轮次，给出判断
  Round 5-10: 动态深入 — 从不同角度继续审视，直到置信度≥阈值或达到上限
    角度: 时间线验证→动机深度→证据链完整性→行为逻辑→间接证据交叉→心理学分析
"""

import json
from typing import Dict, List, Any, Optional
from loguru import logger


class MultiRoundMixin:
    """
    多轮推理Mixin — 增强任何拥有 call_llm() 和 extract_json_from_response() 的Agent
    
    使用方式: 
      class MyExpert(MultiRoundMixin, BaseAgent):
          def process(self, input_data):
              # 构建你的原始prompt
              prompt = self._build_prompt(input_data)
              # 用 multi_round_reasoning() 替代单次 call_llm()
              parsed = self.multi_round_reasoning(
                  initial_prompt=prompt,
                  context=input_data,
                  expert_role="法医专家",
              )
              # parsed 已经是完整的JSON结果
    """

    MAX_ROUNDS = 10
    EARLY_STOP_CONFIDENCE = 0.85
    MIN_REASONING_LEN = 200

    def multi_round_reasoning(
        self,
        initial_prompt: str,
        context: Dict[str, Any],
        expert_role: str = "专家",
        temperature: float = 0.4,
    ) -> Dict[str, Any]:
        """
        多轮推理主入口

        Args:
            initial_prompt: 专家的原始分析prompt（Round 1使用）
            context: 分析上下文（含structured_knowledge, search_results, suspects等）
            expert_role: 专家角色名（用于日志和多轮prompt）
            temperature: LLM温度

        Returns:
            最终分析结果 (JSON dict)，额外包含:
                - _rounds: 各轮次摘要
                - _total_rounds: 实际执行的轮数
                - _early_stop: 是否提前终止
        """
        rounds_log = []
        expert_name = getattr(self, 'name', expert_role)

        # ==========================================
        # Round 1: 初步分析（使用专家原始prompt）
        # ==========================================
        logger.info(f"🔄 [{expert_name}] Round 1/{self.MAX_ROUNDS}: 初步分析")
        r1_response = self.call_llm(initial_prompt, temperature=temperature)
        r1_parsed = self.extract_json_from_response(r1_response)

        if not r1_parsed or not isinstance(r1_parsed, dict):
            logger.warning(f"🔄 [{expert_name}] Round 1 JSON解析失败，回退")
            return {"culprit": "未知", "confidence": 0.2, "reasoning": "初步分析失败",
                    "_rounds": [{"round": 1, "status": "parse_failed"}], "_total_rounds": 1}

        r1_culprit = r1_parsed.get("culprit", "未知")
        r1_confidence = r1_parsed.get("confidence", 0)
        r1_reasoning = r1_parsed.get("reasoning", "")
        r1_unresolved = r1_parsed.get("unresolved_questions", [])
        r1_key_evidence = r1_parsed.get("key_evidence",
                          r1_parsed.get("key_physical_evidence",
                          r1_parsed.get("evidence_chain", [])))

        rounds_log.append({
            "round": 1, "phase": "initial",
            "culprit": r1_culprit, "confidence": r1_confidence,
            "reasoning_len": len(r1_reasoning),
            "unresolved": len(r1_unresolved),
        })
        logger.info(f"🔄 [{expert_name}] R1完成: →{r1_culprit} conf={r1_confidence:.2f} "
                     f"reasoning={len(r1_reasoning)}字 未解决={len(r1_unresolved)}")

        # 提前终止检查
        if (r1_confidence >= self.EARLY_STOP_CONFIDENCE
                and len(r1_reasoning) >= self.MIN_REASONING_LEN):
            logger.info(f"🔄 [{expert_name}] 提前终止: conf={r1_confidence:.2f}≥{self.EARLY_STOP_CONFIDENCE}")
            r1_parsed["_rounds"] = rounds_log
            r1_parsed["_total_rounds"] = 1
            r1_parsed["_early_stop"] = True
            return r1_parsed

        # ==========================================
        # Round 2: 自我审视 — 找弱点、偏见、盲点
        # ==========================================
        logger.info(f"🔄 [{expert_name}] Round 2/{self.MAX_ROUNDS}: 自我审视")

        suspects = context.get("suspects", [])
        suspect_names = [s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in suspects]

        r2_prompt = f"""你是一位严格的推理审查员。请审查以下{expert_role}的初步分析，找出**逻辑漏洞、偏见和被忽略的可能性**。

## 初步分析结论
- 指向嫌疑人: {r1_culprit}
- 置信度: {r1_confidence}
- 推理过程: {r1_reasoning}

## 所有嫌疑人
{', '.join(suspect_names)}

## 关键证据
{json.dumps(r1_key_evidence, ensure_ascii=False)}

## 未解决问题
{json.dumps(r1_unresolved, ensure_ascii=False)}

请严格以JSON格式返回审查结论:
{{
    "logic_flaws": ["逻辑漏洞1", "逻辑漏洞2"],
    "biases_detected": ["可能的偏见"],
    "overlooked_suspects": ["被忽略的嫌疑人及其理由"],
    "weak_evidence": ["证据链中最薄弱的环节"],
    "missing_analysis": ["需要进一步调查的方向"],
    "confidence_assessment": "置信度是否合理(过高/合理/过低)",
    "adjusted_confidence": 0.0-1.0,
    "verdict": "维持/需要深入/需要推翻"
}}

要求:
1. 必须扮演魔鬼代言人，即使结论看起来合理也要找茬
2. 特别关注其他嫌疑人是否被不公平排除
3. 检查是否存在确认偏差
4. 评估置信度是否合理
5. 只返回JSON"""

        r2_response = self.call_llm(r2_prompt, temperature=0.3)
        r2_parsed = self.extract_json_from_response(r2_response)

        if not r2_parsed or not isinstance(r2_parsed, dict):
            logger.warning(f"🔄 [{expert_name}] R2 JSON解析失败，使用R1结果")
            r1_parsed["_rounds"] = rounds_log
            r1_parsed["_total_rounds"] = 1
            return r1_parsed

        r2_verdict = r2_parsed.get("verdict", "维持")
        r2_flaws = r2_parsed.get("logic_flaws", [])
        r2_overlooked = r2_parsed.get("overlooked_suspects", [])
        r2_missing = r2_parsed.get("missing_analysis", [])
        r2_adj_conf = r2_parsed.get("adjusted_confidence", r1_confidence)

        rounds_log.append({
            "round": 2, "phase": "self_review",
            "verdict": r2_verdict, "flaws": len(r2_flaws),
            "overlooked": len(r2_overlooked), "missing": len(r2_missing),
        })
        logger.info(f"🔄 [{expert_name}] R2完成: verdict={r2_verdict} "
                     f"漏洞={len(r2_flaws)} 忽略={len(r2_overlooked)} "
                     f"需深入={len(r2_missing)} adj_conf={r2_adj_conf:.2f}")

        # 审查通过 → 维持Round 1，调整置信度
        if r2_verdict == "维持" and not r2_flaws and not r2_overlooked:
            logger.info(f"🔄 [{expert_name}] R2审查通过，提前结束")
            r1_parsed["confidence"] = r2_adj_conf
            r1_parsed["reasoning"] = (
                f"[多轮推理 R1→R2审查通过]\n\n{r1_reasoning}\n\n"
                f"【自我审视】经过严格审查，未发现重大逻辑漏洞或偏见。"
            )
            r1_parsed["_rounds"] = rounds_log
            r1_parsed["_total_rounds"] = 2
            return r1_parsed

        # ==========================================
        # Round 3: 深入调查 — 聚焦R2发现的疑点
        # ==========================================
        logger.info(f"🔄 [{expert_name}] Round 3/{self.MAX_ROUNDS}: 深入调查")

        # 从context提取原始证据用于深入分析
        sk = context.get("structured_knowledge", {})
        search = context.get("search_results", {})
        ev_data = sk.get("evidence", {}).get("data", {}) if isinstance(sk.get("evidence"), dict) else {}
        tl_data = sk.get("timeline", {}).get("data", {}) if isinstance(sk.get("timeline"), dict) else {}
        pr_data = sk.get("person_relation", {}).get("data", {}) if isinstance(sk.get("person_relation"), dict) else {}

        flaws_text = "\n".join(f"  - {f}" for f in r2_flaws[:5])
        overlooked_text = "\n".join(f"  - {o}" for o in r2_overlooked[:5])
        missing_text = "\n".join(f"  - {m}" for m in r2_missing[:5])

        r3_prompt = f"""你是{expert_role}，现在进入深入调查阶段。
审查员发现了你初步分析中的问题，你需要重新审视。

## 你之前的结论: {r1_culprit} (置信度={r1_confidence})

## 审查发现的逻辑漏洞
{flaws_text if flaws_text else "无"}

## 审查发现被忽略的嫌疑人
{overlooked_text if overlooked_text else "无"}

## 审查建议的深入方向
{missing_text if missing_text else "无"}

## 原始证据（用于重新审视）
物证:
{json.dumps(ev_data.get('physical_evidence', []), ensure_ascii=False, indent=2)[:800]}

数字证据:
{json.dumps(ev_data.get('digital_evidence', []), ensure_ascii=False, indent=2)[:500]}

证言:
{json.dumps(ev_data.get('testimony', []), ensure_ascii=False, indent=2)[:600]}

时间线:
{json.dumps(tl_data.get('events', []), ensure_ascii=False, indent=2)[:600]}

人物关系:
{json.dumps(pr_data.get('relations', []), ensure_ascii=False, indent=2)[:600]}

动机排名:
{json.dumps(search.get('motive', {}).get('data', {}).get('ranking', []), ensure_ascii=False)[:400]}

机会排名:
{json.dumps(search.get('opportunity', {}).get('data', {}).get('ranking', []), ensure_ascii=False)[:400]}

请重新进行深入分析，特别关注:
1. 之前忽略的嫌疑人是否更有嫌疑？
2. 逻辑漏洞是否影响了最终结论？
3. 是否存在未考虑的作案可能性？

请严格以JSON格式返回:
{{
    "culprit": "修正后的真凶（可以改变）",
    "confidence": 0.0-1.0,
    "reasoning": "重新推理过程（必须说明为什么维持或改变结论）",
    "key_changes": ["相比初步分析的主要改变"],
    "evidence_reinterpretation": ["对证据的重新解读"],
    "answer_to_flaws": ["对审查发现的逻辑漏洞的回应"],
    "overlooked_suspect_analysis": "对被忽略嫌疑人的分析结论",
    "key_evidence": ["支撑最终结论的关键证据"],
    "unresolved_questions": ["仍然无法解答的问题"]
}}

要求:
1. 如果审查问题不足以改变结论，维持原判但说明理由
2. 如果发现新的重要证据指向其他人，果断改变结论
3. 必须逐一回应逻辑漏洞
4. 只返回JSON"""

        r3_response = self.call_llm(r3_prompt, temperature=temperature)
        r3_parsed = self.extract_json_from_response(r3_response)

        if not r3_parsed or not isinstance(r3_parsed, dict):
            logger.warning(f"🔄 [{expert_name}] R3 JSON解析失败，使用R1+R2修正")
            r1_parsed["confidence"] = r2_adj_conf
            r1_parsed["reasoning"] = (
                f"[多轮推理 R1→R2发现{len(r2_flaws)}个问题→R3失败]\n\n{r1_reasoning}\n\n"
                f"【自我审视】发现{len(r2_flaws)}个逻辑漏洞但深入调查失败，保守降低置信度。"
            )
            r1_parsed["_rounds"] = rounds_log
            r1_parsed["_total_rounds"] = 2
            return r1_parsed

        r3_culprit = r3_parsed.get("culprit", r1_culprit)
        r3_confidence = r3_parsed.get("confidence", r1_confidence)
        r3_reasoning = r3_parsed.get("reasoning", "")
        changed = r3_culprit != r1_culprit

        rounds_log.append({
            "round": 3, "phase": "deep_investigation",
            "culprit": r3_culprit, "confidence": r3_confidence,
            "changed": changed,
        })

        if changed:
            logger.info(f"🔄 [{expert_name}] R3结论改变! {r1_culprit} → {r3_culprit}")
        else:
            logger.info(f"🔄 [{expert_name}] R3维持: {r3_culprit} conf={r3_confidence:.2f}")

        # ==========================================
        # Round 4: 最终整合
        # ==========================================
        logger.info(f"🔄 [{expert_name}] Round 4/{self.MAX_ROUNDS}: 最终整合")

        r4_prompt = f"""你是{expert_role}，已完成三轮分析。请做最终整合。

## 三轮分析汇总
- R1 初步: {r1_culprit} (conf={r1_confidence})
- R2 审查: {r2_verdict}, {len(r2_flaws)}个漏洞, {len(r2_overlooked)}个被忽略
- R3 深入: {r3_culprit} (conf={r3_confidence}){' ⚠️改变了结论!' if changed else ''}

## R3的推理
{r3_reasoning[:800]}

## R3对逻辑漏洞的回应
{json.dumps(r3_parsed.get('answer_to_flaws', []), ensure_ascii=False)[:400]}

## R3关键证据
{json.dumps(r3_parsed.get('key_evidence', []), ensure_ascii=False)[:400]}

请综合三轮分析给出最终判断:
{{
    "culprit": "最终真凶判断",
    "confidence": 0.0-1.0,
    "reasoning": "综合三轮的完整推理过程",
    "key_evidence": ["支撑结论的关键证据"],
    "unresolved_questions": ["仍无法解答的问题"]
}}

要求:
1. reasoning必须整合三轮关键发现
2. 如果结论在轮次间摇摆，应降低confidence
3. 只返回JSON"""

        r4_response = self.call_llm(r4_prompt, temperature=0.3)
        r4_parsed = self.extract_json_from_response(r4_response)

        if not r4_parsed or not isinstance(r4_parsed, dict):
            logger.warning(f"🔄 [{expert_name}] R4 JSON解析失败，使用R3结果")
            r3_parsed["reasoning"] = (
                f"[多轮推理 R1={r1_culprit}({r1_confidence:.2f})→"
                f"R2({r2_verdict})→"
                f"R3={r3_culprit}({r3_confidence:.2f})]\n\n{r3_reasoning}"
            )
            r3_parsed["_rounds"] = rounds_log
            r3_parsed["_total_rounds"] = 3
            return r3_parsed

        final_culprit = r4_parsed.get("culprit", r3_culprit)
        final_confidence = r4_parsed.get("confidence", r3_confidence)
        final_reasoning = r4_parsed.get("reasoning", r3_reasoning)

        rounds_log.append({
            "round": 4, "phase": "final",
            "culprit": final_culprit, "confidence": final_confidence,
        })

        # 添加多轮推理前缀到reasoning
        r4_parsed["culprit"] = final_culprit
        r4_parsed["confidence"] = final_confidence
        r4_parsed["reasoning"] = (
            f"[多轮推理 R1={r1_culprit}({r1_confidence:.2f})→"
            f"R2({r2_verdict},{len(r2_flaws)}漏洞)→"
            f"R3={r3_culprit}({r3_confidence:.2f})→"
            f"R4={final_culprit}({final_confidence:.2f})]\n\n{final_reasoning}"
        )
        r4_parsed["_rounds"] = rounds_log
        r4_parsed["_total_rounds"] = 4

        logger.info(f"🔄 [{expert_name}] R4完成: {final_culprit} ({final_confidence:.2f})")

        # ==========================================
        # Round 5-10: 动态深入轮 — 当置信度不够高时继续深入
        # 每轮聚焦一个新角度，直到置信度≥阈值或达到MAX_ROUNDS
        # ==========================================
        current_culprit = final_culprit
        current_confidence = final_confidence
        current_reasoning = final_reasoning
        current_result = r4_parsed
        
        # 不同的审视角度，轮流使用
        review_angles = [
            ("时间线验证", "严格验证时间线是否完全排除其他嫌疑人"),
            ("动机深度分析", "重新评估所有嫌疑人的动机强度和可信度"),
            ("证据链完整性", "检查证据链是否有断裂，是否有其他解释"),
            ("行为逻辑审查", "分析嫌疑人案发前后的行为是否合理"),
            ("间接证据交叉验证", "将所有间接证据交叉对比，寻找矛盾"),
            ("心理学分析", "从犯罪心理学角度分析真凶的行为模式"),
        ]

        for round_num in range(5, self.MAX_ROUNDS + 1):
            # 提前终止检查
            if current_confidence >= self.EARLY_STOP_CONFIDENCE:
                logger.info(f"🔄 [{expert_name}] R{round_num}提前终止: conf={current_confidence:.2f}≥{self.EARLY_STOP_CONFIDENCE}")
                break

            angle_name, angle_desc = review_angles[(round_num - 5) % len(review_angles)]
            logger.info(f"🔄 [{expert_name}] Round {round_num}/{self.MAX_ROUNDS}: {angle_name}")

            rN_prompt = f"""你是{expert_role}，已进行{round_num-1}轮分析。现在进行「{angle_name}」。

## 当前结论: {current_culprit} (置信度={current_confidence})
## 已有推理历程:
{current_reasoning[:1000]}

## 审查焦点: {angle_desc}

## 所有嫌疑人
{', '.join(suspect_names)}

请从「{angle_name}」角度重新审视当前结论:
{{
    "culprit": "最终判断（可以维持或改变）",
    "confidence": 0.0-1.0,
    "reasoning": "本轮分析的推理过程",
    "key_insight": "本轮发现的最关键洞察",
    "conclusion_changed": true/false,
    "confidence_trend": "上升/持平/下降"
}}

要求:
1. 聚焦{angle_name}角度，不要泛泛而谈
2. 如果当前结论经得起审查，只微调置信度
3. 如果发现严重问题，果断改变结论
4. 只返回JSON"""

            rN_response = self.call_llm(rN_prompt, temperature=0.3)
            rN_parsed = self.extract_json_from_response(rN_response)

            if not rN_parsed or not isinstance(rN_parsed, dict):
                logger.warning(f"🔄 [{expert_name}] R{round_num} JSON解析失败，保持当前结论")
                continue

            rN_culprit = rN_parsed.get("culprit", current_culprit)
            rN_confidence = rN_parsed.get("confidence", current_confidence)
            changed = rN_culprit != current_culprit

            rounds_log.append({
                "round": round_num, "phase": angle_name,
                "culprit": rN_culprit, "confidence": rN_confidence,
                "changed": changed,
            })

            if changed:
                logger.info(f"🔄 [{expert_name}] R{round_num}结论改变! {current_culprit} → {rN_culprit}")
            else:
                logger.info(f"🔄 [{expert_name}] R{round_num}维持: {rN_culprit} conf={rN_confidence:.2f}")

            current_culprit = rN_culprit
            current_confidence = rN_confidence
            current_reasoning = rN_parsed.get("reasoning", current_reasoning)
            current_result = rN_parsed

        # 最终整合
        actual_rounds = len(rounds_log)
        current_result["culprit"] = current_culprit
        current_result["confidence"] = current_confidence
        
        # 构建推理演进摘要
        evolution = "→".join([f"R{r['round']}({r.get('culprit','?')[:3]})" for r in rounds_log])
        existing_reasoning = current_result.get("reasoning", current_reasoning)
        current_result["reasoning"] = (
            f"[多轮推理 {evolution}]\n\n{existing_reasoning}"
        )
        current_result["_rounds"] = rounds_log
        current_result["_total_rounds"] = actual_rounds
        current_result["_early_stop"] = actual_rounds < self.MAX_ROUNDS

        logger.info(f"🔄 [{expert_name}] 多轮完成: R1={r1_culprit}→R{actual_rounds}={current_culprit} "
                     f"({current_confidence:.2f}), {actual_rounds}轮")

        return current_result
