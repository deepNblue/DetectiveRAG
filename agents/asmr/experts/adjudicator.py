"""
裁判Agent (Adjudicator)
汇总所有Expert意见（含多变体），进行最终裁决
借鉴 Supermemory ASMR 的 "并行集群 + 聚合裁判" 思路
"""

import json
from typing import Dict, List, Any
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from agents.base_agent import BaseAgent


class Adjudicator(BaseAgent):
    """
    裁判Agent — 最终裁决者
    
    职责:
    1. 审查所有Expert结论（含多变体结论）
    2. 识别分歧焦点，分析分歧原因
    3. 结合矛盾搜索结果，做出最终裁决
    4. 给出裁决理由和置信度
    """

    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="ASMR-Adjudicator", config=config, llm_client=llm_client)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        裁决所有Expert意见

        Args:
            input_data: {
                "expert_results": list,          # 所有Expert结果
                "vote_result": dict,             # 投票结果
                "contradiction_data": dict,       # 矛盾搜索结果
                "structured_knowledge": dict,     # 结构化知识
                "suspects": list,
                "tree_result": dict,             # 🆕 v8.1: 推理树结果
            }
        """
        self.log_processing(input_data)
        
        expert_results = input_data.get("expert_results", [])
        vote_result = input_data.get("vote_result", {})
        contradiction_data = input_data.get("contradiction_data", {})
        sk = input_data.get("structured_knowledge", {})
        suspects = input_data.get("suspects", [])
        tree_result = input_data.get("tree_result") or {}  # 🆕 v8.1

        suspect_names = [s.get("name", "?") if isinstance(s, dict) else s for s in suspects]

        # 整理Expert意见
        expert_opinions = []
        for r in expert_results:
            data = r.get("data", r)
            expert_opinions.append({
                "perspective": data.get("perspective", "?"),
                "culprit": data.get("culprit", "未知"),
                "confidence": data.get("confidence", 0),
                "reasoning": str(data.get("reasoning", ""))[:300],
            })

        # 矛盾分析摘要
        contradictions = contradiction_data.get("contradictions", [])
        anomaly_summary = contradiction_data.get("anomaly_summary", {})
        contradiction_ranking = contradiction_data.get("ranking", [])
        contradiction_insight = contradiction_data.get("key_insight", "")

        # 投票分布
        vote_distribution = vote_result.get("vote_distribution", {})
        vote_winner = vote_result.get("winner", "未知")
        consensus = vote_result.get("consensus_level", "未知")

        # 🆕 v9.1: 分层投票详情
        investigation_vote = input_data.get("investigation_vote", {})
        trial_vote = input_data.get("trial_vote", {})

        # 构建调查层投票摘要
        inv_dist = investigation_vote.get("vote_distribution", {})
        inv_winner = investigation_vote.get("winner", "未知")
        investigation_summary_text = f"共识={inv_winner}, 分布={json.dumps(inv_dist, ensure_ascii=False)}"

        # 构建审判层投票摘要 — 特别标注辩护律师意见
        trial_dist = trial_vote.get("vote_distribution", {})
        trial_winner = trial_vote.get("winner", "未知")
        # 从审判层专家中提取辩护律师意见
        defense_opinion = "未知"
        for r in expert_results:
            data = r.get("data", r)
            if data.get("perspective") == "defense_attorney":
                defense_opinion = f"{data.get('culprit', '未知')} (置信度={data.get('confidence', 0):.2f})"
        trial_summary_text = f"共识={trial_winner}, 分布={json.dumps(trial_dist, ensure_ascii=False)}, 辩护律师独立意见={defense_opinion}"

        # 🆕 v8.1: 推理树阶段性结论
        tree_culprit = tree_result.get("tree_culprit", "未知")
        tree_confidence = tree_result.get("tree_confidence", 0)
        tree_different = tree_result.get("is_different_from_vote", False)
        tree_stats = tree_result.get("stats", {}) or {}
        tree_overturned = tree_result.get("tree_overturned", False)
        
        # 构建推理树摘要
        tree_summary_lines = []
        tree_summary_lines.append(f"推理树最终结论: {tree_culprit} (置信度={tree_confidence:.3f})")
        if tree_different:
            tree_summary_lines.append(f"⚠️ 推理树与投票不同! 投票={vote_winner}, 推理树={tree_culprit}")
        if tree_overturned:
            tree_summary_lines.append(f"😈 推理树内部被恶魔代言人推翻过!")
        tree_summary_lines.append(f"与投票结果: {'不同 ❗' if tree_different else '一致 ✅'}")
        
        # 竞争排序各维度赢家
        dimension_winners = tree_result.get("dimension_winners", {})
        if dimension_winners:
            tree_summary_lines.append("各维度竞争赢家:")
            for dim, winner in dimension_winners.items():
                tree_summary_lines.append(f"  {dim}: {winner}")
        
        # 恶魔代言人结果
        devil_result = tree_result.get("devil_result")
        if devil_result:
            tree_summary_lines.append(f"恶魔代言人: {'要求推翻' if devil_result.get('should_overturn') else '维持原判'}"
                                     f" (力度={devil_result.get('overturn_strength', 0):.2f})")
        
        # 尝试获取阶段性信息
        best_path = tree_result.get("best_path", {}) or {}
        if best_path and isinstance(best_path, dict):
            chain_complete = best_path.get("chain_complete", 0)
            hypothesis_valid = best_path.get("hypothesis_valid", True)
            unique_evidence = best_path.get("unique_evidence", []) or []
            chain_gaps = best_path.get("chain_gaps", []) or []
            tree_summary_lines.append(f"证据链完整度: {chain_complete:.2f} (假设={'有效' if hypothesis_valid else '无效'})")
            if unique_evidence and isinstance(unique_evidence, list):
                tree_summary_lines.append(f"独有证据: {'; '.join(str(e) for e in unique_evidence[:3])}")
            if chain_gaps and isinstance(chain_gaps, list):
                tree_summary_lines.append(f"链缺口: {'; '.join(str(g) for g in chain_gaps[:3])}")
        
        # 候选人排名
        candidate_scores = tree_result.get("candidate_scores", {})
        if candidate_scores:
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            tree_summary_lines.append("候选人竞争排名:")
            for i, (name, score) in enumerate(sorted_candidates):
                marker = " 👑" if name == tree_culprit else ""
                if i > 0:
                    gap = sorted_candidates[i-1][1] - score
                    tree_summary_lines.append(f"  {name}: {score:.3f}{marker} (与上一名差距={gap:.3f})")
                else:
                    tree_summary_lines.append(f"  {name}: {score:.3f}{marker}")
            
            # 🆕 v9.1: 如果前两名差距很小，特别标注
            if len(sorted_candidates) >= 2:
                top_gap = sorted_candidates[0][1] - sorted_candidates[1][1]
                if top_gap < 0.1:
                    tree_summary_lines.append(f"⚠️ 前两名差距极小({top_gap:.3f})，证据不足以确认任何一人！")
        
        tree_summary = "\n".join(tree_summary_lines)
        tree_reliable = tree_confidence > 0.4 and tree_stats.get("llm_calls", 0) >= 4
        
        # 从矛盾搜索结果中提取合谋信息
        collusion_detected = contradiction_data.get("collusion_detected", False)
        collusion_suspects = contradiction_data.get("collusion_suspects", [])

        # 🆕 v12.0: 反向排除结果
        elimination_result = input_data.get("elimination_result", {})
        elimination_check = elimination_result.get("elimination_check", []) if elimination_result else []
        blind_spot_found = elimination_result.get("blind_spot_found", False) if elimination_result else False
        blind_spot_detail = elimination_result.get("blind_spot_detail", "") if elimination_result else ""
        vote_winner_weakness = elimination_result.get("vote_winner_weakness", "") if elimination_result else ""
        elimination_recommendation = elimination_result.get("recommendation", "维持原判") if elimination_result else "维持原判"

        # 🆕 v12.0: 分歧分析
        divergence_analysis = input_data.get("divergence_analysis", None)
        divergence_text = ""
        if divergence_analysis and divergence_analysis.get("has_divergence"):
            divergence_text = f"""
⚠️⚠️⚠️ v12.0 分歧检测: 调查层和审判层意见不一致!
- 调查层指向: {divergence_analysis.get('investigation_winner', '?')}
- 审判层指向: {divergence_analysis.get('trial_winner', '?')}
- 分歧类型: {divergence_analysis.get('divergence_type', '?')}
→ 这种分歧往往意味着真凶可能是少数派指向的那个人，请特别关注审判层的独立判断!
"""

        # 🆕 v12.0: 反向排除文本
        elimination_text = ""
        if elimination_check:
            elimination_lines = []
            for ec in elimination_check:
                could_be = "⚠️ 可能是真凶!" if ec.get("could_be_real_culprit") else "✅ 可排除"
                elimination_lines.append(
                    f"  {ec.get('suspect', '?')}: {could_be} "
                    f"(有不在场证明={ec.get('has_alibi', False)}, "
                    f"物证排除={ec.get('physical_exclusion', False)}, "
                    f"排除置信度={ec.get('confidence_in_exclusion', 0):.2f})"
                )
                if ec.get("reason"):
                    elimination_lines.append(f"    原因: {ec['reason']}")
            elimination_text = f"""
=== 🔄 v12.0 反向排除验证结果 (独立于投票的盲点检测) ===
{chr(10).join(elimination_lines)}
投票赢家弱点: {vote_winner_weakness}
发现盲点: {"⚠️ 是 — " + blind_spot_detail if blind_spot_found else "否"}
建议: {elimination_recommendation}
⚠️ 如果blind_spot_found=true或recommendation="需要重新审视"，你必须认真考虑投票赢家可能不是真凶!
"""
        _vote_dist = vote_distribution if vote_distribution else {}
        vote_total_weight = sum(_vote_dist.values()) if _vote_dist else 1
        sorted_votes = sorted(_vote_dist.items(), key=lambda x: x[1], reverse=True) if _vote_dist else []
        top1_name, top1_score = sorted_votes[0] if sorted_votes else ("?", 0)
        top2_name, top2_score = sorted_votes[1] if len(sorted_votes) > 1 else ("?", 0)
        top1_pct = top1_score / vote_total_weight * 100 if vote_total_weight else 0
        gap_pct = (top1_score - top2_score) / vote_total_weight * 100 if vote_total_weight and len(sorted_votes) > 1 else 100

        # 三层一致性
        inv_w = investigation_vote.get("winner", "?")
        trial_w = trial_vote.get("winner", "?")
        three_agree = (inv_w == trial_w == vote_winner)
        all_expert_count = len(expert_opinions)
        winner_support_count = sum(1 for o in expert_opinions if o.get("culprit") == vote_winner)

        prompt = f"""你是最终的裁决法官。你需要谨慎裁决，尊重证据和多数专家的集体智慧。

⚠️ 核心裁决原则（按优先级排序）:
1. **当多层投票一致时，默认信任投票结果** — 如果调查层、审判层、联合投票都指向同一人，除非你有**具体且可验证的物证**反驳，否则不应推翻
2. **推翻需要硬证据，不是推测** — "可能是另一个人"不是推翻理由，你需要指出具体证据矛盾
3. **投票领先幅度大时不要推翻** — 如果赢家得票>60%且领先第二名>20%，推翻需要特别强的理由
4. **专家数量优势很重要** — {winner_support_count}/{all_expert_count}个专家支持{vote_winner}，他们的集体判断不应轻易否定
5. **矛盾分析是补充，不是主导** — 矛盾/异常可以提醒你注意，但不能仅凭"有矛盾"就推翻共识
6. **不要被常见名字误导** — 张伟、张强、李芳、赵敏等常见名字不代表他们就是真凶
7. ⭐ **v12.0: 重视反向排除结果** — 如果反向排除验证发现盲点或建议"需要重新审视"，你必须认真考虑推翻投票
8. ⭐ **v12.0: 分歧意味着可能性** — 如果调查层和审判层指向不同人，真凶可能就在分歧中

⚠️ 当前投票一致性分析:
- 调查层赢家: {inv_w}
- 审判层赢家: {trial_w}
- 联合投票赢家: {vote_winner} (得票{top1_pct:.1f}%)
- 三层是否一致: {"✅ 是 — 三层一致，推翻需要极强理由！" if three_agree else "❌ 否 — 存在分歧，需要仔细判断"}
- 投票领先幅度: {gap_pct:.1f}% {"(大幅领先，不要推翻)" if gap_pct > 30 else "(小幅领先，可以推翻)" if gap_pct < 15 else "(中等领先，需要充分理由才能推翻)"}

⚠️ 审判层角色说明:
- 🔴 检察官: 倾向有罪推定，通常跟从调查层主流
- 🔵 辩护律师: 无罪推定，会提出不同意见（这是正常对抗，不代表调查层错了）
- 🧑‍⚖️ 法官: 中立裁判
- 👥 陪审员: 常识判断

⚠️ 推理树参考（独立验证，但不是绝对权威）:
- 推理树是辅助工具，不是最终答案
- 当推理树与投票一致时，结论更可靠
- 当推理树与投票不同时，**不要自动跟从推理树** — 要看推理树的证据链是否真的比专家集体分析更强
- 推理树置信度<0.5时，其结论参考价值有限

⚠️ 高级推理模式:
- **嫁祸模式**: 物证指向A + A有不在场证明 + 有人能伪造A的证据 → 真凶是伪造者
- **多罪犯合谋**: 两人各掌握不同犯罪要素 → 用"+"连接
- **"狗没叫"原则**: 权限被使用但本人不在场 → 真凶盗用了身份

=== 嫌疑人 ===
{', '.join(suspect_names)}

=== 专家投票结果 (仅供参考，不是最终答案) ===
投票赢家: {vote_winner} (共识: {consensus})
投票分布: {json.dumps(vote_distribution, ensure_ascii=False)}

=== 🔬 分层投票详情 (v9.1新增) ===
调查层(调查专家+名侦探+领域专家): {investigation_summary_text}
审判层(检察官+辩护律师+法官+陪审员): {trial_summary_text}

=== 各专家意见 ===
{json.dumps(expert_opinions, ensure_ascii=False, indent=2)}

=== ⭐ 矛盾/异常分析 (这是最关键的证据) ===
发现矛盾: {len(contradictions)}个
矛盾详情: {json.dumps(contradictions, ensure_ascii=False, indent=2)[:1000]}
最可疑(基于矛盾): {anomaly_summary.get('most_suspicious', '未知')}
关键矛盾: {anomaly_summary.get('key_contradiction', '无')}
隐藏线索: {anomaly_summary.get('hidden_culprit_hint', '无')}
矛盾排名: {contradiction_ranking}
矛盾洞察: {contradiction_insight}
合谋检测: {'检测到合谋! 嫌疑人: ' + ' + '.join(collusion_suspects) if collusion_detected else '未明确检测到合谋，但仍需手动检查'}
合谋证据: {contradiction_data.get('collusion_evidence', '无')}

=== 🌳 推理树验证结论 (独立于专家投票的系统性证据链分析) ===
{tree_summary}
推理树可靠性: {'高 (置信度>{tree_confidence:.2f} 且 LLM调用>{tree_stats.get("llm_calls", 0)}次)' if tree_reliable else '中/低'}
{elimination_text}
{divergence_text}

请独立思考以下问题后再做裁决:
1. 投票赢家是否真的有直接证据链？还是仅仅因为"有能力和动机"？
2. 矛盾分析指向的人是否有更直接的证据(如亲手制作含毒物品)？
3. 是否存在专家集体忽略的关键证据？
4. ⭐ 推理树的结论是否可信？它的证据链（先验→后验→校准）是否逻辑自洽？
5. 如果投票和推理树指向不同人，谁的证据链更强？请具体比较。
6. ⭐ 是否存在两个嫌疑人合谋的可能？检查: 是否有人有经济动机但缺少泄密渠道，另有人有泄密渠道但缺少直接经济动机？合谋是否能让犯罪链条完整？

请严格以JSON格式返回裁决:
{{
    "final_culprit": "最终认定的真凶。多罪犯合谋用+号连接。否则只写一个人名。",
    "is_multi_culprit": true或false,
    "culprit_roles": {{"人A": "角色", "人B": "角色"}},
    "final_confidence": 0.0-1.0,
    "verdict_type": "一致裁决/多数裁决/分歧裁决/推翻裁决/合谋裁决",
    "reasoning": "裁决理由",
    "key_evidence": ["支持裁决的关键证据"],
    "why_not_vote_winner": "如果不选择投票赢家，说明具体原因（必须有具体物证矛盾，不能只是推测）",
    "tree_agreement": "同意推理树结论/不同意推理树结论",
    "tree_disagreement_reason": "如果不同意推理树的理由",
    "divergence_analysis": "专家分歧原因",
    "overlooked_evidence": ["专家忽略的关键证据"],
    "warning": "风险点"
}}

只返回JSON"""

        response = self.call_llm(prompt, temperature=0.2)
        parsed = self.extract_json_from_response(response)

        if parsed and isinstance(parsed, dict):
            final_culprit = parsed.get("final_culprit", vote_winner)
            final_confidence = parsed.get("final_confidence", vote_result.get("confidence", 0.3))
            verdict_type = parsed.get("verdict_type", "未知")
            reasoning = parsed.get("reasoning", "")
            is_multi_culprit = parsed.get("is_multi_culprit", False)
            culprit_roles = parsed.get("culprit_roles", {})
        else:
            final_culprit = vote_winner
            final_confidence = vote_result.get("confidence", 0.3)
            verdict_type = "解析失败"
            reasoning = "裁判Agent解析失败，沿用投票结果"
            is_multi_culprit = False
            culprit_roles = {}

        # 检测多罪犯: 如果final_culprit含+号
        if "+" in str(final_culprit):
            is_multi_culprit = True
            if verdict_type not in ("合谋裁决",):
                verdict_type = "合谋裁决"

        self.logger.info(f"裁判裁决: 真凶={final_culprit}, 置信度={final_confidence}, 类型={verdict_type}, 多罪犯={is_multi_culprit}")
        if final_culprit != vote_winner:
            self.logger.info(f"  ⚡ 裁判推翻了投票结果! 投票={vote_winner} → 裁判={final_culprit}")
        if is_multi_culprit:
            self.logger.info(f"  🔗 裁判检测到多罪犯合谋: {final_culprit}")
        
        # 🆕 v8.1: 裁判推翻推理树+投票一致结论的警告
        tree_culprit = tree_result.get("tree_culprit", "未知")
        if not tree_different and final_culprit != tree_culprit and tree_reliable:
            self.logger.warning(f"  ⚠️ 裁判推翻了投票+推理树一致结论! "
                              f"投票={vote_winner}, 推理树={tree_culprit}(可靠), 裁判={final_culprit}")
            self.logger.warning(f"  ⚠️ 裁判理由: {reasoning[:200]}")
            tree_disagreement = parsed.get("tree_disagreement_reason", "") if parsed else ""
            if tree_disagreement:
                self.logger.warning(f"  ⚠️ 反对推理树的理由: {tree_disagreement[:200]}")

        return self.format_output({
            "perspective": "adjudicator",
            "culprit": final_culprit,
            "confidence": final_confidence,
            "reasoning": reasoning,
            "is_multi_culprit": is_multi_culprit,
            "culprit_roles": culprit_roles,
            "detail": {
                "verdict_type": verdict_type,
                "vote_winner": vote_winner,
                "overturned": final_culprit != vote_winner,
                "overlooked_evidence": parsed.get("overlooked_evidence", []) if parsed else [],
                "warning": parsed.get("warning", "") if parsed else "",
            }
        })
