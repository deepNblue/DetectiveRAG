"""
专家投票引擎
汇总多个Expert的结论，通过加权投票得出最终判断
"""

import json
from typing import Dict, List, Any, Optional
from loguru import logger
from collections import defaultdict
from .name_utils import (normalize_name, is_valid_suspect, merge_name_variants,
                          merge_name_details, split_multiple_names, build_name_alias_map)


class ExpertVotingEngine:
    """
    专家投票引擎
    - 每个专家给出 {culprit, confidence, reasoning}
    - LogicVerifier可调整其他专家的confidence
    - 加权投票得出最终结论
    - v12.1: 名字模糊匹配 — 用嫌疑人列表补全截断名字
    """

    # 专家权重（可配置, v9.1: 三层架构 — 调查层降权 + 审判层大幅加权 + 少数派放大）
    DEFAULT_WEIGHTS = {
        # === 调查层（降低基础权重，因为人数多容易产生群体思维）===
        "forensic": 1.0,                # 法医：物证说话
        "criminal_investigation": 0.9,   # 刑侦：证据链（降权，太容易跟风）
        "psychological_profiling": 0.7,   # 心理画像：主观性较强
        "tech_investigation": 1.0,       # 技侦：数字证据
        "financial_investigation": 0.85,  # 经侦：资金链追踪
        "interrogation_analysis": 0.8,  # 审讯分析：供述可信度
        "intelligence_analysis": 0.85,    # 情报分析：关联研判
        # 名侦探（权重略高，因为他们是独立推理）
        "sherlock_analysis": 1.0,       # 福尔摩斯：演绎推理
        "henry_lee_analysis": 1.0,       # 李昌钰：鉴识科学
        "song_ci_analysis": 0.9,        # 宋慈：传统法医鉴识
        "poirot_analysis": 0.95,          # 波洛：心理侦探
        # 验证
        "logic_verification": 1.1,       # 逻辑验证：修正性（降权，因为LogicVerifier也只是LLM）
        # === 审判层（v9.1: 大幅加权 — 他们看到全部调查层结论后独立判断）===
        "prosecution_review": 1.8,       # 🔴 检察官：综合指控，证据链闭合
        "defense_attorney": 2.5,         # 🔵 辩护律师：对抗确认偏差（最高权重！）
        "judge": 2.0,                    # 🧑‍⚖️ 法官：中立裁判
        "juror": 1.5,                    # 👥 陪审员：常识判断
    }

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.logger = logger.bind(module="ExpertVotingEngine")

    def set_dynamic_weight(self, perspective: str, weight: float):
        """设置动态专家的投票权重（v5新增）"""
        self.weights[perspective] = weight
        self.logger.info(f"动态权重设置: {perspective} = {weight}")

    def vote(self, expert_results: List[Dict[str, Any]], suspect_names: List[str] = None) -> Dict[str, Any]:
        """
        对专家结论进行加权投票

        Args:
            expert_results: 各Expert的输出列表，每个包含 data.culprit, data.confidence 等
            suspect_names: 嫌疑人全名列表，用于名字模糊匹配和截断补全

        Returns:
            投票结果
        """
        self.logger.info(f"开始投票，共 {len(expert_results)} 个专家")
        suspect_names = suspect_names or []

        # Step 1: 提取各专家结论（含名字模糊匹配补全）
        expert_conclusions = []
        for result in expert_results:
            data = result.get("data", result)
            perspective = data.get("perspective", "unknown")
            culprit = data.get("culprit", "未知")
            confidence = float(data.get("confidence", 0.3))
            weight = self.weights.get(perspective, 1.0)

            # 🆕 v12.1: 名字模糊匹配 — 补全截断名字
            # 如果LLM返回的名字是嫌疑人全名的子串，补全为全名
            # 例如: "李" → "李经理", "张" → "张律师", "赵" → "赵商人"
            culprit = self._fuzzy_match_name(culprit, suspect_names)

            expert_conclusions.append({
                "perspective": perspective,
                "culprit": culprit,
                "raw_confidence": confidence,
                "weight": weight,
                "weighted_score": confidence * weight,
                "reasoning": data.get("reasoning", ""),
            })

        self.logger.info(f"专家结论: {[(e['perspective'], e['culprit'], e['raw_confidence']) for e in expert_conclusions]}")

        # Step 2: 检查LogicVerifier是否调整了其他专家的置信度
        logic_verifier_result = None
        for result in expert_results:
            data = result.get("data", result)
            if data.get("perspective") == "logic_verification":
                logic_verifier_result = data
                break

        # 如果LogicVerifier有调整意见，应用调整
        adjusted_conclusions = expert_conclusions.copy()
        if logic_verifier_result:
            detail = logic_verifier_result.get("detail", {})
            if not isinstance(detail, dict):
                detail = {}
            logic_review = detail.get("logic_review", [])
            if isinstance(logic_review, list):
                for review in logic_review:
                    if not isinstance(review, dict):
                        continue
                    expert_perspective = review.get("perspective", review.get("expert", ""))
                    adjusted_conf = review.get("adjusted_confidence", None)
                    if adjusted_conf is not None and expert_perspective:
                        for ac in adjusted_conclusions:
                            if ac["perspective"] == expert_perspective:
                                old_conf = ac["raw_confidence"]
                                ac["raw_confidence"] = float(adjusted_conf)
                                ac["weighted_score"] = float(adjusted_conf) * ac["weight"]
                                self.logger.info(f"  LogicVerifier调整: {expert_perspective} "
                                                 f"{old_conf:.2f} → {float(adjusted_conf):.2f}")
                                break

        # Step 3: 加权投票 — 排除无效结论（"无法确定"等），归一化姓名，拆分多人结果
        # 🆕 v9.1: 少数派放大 — 如果某嫌疑人被少数专家支持但置信度高，给予额外权重
        raw_vote_scores = defaultdict(float)
        raw_vote_details = defaultdict(list)

        # 先收集所有出现的原始姓名，用于构建别名映射
        all_raw_names = [ec["culprit"] for ec in adjusted_conclusions]
        alias_map = build_name_alias_map(all_raw_names)

        # 🆕 v9.1: 统计每个嫌疑人获得的有效票数（用于少数派放大）
        suspect_vote_count = defaultdict(int)
        for ec in adjusted_conclusions:
            for culprit_raw in split_multiple_names(ec["culprit"]):
                culprit = normalize_name(culprit_raw)
                culprit = alias_map.get(culprit, culprit)
                if is_valid_suspect(culprit):
                    suspect_vote_count[culprit] += 1

        total_valid_votes = sum(suspect_vote_count.values())
        minority_threshold = total_valid_votes * 0.4 if total_valid_votes > 0 else 0  # 低于40%算少数派

        for ec in adjusted_conclusions:
            raw_culprit = ec["culprit"]
            # 尝试拆分多人（如 "甲、乙" 或 "甲和乙"）
            split_names = split_multiple_names(raw_culprit)
            
            for culprit_raw in split_names:
                culprit = normalize_name(culprit_raw)
                # 应用别名映射（如 "罗伊洛特" → "格里姆斯比·罗伊洛特"）
                culprit = alias_map.get(culprit, culprit)
                
                # 跳过无效结论（"无法确定"、"未知"等）
                if not is_valid_suspect(culprit):
                    self.logger.info(f"  跳过无效结论: {ec['perspective']} → {raw_culprit}")
                    continue
                
                # 🆕 v9.1: 低置信度惩罚 — 置信度<0.3的结论大幅降权
                confidence = ec["raw_confidence"]
                if confidence < 0.3:
                    confidence_penalty = 0.1  # 几乎不算数
                    self.logger.info(f"  ⚠️ 低置信度惩罚: {ec['perspective']} → {culprit} "
                                     f"(conf={ec['raw_confidence']:.2f}, penalty={confidence_penalty})")
                else:
                    confidence_penalty = 1.0
                
                # 多人拆分时，分数均分
                score = ec["weighted_score"] / len(split_names) * confidence_penalty
                
                # 🆕 v9.1: 少数派放大 — 被少数专家支持但高置信度的结论获得额外权重
                vote_count = suspect_vote_count.get(culprit, 0)
                if 0 < vote_count <= minority_threshold and confidence >= 0.7:
                    minority_boost = 1.0 + (1.0 - vote_count / max(total_valid_votes, 1)) * 0.5
                    score *= minority_boost
                    self.logger.info(f"  🔍 少数派放大: {culprit} "
                                     f"(票数={vote_count}/{total_valid_votes}, "
                                     f"boost={minority_boost:.2f})")
                
                raw_vote_scores[culprit] += score
                raw_vote_details[culprit].append({
                    "perspective": ec["perspective"],
                    "confidence": ec["raw_confidence"],
                    "weight": ec["weight"],
                    "weighted_score": score,
                })

        # 归一化姓名（合并 "王福来" 和 "王福来（管家）" 等变体）
        vote_scores = merge_name_variants(dict(raw_vote_scores))
        vote_details = merge_name_details(dict(raw_vote_details))

        # 记录被跳过的专家数
        skipped_experts = sum(1 for ec in adjusted_conclusions
                             if not is_valid_suspect(normalize_name(ec["culprit"])))

        if skipped_experts > 0:
            self.logger.info(f"  跳过 {skipped_experts} 个无效结论专家")

        # Step 4: 归一化
        total_score = sum(vote_scores.values())
        if total_score > 0:
            normalized = {k: round(v / total_score, 4) for k, v in vote_scores.items()}
        else:
            normalized = {}

        # Step 5: 确定赢家（只在有效嫌疑人中选）
        if normalized:
            winner = max(normalized, key=normalized.get)
            winner_score = normalized[winner]
        else:
            winner = "未知"
            winner_score = 0.0

        # Step 6: 判断共识程度
        consensus_level = self._assess_consensus(normalized, expert_conclusions)

        result = {
            "winner": winner,
            "confidence": round(winner_score, 4),
            "vote_distribution": normalized,
            "vote_details": dict(vote_details),
            "consensus_level": consensus_level,
            "total_experts": len(expert_conclusions),
            "agreeing_experts": len([v for v in vote_details.get(winner, [])]),
            "expert_conclusions": expert_conclusions,
        }

        self.logger.info(f"投票结果: 赢家={winner}, 置信度={winner_score:.4f}, 共识={consensus_level}")

        return result

    def _assess_consensus(self, normalized: Dict[str, float], conclusions: List[Dict]) -> str:
        """评估共识程度"""
        if not normalized:
            return "无共识"

        max_score = max(normalized.values())
        num_candidates = len(normalized)

        if max_score >= 0.75 and num_candidates <= 2:
            return "强共识"
        elif max_score >= 0.5:
            return "基本共识"
        elif max_score >= 0.35:
            return "弱共识"
        else:
            return "无共识"

    def _fuzzy_match_name(self, culprit: str, suspect_names: List[str]) -> str:
        """
        🆕 v12.1: 模糊匹配补全截断名字
        
        当LLM返回的名字不完整时（如"李"而非"李经理"），
        尝试在嫌疑人列表中找到匹配的全名。
        
        规则:
        1. 精确匹配 → 直接使用（无需补全）
        2. 单字/短名是嫌疑人全名的姓 → 补全为全名（但需要唯一匹配）
        3. 名字是嫌疑人全名的子串 → 补全为全名（但需要唯一匹配）
        4. 多个匹配 → 不补全（太模糊）
        5. 无匹配 → 保持原样
        """
        if not suspect_names or not culprit:
            return culprit
        
        # 已经精确匹配，无需补全
        if culprit in suspect_names:
            return culprit
        
        # 归一化比较
        norm_culprit = normalize_name(culprit)
        norm_suspects = {normalize_name(s): s for s in suspect_names}
        
        if norm_culprit in norm_suspects:
            return norm_suspects[norm_culprit]
        
        # 模糊匹配: culprit是否是某个嫌疑人名的子串
        matches = []
        for orig_name in suspect_names:
            norm_name = normalize_name(orig_name)
            # culprit的归一化形式是嫌疑人归一化名的一部分
            if len(norm_culprit) >= 2 and norm_culprit in norm_name:
                matches.append(orig_name)
            # 或者反过来: 嫌疑人名以culprit开头（姓氏匹配）
            elif len(norm_culprit) >= 1 and norm_name.startswith(norm_culprit):
                # 但至少要是姓氏级别（中文姓氏1-2字），嫌疑人名至少比culprit长1字
                if len(norm_name) > len(norm_culprit):
                    matches.append(orig_name)
        
        # 唯一匹配 → 补全
        if len(matches) == 1:
            matched = matches[0]
            self.logger.info(f"  🔤 名字补全: '{culprit}' → '{matched}'")
            return matched
        elif len(matches) > 1:
            self.logger.info(f"  ⚠️ 名字模糊匹配多个候选: '{culprit}' → {matches}, 不补全")
        
        return culprit

    def get_report(self, vote_result: Dict[str, Any]) -> str:
        """生成投票报告文本"""
        lines = [
            "=== ASMR专家投票结果 ===",
            f"最终结论: 真凶为 **{vote_result['winner']}**",
            f"综合置信度: {vote_result['confidence']:.2%}",
            f"共识程度: {vote_result['consensus_level']}",
            f"同意专家数: {vote_result['agreeing_experts']}/{vote_result['total_experts']}",
            "",
            "--- 投票分布 ---",
        ]
        for suspect, score in sorted(vote_result["vote_distribution"].items(), key=lambda x: -x[1]):
            details = vote_result["vote_details"].get(suspect, [])
            expert_names = [d["perspective"] for d in details]
            lines.append(f"  {suspect}: {score:.2%} (支持专家: {', '.join(expert_names)})")

        return "\n".join(lines)
