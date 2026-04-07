#!/usr/bin/env python3
"""
置信度校准器 (Confidence Calibrator)
借鉴 metareflection/llm-mysteries/tms.py 的 bangs-based 置信度标定
+ 统计校准 (Platt Scaling / Isotonic Regression 思想)

核心思路:
  LLM输出的置信度不可靠(几乎都在80-95%)，需要校准:
  
  1. 证据数量校准: 证据越多 → 置信度越高
  2. 证据强度校准: strong证据 → 权重更高 (类似bangs计数)
  3. 共识度校准: 多专家一致 → 置信度提升
  4. 矛盾惩罚: 有反证 → 置信度降低
  5. 基线回归: 极端置信度(>0.95 或 <0.1)向均值回归

借鉴来源:
  - tms.py: 用感叹号数量标定置信度 (len(bangs)*0.1 + 0.5)
  - Platt Scaling: 将分类器输出映射为概率
  - Zadrozny & Elkan (2002): Transforming Classifier Scores into Accurate Multiclass Probability Estimates
"""

import math
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from loguru import logger
from .name_utils import normalize_name


class ConfidenceCalibrator:
    """置信度校准器 — 将LLM原始置信度校准为更准确的概率估计"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger.bind(module="ConfidenceCalibrator")
        
        # 校准参数
        self.regression_strength = self.config.get("regression_strength", 0.3)  # 向均值回归强度
        self.baseline = self.config.get("baseline", 0.5)  # 先验均值
        self.evidence_boost = self.config.get("evidence_boost", 0.05)  # 每条strong证据的增益
        self.contradiction_penalty = self.config.get("contradiction_penalty", 0.15)  # 矛盾惩罚

    def calibrate(
        self,
        raw_confidence: float,
        evidence_count: int = 0,
        strong_evidence_count: int = 0,
        verified_evidence_ratio: float = 0.5,
        expert_agreement_ratio: float = 0.5,
        has_contradiction: bool = False,
        contradiction_count: int = 0,
    ) -> float:
        """
        校准单个置信度
        
        Args:
            raw_confidence: LLM原始置信度 (0-1)
            evidence_count: 支撑证据数量
            strong_evidence_count: 强证据数量
            verified_evidence_ratio: 已验证证据比例 (0-1)
            expert_agreement_ratio: 专家一致比例 (0-1)
            has_contradiction: 是否有矛盾证据
            contradiction_count: 矛盾数量
        
        Returns:
            校准后的置信度 (0-1)
        """
        calibrated = raw_confidence
        
        # 1. 基线回归 — 极端置信度向0.5回归
        calibrated = calibrated + self.regression_strength * (self.baseline - calibrated)
        
        # 2. 证据数量校准 (类似tms.py的bangs标定)
        evidence_boost = strong_evidence_count * self.evidence_boost
        evidence_boost += (evidence_count - strong_evidence_count) * self.evidence_boost * 0.3
        calibrated += evidence_boost
        
        # 3. 验证率校准
        if verified_evidence_ratio < 0.3:
            calibrated -= 0.1
        elif verified_evidence_ratio > 0.7:
            calibrated += 0.05
        
        # 4. 专家一致度校准
        if expert_agreement_ratio > 0.7:
            calibrated += 0.05
        elif expert_agreement_ratio < 0.3:
            calibrated -= 0.1
        
        # 5. 矛盾惩罚
        if has_contradiction:
            calibrated -= self.contradiction_penalty * (1 + contradiction_count * 0.1)
        
        # 截断到 [0.1, 0.99]
        return max(0.1, min(0.99, calibrated))

    def calibrate_batch(
        self,
        expert_analyses: List[Dict],
        bidirectional_evidence: Dict = None,
        contradiction_data: Dict = None,
    ) -> Dict[str, float]:
        """
        批量校准专家置信度
        
        Returns:
            {suspect_name: calibrated_confidence}
        """
        suspect_data = defaultdict(lambda: {
            "raw_confs": [],
            "expert_count": 0,
            "total_evidence": 0,
            "strong_evidence": 0,
        })
        
        # 收集原始数据
        for analysis in expert_analyses:
            data = analysis.get("data", analysis)
            culprit = normalize_name(data.get("culprit", ""))
            conf = float(data.get("confidence", 0.5))
            
            if culprit:
                suspect_data[culprit]["raw_confs"].append(conf)
                suspect_data[culprit]["expert_count"] += 1
        
        # 从双向证据收集
        if bidirectional_evidence:
            for suspect, ev_data in bidirectional_evidence.get("suspects", {}).items():
                norm = normalize_name(suspect)
                inc = ev_data.get("incriminating", [])
                sd = suspect_data[norm]
                sd["total_evidence"] += len(inc)
                sd["strong_evidence"] += sum(1 for e in inc if e.get("strength") == "strong")
        
        # 矛盾数据
        contradiction_suspects = set()
        if contradiction_data:
            for item in contradiction_data.get("ranking", []):
                if isinstance(item, dict) and item.get("score", 0) > 0.5:
                    contradiction_suspects.add(normalize_name(item.get("name", "")))
        
        # 校准
        results = {}
        total_experts = sum(d["expert_count"] for d in suspect_data.values()) or 1
        
        for suspect, data in suspect_data.items():
            raw_avg = sum(data["raw_confs"]) / max(len(data["raw_confs"]), 1)
            expert_agreement = data["expert_count"] / total_experts
            has_contradiction = suspect in contradiction_suspects
            
            calibrated = self.calibrate(
                raw_confidence=raw_avg,
                evidence_count=data["total_evidence"],
                strong_evidence_count=data["strong_evidence"],
                expert_agreement_ratio=expert_agreement,
                has_contradiction=has_contradiction,
            )
            results[suspect] = round(calibrated, 3)
            
            if abs(calibrated - raw_avg) > 0.1:
                self.logger.info(f"  📏 置信度校准: {suspect} {raw_avg:.2f}→{calibrated:.2f} "
                                f"(证据={data['total_evidence']}, 专家={data['expert_count']}, "
                                f"矛盾={'是' if has_contradiction else '否'})")
        
        return results


class MultiMethodFusion:
    """
    多方法投票融合 — 借鉴 graph.py 的3种方法(processCase/2/3)融合
    
    Methods:
      1. 符号求解 (MaxSAT) — Z3最优解
      2. 双向证据 (布尔) — incriminating AND NOT exonerating
      3. 双向证据 (LLM) — 汇总证据让LLM判断
      4. 专家投票 (加权) — 现有投票机制
      5. 推理树 (ToT) — Tree of Thoughts 验证
    
    融合策略:
      - 一致性加权: 多方法一致的结论获得更高置信度
      - 分歧检测: 方法间分歧时，降低最终置信度
      - 排名融合 (Borda Count): 每个方法的排名 → 综合排名
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger.bind(module="MultiMethodFusion")
        
        # 方法权重
        self.method_weights = self.config.get("method_weights", {
            "symbolic": 0.25,
            "bidirectional_bool": 0.20,
            "bidirectional_llm": 0.15,
            "expert_vote": 0.25,
            "reasoning_tree": 0.15,
        })

    def fuse(
        self,
        symbolic_result: Dict = None,
        bidirectional_result: Dict = None,
        vote_result: Dict = None,
        tree_result: Dict = None,
        calibrated_confidences: Dict = None,
    ) -> Dict[str, Any]:
        """
        融合多个方法的结果
        
        Returns:
            {
                "culprit": str,
                "confidence": float,
                "ranking": [{"name": str, "score": float}],
                "method_results": {method_name: culprit},
                "consensus_level": str,
                "divergence": float,
            }
        """
        # 收集各方法结论
        method_conclusions = {}
        method_rankings = {}
        
        if symbolic_result and symbolic_result.get("z3_status") == "sat":
            method_conclusions["symbolic"] = symbolic_result.get("culprit", "?")
            method_rankings["symbolic"] = {
                r["name"]: r["score"] for r in symbolic_result.get("ranking", [])
            }
        
        if bidirectional_result:
            m1 = bidirectional_result.get("method1_result", "?")
            m2 = bidirectional_result.get("method2_result", "?")
            if m1 and m1 != "无法确定":
                method_conclusions["bidirectional_bool"] = m1
            if m2 and m2 != "无法确定":
                method_conclusions["bidirectional_llm"] = m2
            method_rankings["bidirectional"] = {
                r["name"]: r["net_score"] for r in bidirectional_result.get("ranking", [])
            }
        
        if vote_result:
            method_conclusions["expert_vote"] = vote_result.get("winner", "?")
            method_rankings["expert_vote"] = {
                name: score / 100.0 
                for name, score in vote_result.get("vote_distribution", {}).items()
            }
        
        if tree_result:
            tc = tree_result.get("tree_culprit", "?")
            if tc and tc not in ("跳过", "未知", "?"):
                method_conclusions["reasoning_tree"] = tc
                tree_ranking = tree_result.get("tree_ranking", [])
                if tree_ranking:
                    method_rankings["reasoning_tree"] = {
                        r.get("name", r.get("suspect", "?")): r.get("score", 0)
                        for r in tree_ranking
                    }
        
        # 收集所有嫌疑人
        all_suspects = set()
        for ranking in method_rankings.values():
            all_suspects.update(ranking.keys())
        
        if not all_suspects:
            return {
                "culprit": "无法确定",
                "confidence": 0.0,
                "ranking": [],
                "method_results": method_conclusions,
                "consensus_level": "no_data",
                "divergence": 1.0,
            }
        
        # ── Borda Count 排名融合 ──
        borda_scores = defaultdict(float)
        
        for method_name, ranking in method_rankings.items():
            weight = self.method_weights.get(method_name, 0.1)
            sorted_suspects = sorted(ranking.items(), key=lambda x: -x[1])
            n = len(sorted_suspects)
            
            for rank, (suspect, score) in enumerate(sorted_suspects):
                # Borda分数: 排名越高分数越高，加权
                borda_score = (n - rank) / n * weight
                # 加上原始得分
                borda_score += score * weight * 0.5
                borda_scores[suspect] += borda_score
        
        # ── 一致性分析 ──
        conclusion_counts = defaultdict(int)
        for method, culprit in method_conclusions.items():
            norm = normalize_name(culprit)
            if norm:
                conclusion_counts[norm] += 1
        
        max_agreement = max(conclusion_counts.values()) if conclusion_counts else 0
        total_methods = len(method_conclusions)
        
        if total_methods >= 3 and max_agreement >= 3:
            consensus_level = "strong_consensus"
        elif total_methods >= 2 and max_agreement >= 2:
            consensus_level = "moderate_consensus"
        elif total_methods >= 2 and max_agreement == 1:
            consensus_level = "divergent"
        else:
            consensus_level = "weak"
        
        divergence = 1.0 - (max_agreement / max(total_methods, 1))
        
        # ── 最终结论 ──
        sorted_borda = sorted(borda_scores.items(), key=lambda x: -x[1])
        
        if sorted_borda:
            winner = sorted_borda[0][0]
            winner_score = sorted_borda[0][1]
            
            # 置信度 = Borda得分归一化 × 一致性修正
            max_borda = max(borda_scores.values()) if borda_scores else 1.0
            conf = min(winner_score / max(max_borda, 0.01), 0.99)
            
            # 一致性修正
            if consensus_level == "strong_consensus":
                conf = min(conf * 1.1, 0.99)
            elif consensus_level == "divergent":
                conf *= 0.7
        else:
            winner = "无法确定"
            conf = 0.0
        
        ranking = [
            {"name": s, "score": round(sc, 4)} 
            for s, sc in sorted_borda
        ]
        
        self.logger.info(f"📊 多方法融合: {len(method_conclusions)}个方法")
        for method, culprit in method_conclusions.items():
            self.logger.info(f"  {method}: {culprit}")
        self.logger.info(f"  🏁 融合结论: {winner} ({conf:.3f}) [{consensus_level}]")
        self.logger.info(f"  排名: {[(r['name'], r['score']) for r in ranking]}")
        
        return {
            "culprit": winner,
            "confidence": round(conf, 3),
            "ranking": ranking,
            "method_results": method_conclusions,
            "consensus_level": consensus_level,
            "divergence": round(divergence, 3),
        }
