#!/usr/bin/env python3
"""
概率真值维护系统 (Probabilistic Truth Maintenance System, pTMS)
借鉴 metareflection/llm-mysteries/tms.py + Doyle (1979) TMS

核心思路:
  1. 维护信念集 (Belief Base)，每个信念有概率
  2. 当新证据加入时，检查与现有信念的一致性
  3. 发现矛盾 → 回退最低概率的矛盾信念 → 传播修正
  4. Z3 MaxSAT 求解最优信念集

与 symbolic_solver 的区别:
  - symbolic_solver: 一次性构建+求解 (静态)
  - pTMS: 增量式维护 (动态)，支持信念添加/撤销/传播

借鉴来源:
  - metareflection/llm-mysteries/tms.py (概率TMS + MaxSAT)
  - Doyle (1979): Truth Maintenance System
  - de Kleer (1986): Assumption-based TMS
"""

import json
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
from loguru import logger

from z3 import (
    Bool, BoolVal, And, Or, Not, Implies,
    Optimize, If, Sum, sat,
)


class Belief:
    """单个信念"""
    def __init__(self, label: str, proposition: str, probability: float, 
                 source: str = "llm", dependencies: List[str] = None):
        self.label = label
        self.proposition = proposition
        self.probability = probability  # P(belief is true)
        self.source = source
        self.dependencies = dependencies or []  # 依赖的其他信念label
        self.status = "active"  # "active" | "retracted" | "contradicted"
        self.justification = ""  # 为什么这个信念被接受/撤销
    
    def __repr__(self):
        return f"Belief({self.label}, p={self.probability:.2f}, status={self.status})"


class ContradictionRecord:
    """矛盾记录"""
    def __init__(self, belief1_label: str, belief2_label: str, reason: str):
        self.belief1_label = belief1_label
        self.belief2_label = belief2_label
        self.reason = reason
        self.timestamp = time.time()


class ProbabilisticTMS:
    """
    概率真值维护系统
    
    API:
      - add_belief(): 添加信念
      - add_constraint(): 添加约束
      - check_consistency(): 一致性检查
      - resolve(): 求解最优信念集
      - propagate(): 传播概率更新
    """

    def __init__(self):
        self.logger = logger.bind(module="ProbabilisticTMS")
        self.beliefs: Dict[str, Belief] = {}
        self.constraints: List[Dict] = []  # {type, nodes, reason}
        self.contradictions: List[ContradictionRecord] = []
        self.resolution_history: List[Dict] = []

    def add_belief(
        self, label: str, proposition: str, probability: float,
        source: str = "llm", dependencies: List[str] = None
    ) -> Belief:
        """添加信念到TMS"""
        belief = Belief(label, proposition, probability, source, dependencies)
        self.beliefs[label] = belief
        self.logger.debug(f"  + 信念: {label} (p={probability:.2f})")
        return belief

    def add_constraint(self, constraint_type: str, nodes: List[str], reason: str = ""):
        """
        添加约束
        constraint_type:
          - "mutex": 节点互斥 (最多一个为真)
          - "exactly_one": 恰好一个为真
          - "implies": nodes[0] → nodes[1]
          - "not_both": nodes[0] 和 nodes[1] 不能同时为真
        """
        self.constraints.append({
            "type": constraint_type,
            "nodes": nodes,
            "reason": reason,
        })

    def check_consistency(self) -> List[ContradictionRecord]:
        """快速一致性检查 (不求解，仅检查明显矛盾)"""
        new_contradictions = []
        
        for constraint in self.constraints:
            nodes = constraint["nodes"]
            ctype = constraint["type"]
            
            if ctype == "mutex" or ctype == "not_both":
                # 检查是否有两个以上节点概率都>0.5
                high_prob = [n for n in nodes if self.beliefs.get(n, Belief("", "", 0)).probability > 0.5]
                if len(high_prob) > 1:
                    for i in range(len(high_prob)):
                        for j in range(i + 1, len(high_prob)):
                            cr = ContradictionRecord(
                                high_prob[i], high_prob[j],
                                f"{ctype}约束冲突: {constraint.get('reason', '')}"
                            )
                            new_contradictions.append(cr)
            
            elif ctype == "exactly_one":
                high_prob = [n for n in nodes if self.beliefs.get(n, Belief("", "", 0)).probability > 0.5]
                if len(high_prob) == 0:
                    self.logger.warning(f"  ⚠️ exactly_one约束: 无高概率节点 {nodes}")
                elif len(high_prob) > 1:
                    for i in range(len(high_prob)):
                        for j in range(i + 1, len(high_prob)):
                            cr = ContradictionRecord(
                                high_prob[i], high_prob[j],
                                f"exactly_one冲突: 多个高概率节点 {constraint.get('reason', '')}"
                            )
                            new_contradictions.append(cr)
        
        self.contradictions.extend(new_contradictions)
        return new_contradictions

    def resolve(self, suspects: List[str] = None) -> Dict[str, Any]:
        """
        Z3 MaxSAT求解 — 找到最大化满足所有约束的最优信念集
        
        Returns:
            {
                "culprit": str,
                "confidence": float,
                "retracted_beliefs": [label],
                "active_beliefs": [{label, revised_probability}],
                "model": {label: bool},
                "stats": dict,
            }
        """
        start_time = time.time()
        
        if not self.beliefs:
            return {
                "culprit": "无法确定",
                "confidence": 0.0,
                "retracted_beliefs": [],
                "active_beliefs": [],
                "model": {},
                "stats": {"error": "no beliefs"},
            }
        
        opt = Optimize()
        
        # 创建Z3变量
        z3_vars = {}
        for label in self.beliefs:
            z3_vars[label] = Bool(f"b_{label}")
        
        # 硬约束
        for constraint in self.constraints:
            nodes = constraint["nodes"]
            ctype = constraint["type"]
            
            if ctype == "exactly_one":
                node_vars = [z3_vars[n] for n in nodes if n in z3_vars]
                if node_vars:
                    opt.add(Or(*node_vars))
                    for i in range(len(node_vars)):
                        for j in range(i + 1, len(node_vars)):
                            opt.add(Not(And(node_vars[i], node_vars[j])))
            
            elif ctype == "mutex" or ctype == "not_both":
                node_vars = [z3_vars[n] for n in nodes if n in z3_vars]
                for i in range(len(node_vars)):
                    for j in range(i + 1, len(node_vars)):
                        opt.add(Not(And(node_vars[i], node_vars[j])))
            
            elif ctype == "implies" and len(nodes) >= 2:
                if nodes[0] in z3_vars and nodes[1] in z3_vars:
                    opt.add(Implies(z3_vars[nodes[0]], z3_vars[nodes[1]]))
        
        # 软约束: 信念概率 → 权重
        for label, belief in self.beliefs.items():
            if belief.status != "active":
                continue
            var = z3_vars[label]
            prob = belief.probability
            
            # 概率 → 权重: 使用log-odds
            if prob > 0.5:
                weight = self._prob_to_weight(prob)
                opt.add_soft(var, weight)
            elif prob < 0.5:
                weight = self._prob_to_weight(1.0 - prob)
                opt.add_soft(Not(var), weight)
            # prob == 0.5: 不添加软约束
        
        # 求解
        result = opt.check()
        solve_time = time.time() - start_time
        
        if result == sat:
            model = opt.model()
            
            # 提取模型
            model_dict = {}
            for label, var in z3_vars.items():
                val = str(model.evaluate(var))
                model_dict[label] = val == "True"
            
            # 找到信念变化
            retracted = []
            active = []
            for label, belief in self.beliefs.items():
                revised_val = model_dict.get(label, None)
                if revised_val is not None:
                    original_high = belief.probability > 0.5
                    if revised_val != original_high:
                        retracted.append(label)
                        self.beliefs[label].status = "retracted"
                        self.beliefs[label].justification = f"Z3求解撤销: 原概率={belief.probability:.2f}"
                    else:
                        active.append({
                            "label": label,
                            "revised_probability": belief.probability,
                            "z3_value": revised_val,
                        })
            
            # 找真凶
            culprit = "无法确定"
            culprit_conf = 0.0
            if suspects:
                for s in suspects:
                    guilt_labels = [l for l in model_dict if s in l and "guilty" in l and model_dict[l]]
                    if guilt_labels:
                        culprit = s
                        culprit_conf = max(self.beliefs.get(l, Belief("", "", 0.5)).probability for l in guilt_labels)
                        break
            
            self.resolution_history.append({
                "time": time.time(),
                "retracted": retracted,
                "solve_time": solve_time,
            })
            
            return {
                "culprit": culprit,
                "confidence": culprit_conf,
                "retracted_beliefs": retracted,
                "active_beliefs": active,
                "model": model_dict,
                "stats": {
                    "solve_time": round(solve_time, 3),
                    "num_beliefs": len(self.beliefs),
                    "num_constraints": len(self.constraints),
                    "num_retracted": len(retracted),
                    "z3_status": "sat",
                },
            }
        else:
            return {
                "culprit": "无法确定",
                "confidence": 0.0,
                "retracted_beliefs": [],
                "active_beliefs": [],
                "model": {},
                "stats": {
                    "solve_time": round(solve_time, 3),
                    "z3_status": str(result),
                },
            }

    def propagate(self, updated_label: str, new_probability: float):
        """
        概率传播 — 当一个信念概率改变时，更新依赖它的信念
        """
        if updated_label not in self.beliefs:
            return
        
        old_prob = self.beliefs[updated_label].probability
        self.beliefs[updated_label].probability = new_probability
        
        # 找到所有依赖此信念的信念
        dependents = [
            b for b in self.beliefs.values()
            if updated_label in b.dependencies
        ]
        
        for dep in dependents:
            # 简单贝叶斯传播
            if new_probability < 0.3:
                dep.probability *= 0.7
            elif new_probability > 0.7:
                dep.probability *= 1.0 + (new_probability - 0.7) * 0.5
            
            dep.probability = max(0.1, min(0.99, dep.probability))
            self.logger.debug(f"  📡 传播: {updated_label} ({old_prob:.2f}→{new_probability:.2f}) → {dep.label} → {dep.probability:.2f}")

    def get_belief_state(self) -> Dict[str, Any]:
        """获取当前信念状态"""
        return {
            "total": len(self.beliefs),
            "active": sum(1 for b in self.beliefs.values() if b.status == "active"),
            "retracted": sum(1 for b in self.beliefs.values() if b.status == "retracted"),
            "contradictions": len(self.contradictions),
            "avg_probability": sum(b.probability for b in self.beliefs.values()) / max(len(self.beliefs), 1),
        }

    def _prob_to_weight(self, prob: float) -> float:
        """概率 → 约束权重"""
        # log-odds: 越确定的信念权重越高
        if prob <= 0.5:
            return 1.0
        return math.exp((prob - 0.5) * 4.0)  # 0.5→1.0, 0.9→4.95, 0.99→7.0


# ============================================================
# TMS 构建辅助函数 — 从案件数据构建TMS
# ============================================================

def build_tms_from_case(
    suspects: List[str],
    expert_analyses: List[Dict],
    search_results: Dict = None,
    bidirectional_evidence: Dict = None,
) -> ProbabilisticTMS:
    """
    从案件数据构建概率TMS
    
    Args:
        suspects: 嫌疑人列表
        expert_analyses: 专家分析结果
        search_results: 搜索器结果
        bidirectional_evidence: 双向证据采集结果
    
    Returns:
        构建好的 ProbabilisticTMS
    """
    tms = ProbabilisticTMS()
    dimensions = ["motive", "opportunity", "capability", "timeline", "evidence"]
    
    # 1. 从专家分析添加信念
    suspect_votes = defaultdict(lambda: defaultdict(float))
    suspect_counts = defaultdict(int)
    
    for analysis in expert_analyses:
        data = analysis.get("data", analysis)
        culprit = normalize_name(data.get("culprit", ""))
        conf = float(data.get("confidence", 0.5))
        
        if culprit:
            suspect_votes[culprit]["overall"] += conf
            suspect_counts[culprit] += 1
    
    # 为每个嫌疑人的每个维度添加信念
    for suspect in suspects:
        norm_suspect = normalize_name(suspect)
        
        # 总体有罪/无罪信念
        vote_score = suspect_votes.get(norm_suspect, {}).get("overall", 0)
        vote_count = suspect_counts.get(norm_suspect, 0)
        
        guilty_prob = min(vote_score / max(vote_count * 1.0, 1), 0.95) if vote_count > 0 else 0.3
        
        tms.add_belief(
            label=f"{suspect}__guilty",
            proposition=f"{suspect}是真凶",
            probability=guilty_prob,
            source="expert_vote",
        )
        
        tms.add_belief(
            label=f"{suspect}__innocent",
            proposition=f"{suspect}不是真凶",
            probability=1.0 - guilty_prob,
            source="expert_vote_inverse",
        )
        
        # 各维度信念
        for dim in dimensions:
            # 从搜索结果提取维度得分
            dim_score = _extract_dim_score(suspect, norm_suspect, dim, search_results, bidirectional_evidence)
            
            tms.add_belief(
                label=f"{suspect}__{dim}__guilty",
                proposition=f"{suspect}在{dim}维度有罪证据",
                probability=dim_score,
                source="search_evidence",
                dependencies=[f"{suspect}__guilty"],
            )
    
    # 2. 添加约束
    # 恰好一个真凶
    guilty_labels = [f"{s}__guilty" for s in suspects]
    tms.add_constraint("exactly_one", guilty_labels, "恰好一个真凶")
    
    # 每个嫌疑人的有罪/无罪互斥
    for suspect in suspects:
        tms.add_constraint("not_both", 
                          [f"{suspect}__guilty", f"{suspect}__innocent"],
                          f"{suspect}不能既是有罪又是无罪")
    
    return tms


def _extract_dim_score(
    suspect: str, norm_suspect: str, dim: str,
    search_results: Dict = None, bidirectional_evidence: Dict = None
) -> float:
    """从搜索结果/双向证据中提取维度得分"""
    score = 0.4  # 默认
    
    # 从搜索排名提取
    if search_results:
        dim_key_map = {
            "motive": "motive_ranking",
            "opportunity": "opportunity_ranking", 
            "capability": "capability_ranking",
        }
        ranking_key = dim_key_map.get(dim)
        if ranking_key:
            ranking_list = search_results.get(ranking_key, [])
            for item in ranking_list:
                if isinstance(item, dict):
                    item_name = normalize_name(item.get("name", ""))
                    if item_name == norm_suspect:
                        score = float(item.get("score", 0.5))
                        break
    
    # 从双向证据提取 (更可靠)
    if bidirectional_evidence:
        suspect_data = bidirectional_evidence.get("suspects", {}).get(suspect, {})
        inc_score = suspect_data.get("incriminating_score", 0)
        exo_score = suspect_data.get("exonerating_score", 0)
        
        # 找该维度的有罪证据
        for e in suspect_data.get("incriminating", []):
            if e.get("category") == dim:
                strength_map = {"strong": 0.9, "medium": 0.6, "weak": 0.4}
                score = max(score, strength_map.get(e.get("strength", "weak"), 0.4))
        
        # 找该维度的无罪证据 (降低分数)
        for e in suspect_data.get("exonerating", []):
            if e.get("category") == dim:
                score *= 0.6  # 有反证，降低
    
    return score


# import at module level for _extract_dim_score
from .name_utils import normalize_name
