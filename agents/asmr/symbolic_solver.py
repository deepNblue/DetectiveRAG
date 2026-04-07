#!/usr/bin/env python3
"""
符号约束求解器 (Symbolic Constraint Solver) — v11.0
基于 Z3 MaxSAT + Truth-Maintenance System (TMS)
借鉴 metareflection/llm-mysteries 的信念图+MaxSAT方法，适配 DETECTIVE_RAG 架构

核心思路:
  1. 信念图构建: 为每个嫌疑人，LLM评估多维证据的有罪/无罪支持
  2. 符号化约束: 将信念转化为 Z3 布尔变量和软/硬约束
  3. MaxSAT求解: 找到最大化满足所有约束的真值指派
  4. 一致性检验: TMS维护信念一致性，检测矛盾并回退
  5. 与投票融合: 符号解作为独立信号，与投票结果加权融合

对比现有方法:
  - 纯LLM投票: 每个专家独立判断 → 可能一致犯错(群体思维)
  - 推理树: 多假设验证 → LLM自我验证不可靠
  - 符号求解: 将LLM的直觉转化为形式化约束 → Z3保证逻辑一致性

参考文献:
  - metareflection/llm-mysteries (Belief Graph + MaxSAT)
  - Z3: https://github.com/Z3Prover/z3
  - Doyle (1979): Truth Maintenance System
  - MuSR (Sprague et al., 2023): Multistep Soft Reasoning
"""

import json
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from loguru import logger
from .name_utils import normalize_name, is_valid_suspect, build_name_alias_map

# Z3 imports
from z3 import (
    Bool, BoolVal, And, Or, Not, Implies,
    Solver, Optimize, Sum, If,
    sat, unsat,
)


# ============================================================
# 1. 信念节点 & 约束
# ============================================================

class BeliefNode:
    """信念节点 — 代表一个命题及其置信度"""
    def __init__(self, label: str, proposition: str, confidence: float, source: str = "llm"):
        self.label = label
        self.proposition = proposition  # 自然语言描述
        self.confidence = confidence     # 0.0-1.0
        self.source = source            # 来源: llm / evidence / vote

    def __repr__(self):
        return f"Belief({self.label}, conf={self.confidence:.2f})"


class HardConstraint:
    """硬约束 — 必须满足"""
    def __init__(self, name: str, constraint_type: str, nodes: List[str]):
        self.name = name
        self.constraint_type = constraint_type  # "exactly_one", "implies", "mutex"
        self.nodes = nodes


class SoftConstraint:
    """软约束 — 尽量满足，有权重"""
    def __init__(self, name: str, expression: str, weight: float, confidence: float):
        self.name = name
        self.expression = expression  # "positive" | "negative"
        self.weight = weight
        self.confidence = confidence


# ============================================================
# 2. 信念图构建器
# ============================================================

class BeliefGraphBuilder:
    """
    信念图构建器 — 从案件事实+专家分析构建信念图
    
    对每个嫌疑人，从5个维度收集有罪/无罪证据:
      - motive: 作案动机
      - opportunity: 作案机会
      - capability: 作案能力
      - timeline: 时间线一致性
      - evidence: 直接物证
    
    每个维度产生:
      - 有罪信念 P(suspect有dim | evidence)
      - 无罪信念 P(suspect无dim | evidence)
    """

    BELIEF_PROMPT = """你是一名严格的证据分析员。请基于案件事实，评估以下命题的真假。

【案件事实】
{case_summary}

【评估目标】
嫌疑人: {suspect_name}
维度: {dimension_name}
命题: "{proposition}"

【已有专家意见摘要】
{expert_summary}

请给出你的评估。严格输出JSON:
{{
  "is_true": true/false,
  "confidence": 0.0-1.0,
  "key_evidence": ["支撑证据1", "支撑证据2"],
  "counter_evidence": ["反驳证据1"],
  "reasoning": "简要推理过程(100字内)"
}}

规则:
- 只基于案件文本中的客观事实判断
- confidence反映证据的强度，不是猜测的把握
- 如果证据不足，confidence应较低(<0.5)
- 只输出JSON"""

    DIMENSIONS = {
        "motive": {
            "name": "作案动机",
            "guilty_prop": "有充分、具体的作案动机",
            "innocent_prop": "缺乏作案动机或有反证",
            "weight": 0.25,
        },
        "opportunity": {
            "name": "作案机会",
            "guilty_prop": "在案发时有条件实施犯罪(无可靠不在场证明)",
            "innocent_prop": "有可靠的不在场证明或无作案条件",
            "weight": 0.25,
        },
        "capability": {
            "name": "作案能力",
            "guilty_prop": "具备完成犯罪所需的体力、技能、知识",
            "innocent_prop": "不具备犯罪所需的条件",
            "weight": 0.15,
        },
        "timeline": {
            "name": "时间线一致性",
            "guilty_prop": "其时间线与案件事实吻合，行为可解释",
            "innocent_prop": "时间线存在不可解释的空白或矛盾",
            "weight": 0.20,
        },
        "evidence": {
            "name": "直接物证",
            "guilty_prop": "有直接物证指向此人(指纹、DNA、监控等)",
            "innocent_prop": "缺乏直接物证或有排除性证据",
            "weight": 0.15,
        },
    }

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.logger = logger.bind(module="BeliefGraphBuilder")

    def build(
        self,
        case_summary: str,
        suspects: List[str],
        expert_analyses: List[Dict],
        structured_knowledge: Dict = None,
    ) -> Dict[str, Any]:
        """
        构建信念图
        
        Returns:
            {
                "beliefs": {label: BeliefNode},
                "suspect_dims": {suspect: {dim: {guilty: BeliefNode, innocent: BeliefNode}}},
                "expert_summary": str,
            }
        """
        beliefs = {}
        suspect_dims = {}
        expert_summary = self._build_expert_summary(expert_analyses)

        for suspect in suspects:
            suspect_dims[suspect] = {}

            for dim_key, dim_info in self.DIMENSIONS.items():
                # 有罪信念
                guilty_label = f"{suspect}__{dim_key}__guilty"
                guilty_node = self._evaluate_belief(
                    case_summary=case_summary,
                    suspect=suspect,
                    dimension=dim_info,
                    proposition=f"{suspect}{dim_info['guilty_prop']}",
                    label=guilty_label,
                    expert_summary=expert_summary,
                )
                beliefs[guilty_label] = guilty_node

                # 无罪信念
                innocent_label = f"{suspect}__{dim_key}__innocent"
                innocent_node = self._evaluate_belief(
                    case_summary=case_summary,
                    suspect=suspect,
                    dimension=dim_info,
                    proposition=f"{suspect}{dim_info['innocent_prop']}",
                    label=innocent_label,
                    expert_summary=expert_summary,
                )
                beliefs[innocent_label] = innocent_node

                suspect_dims[suspect][dim_key] = {
                    "guilty": guilty_node,
                    "innocent": innocent_node,
                }

        return {
            "beliefs": beliefs,
            "suspect_dims": suspect_dims,
            "expert_summary": expert_summary,
        }

    def _evaluate_belief(
        self,
        case_summary: str,
        suspect: str,
        dimension: Dict,
        proposition: str,
        label: str,
        expert_summary: str,
    ) -> BeliefNode:
        """评估单个信念"""
        prompt = self.BELIEF_PROMPT.format(
            case_summary=case_summary[:1500],
            suspect_name=suspect,
            dimension_name=dimension["name"],
            proposition=proposition,
            expert_summary=expert_summary[:500],
        )

        response = self._call_llm(prompt, temperature=0.2)
        result = self._extract_json(response)

        if result:
            confidence = float(result.get("confidence", 0.3))
            if not result.get("is_true", True):
                confidence = 1.0 - confidence
            return BeliefNode(
                label=label,
                proposition=proposition,
                confidence=confidence,
                source="llm",
            )
        else:
            return BeliefNode(
                label=label,
                proposition=proposition,
                confidence=0.3,
                source="llm_fallback",
            )

    def build_from_experts(
        self,
        suspects: List[str],
        expert_analyses: List[Dict],
        search_results: Dict = None,
    ) -> Dict[str, Any]:
        """
        从现有专家分析构建轻量信念图（不调用LLM，直接用已有结果）
        
        用于快速模式 — 避免额外LLM调用
        """
        beliefs = {}
        suspect_dims = {}
        
        # 提取搜索结果中的排名
        rankings = {
            "motive": search_results.get("motive_ranking", []) if search_results else [],
            "opportunity": search_results.get("opportunity_ranking", []) if search_results else [],
            "capability": search_results.get("capability_ranking", []) if search_results else [],
        }
        
        # 构建排名映射 {suspect: {dim: score}}
        ranking_map = {}
        for dim, ranking_list in rankings.items():
            for item in ranking_list:
                if isinstance(item, dict):
                    name = normalize_name(item.get("name", ""))
                    score = float(item.get("score", 0.5))
                    if name:
                        ranking_map.setdefault(name, {})[dim] = score

        for suspect in suspects:
            norm_name = normalize_name(suspect)
            suspect_dims[suspect] = {}
            dim_scores = ranking_map.get(norm_name, {})

            for dim_key, dim_info in self.DIMENSIONS.items():
                # 有罪维度分数（从搜索排名获取，或从专家投票推断）
                raw_score = dim_scores.get(dim_key, 0.5)
                
                # 从专家分析中获取该维度的支持度
                expert_support = self._get_expert_dim_support(suspect, dim_key, expert_analyses)
                
                # 融合: 搜索排名 × 专家支持
                guilty_conf = (raw_score * 0.4 + expert_support * 0.6) if expert_support > 0 else raw_score
                innocent_conf = 1.0 - guilty_conf

                guilty_label = f"{suspect}__{dim_key}__guilty"
                innocent_label = f"{suspect}__{dim_key}__innocent"

                guilty_node = BeliefNode(
                    label=guilty_label,
                    proposition=f"{suspect}{dim_info['guilty_prop']}",
                    confidence=guilty_conf,
                    source="expert_fusion",
                )
                innocent_node = BeliefNode(
                    label=innocent_label,
                    proposition=f"{suspect}{dim_info['innocent_prop']}",
                    confidence=innocent_conf,
                    source="expert_fusion",
                )

                beliefs[guilty_label] = guilty_node
                beliefs[innocent_label] = innocent_node
                suspect_dims[suspect][dim_key] = {
                    "guilty": guilty_node,
                    "innocent": innocent_node,
                }

        return {
            "beliefs": beliefs,
            "suspect_dims": suspect_dims,
            "expert_summary": "built from expert analyses (no LLM calls)",
        }

    def _get_expert_dim_support(self, suspect: str, dim_key: str, expert_analyses: List[Dict]) -> float:
        """从专家分析中提取某嫌疑人在某维度的支持度"""
        if not expert_analyses:
            return 0.0
        
        norm_suspect = normalize_name(suspect)
        votes_for = 0
        total = 0
        
        for analysis in expert_analyses:
            data = analysis.get("data", analysis)
            culprit = normalize_name(data.get("culprit", ""))
            if culprit == norm_suspect:
                votes_for += 1
            total += 1
        
        return votes_for / total if total > 0 else 0.0

    def _build_expert_summary(self, expert_analyses: List[Dict]) -> str:
        """构建专家意见摘要"""
        if not expert_analyses:
            return "暂无专家意见"
        
        parts = []
        for a in expert_analyses[:10]:
            data = a.get("data", a)
            parts.append(f"- {data.get('perspective', '?')}: 认为凶手是{data.get('culprit', '?')}(置信度={data.get('confidence', 0):.1%})")
        return "\n".join(parts)

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        if self.llm_client is None:
            return '{"is_true": true, "confidence": 0.5, "key_evidence": [], "counter_evidence": [], "reasoning": "LLM不可用"}'
        try:
            return self.llm_client.simple_chat(prompt, temperature=temperature)
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return '{"is_true": true, "confidence": 0.5, "key_evidence": [], "counter_evidence": [], "reasoning": "LLM失败"}'

    def _extract_json(self, text: str) -> Optional[Dict]:
        if not text:
            return None
        import re
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


# ============================================================
# 3. Z3 MaxSAT 求解器
# ============================================================

class MaxSATSolver:
    """
    Z3 MaxSAT 求解器
    
    将信念图转化为 Z3 约束:
    
    硬约束 (必须满足):
      1. 恰好一个真凶: Xor(suspect_1, suspect_2, ...) — 恰好一个为true
      2. 互斥约束: guilty.dim 和 innocent.dim 不能同时为true
      3. 蕴含约束: 如果所有有罪维度都为true，则该嫌疑人是真凶
    
    软约束 (尽量满足，有权重):
      1. 每个信念节点的置信度 → 软约束权重
      2. 维度权重 → 乘以置信度
      3. 专家投票分布 → 额外软约束
    """

    def __init__(self):
        self.logger = logger.bind(module="MaxSATSolver")

    def solve(
        self,
        suspects: List[str],
        belief_graph: Dict[str, Any],
        vote_result: Dict = None,
        fusion_weight: float = 0.3,  # 投票结果在软约束中的权重
    ) -> Dict[str, Any]:
        """
        求解符号约束
        
        Args:
            suspects: 嫌疑人列表
            belief_graph: 信念图 (from BeliefGraphBuilder)
            vote_result: 投票结果 (可选，用于融合)
            fusion_weight: 投票结果在软约束中的权重 (0=纯符号, 1=纯投票)
            
        Returns:
            {
                "culprit": str,
                "confidence": float,
                "ranking": [{"name": str, "score": float}],
                "all_models": [dict],  # 多解情况
                "belief_changes": [dict],  # TMS信念变化
                "stats": dict,
            }
        """
        start_time = time.time()
        beliefs = belief_graph["beliefs"]
        suspect_dims = belief_graph["suspect_dims"]

        opt = Optimize()

        # ── 创建布尔变量 ──
        # 每个嫌疑人一个 "is_culprit" 变量
        culprit_vars = {}
        for s in suspects:
            culprit_vars[s] = Bool(f"culprit_{s}")

        # 每个信念节点一个变量
        belief_vars = {}
        for label, node in beliefs.items():
            belief_vars[label] = Bool(f"belief_{label}")

        # ── 硬约束 ──
        
        # H1: 恰好一个真凶 (使用Xor编码)
        culprit_var_list = [culprit_vars[s] for s in suspects]
        # 恰好一个 = 至少一个 + 至多一个
        opt.add(Or(*culprit_var_list))  # 至少一个
        for i in range(len(suspects)):
            for j in range(i + 1, len(suspects)):
                opt.add(Not(And(culprit_vars[suspects[i]], culprit_vars[suspects[j]])))  # 至多一个

        self.logger.info(f"📏 硬约束: 恰好1个真凶 ({len(suspects)}个嫌疑人)")

        # H2: 有罪/无罪信念互斥
        for suspect in suspects:
            dims = suspect_dims.get(suspect, {})
            for dim_key, dim_data in dims.items():
                guilty_label = f"{suspect}__{dim_key}__guilty"
                innocent_label = f"{suspect}__{dim_key}__innocent"
                if guilty_label in belief_vars and innocent_label in belief_vars:
                    opt.add(Not(And(belief_vars[guilty_label], belief_vars[innocent_label])))

        self.logger.info(f"📏 硬约束: 有罪/无罪互斥 ({len(suspects) * len(BeliefGraphBuilder.DIMENSIONS)}对)")

        # H3: 蕴含约束 — 如果所有有罪维度为true，则该嫌疑人是真凶
        for suspect in suspects:
            dims = suspect_dims.get(suspect, {})
            dim_labels = []
            for dim_key in dims:
                guilty_label = f"{suspect}__{dim_key}__guilty"
                if guilty_label in belief_vars:
                    dim_labels.append(belief_vars[guilty_label])
            
            if dim_labels:
                # 所有有罪维度 → 是真凶
                opt.add(Implies(And(*dim_labels), culprit_vars[suspect]))
                # 不是真凶 → 并非所有有罪维度
                opt.add(Implies(Not(culprit_vars[suspect]), Not(And(*dim_labels))))

        self.logger.info(f"📏 硬约束: 蕴含约束 ({len(suspects)}对)")

        # ── 软约束 ──
        soft_weight_total = 0.0

        # S1: 信念置信度 → 软约束
        for label, node in beliefs.items():
            if label not in belief_vars:
                continue
            var = belief_vars[label]
            confidence = node.confidence
            # 将置信度转化为权重: 高置信度 → 强约束
            weight = self._confidence_to_weight(confidence)
            if weight > 0:
                if confidence > 0.5:
                    opt.add_soft(var, weight)
                else:
                    opt.add_soft(Not(var), weight)
                soft_weight_total += weight

        self.logger.info(f"📏 软约束: 信念置信度 ({len(beliefs)}个, 总权重={soft_weight_total:.1f})")

        # S2: 维度权重 × 有罪信念 → 额外软约束
        for suspect in suspects:
            dims = suspect_dims.get(suspect, {})
            for dim_key, dim_data in dims.items():
                dim_weight = BeliefGraphBuilder.DIMENSIONS.get(dim_key, {}).get("weight", 0.2)
                guilty_label = f"{suspect}__{dim_key}__guilty"
                if guilty_label in belief_vars:
                    # 加权软约束: 高权重维度(如motive)的有罪信念更重要
                    boost = dim_weight * 10.0  # 缩放到与置信度权重可比
                    opt.add_soft(belief_vars[guilty_label], boost)
                    soft_weight_total += boost

        # S3: 投票结果融合 (如果提供)
        if vote_result and fusion_weight > 0:
            vote_dist = vote_result.get("vote_distribution", {})
            alias_map = build_name_alias_map(list(vote_dist.keys()) + suspects)
            
            for suspect in suspects:
                norm = normalize_name(suspect)
                canonical = alias_map.get(norm, norm)
                vote_score = vote_dist.get(canonical, vote_dist.get(norm, 0.0))
                
                if vote_score > 0:
                    weight = vote_score * fusion_weight * 100.0
                    opt.add_soft(culprit_vars[suspect], weight)
                    soft_weight_total += weight

            self.logger.info(f"📏 软约束: 投票融合 (权重={fusion_weight})")

        # ── 求解 ──
        self.logger.info("🔍 Z3 MaxSAT 求解中...")
        result = opt.check()

        stats = {
            "solve_time": round(time.time() - start_time, 3),
            "num_vars": len(belief_vars) + len(culprit_vars),
            "soft_weight_total": round(soft_weight_total, 1),
            "z3_result": str(result),
        }

        if result == sat:
            m = opt.model()
            
            # 提取真凶
            culprit = None
            for s in suspects:
                val = m.evaluate(culprit_vars[s])
                if str(val) == "True":
                    culprit = s
                    break
            
            # 计算每个嫌疑人的综合得分
            ranking = []
            for s in suspects:
                score = self._compute_suspect_score(s, m, belief_vars, culprit_vars, suspect_dims)
                ranking.append({"name": s, "score": round(score, 4)})
            
            ranking.sort(key=lambda x: -x["score"])
            
            # 信念变化分析 (TMS)
            belief_changes = self._analyze_belief_changes(m, beliefs, belief_vars)

            # 提取模型详情
            model_detail = {}
            for s in suspects:
                dims_detail = {}
                for dim_key in BeliefGraphBuilder.DIMENSIONS:
                    guilty_label = f"{s}__{dim_key}__guilty"
                    innocent_label = f"{s}__{dim_key}__innocent"
                    guilty_val = str(m.evaluate(belief_vars.get(guilty_label, BoolVal(False))))
                    innocent_val = str(m.evaluate(belief_vars.get(innocent_label, BoolVal(False))))
                    dims_detail[dim_key] = {
                        "guilty": guilty_val == "True",
                        "innocent": innocent_val == "True",
                    }
                model_detail[s] = {
                    "is_culprit": str(m.evaluate(culprit_vars[s])) == "True",
                    "dimensions": dims_detail,
                }

            stats["num_satisfied"] = self._count_satisfied(opt, m)
            
            self.logger.info(f"✅ Z3求解成功: 真凶={culprit}, 耗时={stats['solve_time']}s")
            self.logger.info(f"  排名: {[(r['name'], r['score']) for r in ranking]}")
            if belief_changes:
                self.logger.info(f"  信念变化: {len(belief_changes)}处")
                for bc in belief_changes[:5]:
                    self.logger.info(f"    ↩️ {bc['label']}: {bc['original']:.2f} → {bc['revised']}")

            return {
                "culprit": culprit or ranking[0]["name"],
                "confidence": ranking[0]["score"] if ranking else 0.0,
                "ranking": ranking,
                "model_detail": model_detail,
                "belief_changes": belief_changes,
                "stats": stats,
                "z3_status": "sat",
            }
        else:
            self.logger.warning(f"⚠️ Z3求解失败: {result}")
            # 回退: 按信念置信度简单排序
            ranking = []
            for s in suspects:
                dims = suspect_dims.get(s, {})
                score = sum(
                    dims.get(dk, {}).get("guilty", BeliefNode("", "", 0.3)).confidence * 
                    BeliefGraphBuilder.DIMENSIONS.get(dk, {}).get("weight", 0.2)
                    for dk in BeliefGraphBuilder.DIMENSIONS
                )
                ranking.append({"name": s, "score": round(score, 4)})
            ranking.sort(key=lambda x: -x["score"])

            return {
                "culprit": ranking[0]["name"] if ranking else "未知",
                "confidence": ranking[0]["score"] if ranking else 0.0,
                "ranking": ranking,
                "model_detail": {},
                "belief_changes": [],
                "stats": stats,
                "z3_status": str(result),
                "fallback": True,
            }

    def _confidence_to_weight(self, confidence: float) -> float:
        """将置信度转化为约束权重"""
        # 使用指数函数放大差异: 0.9→高分, 0.5→低分
        if confidence >= 0.5:
            return math.exp(confidence * 3.0) / math.exp(1.5)  # 归一化
        else:
            return math.exp((1.0 - confidence) * 3.0) / math.exp(1.5)

    def _compute_suspect_score(
        self, suspect: str, model, belief_vars: Dict, culprit_vars: Dict, suspect_dims: Dict
    ) -> float:
        """计算嫌疑人在模型中的综合得分"""
        dims = suspect_dims.get(suspect, {})
        weighted_score = 0.0
        total_weight = 0.0

        for dim_key, dim_info in BeliefGraphBuilder.DIMENSIONS.items():
            dim_weight = dim_info["weight"]
            guilty_label = f"{suspect}__{dim_key}__guilty"

            if guilty_label in belief_vars:
                val = str(model.evaluate(belief_vars[guilty_label]))
                belief_node = dims.get(dim_key, {}).get("guilty")
                base_conf = belief_node.confidence if belief_node else 0.5

                if val == "True":
                    weighted_score += dim_weight * base_conf
                else:
                    weighted_score += dim_weight * (1.0 - base_conf) * 0.3
            else:
                weighted_score += dim_weight * 0.5

            total_weight += dim_weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _count_satisfied(self, opt: Optimize, model) -> int:
        """统计满足的软约束数量"""
        # Z3 Optimize 不直接暴露这个，用近似
        try:
            obj_val = opt.lower_value()
            return int(float(str(obj_val))) if obj_val else 0
        except:
            return -1

    def _analyze_belief_changes(self, model, beliefs: Dict, belief_vars: Dict) -> List[Dict]:
        """
        TMS信念变化分析
        
        找出Z3解与原始信念不同的地方 — 这些是需要修正的不一致信念
        """
        changes = []
        for label, node in beliefs.items():
            if label not in belief_vars:
                continue
            var = belief_vars[label]
            z3_val = str(model.evaluate(var)) == "True"
            original_high = node.confidence > 0.5

            if z3_val != original_high:
                changes.append({
                    "label": label,
                    "original": node.confidence,
                    "revised": z3_val,
                    "proposition": node.proposition[:80],
                    "change_type": "flipped_by_constraint",
                })

        return changes


# ============================================================
# 4. 符号求解器 (对外接口)
# ============================================================

class SymbolicConstraintSolver:
    """
    符号约束求解器 — 完整流水线
    
    Pipeline:
      1. BeliefGraphBuilder → 构建信念图
      2. MaxSATSolver → Z3求解
      3. 融合投票结果 → 最终结论
    
    两种模式:
      - full: 调用LLM构建信念图 (准确但慢)
      - fast: 从已有专家分析构建信念图 (快速，0额外LLM调用)
    """

    def __init__(self, llm_client=None, config: Dict[str, Any] = None):
        self.llm_client = llm_client
        self.config = config or {}
        self.logger = logger.bind(module="SymbolicConstraintSolver")
        
        self.belief_builder = BeliefGraphBuilder(llm_client=llm_client)
        self.maxsat_solver = MaxSATSolver()

    def solve(
        self,
        case_text: str,
        suspects: List[str],
        expert_analyses: List[Dict] = None,
        search_results: Dict = None,
        vote_result: Dict = None,
        structured_knowledge: Dict = None,
        mode: str = "fast",  # "full" | "fast"
    ) -> Dict[str, Any]:
        """
        运行符号约束求解
        
        Args:
            case_text: 案件文本
            suspects: 嫌疑人列表
            expert_analyses: 专家分析结果
            search_results: 搜索器结果
            vote_result: 投票结果 (用于融合)
            structured_knowledge: 结构化知识
            mode: "full" (LLM构建信念图) 或 "fast" (从已有数据构建)
        
        Returns:
            求解结果
        """
        start_time = time.time()
        expert_analyses = expert_analyses or []

        self.logger.info("=" * 50)
        self.logger.info(f"🔮 符号约束求解器启动 (mode={mode})")
        self.logger.info(f"嫌疑人: {suspects}")
        self.logger.info("=" * 50)

        # Step 1: 构建信念图
        belief_start = time.time()
        case_summary = case_text[:2000]

        if mode == "full":
            belief_graph = self.belief_builder.build(
                case_summary=case_summary,
                suspects=suspects,
                expert_analyses=expert_analyses,
                structured_knowledge=structured_knowledge,
            )
        else:
            belief_graph = self.belief_builder.build_from_experts(
                suspects=suspects,
                expert_analyses=expert_analyses,
                search_results=search_results,
            )

        belief_time = time.time() - belief_start
        num_beliefs = len(belief_graph["beliefs"])
        self.logger.info(f"📊 信念图构建完成 ({belief_time:.1f}s): {num_beliefs}个信念节点")

        # 输出信念图摘要
        for suspect in suspects:
            dims = belief_graph["suspect_dims"].get(suspect, {})
            dim_scores = []
            for dk in BeliefGraphBuilder.DIMENSIONS:
                dd = dims.get(dk, {})
                g = dd.get("guilty")
                i = dd.get("innocent")
                g_conf = g.confidence if g else 0.5
                i_conf = i.confidence if i else 0.5
                dim_scores.append(f"{dk[:3]}={g_conf:.2f}/{i_conf:.2f}")
            self.logger.info(f"  📌 {suspect}: {' | '.join(dim_scores)}")

        # Step 2: Z3 MaxSAT 求解
        fusion_weight = self.config.get("vote_fusion_weight", 0.3)
        
        solver_result = self.maxsat_solver.solve(
            suspects=suspects,
            belief_graph=belief_graph,
            vote_result=vote_result,
            fusion_weight=fusion_weight,
        )

        total_time = time.time() - start_time

        # Step 3: 包装结果
        result = {
            "track": "SymbolicSolver",
            "culprit": solver_result["culprit"],
            "confidence": solver_result["confidence"],
            "ranking": solver_result["ranking"],
            "model_detail": solver_result.get("model_detail", {}),
            "belief_changes": solver_result.get("belief_changes", []),
            "z3_status": solver_result.get("z3_status", "unknown"),
            "belief_graph_summary": {
                "num_beliefs": num_beliefs,
                "suspects": suspects,
                "dim_scores": {
                    s: {
                        dk: {
                            "guilty": belief_graph["suspect_dims"].get(s, {}).get(dk, {}).get("guilty", BeliefNode("", "", 0.5)).confidence,
                            "innocent": belief_graph["suspect_dims"].get(s, {}).get(dk, {}).get("innocent", BeliefNode("", "", 0.5)).confidence,
                        }
                        for dk in BeliefGraphBuilder.DIMENSIONS
                    }
                    for s in suspects
                },
            },
            "stats": {
                **solver_result.get("stats", {}),
                "belief_build_time": round(belief_time, 2),
                "total_time": round(total_time, 2),
                "mode": mode,
                "vote_fusion_weight": fusion_weight,
            },
            "is_different_from_vote": False,
            "vote_winner": "",
        }

        # 与投票对比
        if vote_result:
            vote_winner = vote_result.get("winner", "")
            result["vote_winner"] = vote_winner
            result["is_different_from_vote"] = normalize_name(result["culprit"]) != normalize_name(vote_winner)
            if result["is_different_from_vote"]:
                self.logger.info(f"⚡ 符号求解与投票不同! 投票={vote_winner} → 符号={result['culprit']}")

        self.logger.info("=" * 50)
        self.logger.info(f"🔮 符号约束求解完成: 真凶={result['culprit']} "
                         f"(conf={result['confidence']:.3f})")
        self.logger.info(f"  排名: {[(r['name'], r['score']) for r in result['ranking']]}")
        if result["belief_changes"]:
            self.logger.info(f"  信念修正: {len(result['belief_changes'])}处")
        self.logger.info(f"  耗时: {total_time:.1f}s (信念图={belief_time:.1f}s + 求解={solver_result.get('stats', {}).get('solve_time', 0)}s)")
        self.logger.info("=" * 50)

        return result
