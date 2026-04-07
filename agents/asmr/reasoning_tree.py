#!/usr/bin/env python3
"""
刑侦推理树 (Detective Reasoning Tree, DRT)
融合 Tree of Thoughts + MCTS + 刑侦领域知识

核心思路:
  1. 将"谁是凶手"建模为树搜索问题
  2. 每个节点是一个嫌疑人假设 + 部分证据链
  3. 通过多维度验证(动机/机会/能力/时间线/矛盾)动态评估
  4. 发现矛盾 → 回退 → 探索其他嫌疑人
  5. 最终输出带完整证据链的最高得分路径

参考文献:
  - Tree of Thoughts (Yao et al., 2023)
  - Graph of Thoughts (Besta et al., 2024)
  - RAP: Reasoning via Planning (Hao et al., 2023)
  - Tree Search for LM Agents (Koh et al., 2024)
"""

import json
import time
import uuid
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from .name_utils import normalize_name, is_valid_suspect, build_name_alias_map


# ============================================================
# 推理树节点
# ============================================================

@dataclass
class ReasoningNode:
    """推理树节点 — 代表一个假设 + 验证状态"""

    node_id: str
    parent_id: Optional[str] = None
    hypothesis: str = ""                     # 当前假设描述
    suspect_name: str = ""                   # 关联嫌疑人
    depth: int = 0                           # 树深度 (0=根, 1=假设, 2+=验证)

    # 验证维度 — 每个维度记录验证结果
    verification_dimensions: Dict[str, Dict] = field(default_factory=dict)
    # 例: {"motive": {"score": 0.8, "evidence": [...], "reasoning": "..."}}

    # 证据收集
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)

    # 评分
    plausibility_score: float = 0.0          # 可信度 (0-1, LLM评估)
    evidence_chain_score: float = 0.0        # 证据链完整度
    contradiction_penalty: float = 0.0       # 矛盾惩罚

    # 搜索状态
    visits: int = 0
    total_reward: float = 0.0
    is_pruned: bool = False
    prune_reason: str = ""
    is_expanded: bool = False                # 是否已完成子节点展开

    # 子节点ID列表
    children: List[str] = field(default_factory=list)

    # 推理文本
    reasoning_text: str = ""

    @property
    def overall_score(self) -> float:
        """综合得分 = 可信度 × 证据链完整度 - 矛盾惩罚"""
        raw = self.plausibility_score * self.evidence_chain_score - self.contradiction_penalty
        return max(0.0, min(1.0, raw))

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "hypothesis": self.hypothesis,
            "suspect_name": self.suspect_name,
            "depth": self.depth,
            "verification_dimensions": self.verification_dimensions,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "scores": {
                "plausibility": round(self.plausibility_score, 3),
                "evidence_chain": round(self.evidence_chain_score, 3),
                "contradiction_penalty": round(self.contradiction_penalty, 3),
                "overall": round(self.overall_score, 3),
            },
            "visits": self.visits,
            "is_pruned": self.is_pruned,
            "prune_reason": self.prune_reason,
            "reasoning_text": self.reasoning_text[:500],
        }


# ============================================================
# 验证维度定义
# ============================================================

VERIFICATION_DIMENSIONS = {
    "motive": {
        "name": "作案动机",
        "weight": 0.25,
        "description": "该嫌疑人是否有充分、合理的作案动机？",
    },
    "opportunity": {
        "name": "作案机会",
        "weight": 0.25,
        "description": "该嫌疑人在案发时是否有条件实施犯罪？",
    },
    "capability": {
        "name": "作案能力",
        "weight": 0.15,
        "description": "该嫌疑人是否有能力完成犯罪行为？",
    },
    "timeline": {
        "name": "时间线一致性",
        "weight": 0.20,
        "description": "该嫌疑人的时间线是否与案件事实吻合？有无不在场证明？",
    },
    "contradiction": {
        "name": "矛盾检测",
        "weight": 0.15,
        "description": "该嫌疑人的供述/行为是否存在矛盾？是否有未解释的疑点？",
    },
}


# ============================================================
# Prompt模板
# ============================================================

HYPOTHESIS_EVAL_PROMPT = """你是一名刑侦推理专家，正在验证一个嫌疑人假设。

【案件事实】
{case_summary}

【假设】嫌疑人 "{suspect_name}" 是真凶

【验证维度】{dimension_name}
{dimension_desc}

【已有证据分析】
{evidence_context}

【其他专家意见】
{expert_opinions}

请从"{dimension_name}"角度，严格评估该假设。输出JSON:
{{
  "score": 0.0-1.0,
  "supports": ["支持该假设的证据1", ...],
  "contradicts": ["反驳该假设的证据1", ...],
  "reasoning": "推理过程(200字内)",
  "is_hard_contradiction": false,
  "hard_contradiction_reason": ""
}}

规则:
- score=1.0表示完全支持，0.0表示完全反驳
- is_hard_contradiction=true 表示发现铁证般的不在场证明或不可能犯罪证据(如死亡时间不在场)
- 如果is_hard_contradiction=true，必须在hard_contradiction_reason中说明
- 只输出JSON，不要其他内容"""

GLOBAL_REEVAL_PROMPT = """你是一名首席刑侦推理官。你需要对多个嫌疑人假设进行最终评估。

【案件事实】
{case_summary}

【各嫌疑人假设验证结果】
{hypotheses_summary}

【原始投票结果】
{vote_summary}

请综合以上信息，对每个嫌疑人给出最终评分。输出JSON:
{{
  "final_ranking": [
    {{"name": "嫌疑人名", "score": 0.0-1.0, "reasoning": "简述(100字)"}}
  ],
  "overturned": false,
  "overturn_reason": "",
  "confidence_calibration": 0.0-1.0
}}

规则:
- final_ranking按score从高到低排列
- 如果推理树验证结果与原始投票不同，设置overturned=true并说明原因
- confidence_calibration是基于证据链完整度的校准置信度(不要一律0.9)
- 只输出JSON"""

RECONSIDER_PROMPT = """你是一名刑侦推理官。当前的嫌疑人假设都被验证有严重问题，需要重新考虑。

【案件事实】
{case_summary}

【已排除的嫌疑人】
{eliminated_summary}

【案件中的所有人物】
{all_persons}

请从中选出最多3个之前被忽略的可能真凶。输出JSON:
{{
  "new_suspects": [
    {{"name": "人物名", "reason": "为什么此人可能是真凶(100字)"}}
  ]
}}

规则:
- 不要选择已排除的嫌疑人
- 重点关注次要人物、不显眼的角色
- 只输出JSON"""


# ============================================================
# 刑侦推理搜索树
# ============================================================

class DetectiveSearchTree:
    """
    刑侦推理搜索树

    搜索策略: Best-First + 即时剪枝 + 全局回退
    - 用投票结果作为初始优先级(启发函数)
    - 对每个嫌疑人假设做多维度证据链验证
    - 发现硬矛盾立即剪枝并回退
    - 所有Top嫌疑人被排除时，全局回退探索次要人物
    - 最终选择得分最高的完整证据链
    """

    def __init__(self, llm_client=None, config: Dict[str, Any] = None):
        self.llm_client = llm_client
        self.config = config or {}
        self.logger = logger.bind(module="DetectiveSearchTree")

        # 搜索参数
        self.max_depth = self.config.get("max_depth", 3)
        self.prune_threshold = self.config.get("prune_threshold", 0.15)
        self.max_iterations = self.config.get("max_iterations", 20)
        self.expansion_width = self.config.get("expansion_width", 3)  # 每次扩展几个维度

        # 树结构
        self.nodes: Dict[str, ReasoningNode] = {}
        self.root_id: Optional[str] = None

        # 搜索统计
        self.stats = {
            "nodes_created": 0,
            "nodes_pruned": 0,
            "backtracks": 0,
            "global_reconsiders": 0,
            "llm_calls": 0,
        }

    def _new_id(self) -> str:
        return f"n_{uuid.uuid4().hex[:8]}"

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """调用LLM"""
        self.stats["llm_calls"] += 1
        if self.llm_client is None:
            return '{"score": 0.5, "supports": [], "contradicts": [], "reasoning": "LLM不可用", "is_hard_contradiction": false, "hard_contradiction_reason": ""}'

        try:
            response = self.llm_client.simple_chat(prompt, temperature=temperature)
            return response
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return '{"score": 0.5, "supports": [], "contradicts": [], "reasoning": "LLM调用失败", "is_hard_contradiction": false, "hard_contradiction_reason": ""}'

    def _extract_json(self, text: str) -> Optional[Dict]:
        """从LLM响应中提取JSON"""
        if not text:
            return None

        import re

        # Strategy 1: 直接解析
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: 提取代码块
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 3: 找最外层 {...}
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

        self.logger.warning(f"无法提取JSON (长度={len(text)})")
        return None

    # ============================================================
    # 核心搜索流程
    # ============================================================

    def search(
        self,
        case_text: str,
        suspects: List[str],
        structured_knowledge: Dict = None,
        search_results: Dict = None,
        expert_analyses: List[Dict] = None,
        vote_result: Dict = None,
    ) -> Dict[str, Any]:
        """
        运行推理树搜索

        Args:
            case_text: 案件文本
            suspects: 嫌疑人列表
            structured_knowledge: Stage 1结构化知识
            search_results: Stage 2搜索结果
            expert_analyses: Stage 3专家分析
            vote_result: 投票结果(用作初始启发)

        Returns:
            推理树搜索结果
        """
        start_time = time.time()

        self.logger.info("=" * 50)
        self.logger.info("🌳 刑侦推理树搜索启动")
        self.logger.info(f"嫌疑人: {suspects}, 最大迭代: {self.max_iterations}")
        self.logger.info("=" * 50)

        # 重置状态
        self.nodes.clear()
        self.stats = {"nodes_created": 0, "nodes_pruned": 0, "backtracks": 0, "global_reconsiders": 0, "llm_calls": 0}

        # 构建案件摘要
        case_summary = self._build_case_summary(case_text, structured_knowledge)
        evidence_context = self._build_evidence_context(structured_knowledge, search_results)
        expert_opinions = self._build_expert_opinions(expert_analyses)

        # Phase 1: 创建根节点
        self.root_id = self._new_id()
        root = ReasoningNode(
            node_id=self.root_id,
            hypothesis="案件事实",
            depth=0,
        )
        self.nodes[self.root_id] = root
        self.stats["nodes_created"] += 1

        # Phase 2: 按投票优先级排序嫌疑人
        prioritized_suspects = self._prioritize_suspects(suspects, vote_result)
        self.logger.info(f"📋 嫌疑人优先级: {prioritized_suspects}")

        # Phase 3: 为每个嫌疑人创建假设节点
        hypothesis_nodes = []
        for suspect in prioritized_suspects:
            h_node = self._create_hypothesis_node(
                parent_id=self.root_id,
                suspect_name=suspect,
                case_summary=case_summary,
            )
            if h_node:
                hypothesis_nodes.append(h_node)
                root.children.append(h_node.node_id)

        if not hypothesis_nodes:
            self.logger.warning("没有创建任何假设节点")
            return self._build_result(case_text, vote_result, time.time() - start_time)

        # Phase 4: 逐个验证假设 (Best-First)
        best_path = None
        best_score = -1.0
        eliminated_suspects = []

        for h_node in hypothesis_nodes:
            if h_node.is_pruned:
                continue

            self.logger.info(f"🔍 验证假设: {h_node.suspect_name} (初始得分={h_node.plausibility_score:.2f})")

            # 对当前假设做多维度验证
            score, hard_contradiction = self._verify_hypothesis(
                node=h_node,
                case_summary=case_summary,
                evidence_context=evidence_context,
                expert_opinions=expert_opinions,
            )

            h_node.plausibility_score = score
            h_node.is_expanded = True

            if hard_contradiction:
                self.logger.info(f"  ❌ 硬矛盾! 剪枝 {h_node.suspect_name}: {h_node.prune_reason}")
                h_node.is_pruned = True
                eliminated_suspects.append(h_node.suspect_name)
                self.stats["nodes_pruned"] += 1
                self.stats["backtracks"] += 1
                continue

            if score < self.prune_threshold:
                self.logger.info(f"  ⚠️ 得分过低({score:.2f}), 剪枝 {h_node.suspect_name}")
                h_node.is_pruned = True
                h_node.prune_reason = f"综合得分过低({score:.2f})"
                eliminated_suspects.append(h_node.suspect_name)
                self.stats["nodes_pruned"] += 1
                self.stats["backtracks"] += 1
                continue

            self.logger.info(f"  ✅ {h_node.suspect_name} 验证通过, 得分={score:.3f}")

            if score > best_score:
                best_score = score
                best_path = h_node

        # Phase 5: 全局回退 — 如果所有Top嫌疑人都被排除
        if best_path is None and eliminated_suspects:
            self.logger.info("🔄 所有Top嫌疑人被排除! 启动全局回退...")
            self.stats["global_reconsiders"] += 1

            reconsidered = self._global_reconsider(
                case_summary=case_summary,
                eliminated_suspects=eliminated_suspects,
                case_text=case_text,
                structured_knowledge=structured_knowledge,
            )

            for new_suspect in reconsidered:
                h_node = self._create_hypothesis_node(
                    parent_id=self.root_id,
                    suspect_name=new_suspect["name"],
                    case_summary=case_summary,
                )
                if h_node:
                    root.children.append(h_node.node_id)
                    score, hard_contradiction = self._verify_hypothesis(
                        node=h_node,
                        case_summary=case_summary,
                        evidence_context=evidence_context,
                        expert_opinions=expert_opinions,
                    )
                    h_node.plausibility_score = score
                    h_node.is_expanded = True

                    if not hard_contradiction and score >= self.prune_threshold:
                        if score > best_score:
                            best_score = score
                            best_path = h_node
                            self.logger.info(f"  🎯 全局回退发现新嫌疑人: {new_suspect['name']}, 得分={score:.3f}")

        # Phase 6: 最终评估 — 如果有多个未剪枝假设，做全局比较
        active_hypotheses = [
            n for n in self.nodes.values()
            if n.depth == 1 and not n.is_pruned and n.is_expanded
        ]

        if len(active_hypotheses) >= 2:
            self.logger.info("⚖️ 多个假设存活，进行全局对比评估...")
            final_result = self._global_reevaluate(
                hypotheses=active_hypotheses,
                case_summary=case_summary,
                vote_result=vote_result,
            )
        elif best_path:
            final_result = {
                "final_ranking": [
                    {"name": best_path.suspect_name, "score": best_path.overall_score, "reasoning": best_path.reasoning_text[:200]}
                ],
                "overturned": False,
                "overturn_reason": "",
                "confidence_calibration": best_path.overall_score,
            }
        else:
            # 全部失败，回退到投票结果
            final_result = {
                "final_ranking": [],
                "overturned": False,
                "overturn_reason": "推理树未能找到有效假设，回退到投票结果",
                "confidence_calibration": 0.0,
            }

        elapsed = time.time() - start_time

        self.logger.info("=" * 50)
        if final_result.get("final_ranking"):
            top = final_result["final_ranking"][0]
            self.logger.info(f"🌳 推理树结论: {top['name']} (得分={top['score']:.3f})")
        if final_result.get("overturned"):
            self.logger.info(f"⚡ 推理树推翻了投票结果! 原因: {final_result['overturn_reason']}")
        self.logger.info(f"📊 统计: 创建{self.stats['nodes_created']}节点, "
                         f"剪枝{self.stats['nodes_pruned']}, "
                         f"回退{self.stats['backtracks']}, "
                         f"全局重考虑{self.stats['global_reconsiders']}, "
                         f"LLM调用{self.stats['llm_calls']}次")
        self.logger.info("=" * 50)

        return self._build_result(
            case_text=case_text,
            vote_result=vote_result,
            final_result=final_result,
            best_path=best_path,
            elapsed=elapsed,
        )

    # ============================================================
    # 辅助方法
    # ============================================================

    def _build_case_summary(self, case_text: str, structured_knowledge: Dict) -> str:
        """构建案件摘要"""
        parts = [f"案件原文(节选): {case_text[:1500]}"]

        if structured_knowledge and isinstance(structured_knowledge, dict):
            timeline = structured_knowledge.get("timeline", {})
            if isinstance(timeline, dict):
                tdata = timeline.get("data", {})
                if isinstance(tdata, dict):
                    events = tdata.get("events", [])
                    if events:
                        parts.append(f"\n时间线({len(events)}个事件): " + "; ".join(
                            [f"{e.get('time', '?')}: {e.get('event', '?')}" for e in events[:10]]
                        ))

            persons_data = structured_knowledge.get("person_relation", {})
            if isinstance(persons_data, dict):
                pdata = persons_data.get("data", {})
                if isinstance(pdata, dict):
                    p_list = pdata.get("persons", [])
                    if p_list:
                        parts.append(f"\n涉案人物({len(p_list)}人): " + ", ".join(
                            [p.get("name", "?") for p in p_list[:15]]
                        ))

        return "\n".join(parts)

    def _build_evidence_context(self, structured_knowledge: Dict, search_results: Dict) -> str:
        """构建证据上下文"""
        parts = []

        if search_results and isinstance(search_results, dict):
            for key in ["motive_ranking", "opportunity_ranking", "capability_ranking"]:
                ranking = search_results.get(key, [])
                if ranking and isinstance(ranking, list):
                    dim_name = key.replace("_ranking", "")
                    top3 = ranking[:3]
                    parts.append(f"{dim_name}排名Top3: " + ", ".join(
                        [f"{r.get('name', '?')}(score={r.get('score', '?')})" for r in top3 if isinstance(r, dict)]
                    ))

            contradiction = search_results.get("contradiction_insight", "")
            if contradiction:
                parts.append(f"矛盾发现: {contradiction}")

        if structured_knowledge and isinstance(structured_knowledge, dict):
            evidence_data = structured_knowledge.get("evidence", {})
            if isinstance(evidence_data, dict):
                edata = evidence_data.get("data", {})
                if isinstance(edata, dict):
                    physical = edata.get("physical_evidence", [])
                    if physical and isinstance(physical, list):
                        parts.append(f"物证({len(physical)}件): " + "; ".join(
                            [str(e)[:80] for e in physical[:5]]
                        ))

        return "\n".join(parts) if parts else "暂无额外证据信息"

    def _build_expert_opinions(self, expert_analyses: List[Dict]) -> str:
        """构建专家意见摘要"""
        if not expert_analyses:
            return "暂无专家意见"

        parts = []
        for analysis in expert_analyses[:8]:
            perspective = analysis.get("perspective", "?")
            culprit = analysis.get("culprit", "?")
            confidence = analysis.get("confidence", 0)
            parts.append(f"- {perspective}: 认为凶手是{culprit}(置信度={confidence})")

        return "\n".join(parts)

    def _prioritize_suspects(self, suspects: List[str], vote_result: Dict) -> List[str]:
        """按投票优先级排序嫌疑人"""
        if not vote_result:
            return suspects

        vote_dist = vote_result.get("vote_distribution", {})
        alias_map = build_name_alias_map(list(vote_dist.keys()) + suspects)

        # 按投票分数排序
        scored = []
        for s in suspects:
            norm = normalize_name(s)
            canonical = alias_map.get(norm, norm)
            score = vote_dist.get(canonical, vote_dist.get(norm, 0.0))
            scored.append((s, score))

        scored.sort(key=lambda x: -x[1])
        return [s for s, _ in scored]

    def _create_hypothesis_node(
        self, parent_id: str, suspect_name: str, case_summary: str
    ) -> Optional[ReasoningNode]:
        """创建假设节点并进行初步评估"""
        node_id = self._new_id()
        node = ReasoningNode(
            node_id=node_id,
            parent_id=parent_id,
            hypothesis=f"假设: {suspect_name} 是真凶",
            suspect_name=suspect_name,
            depth=1,
        )
        self.nodes[node_id] = node
        self.stats["nodes_created"] += 1

        # 初步评估 — 快速判断这个假设是否值得深入
        prompt = f"""你是刑侦专家。请快速评估以下假设的初步可行性。

【案件概要】
{case_summary[:800]}

【假设】"{suspect_name}" 是真凶

请给出0-1的初步可行性分数。输出JSON:
{{"initial_score": 0.0-1.0, "brief_reason": "50字以内的理由"}}

只输出JSON。"""

        response = self._call_llm(prompt, temperature=0.2)
        result = self._extract_json(response)

        if result:
            initial_score = float(result.get("initial_score", 0.3))
            node.plausibility_score = initial_score
            node.reasoning_text = result.get("brief_reason", "")
        else:
            node.plausibility_score = 0.3

        self.logger.info(f"  📌 创建假设节点: {suspect_name} (初始得分={node.plausibility_score:.2f})")
        return node

    def _verify_hypothesis(
        self,
        node: ReasoningNode,
        case_summary: str,
        evidence_context: str,
        expert_opinions: str,
    ) -> Tuple[float, bool]:
        """
        对假设节点做多维度验证

        Returns:
            (综合得分, 是否有硬矛盾)
        """
        total_weighted_score = 0.0
        total_weight = 0.0
        hard_contradiction = False

        for dim_key, dim_info in VERIFICATION_DIMENSIONS.items():
            prompt = HYPOTHESIS_EVAL_PROMPT.format(
                case_summary=case_summary[:1200],
                suspect_name=node.suspect_name,
                dimension_name=dim_info["name"],
                dimension_desc=dim_info["description"],
                evidence_context=evidence_context[:600],
                expert_opinions=expert_opinions[:400],
            )

            response = self._call_llm(prompt, temperature=0.2)
            result = self._extract_json(response)

            if result is None:
                result = {
                    "score": 0.3,
                    "supports": [],
                    "contradicts": [],
                    "reasoning": "无法解析LLM响应",
                    "is_hard_contradiction": False,
                }

            score = float(result.get("score", 0.3))
            supports = result.get("supports", [])
            contradicts = result.get("contradicts", [])
            reasoning = result.get("reasoning", "")
            is_hard = result.get("is_hard_contradiction", False)
            hard_reason = result.get("hard_contradiction_reason", "")

            # 记录验证维度
            node.verification_dimensions[dim_key] = {
                "dimension": dim_info["name"],
                "score": round(score, 3),
                "supports": supports,
                "contradicts": contradicts,
                "reasoning": reasoning[:200],
                "is_hard_contradiction": is_hard,
            }

            node.supporting_evidence.extend(supports)
            node.contradicting_evidence.extend(contradicts)

            # 累计加权分数
            total_weighted_score += score * dim_info["weight"]
            total_weight += dim_info["weight"]

            self.logger.info(f"    📏 {dim_info['name']}: {score:.2f} "
                             f"(支持{len(supports)}项, 矛盾{len(contradicts)}项)"
                             f"{' ⚠️硬矛盾!' if is_hard else ''}")

            # 硬矛盾检测 — 立即剪枝
            if is_hard and hard_reason:
                hard_contradiction = True
                node.is_pruned = True
                node.prune_reason = f"[{dim_info['name']}] {hard_reason}"
                node.contradiction_penalty = 0.5
                return 0.0, True

        # 计算综合得分
        avg_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

        # 证据链完整度 — 有多少维度给出了正面评估
        positive_dims = sum(
            1 for d in node.verification_dimensions.values()
            if d.get("score", 0) >= 0.5
        )
        evidence_chain_score = positive_dims / len(VERIFICATION_DIMENSIONS)

        # 矛盾惩罚 — 有矛盾证据的维度数
        contradiction_dims = sum(
            1 for d in node.verification_dimensions.values()
            if d.get("score", 0) < 0.3
        )
        contradiction_penalty = contradiction_dims * 0.1

        node.plausibility_score = avg_score
        node.evidence_chain_score = evidence_chain_score
        node.contradiction_penalty = contradiction_penalty
        node.visits += 1

        # 综合推理文本
        node.reasoning_text = f"假设'{node.suspect_name}'是真凶。"
        for dim_key, dim_data in node.verification_dimensions.items():
            node.reasoning_text += f"\n{dim_data['dimension']}: {dim_data['reasoning']}"

        return node.overall_score, False

    def _global_reconsider(
        self,
        case_summary: str,
        eliminated_suspects: List[str],
        case_text: str,
        structured_knowledge: Dict,
    ) -> List[Dict]:
        """全局回退: 所有Top嫌疑人被排除时，重新考虑次要人物"""

        # 提取所有人物
        all_persons = []
        if structured_knowledge and isinstance(structured_knowledge, dict):
            pr = structured_knowledge.get("person_relation", {})
            if isinstance(pr, dict):
                pdata = pr.get("data", {})
                if isinstance(pdata, dict):
                    persons = pdata.get("persons", [])
                    all_persons = [p.get("name", "") for p in persons if isinstance(p, dict) and p.get("name")]

        if not all_persons:
            # 从文本中提取(简单方法)
            all_persons = [s for s in eliminated_suspects]  # fallback

        prompt = RECONSIDER_PROMPT.format(
            case_summary=case_summary[:1000],
            eliminated_summary="\n".join(
                [f"- {s}: 已排除" for s in eliminated_suspects]
            ),
            all_persons=", ".join(all_persons[:20]),
        )

        response = self._call_llm(prompt, temperature=0.5)
        result = self._extract_json(response)

        if result and "new_suspects" in result:
            new_suspects = result["new_suspects"]
            self.logger.info(f"  🔄 全局回退建议: {new_suspects}")
            return new_suspects[:3]

        return []

    def _global_reevaluate(
        self,
        hypotheses: List[ReasoningNode],
        case_summary: str,
        vote_result: Dict,
    ) -> Dict:
        """全局对比评估 — 对多个存活假设做最终比较"""

        # 构建假设摘要
        hypotheses_summary = []
        for h in hypotheses:
            dims_str = "; ".join(
                [f"{d['dimension']}={d['score']:.2f}" for d in h.verification_dimensions.values()]
            )
            hypotheses_summary.append(
                f"假设: {h.suspect_name} | 综合得分={h.overall_score:.3f} | "
                f"证据链={h.evidence_chain_score:.2f} | 矛盾惩罚={h.contradiction_penalty:.2f}\n"
                f"  维度: {dims_str}\n"
                f"  推理: {h.reasoning_text[:300]}"
            )

        vote_summary = "无投票数据"
        if vote_result:
            vote_dist = vote_result.get("vote_distribution", {})
            vote_summary = "; ".join([f"{k}={v:.2%}" for k, v in sorted(vote_dist.items(), key=lambda x: -x[1])])

        prompt = GLOBAL_REEVAL_PROMPT.format(
            case_summary=case_summary[:1200],
            hypotheses_summary="\n\n".join(hypotheses_summary),
            vote_summary=vote_summary,
        )

        response = self._call_llm(prompt, temperature=0.2)
        result = self._extract_json(response)

        if result and "final_ranking" in result:
            return result

        # fallback: 按overall_score排序
        sorted_h = sorted(hypotheses, key=lambda h: -h.overall_score)
        return {
            "final_ranking": [
                {"name": h.suspect_name, "score": h.overall_score, "reasoning": h.reasoning_text[:200]}
                for h in sorted_h
            ],
            "overturned": False,
            "overturn_reason": "",
            "confidence_calibration": sorted_h[0].overall_score if sorted_h else 0.0,
        }

    def _build_result(
        self,
        case_text: str,
        vote_result: Dict,
        final_result: Dict = None,
        best_path: ReasoningNode = None,
        elapsed: float = 0.0,
    ) -> Dict[str, Any]:
        """构建最终输出"""

        # 确定推理树结论
        if final_result and final_result.get("final_ranking"):
            top = final_result["final_ranking"][0]
            tree_culprit = top["name"]
            tree_confidence = top["score"]
            tree_overturned = final_result.get("overturned", False)
        elif best_path:
            tree_culprit = best_path.suspect_name
            tree_confidence = best_path.overall_score
            tree_overturned = False
        else:
            tree_culprit = vote_result.get("winner", "未知") if vote_result else "未知"
            tree_confidence = vote_result.get("confidence", 0) if vote_result else 0
            tree_overturned = False

        # 与投票结果对比
        vote_winner = vote_result.get("winner", "") if vote_result else ""
        is_different = normalize_name(tree_culprit) != normalize_name(vote_winner)

        return {
            "track": "ReasoningTree",
            "tree_culprit": tree_culprit,
            "tree_confidence": round(tree_confidence, 4),
            "tree_overturned": tree_overturned,
            "is_different_from_vote": is_different,
            "vote_winner": vote_winner,
            "final_result": final_result or {},
            "best_path": best_path.to_dict() if best_path else None,
            "tree_structure": {
                "total_nodes": self.stats["nodes_created"],
                "pruned_nodes": self.stats["nodes_pruned"],
                "backtracks": self.stats["backtracks"],
                "global_reconsiders": self.stats["global_reconsiders"],
            },
            "all_nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "stats": self.stats,
            "timing": round(elapsed, 1),
        }
