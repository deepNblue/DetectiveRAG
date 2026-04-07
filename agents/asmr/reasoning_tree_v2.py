
import json
import time
import uuid
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from .name_utils import normalize_name, is_valid_suspect, build_name_alias_map

EVIDENCE_PHASES = {
    "phase1_hard": {
        "name": "硬证据阶段",
        "description": "时间线约束 + 物证分析 — 建立先验概率",
        "dimensions": ["timeline", "physical_evidence"],
        "weight": 0.35,
        "can_hard_exclude": True,
    },
    "phase2_behavioral": {
        "name": "行为证据阶段",
        "description": "动机 + 机会 + 能力 — 贝叶斯更新",
        "dimensions": ["motive", "opportunity", "capability"],
        "weight": 0.40,
        "can_hard_exclude": False,
    },
    "phase3_contradiction": {
        "name": "矛盾证据阶段",
        "description": "交叉验证 + 反向论证 — 动态校准",
        "dimensions": ["cross_validation", "devil_advocate"],
        "weight": 0.25,
        "can_hard_exclude": True,
    },
}

PHASE_DIMENSIONS = {
    "timeline": {
        "name": "时间线一致性",
        "phase": "phase1_hard",
        "prompt": "该嫌疑人的时间线是否与案件事实吻合？有无不在场证明？案发时间段内的活动是否可验证？",
    },
    "physical_evidence": {
        "name": "物证关联度",
        "phase": "phase1_hard",
        "prompt": "物证(指纹、DNA、监控等)是否指向该嫌疑人？有无直接/间接物证支持或排除？",
    },
    "motive": {
        "name": "作案动机",
        "phase": "phase2_behavioral",
        "prompt": "该嫌疑人是否有充分、合理的作案动机？动机强度如何？是否有隐藏动机？",
    },
    "opportunity": {
        "name": "作案机会",
        "phase": "phase2_behavioral",
        "prompt": "该嫌疑人在案发时是否有条件实施犯罪？是否能接触被害人/犯罪现场/作案工具？",
    },
    "capability": {
        "name": "作案能力",
        "phase": "phase2_behavioral",
        "prompt": "该嫌疑人是否有能力完成犯罪行为？体力、技能、知识是否匹配？",
    },
    "cross_validation": {
        "name": "交叉验证",
        "phase": "phase3_contradiction",
        "prompt": "各维度证据之间是否一致？有无自相矛盾？其他嫌疑人是否有更强解释？",
    },
    "devil_advocate": {
        "name": "反向论证(恶魔代言人)",
        "phase": "phase3_contradiction",
        "prompt": "假设该嫌疑人不是真凶，能否找到其他合理解释？对其不利的证据是否有其他解读？",
    },
}

@dataclass
class ReasoningNodeV2:
    """推理树节点v2 — 支持阶段性证据累积"""
    node_id: str
    parent_id: Optional[str] = None
    hypothesis: str = ""
    suspect_name: str = ""
    depth: int = 0

    phase_scores: Dict[str, float] = field(default_factory=lambda: {
        "phase1_hard": 0.5, "phase2_behavioral": 0.5, "phase3_contradiction": 0.5
    })
    prior_probability: float = 0.5
    posterior_probability: float = 0.5
    calibrated_probability: float = 0.5

    dimension_results: Dict[str, Dict] = field(default_factory=dict)
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    evidence_conflicts: List[Dict] = field(default_factory=list)

    visits: int = 0
    is_pruned: bool = False
    prune_reason: str = ""
    is_expanded: bool = False
    current_phase: str = ""
    children: List[str] = field(default_factory=list)
    reasoning_text: str = ""

    @property
    def overall_score(self) -> float:
        return max(0.0, min(1.0, self.calibrated_probability))

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "suspect_name": self.suspect_name,
            "phase_scores": {k: round(v, 3) for k, v in self.phase_scores.items()},
            "prior": round(self.prior_probability, 3),
            "posterior": round(self.posterior_probability, 3),
            "calibrated": round(self.calibrated_probability, 3),
            "dimension_results": {k: {kk: str(vv)[:100] for kk, vv in v.items()} for k, v in self.dimension_results.items()},
            "supporting_count": len(self.supporting_evidence),
            "contradicting_count": len(self.contradicting_evidence),
            "conflicts": self.evidence_conflicts,
            "is_pruned": self.is_pruned,
            "current_phase": self.current_phase,
        }



PHASE1_PROMPT = """你是一名刑侦鉴识专家，正在做第一阶段评估：硬证据分析。

【案件事实】
{case_summary}

【假设】嫌疑人 "{suspect_name}" 是真凶

【验证维度】{dimension_name}
{dimension_prompt}

【物证与时间线数据】
{evidence_context}

请严格从"{dimension_name}"角度评估。这是硬证据阶段，重点关注客观事实。
输出JSON:
{{
  "score": 0.0-1.0,
  "evidence": ["具体证据1", ...],
  "counter_evidence": ["反证1", ...],
  "reasoning": "推理过程(200字内)",
  "is_exclusionary": false,
  "exclusionary_reason": ""
}}

规则:
- score: 1.0=铁证指向该嫌疑人, 0.0=铁证排除该嫌疑人
- is_exclusionary=true 表示发现铁证般的不在场证明或物证排除(如DNA不匹配)
- 只输出JSON"""

PHASE2_PROMPT = """你是一名刑侦心理画像专家，正在做第二阶段评估：行为证据分析。

【案件事实】
{case_summary}

【假设】嫌疑人 "{suspect_name}" 是真凶

【验证维度】{dimension_name}
{dimension_prompt}

【第一阶段硬证据结论】
{phase1_summary}

【其他专家意见】
{expert_opinions}

基于第一阶段已经建立的先验概率({prior_prob:.2f})，请从"{dimension_name}"角度做贝叶斯更新。
输出JSON:
{{
  "likelihood_ratio": 0.1-10.0,
  "score": 0.0-1.0,
  "evidence": ["行为证据1", ...],
  "counter_evidence": ["反行为证据1", ...],
  "reasoning": "推理过程(200字内)",
  "bayesian_adjustment": "说明如何调整先验概率(100字)"
}}

规则:
- likelihood_ratio: >1表示证据支持假设, <1表示削弱
- score: 该维度的独立评分
- 必须考虑第一阶段结论 — 如果硬证据已经排除，行为证据不能无条件推翻
- 如果先验很低(<0.3)，需要更强的行为证据才能提升
- 只输出JSON"""

PHASE3_PROMPT = """你是一名刑侦审查官，正在做第三阶段评估：矛盾检测和反向论证。

【案件事实】
{case_summary}

【假设】嫌疑人 "{suspect_name}" 是真凶

【验证维度】{dimension_name}
{dimension_prompt}

【Phase1硬证据结论】
{phase1_summary}

【Phase2行为证据结论】
{phase2_summary}

【当前后验概率】{posterior_prob:.2f}

现在做最终校准。请特别关注:
1. 各阶段证据是否冲突？(如: 时间线排除但动机很强)
2. 反向论证 — 如果该嫌疑人不是真凶，是否有合理解释？
3. 证据链的完整性 — 是否有逻辑闭环？

输出JSON:
{{
  "calibration_score": 0.0-1.0,
  "conflicts_found": ["冲突描述1", ...],
  "devil_advocate_args": ["反向论证1", ...],
  "evidence": ["校准证据1", ...],
  "counter_evidence": ["反校准证据1", ...],
  "reasoning": "推理过程(200字内)",
  "final_adjustment": "上调/下调/维持",
  "adjustment_reason": "原因(100字)",
  "is_fatal_contradiction": false,
  "fatal_reason": ""
}}

规则:
- calibration_score: 0=完全推翻, 1=完全确认
- is_fatal_contradiction=true 表示发现无法调和的矛盾(如证据链断裂)
- 只输出JSON"""

GLOBAL_REEVAL_PROMPT_V2 = """你是一名首席刑侦推理官，需要对多个嫌疑人做最终裁决。

【案件事实】
{case_summary}

【各嫌疑人三阶段评估结果】
{hypotheses_summary}

【原始投票结果】
{vote_summary}

每个嫌疑人都经过了三个阶段的评估：
- Phase1(硬证据): 时间线+物证 → 先验概率
- Phase2(行为证据): 动机+机会+能力 → 后验概率
- Phase3(矛盾证据): 交叉验证+反向论证 → 校准概率

请综合所有阶段的结果，给出最终排名。注意:
1. 硬证据(Phase1)是最可靠的，行为证据(Phase2)次之，矛盾检测(Phase3)用于校准
2. 如果一个嫌疑人Phase1得分很高但Phase3发现致命矛盾，应当谨慎
3. 如果投票结果与推理树结论不同，说明理由
4. 置信度要基于证据链的完整性和一致性，不要一律给0.9

输出JSON:
{{
  "final_ranking": [
    {{"name": "嫌疑人名", "score": 0.0-1.0, "reasoning": "综合理由(150字)", "confidence": 0.0-1.0}}
  ],
  "overturned": false,
  "overturn_reason": "",
  "confidence_calibration": 0.0-1.0,
  "evidence_chain_assessment": "整体证据链评估(200字)"
}}

只输出JSON"""

RECONSIDER_PROMPT_V2 = """你是一名刑侦推理官。当前所有嫌疑人假设在证据阶段性验证中都被排除，需要重新考虑。

【案件事实】
{case_summary}

【已排除的嫌疑人及原因】
{eliminated_summary}

【案件中的所有人物】
{all_persons}

请从中选出最多3个之前被忽略的可能真凶。注意:
- 重点关注次要人物、不显眼的角色
- 考虑之前排除嫌疑人的原因中是否有遗漏
- 新嫌疑人应该能解释已有证据

输出JSON:
{{
  "new_suspects": [
    {{"name": "人物名", "reason": "为什么此人可能是真凶(100字)"}}
  ]
}}

只输出JSON"""



class DetectiveSearchTreeV2:
    """
    刑侦推理搜索树 v2 - 证据阶段性动态调整

    核心升级:
    1. 证据分3阶段注入（硬证据→行为证据→矛盾证据）
    2. 每阶段结束后做贝叶斯式后验更新
    3. 后一阶段可以放大/缩小前一阶段结论
    4. 发现阶段间冲突时动态降权/升权
    """

    def __init__(self, llm_client=None, config: Dict[str, Any] = None):
        self.llm_client = llm_client
        self.config = config or {}
        self.logger = logger.bind(module="DetectiveSearchTreeV2")
        
        self.max_depth = self.config.get("max_depth", 3)
        self.prune_threshold = self.config.get("prune_threshold", 0.15)
        self.max_iterations = self.config.get("max_iterations", 20)
        
        self.nodes: Dict[str, ReasoningNodeV2] = {}
        self.root_id: Optional[str] = None
        self.stats = {
            "nodes_created": 0, "nodes_pruned": 0, "backtracks": 0,
            "global_reconsiders": 0, "llm_calls": 0,
            "phase1_time": 0, "phase2_time": 0, "phase3_time": 0,
        }
    
    def _new_id(self) -> str:
        return f"n_{uuid.uuid4().hex[:8]}"
    
    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        self.stats["llm_calls"] += 1
        if self.llm_client is None:
            return '{"score": 0.5, "evidence": [], "counter_evidence": [], "reasoning": "LLM不可用"}'
        try:
            return self.llm_client.simple_chat(prompt, temperature=temperature)
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return '{"score": 0.5, "evidence": [], "counter_evidence": [], "reasoning": "LLM调用失败"}'
    
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
        self.logger.warning(f"无法提取JSON (长度={len(text)})")
        return None
    
    
    def search(
        self,
        case_text: str,
        suspects: List[str],
        structured_knowledge: Dict = None,
        search_results: Dict = None,
        expert_analyses: List[Dict] = None,
        vote_result: Dict = None,
    ) -> Dict[str, Any]:
        """运行推理树搜索 (v2 三阶段)"""
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("🌳 刑侦推理树v2 (证据阶段性) 搜索启动")
        self.logger.info(f"嫌疑人: {suspects}, 阶段数: 3")
        self.logger.info("=" * 60)
        
        self.nodes.clear()
        self.stats = {
            "nodes_created": 0, "nodes_pruned": 0, "backtracks": 0,
            "global_reconsiders": 0, "llm_calls": 0,
            "phase1_time": 0, "phase2_time": 0, "phase3_time": 0,
        }
        
        case_summary = self._build_case_summary(case_text, structured_knowledge)
        evidence_context = self._build_evidence_context(structured_knowledge, search_results)
        expert_opinions = self._build_expert_opinions(expert_analyses)
        
        # 创建根节点
        self.root_id = self._new_id()
        root = ReasoningNodeV2(node_id=self.root_id, hypothesis="案件事实", depth=0)
        self.nodes[self.root_id] = root
        self.stats["nodes_created"] += 1
        
        # 按投票优先级排序
        prioritized = self._prioritize_suspects(suspects, vote_result)
        self.logger.info(f"📋 嫌疑人优先级: {prioritized}")
        
        # 创建假设节点
        hypothesis_nodes = []
        for suspect in prioritized:
            h = self._create_hypothesis_node(self.root_id, suspect, case_summary)
            if h:
                hypothesis_nodes.append(h)
                root.children.append(h.node_id)
        
        if not hypothesis_nodes:
            return self._build_result(case_text, vote_result, time.time() - start_time)
        
        eliminated = []
        
        # ========== Phase 1: 硬证据 (时间线+物证) → 先验概率 ==========
        phase1_start = time.time()
        self.logger.info("🔬 Phase 1: 硬证据分析 (时间线 + 物证) → 建立先验概率")
        
        for h in hypothesis_nodes:
            if h.is_pruned:
                continue
            score, excluded = self._run_phase1(
                node=h, case_summary=case_summary, evidence_context=evidence_context
            )
            h.phase_scores["phase1_hard"] = score
            h.prior_probability = score  # Phase 1 的结果就是先验
            h.current_phase = "phase1_hard"
            
            if excluded:
                self.logger.info(f"  ❌ Phase1排除: {h.suspect_name} ({h.prune_reason})")
                eliminated.append(h.suspect_name)
            else:
                self.logger.info(f"  ✅ Phase1: {h.suspect_name} 先验={score:.3f}")
        
        self.stats["phase1_time"] = time.time() - phase1_start
        
        # ========== Phase 2: 行为证据 (动机+机会+能力) → 贝叶斯更新 ==========
        phase2_start = time.time()
        self.logger.info("🧠 Phase 2: 行为证据分析 (动机 + 机会 + 能力) → 贝叶斯更新")
        
        for h in hypothesis_nodes:
            if h.is_pruned:
                continue
            
            phase1_summary = self._get_phase_summary(h, "phase1_hard")
            score, bayesian = self._run_phase2(
                node=h, case_summary=case_summary,
                phase1_summary=phase1_summary,
                expert_opinions=expert_opinions
            )
            h.phase_scores["phase2_behavioral"] = score
            
            # 贝叶斯更新: posterior = prior * adjustment
            # 如果行为证据支持(likelihood>1)，后验上升；否则下降
            h.posterior_probability = self._bayesian_update(
                prior=h.prior_probability,
                likelihood_ratio=bayesian.get("likelihood_ratio", 1.0),
                phase2_score=score
            )
            h.current_phase = "phase2_behavioral"
            
            self.logger.info(f"  📊 Phase2: {h.suspect_name} "
                           f"先验={h.prior_probability:.3f} → 后验={h.posterior_probability:.3f} "
                           f"(LR={bayesian.get('likelihood_ratio', '?')})")
        
        self.stats["phase2_time"] = time.time() - phase2_start
        
        # ========== Phase 3: 矛盾证据 (交叉验证+反向论证) → 校准 ==========
        phase3_start = time.time()
        self.logger.info("⚖️ Phase 3: 矛盾证据分析 (交叉验证 + 反向论证) → 校准")
        
        for h in hypothesis_nodes:
            if h.is_pruned:
                continue
            
            phase1_s = self._get_phase_summary(h, "phase1_hard")
            phase2_s = self._get_phase_summary(h, "phase2_behavioral")
            score, calibration = self._run_phase3(
                node=h, case_summary=case_summary,
                phase1_summary=phase1_s, phase2_summary=phase2_s
            )
            h.phase_scores["phase3_contradiction"] = score
            
            # 校准: 根据矛盾证据调整后验
            h.calibrated_probability = self._calibrate(
                posterior=h.posterior_probability,
                calibration_score=score,
                calibration_data=calibration
            )
            h.current_phase = "phase3_contradiction"
            
            if calibration.get("is_fatal_contradiction"):
                h.is_pruned = True
                h.prune_reason = f"[Phase3致命矛盾] {calibration.get('fatal_reason', '')}"
                eliminated.append(h.suspect_name)
                self.stats["nodes_pruned"] += 1
                self.logger.info(f"  ❌ Phase3致命矛盾: {h.suspect_name} — {calibration.get('fatal_reason', '')}")
            elif h.calibrated_probability < self.prune_threshold:
                h.is_pruned = True
                h.prune_reason = f"校准后概率过低({h.calibrated_probability:.3f})"
                eliminated.append(h.suspect_name)
                self.stats["nodes_pruned"] += 1
                self.logger.info(f"  ⚠️ Phase3排除: {h.suspect_name} (校准={h.calibrated_probability:.3f})")
            else:
                self.logger.info(f"  ✅ Phase3: {h.suspect_name} 校准={h.calibrated_probability:.3f} "
                               f"(调整: {calibration.get('final_adjustment', '?')})")
        
        self.stats["phase3_time"] = time.time() - phase3_start
        
        # ========== 最终评估 ==========
        active = [n for n in self.nodes.values() if n.depth == 1 and not n.is_pruned]
        best_path = max(active, key=lambda n: n.overall_score) if active else None
        
        if best_path is None and eliminated:
            # 全局回退
            self.stats["global_reconsiders"] += 1
            reconsidered = self._global_reconsider(case_summary, eliminated, case_text, structured_knowledge)
            for ns in reconsidered:
                h = self._create_hypothesis_node(self.root_id, ns["name"], case_summary)
                if h:
                    root.children.append(h.node_id)
                    # 跑完整3阶段
                    self._run_all_phases(h, case_summary, evidence_context, expert_opinions)
                    if not h.is_pruned and h.overall_score > (best_path.overall_score if best_path else 0):
                        best_path = h
        
        if len(active) >= 2:
            final_result = self._global_reevaluate(active, case_summary, vote_result)
        elif best_path:
            final_result = {
                "final_ranking": [{"name": best_path.suspect_name, "score": best_path.overall_score,
                                   "reasoning": best_path.reasoning_text[:200], "confidence": best_path.calibrated_probability}],
                "overturned": False, "overturn_reason": "",
                "confidence_calibration": best_path.calibrated_probability,
            }
        else:
            final_result = {"final_ranking": [], "overturned": False,
                           "overturn_reason": "推理树未能找到有效假设", "confidence_calibration": 0.0}
        
        elapsed = time.time() - start_time
        
        self.logger.info("=" * 60)
        if final_result.get("final_ranking"):
            top = final_result["final_ranking"][0]
            self.logger.info(f"🌳 推理树v2结论: {top['name']} (得分={top['score']:.3f})")
        self.logger.info(f"📊 统计: 创建{self.stats['nodes_created']}节点, "
                        f"剪枝{self.stats['nodes_pruned']}, "
                        f"P1={self.stats['phase1_time']:.1f}s, "
                        f"P2={self.stats['phase2_time']:.1f}s, "
                        f"P3={self.stats['phase3_time']:.1f}s, "
                        f"LLM={self.stats['llm_calls']}次")
        self.logger.info("=" * 60)
        
        return self._build_result(case_text, vote_result, final_result, best_path, elapsed)
    
    def _run_all_phases(self, node, case_summary, evidence_context, expert_opinions):
        """对节点跑完整3阶段（用于全局回退的新嫌疑人）"""
        # Phase 1
        s1, ex = self._run_phase1(node, case_summary, evidence_context)
        node.phase_scores["phase1_hard"] = s1
        node.prior_probability = s1
        if ex:
            node.is_pruned = True
            return
        
        # Phase 2
        p1s = self._get_phase_summary(node, "phase1_hard")
        s2, bay = self._run_phase2(node, case_summary, p1s, expert_opinions)
        node.phase_scores["phase2_behavioral"] = s2
        node.posterior_probability = self._bayesian_update(s1, bay.get("likelihood_ratio", 1.0), s2)
        
        # Phase 3
        p1s = self._get_phase_summary(node, "phase1_hard")
        p2s = self._get_phase_summary(node, "phase2_behavioral")
        s3, cal = self._run_phase3(node, case_summary, p1s, p2s)
        node.phase_scores["phase3_contradiction"] = s3
        node.calibrated_probability = self._calibrate(node.posterior_probability, s3, cal)
        
        if cal.get("is_fatal_contradiction") or node.calibrated_probability < self.prune_threshold:
            node.is_pruned = True


    def _run_phase1(self, node, case_summary, evidence_context):
        """Phase 1: 硬证据 (时间线 + 物证)"""
        dims = EVIDENCE_PHASES["phase1_hard"]["dimensions"]
        total_score = 0.0
        total_weight = 0.0
        excluded = False
        
        for dim_key in dims:
            dim_info = PHASE_DIMENSIONS[dim_key]
            prompt = PHASE1_PROMPT.format(
                case_summary=case_summary[:1200],
                suspect_name=node.suspect_name,
                dimension_name=dim_info["name"],
                dimension_prompt=dim_info["prompt"],
                evidence_context=evidence_context[:600],
            )
            response = self._call_llm(prompt, temperature=0.2)
            result = self._extract_json(response)
            if result is None:
                result = {"score": 0.5, "evidence": [], "counter_evidence": [], "reasoning": "无法解析", "is_exclusionary": False}
            
            score = float(result.get("score", 0.5))
            node.dimension_results[dim_key] = {
                "dimension": dim_info["name"], "score": round(score, 3),
                "evidence": result.get("evidence", []),
                "counter_evidence": result.get("counter_evidence", []),
                "reasoning": result.get("reasoning", "")[:200],
                "phase": "phase1_hard",
            }
            node.supporting_evidence.extend(result.get("evidence", []))
            node.contradicting_evidence.extend(result.get("counter_evidence", []))
            
            total_score += score
            total_weight += 1.0
            
            self.logger.info(f"    🔬 P1-{dim_info['name']}: {score:.2f} "
                           f"(支持{len(result.get('evidence',[]))}项, "
                           f"反证{len(result.get('counter_evidence',[]))}项)"
                           f"{' 🚫排除!' if result.get('is_exclusionary') else ''}")
            
            if result.get("is_exclusionary"):
                excluded = True
                node.is_pruned = True
                node.prune_reason = f"[P1-{dim_info['name']}] {result.get('exclusionary_reason', '')}"
                return 0.0, True
        
        return total_score / total_weight if total_weight > 0 else 0.5, False
    
    def _run_phase2(self, node, case_summary, phase1_summary, expert_opinions):
        """Phase 2: 行为证据 (动机 + 机会 + 能力) — 贝叶斯更新"""
        dims = EVIDENCE_PHASES["phase2_behavioral"]["dimensions"]
        total_lr = 1.0
        total_score = 0.0
        total_weight = 0.0
        
        for dim_key in dims:
            dim_info = PHASE_DIMENSIONS[dim_key]
            prompt = PHASE2_PROMPT.format(
                case_summary=case_summary[:1200],
                suspect_name=node.suspect_name,
                dimension_name=dim_info["name"],
                dimension_prompt=dim_info["prompt"],
                phase1_summary=phase1_summary[:500],
                expert_opinions=expert_opinions[:400],
                prior_prob=node.prior_probability,
            )
            response = self._call_llm(prompt, temperature=0.2)
            result = self._extract_json(response)
            if result is None:
                result = {"likelihood_ratio": 1.0, "score": 0.5, "evidence": [], "counter_evidence": [], "reasoning": "无法解析"}
            
            lr = float(result.get("likelihood_ratio", 1.0))
            lr = max(0.1, min(10.0, lr))  # clamp
            score = float(result.get("score", 0.5))
            
            node.dimension_results[dim_key] = {
                "dimension": dim_info["name"], "score": round(score, 3),
                "likelihood_ratio": round(lr, 3),
                "evidence": result.get("evidence", []),
                "counter_evidence": result.get("counter_evidence", []),
                "reasoning": result.get("reasoning", "")[:200],
                "bayesian_adjustment": result.get("bayesian_adjustment", "")[:200],
                "phase": "phase2_behavioral",
            }
            node.supporting_evidence.extend(result.get("evidence", []))
            node.contradicting_evidence.extend(result.get("counter_evidence", []))
            
            total_lr *= lr  # 各维度LR相乘
            total_score += score
            total_weight += 1.0
            
            self.logger.info(f"    🧠 P2-{dim_info['name']}: {score:.2f} "
                           f"(LR={lr:.2f}, "
                           f"支持{len(result.get('evidence',[]))}项)")
        
        avg_score = total_score / total_weight if total_weight > 0 else 0.5
        return avg_score, {"likelihood_ratio": total_lr}
    
    def _run_phase3(self, node, case_summary, phase1_summary, phase2_summary):
        """Phase 3: 矛盾证据 (交叉验证 + 反向论证)"""
        dims = EVIDENCE_PHASES["phase3_contradiction"]["dimensions"]
        total_score = 0.0
        total_weight = 0.0
        all_calibration = {}
        
        for dim_key in dims:
            dim_info = PHASE_DIMENSIONS[dim_key]
            prompt = PHASE3_PROMPT.format(
                case_summary=case_summary[:1200],
                suspect_name=node.suspect_name,
                dimension_name=dim_info["name"],
                dimension_prompt=dim_info["prompt"],
                phase1_summary=phase1_summary[:500],
                phase2_summary=phase2_summary[:500],
                posterior_prob=node.posterior_probability,
            )
            response = self._call_llm(prompt, temperature=0.2)
            result = self._extract_json(response)
            if result is None:
                result = {"calibration_score": 0.5, "conflicts_found": [], "devil_advocate_args": [],
                         "reasoning": "无法解析", "final_adjustment": "维持", "is_fatal_contradiction": False}
            
            cal_score = float(result.get("calibration_score", 0.5))
            
            node.dimension_results[dim_key] = {
                "dimension": dim_info["name"], "calibration_score": round(cal_score, 3),
                "conflicts": result.get("conflicts_found", []),
                "devil_advocate": result.get("devil_advocate_args", []),
                "reasoning": result.get("reasoning", "")[:200],
                "adjustment": result.get("final_adjustment", "维持"),
                "phase": "phase3_contradiction",
            }
            
            # 记录冲突
            for c in result.get("conflicts_found", []):
                node.evidence_conflicts.append({"dimension": dim_info["name"], "conflict": c})
            
            total_score += cal_score
            total_weight += 1.0
            
            self.logger.info(f"    ⚖️ P3-{dim_info['name']}: {cal_score:.2f} "
                           f"(调整:{result.get('final_adjustment','?')}, "
                           f"冲突{len(result.get('conflicts_found',[]))}个"
                           f"{' ☠️致命!' if result.get('is_fatal_contradiction') else ''})")
            
            if result.get("is_fatal_contradiction"):
                all_calibration["is_fatal_contradiction"] = True
                all_calibration["fatal_reason"] = result.get("fatal_reason", "")
            
            # 收集校准信息
            all_calibration.setdefault("final_adjustment", result.get("final_adjustment", "维持"))
            all_calibration.setdefault("adjustment_reason", result.get("adjustment_reason", ""))
        
        avg_cal = total_score / total_weight if total_weight > 0 else 0.5
        return avg_cal, all_calibration
    
    def _bayesian_update(self, prior: float, likelihood_ratio: float, phase2_score: float) -> float:
        """
        贝叶斯更新: P(guilty|evidence) = P(evidence|guilty) * P(guilty) / P(evidence)
        
        简化实现:
        - 使用 odds ratio 形式: posterior_odds = prior_odds * LR
        - 再融合 phase2_score 作为软约束
        """
        # odds = p / (1 - p)
        prior = max(0.01, min(0.99, prior))
        prior_odds = prior / (1 - prior)
        
        # posterior_odds = prior_odds * LR
        posterior_odds = prior_odds * likelihood_ratio
        
        # 转回概率
        posterior = posterior_odds / (1 + posterior_odds)
        
        # 融合 phase2_score (加权混合，避免极端值)
        # 如果 phase2_score 和 posterior 方向一致，增强；否则取中间值
        alpha = 0.6  # posterior 权重
        blended = alpha * posterior + (1 - alpha) * phase2_score
        
        return max(0.01, min(0.99, blended))
    
    def _calibrate(self, posterior: float, calibration_score: float, calibration_data: Dict) -> float:
        """
        Phase 3 校准: 根据矛盾证据调整后验概率
        
        策略:
        - calibration_score > 0.5: 轻微上调（矛盾少，确认证据链）
        - calibration_score 0.3-0.5: 轻微下调（有矛盾但不致命）
        - calibration_score < 0.3: 显著下调（矛盾严重）
        - 致命矛盾: 直接降到极低值
        """
        if calibration_data.get("is_fatal_contradiction"):
            return 0.05  # 致命矛盾，几乎排除
        
        # 校准因子: calibration_score 作为乘数
        if calibration_score >= 0.7:
            factor = 1.0 + (calibration_score - 0.7) * 0.5  # 1.0 ~ 1.15
        elif calibration_score >= 0.5:
            factor = 1.0  # 维持
        elif calibration_score >= 0.3:
            factor = 0.5 + calibration_score  # 0.8 ~ 1.0
        else:
            factor = calibration_score * 2  # 0 ~ 0.6, 大幅下调
        
        calibrated = posterior * factor
        
        # 约束范围
        return max(0.01, min(0.99, calibrated))
    
    def _get_phase_summary(self, node, phase_key: str) -> str:
        """获取某阶段的摘要"""
        dims = EVIDENCE_PHASES.get(phase_key, {}).get("dimensions", [])
        parts = []
        for dk in dims:
            dr = node.dimension_results.get(dk, {})
            if dr:
                parts.append(f"- {dr.get('dimension', dk)}: 得分={dr.get('score', dr.get('calibration_score', '?'))}, "
                           f"推理={dr.get('reasoning', '无')[:100]}")
        if phase_key == "phase1_hard":
            parts.append(f"先验概率: {node.prior_probability:.3f}")
        elif phase_key == "phase2_behavioral":
            parts.append(f"后验概率: {node.posterior_probability:.3f}")
        return "\n".join(parts) if parts else "该阶段尚无结果"


    def _build_case_summary(self, case_text: str, structured_knowledge: Dict) -> str:
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
                        parts.append(f"物证({len(physical)}件): " + "; ".join([str(e)[:80] for e in physical[:5]]))
        return "\n".join(parts) if parts else "暂无额外证据信息"

    def _build_expert_opinions(self, expert_analyses: List[Dict]) -> str:
        if not expert_analyses:
            return "暂无专家意见"
        parts = []
        for a in expert_analyses[:8]:
            parts.append(f"- {a.get('perspective', '?')}: 认为凶手是{a.get('culprit', '?')}(置信度={a.get('confidence', 0)})")
        return "\n".join(parts)

    def _prioritize_suspects(self, suspects: List[str], vote_result: Dict) -> List[str]:
        if not vote_result:
            return suspects
        vote_dist = vote_result.get("vote_distribution", {})
        alias_map = build_name_alias_map(list(vote_dist.keys()) + suspects)
        scored = []
        for s in suspects:
            norm = normalize_name(s)
            canonical = alias_map.get(norm, norm)
            score = vote_dist.get(canonical, vote_dist.get(norm, 0.0))
            scored.append((s, score))
        scored.sort(key=lambda x: -x[1])
        return [s for s, _ in scored]

    def _create_hypothesis_node(self, parent_id: str, suspect_name: str, case_summary: str) -> Optional[ReasoningNodeV2]:
        node_id = self._new_id()
        node = ReasoningNodeV2(
            node_id=node_id, parent_id=parent_id,
            hypothesis=f"假设: {suspect_name} 是真凶",
            suspect_name=suspect_name, depth=1,
        )
        self.nodes[node_id] = node
        self.stats["nodes_created"] += 1
        
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
            node.prior_probability = float(result.get("initial_score", 0.3))
            node.posterior_probability = node.prior_probability
            node.calibrated_probability = node.prior_probability
            node.reasoning_text = result.get("brief_reason", "")
        else:
            node.prior_probability = 0.3
        
        self.logger.info(f"  📌 创建假设: {suspect_name} (初始得分={node.prior_probability:.2f})")
        return node

    def _global_reconsider(self, case_summary, eliminated, case_text, structured_knowledge):
        all_persons = []
        if structured_knowledge and isinstance(structured_knowledge, dict):
            pr = structured_knowledge.get("person_relation", {})
            if isinstance(pr, dict):
                pdata = pr.get("data", {})
                if isinstance(pdata, dict):
                    persons = pdata.get("persons", [])
                    all_persons = [p.get("name", "") for p in persons if isinstance(p, dict) and p.get("name")]
        if not all_persons:
            all_persons = [s for s in eliminated]
        
        prompt = RECONSIDER_PROMPT_V2.format(
            case_summary=case_summary[:1000],
            eliminated_summary="\n".join([f"- {s}: 已排除" for s in eliminated]),
            all_persons=", ".join(all_persons[:20]),
        )
        response = self._call_llm(prompt, temperature=0.5)
        result = self._extract_json(response)
        if result and "new_suspects" in result:
            self.logger.info(f"  🔄 全局回退: {result['new_suspects']}")
            return result["new_suspects"][:3]
        return []

    def _global_reevaluate(self, hypotheses, case_summary, vote_result):
        summaries = []
        for h in hypotheses:
            dims = []
            for dk, dv in h.dimension_results.items():
                s = dv.get("score", dv.get("calibration_score", "?"))
                dims.append(f"{dv.get('dimension', dk)}={s}")
            summaries.append(
                f"假设: {h.suspect_name}\n"
                f"  Phase1(硬证据→先验): {h.phase_scores['phase1_hard']:.3f} → prior={h.prior_probability:.3f}\n"
                f"  Phase2(行为→后验): {h.phase_scores['phase2_behavioral']:.3f} → posterior={h.posterior_probability:.3f}\n"
                f"  Phase3(矛盾→校准): {h.phase_scores['phase3_contradiction']:.3f} → calibrated={h.calibrated_probability:.3f}\n"
                f"  冲突: {len(h.evidence_conflicts)}个\n"
                f"  维度: {'; '.join(dims)}"
            )
        
        vote_summary = "无投票"
        if vote_result:
            vd = vote_result.get("vote_distribution", {})
            vote_summary = "; ".join([f"{k}={v:.2%}" for k, v in sorted(vd.items(), key=lambda x: -x[1])])
        
        prompt = GLOBAL_REEVAL_PROMPT_V2.format(
            case_summary=case_summary[:1200],
            hypotheses_summary="\n\n".join(summaries),
            vote_summary=vote_summary,
        )
        response = self._call_llm(prompt, temperature=0.2)
        result = self._extract_json(response)
        if result and "final_ranking" in result:
            return result
        
        sorted_h = sorted(hypotheses, key=lambda h: -h.overall_score)
        return {
            "final_ranking": [{"name": h.suspect_name, "score": h.overall_score,
                              "reasoning": h.reasoning_text[:200], "confidence": h.calibrated_probability}
                             for h in sorted_h],
            "overturned": False, "overturn_reason": "",
            "confidence_calibration": sorted_h[0].calibrated_probability if sorted_h else 0.0,
        }

    def _build_result(self, case_text, vote_result, final_result=None, best_path=None, elapsed=0.0):
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
        
        vote_winner = vote_result.get("winner", "") if vote_result else ""
        is_different = normalize_name(tree_culprit) != normalize_name(vote_winner)
        
        return {
            "track": "ReasoningTreeV2",
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
                "phase1_time": round(self.stats["phase1_time"], 1),
                "phase2_time": round(self.stats["phase2_time"], 1),
                "phase3_time": round(self.stats["phase3_time"], 1),
            },
            "all_nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "stats": self.stats,
            "timing": round(elapsed, 1),
        }
