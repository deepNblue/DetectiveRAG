"""
刑侦推理搜索树 v3 — 竞争假设搜索 (Competitive Hypothesis Search)

核心算法升级 (vs v2):
  v2: 对每个嫌疑人独立算绝对概率 → argmax
  v3: 嫌疑人互斥竞争 → 跨嫌疑人排序 → 假设验证 → 反向论证

四阶段:
  Phase 1 (Scatter):  并行收集所有嫌疑人在所有维度的原始证据
  Phase 2 (Compete):  跨嫌疑人竞争排序 — 每个维度内排名，而非绝对分
  Phase 3 (Branch):   对Top-2候选人展开多假设分支，深度验证
  Phase 4 (Devil):    对最终赢家做反向论证，如被推翻则选第二

数学基础:
  P(嫌疑人i是真凶|证据) ∝ P(证据|嫌疑人i) / Σ_j P(证据|嫌疑人j)
  即: 不是给每个人打绝对分，而是在竞争中比较相对强度
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
# Prompt 模板
# ============================================================

SCATTER_DIMENSION_PROMPT = """你是一名刑侦鉴识专家，正在做独立的证据评估。

【案件事实】
{case_summary}

【当前嫌疑人】{suspect_name}

【评估维度】{dimension_name}
{dimension_prompt}

【可用证据】
{evidence_context}

⚠️ 请参考上方"专家分析要点"中的具体发现，但你要独立评估，不要简单跟从专家结论。

请从"{dimension_name}"角度，仅针对嫌疑人"{suspect_name}"进行评估。
输出JSON:
{{
  "raw_score": 0.0-1.0,
  "evidence_for": ["支持该假设的具体证据（引用案件中的具体事实，不要泛泛而谈）"],
  "evidence_against": ["反对该假设的具体证据（引用案件中的具体事实）"],
  "reasoning": "评估理由(150字内，必须引用具体证据)",
  "key_finding": "一句话核心发现(30字内)"
}}
只输出JSON。"""

COMPETE_RANKING_PROMPT = """你是一名首席刑侦分析师，需要从竞争假设的角度做跨嫌疑人排序。

【案件事实】
{case_summary}

【评估维度】{dimension_name}
{dimension_prompt}

【各嫌疑人在该维度的原始评估】
{raw_assessments}

请对这 {n_suspects} 个嫌疑人在"{dimension_name}"维度上进行竞争排名。
注意: 这是相对排名，不是绝对评分。真凶只有一个，所以排名反映了谁在该维度上嫌疑更大。

输出JSON:
{{
  "ranking": [
    {{"name": "嫌疑人名", "rank_score": 0.0-1.0, "advantage": "该嫌疑人在此维度的优势", "weakness": "该嫌疑人在此维度的弱点"}}
  ],
  "dimension_winner": "该维度嫌疑最大的人",
  "dimension_loser": "该维度嫌疑最小的人",
  "key_insight": "该维度的关键发现(50字内)"
}}

排名规则:
- rank_score 是相对分数，最高分者=1.0时，其他人按比例缩放
- 如果某人在该维度明显领先，分数差距应该拉大
- 如果多人接近，分数差距应该缩小
只输出JSON。"""

BRANCH_HYPOTHESIS_PROMPT = """你是一名刑侦推理官，正在做深度假设验证。

【案件事实】
{case_summary}

【当前假设】"{suspect_name}" 是真凶
【假设来源】该嫌疑人在竞争排序中排名第{rank_position}

【该嫌疑人的证据概况】
{evidence_profile}

【竞争嫌疑人概况】
{competitor_profile}

请深度验证这个假设。考虑:
1. 如果该嫌疑人是真凶，完整的犯罪链条是什么？（动机→计划→实施→掩盖）
2. 证据链是否闭环？有没有断裂环节？
3. 该嫌疑人比竞争者强在哪里？弱在哪里？
4. 有没有"只有TA才能解释"的证据？

输出JSON:
{{
  "hypothesis_valid": true/false,
  "crime_chain": "完整犯罪链条描述(200字内)",
  "chain_complete": 0.0-1.0,
  "unique_evidence": ["只有该嫌疑人能解释的证据"],
  "chain_gaps": ["证据链断裂处"],
  "vs_competitor": "与竞争者比较分析(100字)",
  "confidence": 0.0-1.0,
  "reasoning": "推理过程(200字内)"
}}
只输出JSON。"""

DEVIL_ADVOCATE_PROMPT = """你是"恶魔代言人"——你的职责是尽最大努力推翻当前的结论。

【案件事实】
{case_summary}

【当前结论】"{winner}" 被认定为真凶（竞争得分={winner_score:.3f}）
【第二名】"{runner_up}"（竞争得分={runner_up_score:.3f}）

【赢家证据链】
{winner_profile}

【第二名证据链】
{runner_up_profile}

【竞争排序各维度得分】
{dimension_rankings}

请竭尽全力为"{runner_up}"辩护、攻击"{winner}"的结论:
1. 赢家的证据链中最薄弱的环节是什么？
2. 第二名有什么被忽略的强项？
3. 是否存在其他合理解释？
4. 如果要选第二名为真凶，需要什么额外证据？

输出JSON:
{{
  "should_overturn": true/false,
  "overturn_strength": 0.0-1.0,
  "winner_weaknesses": ["赢家证据链的弱点"],
  "runner_up_strengths": ["第二名的被忽略强项"],
  "alternative_theory": "如果第二名是真凶的替代犯罪理论(200字)",
  "missing_evidence_for_runner_up": "第二名需要什么额外证据才能确认",
  "final_verdict": "维持winner/推翻选runner_up",
  "confidence_in_verdict": 0.0-1.0,
  "reasoning": "论证过程(300字内)"
}}
只输出JSON。"""

FINAL_RANKING_PROMPT = """你是首席刑侦裁决官，基于竞争假设搜索结果做最终排名。

【案件事实】
{case_summary}

【竞争排序结果（各维度）】
{dimension_rankings}

【假设验证结果】
{branch_results}

【恶魔代言人挑战结果】
{devil_result}

请给出最终排名。注意:
1. 优先相信竞争排序一致指向的人
2. 如果假设验证发现某人的证据链断裂，应大幅降低其排名
3. 如果恶魔代言人成功挑战了第一名，考虑选第二名
4. 置信度应反映证据链的完整性，不是固定的0.9

输出JSON:
{{
  "final_ranking": [
    {{"name": "嫌疑人名", "score": 0.0-1.0, "confidence": 0.0-1.0, "reasoning": "综合理由(100字)"}}
  ],
  "winner_evidence_chain": "最终赢家的完整证据链(200字)",
  "confidence_explanation": "为什么给这个置信度(100字)",
  "key_differentiator": "区分真凶与其他嫌疑人的关键证据(50字)"
}}
只输出JSON。"""


# ============================================================
# 数据结构
# ============================================================

EVIDENCE_DIMENSIONS = {
    "timeline": {
        "name": "时间线一致性",
        "prompt": "该嫌疑人的时间线是否与案件事实吻合？有无不在场证明？案发时间段内的活动是否可验证？时间线中有无矛盾或空白？",
    },
    "physical_evidence": {
        "name": "物证关联度",
        "prompt": "物证(指纹、DNA、监控、痕迹等)是否指向该嫌疑人？有无直接/间接物证？物证排除可能性？",
    },
    "motive": {
        "name": "作案动机强度",
        "prompt": "该嫌疑人的作案动机有多强？是明确的经济/情感/权力动机，还是推测性的？动机是否有具体证据支撑？",
    },
    "opportunity": {
        "name": "作案机会控制",
        "prompt": "该嫌疑人是否控制了作案的关键环节（如投毒载体、入口、工具）？是直接控制还是间接接触？",
    },
    "capability": {
        "name": "作案能力匹配",
        "prompt": "该嫌疑人是否有能力完成犯罪的具体手法？需要什么特定技能/知识/体力？TA是否具备？",
    },
    "behavioral_contradiction": {
        "name": "行为矛盾度",
        "prompt": "该嫌疑人的行为/陈述中是否有矛盾？案发前后的行为是否异常？是否有掩盖行为的迹象？",
    },
}

RANK_NORMALIZATION = {
    # 将排名位置映射为归一化分数: rank 1 → 1.0, rank N → lower
    # 使用指数衰减: score = exp(-decay * (rank - 1))
    "decay": 1.2,  # 控制衰减速度
}


@dataclass
class SuspectProfile:
    """嫌疑人证据画像 — 收集所有维度的原始评估"""
    name: str
    raw_scores: Dict[str, float] = field(default_factory=dict)       # dim → raw_score
    rank_scores: Dict[str, float] = field(default_factory=dict)      # dim → rank_score (竞争后)
    evidence_for: Dict[str, List[str]] = field(default_factory=dict)  # dim → evidence list
    evidence_against: Dict[str, List[str]] = field(default_factory=dict)
    key_findings: Dict[str, str] = field(default_factory=dict)       # dim → key finding
    reasonings: Dict[str, str] = field(default_factory=dict)         # dim → reasoning

    # 假设验证结果
    hypothesis_valid: bool = False
    chain_complete: float = 0.0
    unique_evidence: List[str] = field(default_factory=list)
    chain_gaps: List[str] = field(default_factory=list)
    branch_confidence: float = 0.0

    @property
    def competitive_score(self) -> float:
        """综合竞争得分 — 所有维度rank_scores的加权平均"""
        if not self.rank_scores:
            return 0.0
        return sum(self.rank_scores.values()) / len(self.rank_scores)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "competitive_score": round(self.competitive_score, 4),
            "raw_scores": {k: round(v, 3) for k, v in self.raw_scores.items()},
            "rank_scores": {k: round(v, 3) for k, v in self.rank_scores.items()},
            "chain_complete": round(self.chain_complete, 3),
            "hypothesis_valid": self.hypothesis_valid,
            "unique_evidence": self.unique_evidence,
            "chain_gaps": self.chain_gaps,
            "branch_confidence": round(self.branch_confidence, 3),
        }


@dataclass
class DevilAdvocateResult:
    """恶魔代言人挑战结果"""
    should_overturn: bool = False
    overturn_strength: float = 0.0
    winner_weaknesses: List[str] = field(default_factory=list)
    runner_up_strengths: List[str] = field(default_factory=list)
    alternative_theory: str = ""
    final_verdict: str = "维持winner"
    confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> Dict:
        return {
            "should_overturn": self.should_overturn,
            "overturn_strength": round(self.overturn_strength, 3),
            "final_verdict": self.final_verdict,
            "confidence": round(self.confidence, 3),
        }


# ============================================================
# 主搜索类
# ============================================================

class DetectiveSearchTreeV3:
    """
    刑侦推理搜索树 v3 — 竞争假设搜索

    vs v2 的核心区别:
    1. Scatter阶段: 所有嫌疑人在同一维度并行评估 → 避免串行偏差
    2. 竞争归一化: 每个维度内做跨嫌疑人排序 → "抬一个压另一个"
    3. 假设分支: 对Top-2展开深度验证 → 不是简单取最大值
    4. 反向论证: 恶魔代言人机制 → 能推翻错误的Top-1
    """

    def __init__(self, llm_client=None, config: Dict[str, Any] = None):
        self.llm_client = llm_client
        self.config = config or {}
        self.logger = logger.bind(module="DetectiveSearchTreeV3")

        self.max_iterations = self.config.get("max_iterations", 30)

        self.profiles: Dict[str, SuspectProfile] = {}
        self.devil_result: Optional[DevilAdvocateResult] = None
        self.dimension_winners: Dict[str, str] = {}  # dim → winner name
        self.stats = {
            "nodes_created": 0, "nodes_pruned": 0, "llm_calls": 0,
            "scatter_time": 0, "compete_time": 0,
            "branch_time": 0, "devil_time": 0,
        }

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        self.stats["llm_calls"] += 1
        if self.llm_client is None:
            return '{"raw_score": 0.5, "evidence_for": [], "evidence_against": [], "reasoning": "LLM不可用"}'
        try:
            return self.llm_client.simple_chat(prompt, temperature=temperature)
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return '{"raw_score": 0.5, "evidence_for": [], "evidence_against": [], "reasoning": "LLM调用失败"}'

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
        """运行竞争假设搜索"""
        start_time = time.time()

        self.logger.info("=" * 60)
        self.logger.info("🌳 刑侦推理树v3 (竞争假设搜索) 启动")
        self.logger.info(f"嫌疑人: {suspects}, 维度数: {len(EVIDENCE_DIMENSIONS)}")
        self.logger.info("=" * 60)

        # 重置状态
        self.profiles.clear()
        self.devil_result = None
        self.dimension_winners.clear()
        self.stats = {
            "nodes_created": 0, "nodes_pruned": 0, "llm_calls": 0,
            "scatter_time": 0, "compete_time": 0,
            "branch_time": 0, "devil_time": 0,
        }

        # 构建上下文
        case_summary = self._build_case_summary(case_text, structured_knowledge, expert_analyses)
        evidence_context = self._build_evidence_context(structured_knowledge, search_results)

        # 初始化嫌疑人画像
        for s in suspects:
            self.profiles[s] = SuspectProfile(name=s)
            self.stats["nodes_created"] += 1

        # ========================================
        # Phase 1: Scatter — 并行证据收集
        # ========================================
        scatter_start = time.time()
        self.logger.info("📡 Phase 1: Scatter — 并行证据收集")

        for dim_key, dim_info in EVIDENCE_DIMENSIONS.items():
            self.logger.info(f"  🔍 维度: {dim_info['name']}")
            for suspect_name, profile in self.profiles.items():
                prompt = SCATTER_DIMENSION_PROMPT.format(
                    case_summary=case_summary[:1500],
                    suspect_name=suspect_name,
                    dimension_name=dim_info["name"],
                    dimension_prompt=dim_info["prompt"],
                    evidence_context=evidence_context[:600],
                )
                response = self._call_llm(prompt, temperature=0.2)
                result = self._extract_json(response)
                if result is None:
                    result = {"raw_score": 0.5, "evidence_for": [], "evidence_against": [],
                              "reasoning": "无法解析", "key_finding": "无"}

                score = float(result.get("raw_score", 0.5))
                profile.raw_scores[dim_key] = score
                profile.evidence_for[dim_key] = result.get("evidence_for", [])
                profile.evidence_against[dim_key] = result.get("evidence_against", [])
                profile.key_findings[dim_key] = result.get("key_finding", "")
                profile.reasonings[dim_key] = result.get("reasoning", "")[:200]

            # 输出该维度的原始分数
            scores_str = ", ".join([f"{s}={p.raw_scores[dim_key]:.2f}" for s, p in self.profiles.items()])
            self.logger.info(f"    原始分: {scores_str}")

        self.stats["scatter_time"] = time.time() - scatter_start

        # ========================================
        # Phase 2: Compete — 竞争归一化排序
        # ========================================
        compete_start = time.time()
        self.logger.info("⚔️ Phase 2: Compete — 跨嫌疑人竞争排序")

        for dim_key, dim_info in EVIDENCE_DIMENSIONS.items():
            # 构建该维度所有嫌疑人的原始评估摘要
            raw_assessments = []
            for s, p in self.profiles.items():
                raw_assessments.append(
                    f"- {s}: 原始分={p.raw_scores.get(dim_key, 0.5):.2f}, "
                    f"支持证据{len(p.evidence_for.get(dim_key, []))}项, "
                    f"反对证据{len(p.evidence_against.get(dim_key, []))}项, "
                    f"发现: {p.key_findings.get(dim_key, '无')}"
                )

            prompt = COMPETE_RANKING_PROMPT.format(
                case_summary=case_summary[:1200],
                dimension_name=dim_info["name"],
                dimension_prompt=dim_info["prompt"],
                raw_assessments="\n".join(raw_assessments),
                n_suspects=len(self.profiles),
            )
            response = self._call_llm(prompt, temperature=0.2)
            result = self._extract_json(response)

            if result and "ranking" in result:
                for item in result["ranking"]:
                    name = item.get("name", "")
                    # 归一化: 确保rank_score在0-1之间
                    rank_score = max(0.0, min(1.0, float(item.get("rank_score", 0.5))))
                    if name in self.profiles:
                        self.profiles[name].rank_scores[dim_key] = rank_score

                winner = result.get("dimension_winner", "")
                if winner:
                    self.dimension_winners[dim_key] = winner
                self.logger.info(f"    ⚔️ {dim_info['name']} → 赢家: {winner}")
                for item in result["ranking"]:
                    self.logger.info(f"      {item.get('name', '?')}: rank={item.get('rank_score', '?'):.3f}")
            else:
                # Fallback: 用原始分数做排序归一化
                self.logger.warning(f"    ⚠️ 维度{dim_key}竞争排序失败，使用原始分数归一化")
                self._fallback_rank(dim_key)

        # 输出竞争排序后的综合得分
        self.logger.info("  📊 竞争排序综合得分:")
        sorted_profiles = sorted(self.profiles.values(), key=lambda p: -p.competitive_score)
        for i, p in enumerate(sorted_profiles):
            dims_won = sum(1 for w in self.dimension_winners.values() if w == p.name)
            self.logger.info(f"    #{i+1} {p.name}: {p.competitive_score:.3f} (赢{dims_won}个维度)")

        self.stats["compete_time"] = time.time() - compete_start

        # 如果少于2个嫌疑人，跳过Phase 3/4
        if len(sorted_profiles) < 2:
            winner = sorted_profiles[0] if sorted_profiles else None
            return self._build_result(winner, None, None, vote_result, time.time() - start_time, False)

        # ========================================
        # Phase 3: Branch — Top-2 深度假设验证
        # ========================================
        branch_start = time.time()
        top1 = sorted_profiles[0]
        top2 = sorted_profiles[1]
        self.logger.info(f"🌿 Phase 3: Branch — 假设验证: {top1.name}(#{1}) vs {top2.name}(#{2})")

        for rank_pos, profile in enumerate([top1, top2], 1):
            competitor = top2 if profile == top1 else top1
            evidence_profile = self._build_profile_summary(profile)
            competitor_profile = self._build_profile_summary(competitor)

            prompt = BRANCH_HYPOTHESIS_PROMPT.format(
                case_summary=case_summary[:1200],
                suspect_name=profile.name,
                rank_position=rank_pos,
                evidence_profile=evidence_profile,
                competitor_profile=competitor_profile,
            )
            response = self._call_llm(prompt, temperature=0.2)
            result = self._extract_json(response)
            if result is None:
                result = {
                    "hypothesis_valid": True, "chain_complete": 0.5,
                    "unique_evidence": [], "chain_gaps": ["无法解析"],
                    "confidence": 0.5, "reasoning": "解析失败"
                }

            profile.hypothesis_valid = result.get("hypothesis_valid", True)
            profile.chain_complete = float(result.get("chain_complete", 0.5))
            profile.unique_evidence = result.get("unique_evidence", [])
            profile.chain_gaps = result.get("chain_gaps", [])
            profile.branch_confidence = float(result.get("confidence", 0.5))

            valid_str = "✅ 有效" if profile.hypothesis_valid else "❌ 无效"
            self.logger.info(f"  🌿 {profile.name}(#{rank_pos}): {valid_str}, "
                           f"链完整度={profile.chain_complete:.2f}, "
                           f"置信度={profile.branch_confidence:.2f}")
            if profile.chain_gaps:
                self.logger.info(f"    缺口: {'; '.join(profile.chain_gaps[:3])}")
            if profile.unique_evidence:
                self.logger.info(f"    独有证据: {'; '.join(profile.unique_evidence[:3])}")

        self.stats["branch_time"] = time.time() - branch_start

        # ========================================
        # Phase 4: Devil's Advocate — 反向论证
        # ========================================
        devil_start = time.time()
        self.logger.info(f"😈 Phase 4: Devil's Advocate — 挑战 {top1.name}")

        # 如果Top-1的证据链不完整但Top-2完整，交换 — 但v9.1增加保守门槛
        if not top1.hypothesis_valid and top2.hypothesis_valid:
            # 🆕 v9.1: 只有当链完整度差距足够大时才交换（>0.25），否则维持竞争排序
            chain_gap = top2.chain_complete - top1.chain_complete
            if chain_gap > 0.25:
                self.logger.info(f"  🔄 Top-1假设无效且链完整度差距大({chain_gap:.2f})，交换！")
                top1, top2 = top2, top1
                sorted_profiles[0], sorted_profiles[1] = sorted_profiles[1], sorted_profiles[0]
            else:
                self.logger.info(f"  ⚠️ Top-1假设无效但链完整度差距不大({chain_gap:.2f})，维持竞争排序")
                self.logger.info(f"    竞争排序: {top1.name}={top1.competitive_score:.3f} > {top2.name}={top2.competitive_score:.3f}")

        winner_profile = self._build_profile_summary(top1)
        runner_up_profile = self._build_profile_summary(top2)
        dimension_rankings = self._build_dimension_rankings_summary()

        prompt = DEVIL_ADVOCATE_PROMPT.format(
            case_summary=case_summary[:1200],
            winner=top1.name,
            winner_score=top1.competitive_score,
            runner_up=top2.name,
            runner_up_score=top2.competitive_score,
            winner_profile=winner_profile,
            runner_up_profile=runner_up_profile,
            dimension_rankings=dimension_rankings,
        )
        response = self._call_llm(prompt, temperature=0.4)  # 更高温度鼓励创造性反驳
        result = self._extract_json(response)
        if result is None:
            result = {"should_overturn": False, "overturn_strength": 0.0,
                      "final_verdict": "维持winner", "confidence": 0.5, "reasoning": "解析失败"}

        devil = DevilAdvocateResult(
            should_overturn=result.get("should_overturn", False),
            overturn_strength=float(result.get("overturn_strength", 0.0)),
            winner_weaknesses=result.get("winner_weaknesses", []),
            runner_up_strengths=result.get("runner_up_strengths", []),
            alternative_theory=result.get("alternative_theory", ""),
            final_verdict=result.get("final_verdict", "维持winner"),
            confidence=float(result.get("confidence_in_verdict", 0.5)),
            reasoning=result.get("reasoning", ""),
        )
        self.devil_result = devil

        # 🆕 v3.1: 恶魔代言人推翻门槛 — 不能轻易推翻竞争排序#1
        # 推翻条件: should_overturn=True 且 overturn_strength > 0.7
        # 且至少满足以下之一:
        #   a) Top-1证据链完整度 < 0.7
        #   b) Top-1假设被标记为invalid
        #   c) Top-2的证据链完整度 > Top-1 且差距 > 0.2
        actually_overturn = False
        if devil.should_overturn and devil.overturn_strength > 0.7:
            if not top1.hypothesis_valid:
                actually_overturn = True
                self.logger.info(f"  😈 推翻理由: Top-1假设无效")
            elif top1.chain_complete < 0.7:
                actually_overturn = True
                self.logger.info(f"  😈 推翻理由: Top-1证据链完整度低({top1.chain_complete:.2f})")
            elif top2.chain_complete > top1.chain_complete + 0.2:
                actually_overturn = True
                self.logger.info(f"  😈 推翻理由: Top-2链完整度远超Top-1 ({top2.chain_complete:.2f} vs {top1.chain_complete:.2f})")
            else:
                self.logger.info(f"  😈 推翻被拒绝! Top-1证据链健全(完整度={top1.chain_complete:.2f})，不足以推翻")

        if actually_overturn:
            self.logger.info(f"  😈恶魔代言人推翻！{top1.name} → {top2.name}")
            self.logger.info(f"    推翻力度={devil.overturn_strength:.2f}")
            self.logger.info(f"    赢家弱点: {'; '.join(devil.winner_weaknesses[:2])}")
            self.logger.info(f"    第二名强项: {'; '.join(devil.runner_up_strengths[:2])}")
            top1, top2 = top2, top1  # 交换
            sorted_profiles[0], sorted_profiles[1] = sorted_profiles[1], sorted_profiles[0]
        else:
            if devil.should_overturn:
                self.logger.info(f"  😈恶魔代言人尝试推翻但未达门槛，{top1.name} 维持")
            else:
                self.logger.info(f"  😈恶魔代言人确认: {top1.name} 维持")

        self.stats["devil_time"] = time.time() - devil_start

        # ========================================
        # 最终排名
        # ========================================
        elapsed = time.time() - start_time

        # 用最终赢家构建结果
        final_winner = sorted_profiles[0]

        # 计算最终置信度: 竞争得分 * 链完整度 * (1 - 推翻力度)
        confidence = final_winner.competitive_score
        confidence *= (0.5 + 0.5 * final_winner.chain_complete)
        if self.devil_result and self.devil_result.should_overturn:
            confidence *= (1.0 - self.devil_result.overturn_strength * 0.3)

        self.logger.info("=" * 60)
        self.logger.info(f"🌳 推理树v3结论: {final_winner.name} (竞争分={final_winner.competitive_score:.3f}, "
                        f"链完整={final_winner.chain_complete:.2f}, "
                        f"置信度={confidence:.3f})")
        self.logger.info(f"📊 统计: LLM={self.stats['llm_calls']}次, "
                        f"Scatter={self.stats['scatter_time']:.1f}s, "
                        f"Compete={self.stats['compete_time']:.1f}s, "
                        f"Branch={self.stats['branch_time']:.1f}s, "
                        f"Devil={self.stats['devil_time']:.1f}s")
        self.logger.info("=" * 60)

        return self._build_result(final_winner, confidence, sorted_profiles, vote_result, elapsed, actually_overturn)

    # ============================================================
    # 辅助方法
    # ============================================================

    def _fallback_rank(self, dim_key: str):
        """当竞争排序LLM失败时，用原始分数做归一化"""
        scores = [(s, p.raw_scores.get(dim_key, 0.5)) for s, p in self.profiles.items()]
        scores.sort(key=lambda x: -x[1])

        n = len(scores)
        decay = RANK_NORMALIZATION["decay"]
        for rank, (name, raw) in enumerate(scores):
            rank_score = math.exp(-decay * rank)
            self.profiles[name].rank_scores[dim_key] = rank_score

        if scores:
            self.dimension_winners[dim_key] = scores[0][0]

    def _build_profile_summary(self, profile: SuspectProfile) -> str:
        """构建嫌疑人画像摘要"""
        lines = [f"嫌疑人: {profile.name}"]
        lines.append(f"竞争得分: {profile.competitive_score:.3f}")
        lines.append("各维度:")
        for dim_key, dim_info in EVIDENCE_DIMENSIONS.items():
            raw = profile.raw_scores.get(dim_key, "?")
            rank = profile.rank_scores.get(dim_key, "?")
            won = "👑" if self.dimension_winners.get(dim_key) == profile.name else "  "
            lines.append(f"  {won} {dim_info['name']}: 原始={raw}, 竞争={rank}")
            finding = profile.key_findings.get(dim_key, "")
            if finding:
                lines.append(f"     发现: {finding}")
            ev_for = profile.evidence_for.get(dim_key, [])
            if ev_for:
                lines.append(f"     支持: {'; '.join(ev_for[:2])}")
            ev_against = profile.evidence_against.get(dim_key, [])
            if ev_against:
                lines.append(f"     反对: {'; '.join(ev_against[:2])}")

        if profile.chain_complete > 0:
            lines.append(f"证据链完整度: {profile.chain_complete:.2f}")
            if profile.unique_evidence:
                lines.append(f"独有证据: {'; '.join(profile.unique_evidence[:3])}")
            if profile.chain_gaps:
                lines.append(f"链缺口: {'; '.join(profile.chain_gaps[:3])}")

        return "\n".join(lines)

    def _build_dimension_rankings_summary(self) -> str:
        """构建各维度排名摘要"""
        lines = []
        for dim_key, dim_info in EVIDENCE_DIMENSIONS.items():
            winner = self.dimension_winners.get(dim_key, "?")
            sorted_p = sorted(self.profiles.values(), key=lambda p: -p.rank_scores.get(dim_key, 0))
            ranking_str = " > ".join([f"{p.name}({p.rank_scores.get(dim_key, 0):.2f})" for p in sorted_p])
            lines.append(f"{dim_info['name']}: {ranking_str} (赢家: {winner})")
        return "\n".join(lines)

    def _build_case_summary(self, case_text: str, structured_knowledge: Dict,
                             expert_analyses: List[Dict] = None) -> str:
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

        # 🆕 v9.1: 注入专家分析的关键发现（不超过500字，避免token爆炸）
        if expert_analyses and isinstance(expert_analyses, list):
            expert_insights = []
            for analysis in expert_analyses[:10]:
                if not isinstance(analysis, dict):
                    continue
                data = analysis.get("data", analysis)
                perspective = data.get("perspective", "?")
                culprit = data.get("culprit", "未知")
                reasoning = data.get("reasoning", "")
                confidence = data.get("confidence", 0)
                if reasoning:
                    # 只取每个专家推理的核心部分（前80字）
                    expert_insights.append(f"[{perspective}]→{culprit}({confidence:.0%}): {reasoning[:120]}")
            if expert_insights:
                parts.append(f"\n=== 专家分析要点({len(expert_insights)}条) ===")
                parts.append("\n".join(expert_insights[:12]))

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

    def _build_result(
        self,
        winner: Optional[SuspectProfile],
        confidence: Optional[float],
        sorted_profiles: Optional[List[SuspectProfile]],
        vote_result: Dict,
        elapsed: float,
        actually_overturn: bool = False,
    ) -> Dict[str, Any]:
        if winner is None:
            return {
                "track": "ReasoningTreeV3",
                "tree_culprit": vote_result.get("winner", "未知") if vote_result else "未知",
                "tree_confidence": 0.0,
                "is_different_from_vote": False,
                "vote_winner": vote_result.get("winner", "") if vote_result else "",
                "stats": self.stats,
                "timing": round(elapsed, 1),
            }

        if confidence is None:
            confidence = winner.competitive_score

        vote_winner = vote_result.get("winner", "") if vote_result else ""
        is_different = normalize_name(winner.name) != normalize_name(vote_winner)

        # 构建所有候选人得分
        candidate_scores = {}
        if sorted_profiles:
            for p in sorted_profiles:
                candidate_scores[p.name] = p.competitive_score
        else:
            for p in self.profiles.values():
                candidate_scores[p.name] = p.competitive_score

        return {
            "track": "ReasoningTreeV3",
            "tree_culprit": winner.name,
            "tree_confidence": round(confidence, 4),
            "tree_overturned": actually_overturn,
            "is_different_from_vote": is_different,
            "vote_winner": vote_winner,
            "best_path": winner.to_dict() if winner else None,
            "candidate_scores": {k: round(v, 4) for k, v in sorted(
                candidate_scores.items(), key=lambda x: -x[1]
            )},
            "dimension_winners": self.dimension_winners,
            "devil_result": self.devil_result.to_dict() if self.devil_result else None,
            "all_profiles": {name: p.to_dict() for name, p in self.profiles.items()},
            "stats": self.stats,
            "timing": round(elapsed, 1),
        }
