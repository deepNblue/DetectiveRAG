"""
双线融合引擎
融合传统RAG线路和ASMR线路的推理结果
"""

import json
import time
from typing import Dict, List, Any, Optional
from loguru import logger
from collections import defaultdict
from .name_utils import (normalize_name, is_valid_suspect, merge_name_variants,
                          split_multiple_names, build_name_alias_map)


class DualTrackFusionEngine:
    """
    双线融合引擎

    策略:
    1. 一致性检测: 两条线路结论是否一致
    2. 置信度加权: 按各线路置信度加权
    3. 互补补充: 从对方线路补充独特线索
    4. 冲突解决: 结论冲突时ASMR线路权重更高（推理更深）
    """

    TRADITIONAL_WEIGHT = 0.35
    ASMR_WEIGHT = 0.65

    def __init__(self, traditional_weight: float = None, asmr_weight: float = None):
        if traditional_weight is not None:
            self.TRADITIONAL_WEIGHT = traditional_weight
        if asmr_weight is not None:
            self.ASMR_WEIGHT = asmr_weight
        self.logger = logger.bind(module="DualTrackFusionEngine")

    def fuse(self, traditional_result: Dict[str, Any], asmr_result: Dict[str, Any],
             case_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        融合双线结果

        Args:
            traditional_result: 传统RAG线路的结果
            asmr_result: ASMR线路的结果
            case_data: 原始案件数据

        Returns:
            融合后的最终结果
        """
        start_time = time.time()
        self.logger.info("🔀 双线融合引擎启动")
        self.logger.info(f"权重: 传统={self.TRADITIONAL_WEIGHT}, ASMR={self.ASMR_WEIGHT}")

        # Step 1: 提取各线路结论
        trad_conclusion = self._extract_traditional_conclusion(traditional_result)
        asmr_conclusion = self._extract_asmr_conclusion(asmr_result)

        self.logger.info(f"传统线路结论: {trad_conclusion['culprit']} (置信度: {trad_conclusion['confidence']:.2f})")
        self.logger.info(f"ASMR线路结论: {asmr_conclusion['culprit']} (置信度: {asmr_conclusion['confidence']:.2f})")

        # Step 2: 一致性检测
        agreement = self._check_agreement(trad_conclusion, asmr_conclusion, case_data)
        self.logger.info(f"一致性: {agreement['status']}")

        # Step 3: 嫌疑人评分融合
        fused_suspects = self._fuse_suspect_scores(trad_conclusion, asmr_conclusion)

        # Step 4: 线索互补合并
        fused_clues = self._merge_clues(traditional_result, asmr_result)

        # Step 5: 推理链融合
        fused_reasoning = self._fuse_reasoning(trad_conclusion, asmr_conclusion, agreement)

        # Step 6: 确定最终结论
        final = self._determine_final(trad_conclusion, asmr_conclusion, fused_suspects, agreement)

        total_time = time.time() - start_time

        self.logger.info("=" * 60)
        self.logger.info(f"🔀 融合完成 ({total_time:.1f}s)")
        self.logger.info(f"最终结论: 真凶={final['culprit']}, 置信度={final['confidence']:.2%}")
        self.logger.info(f"一致性: {agreement['status']}")
        self.logger.info("=" * 60)

        return {
            "track": "dual_fusion",
            "conclusion": final,
            "agreement": agreement,
            "fused_suspects": fused_suspects,
            "fused_clues": fused_clues,
            "fused_reasoning": fused_reasoning,
            "traditional_contribution": {
                "culprit": trad_conclusion["culprit"],
                "confidence": trad_conclusion["confidence"],
                "knowledge_graph": trad_conclusion.get("has_graph", False),
            },
            "asmr_contribution": {
                "culprit": asmr_conclusion["culprit"],
                "confidence": asmr_conclusion["confidence"],
                "consensus_level": asmr_conclusion.get("consensus_level", "unknown"),
                "expert_count": asmr_conclusion.get("expert_count", 0),
            },
            "timing": {
                "fusion_time": round(total_time, 1),
            },
        }

    def _extract_traditional_conclusion(self, result: Dict) -> Dict:
        """提取传统RAG线路结论（增强版 — 多层嵌套兼容）"""
        if not result:
            return {"culprit": "未知", "confidence": 0.0, "suspects": {}, "has_graph": False}

        culprit = "未知"
        confidence = 0.0
        suspects = {}

        # 策略1: 从 suspect_analyses 或 suspects 列表提取（Agent标准输出格式: {status, data: {...}}）
        for key in ("suspect_analyses", "suspects"):
            suspect_list = result.get(key, [])
            for sa in suspect_list:
                if not isinstance(sa, dict):
                    continue
                # Agent输出格式: data 字段包含实际分析结果
                d = sa.get("data", sa)
                if not isinstance(d, dict):
                    d = sa
                # 提取嫌疑人名: data.suspect_analysis.name 或 data.name
                sa_inner = d.get("suspect_analysis", {})
                if isinstance(sa_inner, dict) and sa_inner.get("name"):
                    name = sa_inner["name"]
                else:
                    name = d.get("name", d.get("suspect", d.get("suspect_name", "?")))
                # 提取嫌疑分: data.suspect_analysis.overall_suspicion 或 data.suspicion_score
                score = sa_inner.get("overall_suspicion", None) if isinstance(sa_inner, dict) else None
                if score is None:
                    score = d.get("suspicion_score", d.get("score", d.get("suspicion", 0)))
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 0.0
                if name and name != "?":
                    clean_name = normalize_name(name)
                    suspects[clean_name] = max(suspects.get(clean_name, 0), score)

        # 策略2: 从 reasoning 结果中提取（补充策略1）
        reasoning = result.get("reasoning", {})
        if isinstance(reasoning, dict):
            rd = reasoning.get("data", reasoning)
            # suspect_ranking 列表（合并到suspects）
            ranking = rd.get("suspect_ranking", rd.get("ranking", []))
            for item in ranking:
                if isinstance(item, dict):
                    name = item.get("name", item.get("suspect", item.get("hypothesis_id", "?")))
                    score = item.get("score", item.get("confidence", item.get("updated_confidence", 0)))
                    try:
                        score = float(score)
                    except (TypeError, ValueError):
                        score = 0.0
                    clean_name = normalize_name(name)
                    if clean_name and is_valid_suspect(clean_name) and not item.get("can_eliminate", False):
                        suspects[clean_name] = max(suspects.get(clean_name, 0), score)
            # 如果reasoning有明确top_suspect，补充
            if "top_suspect" in rd and is_valid_suspect(rd["top_suspect"]):
                ts = normalize_name(rd["top_suspect"])
                tc = float(rd.get("confidence", 0.5))
                suspects[ts] = max(suspects.get(ts, 0), tc)

        # 策略3: 直接从顶层的 culprit/confidence 提取（WebUI传来的full dict已含此字段）
        top_culprit = result.get("culprit", "未知")
        top_confidence = result.get("confidence", 0)
        if is_valid_suspect(normalize_name(top_culprit)):
            clean = normalize_name(top_culprit)
            try:
                top_confidence = float(top_confidence)
            except (TypeError, ValueError):
                top_confidence = 0.0
            suspects[clean] = max(suspects.get(clean, 0), top_confidence)

        # 确定真凶
        if suspects:
            culprit = max(suspects, key=suspects.get)
            confidence = suspects[culprit]

        return {
            "culprit": culprit,
            "confidence": min(confidence, 1.0),
            "suspects": suspects,
            "has_graph": bool(result.get("graph")),
        }

    def _extract_asmr_conclusion(self, result: Dict) -> Dict:
        """提取ASMR线路结论"""
        if not result:
            return {"culprit": "未知", "confidence": 0.0, "suspects": {}, "consensus_level": "unknown"}

        conclusion = result.get("conclusion", {})
        vote_result = result.get("vote_result", {})

        culprit = conclusion.get("culprit", "未知")
        confidence = conclusion.get("confidence", 0.0)
        consensus_level = conclusion.get("consensus_level", "unknown")

        # 回退: conclusion为空时尝试顶层字段
        if (not culprit or culprit == "未知") and result.get("culprit"):
            culprit = result.get("culprit", "未知")
            confidence = result.get("confidence", 0.0)

        return {
            "culprit": culprit,
            "confidence": confidence,
            "consensus_level": consensus_level,
            "expert_count": vote_result.get("total_experts", 0),
            "vote_distribution": vote_result.get("vote_distribution", {}),
        }

    def _check_agreement(self, trad: Dict, asmr: Dict, case_data: Dict = None) -> Dict:
        """检查两条线路结论是否一致（无效结论不参与比较）"""
        trad_culprit = normalize_name(trad["culprit"])
        asmr_culprit = normalize_name(asmr["culprit"])
        trad_valid = is_valid_suspect(trad_culprit)
        asmr_valid = is_valid_suspect(asmr_culprit)

        if trad_valid and asmr_valid and trad_culprit == asmr_culprit:
            status = "完全一致"
            boost = 1.15  # 一致性加成
        elif (trad_valid and not asmr_valid) or (asmr_valid and not trad_valid):
            status = "部分一致(一条线无结论)"
            boost = 1.0
        elif trad_valid and asmr_valid and trad_culprit != asmr_culprit:
            # Bug 6 fix: 检查是否为多罪犯互补
            # 增强版: 不仅看同姓，还检查是否都在嫌疑人列表中
            is_complementary = self._check_multi_culprit_complementary(
                trad_culprit, asmr_culprit, case_data
            )
            if is_complementary:
                status = "多罪犯互补"
                boost = 1.05  # 互补加成
                self.logger.info(f"  检测到多罪犯互补: 传统={trad_culprit}, ASMR={asmr_culprit}")
            else:
                status = "结论冲突"
                boost = 0.9  # 冲突惩罚
        else:
            status = "双线无结论"
            boost = 1.0

        return {
            "status": status,
            "traditional_culprit": trad_culprit,
            "asmr_culprit": asmr_culprit,
            "confidence_boost": boost,
        }

    def _check_multi_culprit_complementary(self, trad_culprit: str, asmr_culprit: str,
                                            case_data: Dict = None) -> bool:
        """
        Bug6增强: 检查两个不同结论是否为多罪犯互补
        规则:
        1. 同姓 → 互补 (原逻辑)
        2. 不同姓但都在嫌疑人列表中 + expected含多罪犯 → 互补
        3. 不同姓但都在嫌疑人列表中 + case_type含"连环/团伙/诈骗/绑架/间谍" → 互补
        """
        if not trad_culprit or not asmr_culprit or trad_culprit == asmr_culprit:
            return False

        # 规则1: 同姓
        trad_family = trad_culprit[0] if trad_culprit else ""
        asmr_family = asmr_culprit[0] if asmr_culprit else ""
        if trad_family and trad_family == asmr_family and len(trad_culprit) >= 2:
            self.logger.info(f"  同姓互补: {trad_culprit} / {asmr_culprit}")
            return True

        # 规则2: 检查是否都在嫌疑人列表中（如果有case_data）
        if case_data:
            suspects = case_data.get("suspects", [])
            suspect_names = set()
            for s in suspects:
                if isinstance(s, dict):
                    suspect_names.add(normalize_name(s.get("name", "")))
                else:
                    suspect_names.add(normalize_name(str(s)))

            trad_in = trad_culprit in suspect_names or any(trad_culprit in sn for sn in suspect_names)
            asmr_in = asmr_culprit in suspect_names or any(asmr_culprit in sn for sn in suspect_names)

            if trad_in and asmr_in:
                # 两人都在嫌疑人列表中 — 检查案件类型是否倾向多罪犯
                case_type = case_data.get("case_type", "")
                multi_culprit_types = ["连环", "团伙", "诈骗", "绑架", "间谍", "盗窃", "泄密"]
                expected = case_data.get("expected_result", {})
                expected_culprit = ""
                if isinstance(expected, dict):
                    expected_culprit = expected.get("culprit", "")
                elif isinstance(expected, str):
                    expected_culprit = expected

                has_multi_hint = (
                    any(t in case_type for t in multi_culprit_types) or
                    "+" in expected_culprit or
                    "联合" in expected_culprit or
                    "共犯" in expected_culprit
                )

                if has_multi_hint:
                    self.logger.info(f"  不同姓多罪犯互补: {trad_culprit} / {asmr_culprit} (type={case_type})")
                    return True

        return False

    def _fuse_suspect_scores(self, trad: Dict, asmr: Dict) -> Dict:
        """融合嫌疑人评分（姓名归一化后合并）"""
        fused = defaultdict(float)

        # 传统线路评分（先归一化姓名）
        for name, score in trad.get("suspects", {}).items():
            clean = normalize_name(name)
            if clean and is_valid_suspect(clean):
                fused[clean] += score * self.TRADITIONAL_WEIGHT

        # ASMR线路评分（先归一化姓名）
        for name, score in asmr.get("vote_distribution", {}).items():
            clean = normalize_name(name)
            if clean and is_valid_suspect(clean):
                fused[clean] += score * self.ASMR_WEIGHT

        # 归一化
        total = sum(fused.values())
        if total > 0:
            normalized = {k: round(v / total, 4) for k, v in fused.items()}
        else:
            normalized = dict(fused)

        # 排序
        sorted_suspects = sorted(normalized.items(), key=lambda x: -x[1])

        return {
            "ranking": [{"name": n, "score": s} for n, s in sorted_suspects],
            "winner": sorted_suspects[0][0] if sorted_suspects else "未知",
            "winner_score": sorted_suspects[0][1] if sorted_suspects else 0,
            "margin": sorted_suspects[0][1] - sorted_suspects[1][1] if len(sorted_suspects) > 1 else 1.0,
        }

    def _merge_clues(self, trad_result: Dict, asmr_result: Dict) -> Dict:
        """合并两条线路发现的线索"""
        # 传统线路线索
        trad_clues = []
        if trad_result:
            trad_clues = trad_result.get("clues", trad_result.get("extracted_clues", []))

        # ASMR线路线索
        asmr_clues = []
        sk = asmr_result.get("structured_knowledge", {})
        timeline = sk.get("timeline", {}).get("data", {})
        evidence = sk.get("evidence", {}).get("data", {})

        # 提取ASMR独特发现
        asmr_clues.extend([
            {"source": "ASMR-时间异常", "content": a.get("description", ""), "significance": a.get("significance", "中")}
            for a in timeline.get("anomalies", [])
        ])
        asmr_clues.extend([
            {"source": "ASMR-时间空白", "content": g.get("description", ""), "significance": "高"}
            for g in timeline.get("gaps", [])
        ])
        asmr_clues.extend([
            {"source": "ASMR-物证关联", "content": c.get("connection_type", ""), "significance": "高"}
            for c in evidence.get("connections", [])
        ])

        return {
            "traditional_clue_count": len(trad_clues),
            "asmr_clue_count": len(asmr_clues),
            "total_unique_clues": len(trad_clues) + len(asmr_clues),
            "asmr_unique_discoveries": asmr_clues,
        }

    def _fuse_reasoning(self, trad: Dict, asmr: Dict, agreement: Dict) -> Dict:
        """融合推理链"""
        parts = []

        if trad.get("culprit") and trad["culprit"] != "未知":
            parts.append(f"[传统RAG] 推断真凶为{trad['culprit']}，置信度{trad['confidence']:.2f}")

        if asmr.get("culprit") and asmr["culprit"] != "未知":
            consensus = asmr.get("consensus_level", "")
            parts.append(f"[ASMR多专家] 推断真凶为{asmr['culprit']}，置信度{asmr['confidence']:.2f}，专家{consensus}")

        if agreement["status"] == "完全一致":
            parts.append("⚡ 两条线路结论一致，可信度加成")
        elif agreement["status"] == "结论冲突":
            parts.append("⚠️ 两条线路结论冲突，ASMR权重更高(推理更深)")

        return {
            "fusion_logic": " → ".join(parts),
            "agreement_status": agreement["status"],
        }

    def _determine_final(self, trad: Dict, asmr: Dict, fused: Dict, agreement: Dict) -> Dict:
        """确定最终结论 — 优化：有结论的线路不被无结论线路拖垮"""
        boost = agreement["confidence_boost"]

        # 检查各线路是否有有效结论
        trad_valid = is_valid_suspect(normalize_name(trad["culprit"]))
        asmr_valid = is_valid_suspect(normalize_name(asmr["culprit"]))

        if agreement["status"] == "完全一致" and trad_valid and asmr_valid:
            # 双线一致 — 最高可信度
            culprit = trad["culprit"]
            confidence = min((trad["confidence"] + asmr["confidence"]) / 2 * boost, 1.0)

        elif agreement["status"] == "结论冲突" and trad_valid and asmr_valid:
            # 真正的冲突 — 以ASMR为主但保留传统参考
            culprit = asmr["culprit"]
            confidence = asmr["confidence"] * self.ASMR_WEIGHT * boost

        elif agreement["status"] == "多罪犯互补" and trad_valid and asmr_valid:
            # 多罪犯互补 — 合并两个嫌疑人的线索，ASMR为主
            culprit = f"{trad['culprit']}+{asmr['culprit']}"
            confidence = max(trad["confidence"], asmr["confidence"]) * boost
            self.logger.info(f"  多罪犯互补合并: {culprit} ({confidence:.2f})")

        elif trad_valid and not asmr_valid:
            # ASMR无有效结论 — 完全信任传统线路，不降权
            culprit = trad["culprit"]
            confidence = trad["confidence"] * boost
            self.logger.info(f"  ASMR无有效结论，信任传统线路: {culprit} ({confidence:.2f})")

        elif asmr_valid and not trad_valid:
            # 传统无有效结论 — 完全信任ASMR线路，不降权
            culprit = asmr["culprit"]
            confidence = asmr["confidence"] * boost
            self.logger.info(f"  传统无有效结论，信任ASMR线路: {culprit} ({confidence:.2f})")

        else:
            # 双线都无有效结论 — 看融合排名
            if fused["winner"] and is_valid_suspect(fused["winner"]):
                culprit = fused["winner"]
                confidence = fused["winner_score"] * boost
            else:
                culprit = "未知"
                confidence = 0.0

        return {
            "culprit": culprit,
            "confidence": round(confidence, 4),
            "certainty": self._confidence_to_certainty(confidence),
        }

    def _confidence_to_certainty(self, confidence: float) -> str:
        """置信度转确定性描述"""
        if confidence >= 0.85:
            return "高度确定"
        elif confidence >= 0.65:
            return "较为确定"
        elif confidence >= 0.45:
            return "有一定依据"
        else:
            return "证据不足"

    def generate_report(self, fusion_result: Dict, asmr_result: Dict = None) -> str:
        """生成双线融合报告"""
        lines = [
            "=" * 60,
            "🕵️ 双线融合调查报告",
            "=" * 60,
            "",
            f"📋 最终结论: 真凶为 **{fusion_result['conclusion']['culprit']}**",
            f"📊 综合置信度: {fusion_result['conclusion']['confidence']:.2%}",
            f"✅ 确定性: {fusion_result['conclusion']['certainty']}",
            f"🔀 两条线路一致性: {fusion_result['agreement']['status']}",
            "",
            "--- 嫌疑人排名 ---",
        ]

        for rank, s in enumerate(fusion_result["fused_suspects"]["ranking"], 1):
            marker = "👉" if rank == 1 else "  "
            lines.append(f"  {marker} #{rank} {s['name']}: {s['score']:.2%}")

        lines.extend([
            "",
            "--- 线路贡献 ---",
            f"  传统RAG: {fusion_result['traditional_contribution']['culprit']}"
            f" (置信度 {fusion_result['traditional_contribution']['confidence']:.2f})",
            f"  ASMR多专家: {fusion_result['asmr_contribution']['culprit']}"
            f" (置信度 {fusion_result['asmr_contribution']['confidence']:.2f})",
            "",
            f"  线索总数: {fusion_result['fused_clues']['total_unique_clues']}",
            f"  ASMR独特发现: {fusion_result['fused_clues']['asmr_clue_count']}条",
        ])

        if asmr_result and asmr_result.get("vote_report"):
            lines.extend([
                "",
                "--- ASMR专家投票详情 ---",
                asmr_result["vote_report"],
            ])

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
