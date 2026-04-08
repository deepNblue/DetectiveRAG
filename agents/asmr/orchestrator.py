"""
ASMR编排器 (Orchestrator) - v12.0 精准增强版
四阶段流水线 + 矛盾搜索 + 裁判Agent + 多变体机制 + 时序侦破引擎 + 推理树验证
借鉴 Supermemory ASMR 的 "并行集群 + 聚合裁判" 思路
v3新增: run_staged() 方法，集成 StageEngine 时序侦破推理
v8新增: Stage 3.5 推理树验证 (Detective Reasoning Tree)，多假设验证+回退机制
v12.0新增: 
  - Stage 3.3: 反向排除验证 (Reverse Elimination Check) — 独立验证"为什么不选其他人"
  - 分歧焦点放大: 调查层≠审判层时，自动标记分歧焦点给裁判
  - 名字去偏校验: 检测常见名字偏好，触发二次确认
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from agents.asmr.readers import TimelineReader, PersonRelationReader, EvidenceReader
from agents.asmr.searchers import (MotiveSearcher, OpportunitySearcher, CapabilitySearcher,
                                    TemporalSearcher, ContradictionSearcher)
from agents.asmr.experts import (ForensicExpert, CriminalExpert, PsychologicalProfiler,
                                  LogicVerifier, Adjudicator,
                                  TechInvestigator, DefenseAttorney,
                                  FinancialInvestigator, InterrogationAnalyst,
                                  ProsecutionReviewer, IntelligenceAnalyst,
                                  SherlockAnalyst, HenryLeeAnalyst, SongCiAnalyst, PoirotAnalyst,
                                  Judge, Juror)
from agents.asmr.voting import ExpertVotingEngine
from agents.asmr.evidence_graph import EvidenceGraphBuilder, ContextExtractor  # 🆕 v13: 图片证据图谱
from agents.asmr.name_utils import normalize_name  # 名字标准化工具
from agents.asmr.stage_engine import StageEngine, InvestigationStage
from agents.asmr.reasoning_tree import DetectiveSearchTree  # 🆕 v8: 推理树
from agents.asmr.reasoning_tree_v2 import DetectiveSearchTreeV2  # 🆕 v8.1: 阶段性推理树
from agents.asmr.reasoning_tree_v3 import DetectiveSearchTreeV3  # 🆕 v8.2: 竞争假设搜索
from agents.memory import MemoryStore, SkillRegistry, PatternLibrary
from agents.memory.skill_learner import SkillLearner
from agents.domain_experts import DomainExpertFactory  # 🆕 v5: 动态领域专家


class ASMROrchestrator:
    """
    ASMR编排器 v2 — 增强版
    
    Stage 1: 3个Reader并行摄取 → 结构化知识
    Stage 2: 5个Searcher并行检索 (含矛盾搜索) → 多维度分析
    Stage 3: 3个Expert并行推理 + LogicVerifier串行验证 → 投票
    Stage 4: 裁判Agent — 综合投票+矛盾分析，最终裁决
    """

    def __init__(self, llm_client=None, config: Dict[str, Any] = None, max_workers: int = 2, event_bus=None, conversation_store=None):
        self.llm_client = llm_client
        self.config = config or {}
        self.max_workers = max_workers
        self.event_bus = event_bus  # 🆕 v10.0: 推理事件总线
        self.conversation_store = conversation_store  # 🆕 v14.0: 对话持久化存储
        self.logger = logger.bind(module="ASMROrchestrator")

        # Stage 1: Readers
        self.timeline_reader = TimelineReader(llm_client=llm_client)
        self.person_relation_reader = PersonRelationReader(llm_client=llm_client)
        self.evidence_reader = EvidenceReader(llm_client=llm_client)

        # Stage 2: Searchers (v2: +ContradictionSearcher)
        self.motive_searcher = MotiveSearcher(llm_client=llm_client)
        self.opportunity_searcher = OpportunitySearcher(llm_client=llm_client)
        self.capability_searcher = CapabilitySearcher(llm_client=llm_client)
        self.temporal_searcher = TemporalSearcher(llm_client=llm_client)
        self.contradiction_searcher = ContradictionSearcher(llm_client=llm_client)

        # Stage 3: Experts (v4: 5分析+1对抗 并行, 1验证串行)
        self.forensic_expert = ForensicExpert(llm_client=llm_client)
        self.criminal_expert = CriminalExpert(llm_client=llm_client)
        self.psychological_profiler = PsychologicalProfiler(llm_client=llm_client)
        self.tech_investigator = TechInvestigator(llm_client=llm_client)  # 🆕 v4
        self.defense_attorney = DefenseAttorney(llm_client=llm_client)    # 🆕 v4
        self.logic_verifier = LogicVerifier(llm_client=llm_client)

        # Stage 3: Phase 2+3 新增Agent (v6)
        self.financial_investigator = FinancialInvestigator(llm_client=llm_client)      # 🆕 Phase 2
        self.interrogation_analyst = InterrogationAnalyst(llm_client=llm_client)        # 🆕 Phase 2
        self.prosecution_reviewer = ProsecutionReviewer(llm_client=llm_client)          # 🆕 Phase 3
        self.intelligence_analyst = IntelligenceAnalyst(llm_client=llm_client)          # 🆕 Phase 3

        # Stage 3: Phase 4 名侦探专家 (v7)
        self.sherlock_analyst = SherlockAnalyst(llm_client=llm_client)                  # 🆕 Phase 4
        self.henry_lee_analyst = HenryLeeAnalyst(llm_client=llm_client)                 # 🆕 Phase 4
        self.song_ci_analyst = SongCiAnalyst(llm_client=llm_client)                     # 🆕 Phase 4
        self.poirot_analyst = PoirotAnalyst(llm_client=llm_client)                      # 🆕 Phase 4

        # Stage 4: Adjudicator (v2新增)
        self.adjudicator = Adjudicator(llm_client=llm_client)

        # 🆕 v9: Stage 3.2 审判层角色
        self.judge = Judge(llm_client=llm_client)                              # 🆕 法官
        self.juror = Juror(llm_client=llm_client)                              # 🆕 陪审员

        # 投票引擎
        self.voting_engine = ExpertVotingEngine()

        # v3新增: 时序侦破引擎
        self.stage_engine = StageEngine(llm_client=llm_client)

        # 🆕 v8: 推理树搜索引擎 (支持v1/v2/v3切换)
        tree_version = self.config.get("tree_version", "v1")
        if tree_version == "v3":
            self.reasoning_tree = DetectiveSearchTreeV3(
                llm_client=llm_client,
                config=self.config.get("reasoning_tree", {}),
            )
            self.logger.info("🌳 推理树版本: v3 (竞争假设搜索)")
        elif tree_version == "v2":
            self.reasoning_tree = DetectiveSearchTreeV2(
                llm_client=llm_client,
                config=self.config.get("reasoning_tree", {}),
            )
            self.logger.info("🌳 推理树版本: v2 (证据阶段性动态调整)")
        else:
            self.reasoning_tree = DetectiveSearchTree(
                llm_client=llm_client,
                config=self.config.get("reasoning_tree", {}),
            )
            self.logger.info("🌳 推理树版本: v1 (原始)")

        # v4新增: 记忆与技能学习系统
        self.memory_store = MemoryStore()
        self.skill_registry = SkillRegistry()
        self.pattern_library = PatternLibrary()
        self.skill_learner = SkillLearner(
            memory_store=self.memory_store,
            skill_registry=self.skill_registry,
            pattern_library=self.pattern_library,
            llm_client=llm_client,
        )

        # 🆕 v5: 动态领域专家工厂
        self.domain_expert_factory = DomainExpertFactory(llm_client=llm_client)

    def _emit(self, event_type: str, data: dict = None, stage: str = "", agent_id: str = ""):
        """向事件总线推送事件（自动构造 ReasoningEvent）"""
        if self.event_bus:
            from ui.reasoning_event_bus import ReasoningEvent, EventType
            # 容错: event_type 可能是字符串或枚举
            try:
                et = EventType(event_type)
            except ValueError:
                et = EventType.PROGRESS
            self.event_bus.push(ReasoningEvent(
                event_type=et,
                stage=stage,
                agent_id=agent_id,
                data=data or {},
            ))

    def _run_agent_safe(self, agent, input_data: Dict, label: str) -> tuple:
        """安全运行单个Agent，返回 (label, result, error)"""
        try:
            self._emit("agent_start", {"agent": label}, agent_id=label)
            
            # 🆕 v14.0: 保存Agent启动事件到对话存储
            if self.conversation_store:
                input_summary = json.dumps(input_data, ensure_ascii=False, default=str)[:800]
                self.conversation_store.save_event("agent_start", label, {
                    "input_summary": input_summary[:400],
                })
            
            result = agent.process(input_data)
            
            # 🆕 v14.0: 保存Agent完成事件到对话存储
            if self.conversation_store and result:
                resp = result.get("data", result) if isinstance(result, dict) else result
                self.conversation_store.save_event("agent_done", label, {
                    "culprit": resp.get("culprit", "?") if isinstance(resp, dict) else "?",
                    "confidence": resp.get("confidence", 0) if isinstance(resp, dict) else 0,
                    "reasoning": str(resp.get("reasoning", ""))[:500] if isinstance(resp, dict) else str(resp)[:500],
                })
                
                # 保存多轮推理详情
                if isinstance(resp, dict):
                    detail = resp.get("detail", {})
                    if isinstance(detail, dict):
                        mr = detail.get("multi_round", {})
                        if isinstance(mr, dict) and mr.get("total_rounds", 0) > 1:
                            self.conversation_store.save_agent_round(label, {
                                "rounds": mr.get("rounds", []),
                                "total_rounds": mr.get("total_rounds", 0),
                                "culprit": resp.get("culprit", "?"),
                                "confidence": resp.get("confidence", 0),
                            })
            
            # 提取Agent结论用于事件推送
            if isinstance(result, dict):
                rd = result.get("data", result)
                
                # 🆕 v14: 提取多轮推理信息并发射round事件
                detail = rd.get("detail", {})
                if isinstance(detail, dict):
                    mr = detail.get("multi_round", {})
                    if isinstance(mr, dict) and mr.get("total_rounds", 0) > 1:
                        rounds = mr.get("rounds", [])
                        for r_info in rounds:
                            self._emit("agent_round_done", {
                                "round": r_info.get("round", 0),
                                "phase": r_info.get("phase", ""),
                                "culprit": r_info.get("culprit", "?"),
                                "confidence": r_info.get("confidence", 0),
                                "changed": r_info.get("changed", r_info.get("culprit_changed", False)),
                            }, agent_id=label)
                
                self._emit("agent_done", {
                    "agent": label,
                    "culprit": rd.get("culprit", "?"),
                    "confidence": rd.get("confidence", 0),
                    "reasoning": str(rd.get("reasoning", ""))[:200],
                    "multi_round": rd.get("detail", {}).get("multi_round", {}) if isinstance(rd.get("detail"), dict) else {},
                }, agent_id=label)
            return (label, result, None)
        except Exception as e:
            self.logger.error(f"Agent {label} 异常: {e}")
            self._emit("agent_done", {"agent": label, "error": str(e)}, agent_id=label)
            if self.conversation_store:
                self.conversation_store.save_event("agent_error", label, {"error": str(e)})
            return (label, None, str(e))

    def _parse_json_safely(self, text: str) -> Optional[Dict]:
        """安全解析LLM返回的JSON"""
        if not text:
            return None
        try:
            # 尝试直接解析
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        try:
            # 尝试提取JSON块
            import re
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    def _run_parallel(self, tasks: List[tuple], desc: str) -> tuple:
        """并行运行多个Agent任务"""
        results = {}
        errors = []
        
        if len(tasks) == 1:
            label, result, error = self._run_agent_safe(tasks[0][0], tasks[0][1], tasks[0][2])
            if error:
                errors.append(f"{label}: {error}")
            else:
                results[label] = result
            return results, errors
        
        self.logger.info(f"  {desc}: 并行启动 {len(tasks)} 个Agent (workers={self.max_workers})")
        
        with ThreadPoolExecutor(max_workers=min(len(tasks), self.max_workers)) as executor:
            futures = {}
            for agent, input_data, label in tasks:
                future = executor.submit(self._run_agent_safe, agent, input_data, label)
                futures[future] = label
            
            for future in as_completed(futures):
                label, result, error = future.result()
                if error:
                    errors.append(f"{label}: {error}")
                    self.logger.warning(f"  ❌ {label} 失败: {error[:80]}")
                else:
                    results[label] = result
                    self.logger.info(f"  ✅ {label} 完成")
        
        return results, errors

    def run(self, case_text: str, suspects: List[Any] = None, case_type: str = "modern", images: List[Dict] = None) -> Dict[str, Any]:
        """
        运行完整ASMR v8 推理树增强流水线
        """
        start_time = time.time()
        suspects = suspects or []
        images = images or []

        self.logger.info("=" * 60)
        self.logger.info("🚀 ASMR v9 三层架构流水线启动")
        self.logger.info(f"案件文本: {len(case_text)}字, 嫌疑人: {len(suspects)}人, 🖼️图片: {len(images)}张, 并行度: {self.max_workers}")
        self.logger.info("=" * 60)
        self._emit("stage_start", {"stage": "pipeline", "text_len": len(case_text), "images": len(images)}, stage="pipeline")

        all_errors = []

        # ==========================================
        # 🆕 v13: Stage 0 图片预分析 (Vision Model → Text Description)
        # ==========================================
        image_context = ""
        image_paths = []
        image_analysis_time = 0
        image_descriptions = []

        if images:
            img_start = time.time()
            self.logger.info(f"🖼️ Stage 0: 视觉模型分析 {len(images)} 张图片")
            self._emit("stage_start", {"stage": "stage_0_vision", "image_count": len(images)}, stage="stage_0_vision")

            for i, img in enumerate(images):
                img_path = img.get("path", "")
                caption = img.get("caption", f"图片{i+1}")
                if not os.path.exists(img_path):
                    self.logger.warning(f"  ⚠️ 图片不存在: {img_path}")
                    continue
                image_paths.append(img_path)

                try:
                    # 用视觉模型分析图片
                    vision_prompt = (
                        f"你是一位刑侦证据分析师。请仔细分析这张案件图片: {caption}\n"
                        "请用中文描述图片中的所有关键细节，特别关注：\n"
                        "1. 任何文字、数字、标识\n"
                        "2. 人物特征或动作\n"
                        "3. 物品、颜色、位置关系\n"
                        "4. 任何异常或不寻常的细节\n"
                        "请简洁准确地描述。"
                    )
                    description = self.llm_client.chat_with_images(
                        prompt=vision_prompt,
                        image_paths=[img_path],
                        temperature=0.3,
                        max_tokens=500,
                        timeout=60,
                    )
                    if description:
                        image_descriptions.append({
                            "index": i + 1,
                            "caption": caption,
                            "path": img_path,
                            "analysis": description.strip(),
                        })
                        self.logger.info(f"  ✅ 图片{i+1}分析完成: {description.strip()[:80]}...")
                    else:
                        self.logger.warning(f"  ⚠️ 图片{i+1}分析返回空")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ 图片{i+1}分析失败: {e}")

            image_analysis_time = time.time() - img_start

            # 构建图片分析上下文
            if image_descriptions:
                image_context = "\n\n【🖼️ 案件图片证据 — 视觉模型分析结果】\n"
                for desc in image_descriptions:
                    image_context += f"\n--- 图片{desc['index']}: {desc['caption']} ---\n"
                    image_context += f"{desc['analysis']}\n"
                image_context += "\n以上是视觉模型对案件图片的分析结果，请作为证据参考。\n"
                self.logger.info(f"🖼️ Stage 0 完成 ({image_analysis_time:.1f}s): "
                                 f"成功分析 {len(image_descriptions)}/{len(images)} 张图片")
            else:
                # fallback: 只写 caption
                image_context = "\n\n【🖼️ 案件图片资料】\n"
                for i, img in enumerate(images):
                    image_context += f"  - 图片{i+1}: {img.get('caption', f'图片{i+1}')}\n"
                image_context += "\n（视觉模型分析失败，仅展示图片标题）\n"

            self._emit("stage_done", {
                "stage": "stage_0_vision",
                "timing": round(image_analysis_time, 1),
                "analyzed": len(image_descriptions),
                "total": len(images),
            }, stage="stage_0_vision")

        enhanced_text = case_text + image_context

        # ==========================================
        # 🆕 v15.2: Stage 0.5 证据图谱构建 (文本+图片)
        # ==========================================
        evidence_graph_data = {}
        try:
            graph_builder = EvidenceGraphBuilder(
                llm_client=self.llm_client,
                context_extractor=ContextExtractor(),
            )

            if image_descriptions:
                # 有图片：构建完整图谱 (图片实体 + 跨模态关联)
                evidence_graph_data = graph_builder.build_from_image_analyses(
                    image_descriptions=image_descriptions,
                    suspects=suspects,
                    case_text=case_text,
                )
                self.logger.info(f"🕸️ Stage 0.5: 图片图谱构建完成 — "
                                 f"节点={evidence_graph_data.get('stats', {}).get('total_nodes', 0)}, "
                                 f"边={evidence_graph_data.get('stats', {}).get('total_edges', 0)}")
            else:
                # 无图片：从案件文本构建图谱 (v15.2 新增)
                evidence_graph_data = graph_builder.build_from_text(
                    case_text=enhanced_text or case_text,
                    suspects=suspects,
                )
                self.logger.info(f"🕸️ Stage 0.5: 文本图谱构建完成 — "
                                 f"节点={evidence_graph_data.get('stats', {}).get('total_nodes', 0)}, "
                                 f"边={evidence_graph_data.get('stats', {}).get('total_edges', 0)}, "
                                 f"文本关系={evidence_graph_data.get('stats', {}).get('text_relations', 0)}")

            # 将图谱文本注入 enhanced_text
            graph_text = evidence_graph_data.get("graph_text", "")
            if graph_text:
                enhanced_text += graph_text

            graph_stats = evidence_graph_data.get("stats", {})
            cross_links = evidence_graph_data.get("cross_modal_links", [])
            contradictions = evidence_graph_data.get("contradiction_hints", [])
            retrieval_chunks = evidence_graph_data.get("retrieval_chunks", [])
            self.logger.info(f"🕸️ Stage 0.5 汇总: "
                             f"节点={graph_stats.get('total_nodes', 0)}, "
                             f"边={graph_stats.get('total_edges', 0)}, "
                             f"跨模态关联={len(cross_links)}, "
                             f"检索chunks={len(retrieval_chunks)}")
            if cross_links:
                for link in cross_links[:5]:
                    self.logger.info(f"  🔗 {link['image_entity']} ↔ {link['suspect_name']} "
                                     f"({link['link_type']}, {link['confidence']:.0%})")
            if contradictions:
                for c in contradictions[:3]:
                    self.logger.warning(f"  ⚠️ 图谱矛盾: {c['type']} — {c.get('entity', '?')}")
        except Exception as e:
            self.logger.warning(f"  ⚠️ 证据图谱构建失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

        # ==========================================
        # Stage 1: 3个Reader并行摄取
        # ==========================================
        self.logger.info("📚 Stage 1: 3个Reader并行摄取")
        self._emit("stage_start", {"stage": "stage_1_readers"}, stage="stage_1_readers")
        stage1_start = time.time()

        reader_input = {
            "case_text": enhanced_text,
            "suspects": suspects,
            "case_type": case_type,
            "image_paths": image_paths,
        }

        reader_results, reader_errors = self._run_parallel([
            (self.timeline_reader, reader_input, "timeline"),
            (self.person_relation_reader, reader_input, "person_relation"),
            (self.evidence_reader, reader_input, "evidence"),
        ], desc="Readers")
        all_errors.extend(reader_errors)

        structured_knowledge = reader_results
        stage1_time = time.time() - stage1_start

        t_events = structured_knowledge.get("timeline", {}).get("data", {}).get("event_count", "?")
        p_count = structured_knowledge.get("person_relation", {}).get("data", {}).get("person_count", "?")
        e_count = structured_knowledge.get("evidence", {}).get("data", {}).get("physical_count", "?")
        self.logger.info(f"📚 Stage 1 完成 ({stage1_time:.1f}s): "
                         f"时间线={t_events}事件, 人物={p_count}人, 物证={e_count}件")
        self._emit("stage_done", {"stage": "stage_1_readers", "timing": round(stage1_time, 1)}, stage="stage_1_readers")

        # ==========================================
        # Stage 2: 5个Searcher并行检索 (v2: +矛盾搜索)
        # ==========================================
        self.logger.info("🔍 Stage 2: 5个Searcher并行检索 (含矛盾搜索)")
        self._emit("stage_start", {"stage": "stage_2_searchers"}, stage="stage_2_searchers")
        stage2_start = time.time()

        searcher_input = {
            "structured_knowledge": structured_knowledge,
            "suspects": suspects,
            "case_text": enhanced_text,
        }

        searcher_results, searcher_errors = self._run_parallel([
            (self.motive_searcher, searcher_input, "motive"),
            (self.opportunity_searcher, searcher_input, "opportunity"),
            (self.capability_searcher, searcher_input, "capability"),
            (self.temporal_searcher, searcher_input, "temporal"),
            (self.contradiction_searcher, searcher_input, "contradiction"),
        ], desc="Searchers")
        all_errors.extend(searcher_errors)

        search_results = searcher_results
        stage2_time = time.time() - stage2_start

        # 提取矛盾搜索结果
        contradiction_data = {}
        if "contradiction" in search_results:
            cd = search_results["contradiction"].get("data", search_results["contradiction"])
            contradiction_data = cd if isinstance(cd, dict) else {}
            c_count = len(contradiction_data.get("contradictions", []))
            self.logger.info(f"🔍 Stage 2 完成 ({stage2_time:.1f}s): 发现 {c_count} 个矛盾/异常")
        else:
            self.logger.info(f"🔍 Stage 2 完成 ({stage2_time:.1f}s): 矛盾搜索未完成")
        self._emit("stage_done", {"stage": "stage_2_searchers", "timing": round(stage2_time, 1), "contradictions": c_count if 'c_count' in dir() else 0}, stage="stage_2_searchers")

        # ==========================================
        # Stage 3.1: 调查层 — 9分析专家 + 4名侦探 (搜集证据、构建推理链)
        # ==========================================
        self.logger.info("🔬 Stage 3.1: 调查层 — 9分析专家 + 4名侦探并行分析")
        stage31_start = time.time()

        expert_input = {
            "structured_knowledge": structured_knowledge,
            "search_results": search_results,
            "suspects": suspects,
        }

        # 调查层: 9分析专家 + 4名侦探 (不含检察官/辩护律师 — 他们是审判层)
        investigation_results_raw, expert_errors = self._run_parallel([
            (self.forensic_expert, expert_input, "forensic"),
            (self.criminal_expert, expert_input, "criminal"),
            (self.psychological_profiler, expert_input, "profiler"),
            (self.tech_investigator, expert_input, "tech"),
            (self.financial_investigator, expert_input, "financial"),
            (self.interrogation_analyst, expert_input, "interrogation"),
            (self.intelligence_analyst, expert_input, "intelligence"),
            (self.sherlock_analyst, expert_input, "sherlock"),
            (self.henry_lee_analyst, expert_input, "henry_lee"),
            (self.song_ci_analyst, expert_input, "song_ci"),
            (self.poirot_analyst, expert_input, "poirot"),
        ], desc="Investigation")
        all_errors.extend(expert_errors)

        investigation_results = []
        for label in ["forensic", "criminal", "profiler", "tech",
                       "financial", "interrogation", "intelligence",
                       "sherlock", "henry_lee", "song_ci", "poirot"]:
            if label in investigation_results_raw:
                investigation_results.append(investigation_results_raw[label])

        # LogicVerifier串行验证（依赖调查层结果）
        logic_input = {
            "expert_results": investigation_results,
            "structured_knowledge": structured_knowledge,
            "search_results": search_results,
        }
        logic_label, logic_result, logic_error = self._run_agent_safe(
            self.logic_verifier, logic_input, "logic_verifier"
        )
        if logic_error:
            all_errors.append(f"logic_verifier: {logic_error}")
            self.logger.warning(f"  ⚠️ LogicVerifier异常，跳过: {logic_error[:100]}")
        else:
            investigation_results.append(logic_result)

        # 🆕 v5: 动态领域专家（调查层补充）
        domain_expert_results = []
        try:
            matched_domains = self.domain_expert_factory.analyze_case(case_text, case_type)
            if matched_domains:
                domain_experts = self.domain_expert_factory.create_experts(matched_domains)
                domain_expert_input = {
                    "structured_knowledge": structured_knowledge,
                    "search_results": search_results,
                    "suspects": suspects,
                    "case_text": enhanced_text,
                }
                domain_tasks = [
                    (expert, domain_expert_input, f"domain_{expert.template.expert_id}")
                    for expert in domain_experts
                ]
                domain_results, domain_errors = self._run_parallel(domain_tasks, desc="DomainExperts")
                all_errors.extend(domain_errors)
                
                for label, result in domain_results.items():
                    domain_expert_results.append(result)
                    investigation_results.append(result)
                    self.logger.info(f"  🔬 领域专家 {label} 完成")
                
                for m in matched_domains:
                    t = m["template"]
                    self.voting_engine.set_dynamic_weight(
                        f"domain_{t.expert_id}", t.voting_weight
                    )
            else:
                self.logger.info("  📋 未匹配到领域专家，使用核心专家阵容")
        except Exception as e:
            self.logger.warning(f"  ⚠️ 动态领域专家异常: {e}")
            all_errors.append(f"domain_experts: {e}")

        stage31_time = time.time() - stage31_start

        # 🆕 v12.1: 提前构建嫌疑人全名列表，传给投票引擎做名字补全
        suspect_name_list = [s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in suspects]

        # 调查层投票（内部共识参考，非最终投票）
        investigation_vote = self.voting_engine.vote(investigation_results, suspect_names=suspect_name_list)
        self.logger.info(f"🔬 Stage 3.1 完成 ({stage31_time:.1f}s): "
                         f"调查层共识={investigation_vote.get('winner', '?')}")
        self._emit("stage_done", {"stage": "stage_31_investigation", "timing": round(stage31_time, 1), "winner": investigation_vote.get("winner", "?")}, stage="stage_31_investigation")

        # ==========================================
        # Stage 3.2: 审判层 — 检察官 + 辩护律师 + 法官 + 陪审员 (控辩对抗投票)
        # ==========================================
        self.logger.info("⚖️ Stage 3.2: 审判层 — 控辩对抗 + 中立裁判")
        stage32_start = time.time()

        trial_input = {
            "structured_knowledge": structured_knowledge,
            "search_results": search_results,
            "suspects": suspects,
            "investigation_results": investigation_results,  # 🆕 调查层结论作为输入
        }

        # 审判层: 4角色并行 (检察官=控方, 辩护律师=辩方, 法官=中立, 陪审员=常识)
        trial_results_raw, trial_errors = self._run_parallel([
            (self.prosecution_reviewer, trial_input, "prosecution"),    # 🔴 检察官
            (self.defense_attorney, trial_input, "defense"),            # 🔵 辩护律师
            (self.judge, trial_input, "judge"),                         # 🧑‍⚖️ 法官
            (self.juror, trial_input, "juror"),                         # 👥 陪审员
        ], desc="Trial")
        all_errors.extend(trial_errors)

        trial_results = []
        for label in ["prosecution", "defense", "judge", "juror"]:
            if label in trial_results_raw:
                trial_results.append(trial_results_raw[label])

        stage32_time = time.time() - stage32_start

        # 审判层投票（使用审判层专用权重）
        trial_vote = self.voting_engine.vote(trial_results, suspect_names=suspect_name_list)
        self.logger.info(f"⚖️ Stage 3.2 完成 ({stage32_time:.1f}s): "
                         f"审判团投票={trial_vote.get('winner', '?')}")
        self._emit("stage_done", {"stage": "stage_32_trial", "timing": round(stage32_time, 1), "winner": trial_vote.get("winner", "?")}, stage="stage_32_trial")

        # 合并所有结果用于统计
        all_expert_results = investigation_results + trial_results
        stage3_time = stage31_time + stage32_time

        # 最终投票: 调查层 + 审判层联合投票（审判层权重更高）
        vote_result = self.voting_engine.vote(all_expert_results, suspect_names=suspect_name_list)
        vote_report = self.voting_engine.get_report(vote_result)

        # 🆕 v10.0: 向事件总线推送投票结果
        self._emit("vote_cast", {
            "expert": "combined_vote",
            "culprit": vote_result.get("winner", "?"),
            "confidence": vote_result.get("confidence", 0),
            "weight": vote_result.get("confidence", 1.0),
            "distribution": vote_result.get("vote_distribution", {}),
        }, stage="vote")

        self.logger.info(f"🧠 Stage 3 总完成 ({stage3_time:.1f}s): "
                         f"调查层={investigation_vote.get('winner', '?')}, "
                         f"审判层={trial_vote.get('winner', '?')}, "
                         f"联合投票={vote_result.get('winner', '?')}")

        # ==========================================
        # 🆕 v12.0 Stage 3.3: 反向排除验证 (Reverse Elimination Check)
        # ==========================================
        # 独立LLM调用，强制分析"为什么不选其他人"，检测盲点
        elimination_result = None
        stage33_time = 0
        try:
            self.logger.info("🔄 Stage 3.3: 反向排除验证 — 检查'为什么不选其他人'")
            stage33_start = time.time()

            vote_winner = vote_result.get("winner", "未知")
            inv_winner = investigation_vote.get("winner", "未知")
            trial_winner = trial_vote.get("winner", "未知")
            suspect_names = suspect_name_list  # 🆕 v12.1: 使用提前构建的列表
            
            # 其他嫌疑人 = 全部嫌疑人 - 投票赢家
            other_suspects = [s for s in suspect_names if s != vote_winner]
            
            if other_suspects and vote_winner != "无法确定":
                elimination_prompt = f"""你是一个独立的证据审查员。当前调查团一致认为"{vote_winner}"是真凶。
但你需要扮演魔鬼代言人，严格审查每一个被排除的嫌疑人。

案件嫌疑人: {', '.join(suspect_names)}
调查层指向: {inv_winner}
审判层指向: {trial_winner}  
联合投票赢家: {vote_winner}

被排除的嫌疑人: {', '.join(other_suspects)}

对于每一个被排除的嫌疑人，请回答:
1. 是否有直接物证证明他/她不是凶手？（不是"看起来不像"，而是"有不在场证明"或"物证排除"）
2. 是否有可能他/她才是真凶，但被专家们忽略了？

请以JSON格式返回:
{{
    "elimination_check": [
        {{
            "suspect": "嫌疑人名",
            "has_alibi": true/false,
            "alibi_detail": "不在场证明详情",
            "physical_exclusion": true/false,
            "physical_exclusion_detail": "物证排除详情", 
            "could_be_real_culprit": true/false,
            "reason": "如果可能是真凶，说明为什么专家可能忽略了此人",
            "confidence_in_exclusion": 0.0-1.0
        }}
    ],
    "vote_winner_weakness": "投票赢家的证据链中最薄弱的一环",
    "blind_spot_found": true/false,
    "blind_spot_detail": "发现的盲点详情（如果有）",
    "recommendation": "维持原判/需要重新审视"
}}"""

                elimination_response = self.llm_client.chat_completion(
                    messages=[{"role": "user", "content": elimination_prompt}],
                    temperature=0.3,
                )
                if isinstance(elimination_response, str):
                    elimination_result = self._parse_json_safely(elimination_response)
                else:
                    elimination_result = None
                
                if elimination_result:
                    blind_spot = elimination_result.get("blind_spot_found", False)
                    recommendation = elimination_result.get("recommendation", "维持原判")
                    self.logger.info(f"  🔄 反向排除完成: blind_spot={blind_spot}, recommendation={recommendation}")
                    if blind_spot:
                        self.logger.warning(f"  ⚠️ 发现盲点! {elimination_result.get('blind_spot_detail', '')[:100]}")
                else:
                    self.logger.info("  🔄 反向排除完成: JSON解析失败，跳过")
            else:
                self.logger.info("  🔄 反向排除跳过: 无足够嫌疑人或投票结果不确定")

            stage33_time = time.time() - stage33_start
        except Exception as e:
            self.logger.warning(f"  ⚠️ 反向排除验证异常: {e}")
            stage33_time = time.time() - stage33_start if 'stage33_start' in dir() else 0

        # 🆕 v12.0: 分歧焦点分析
        divergence_analysis = None
        if inv_winner != trial_winner:
            self.logger.info(f"  ⚡ 检测到分歧: 调查层={inv_winner} vs 审判层={trial_winner}")
            divergence_analysis = {
                "has_divergence": True,
                "investigation_winner": inv_winner,
                "trial_winner": trial_winner,
                "divergence_type": "investigation_disagrees" if inv_winner != vote_winner else "trial_disagrees",
            }

        # ==========================================
        # Stage 3.5: 🆕 v8 推理树验证 (多假设验证 + 回退)
        # ==========================================
        self.logger.info("🌳 Stage 3.5: 推理树验证 (多假设 + 证据链 + 回退)")
        stage35_start = time.time()

        # 提取嫌疑人列表
        # 🆕 v12.1: 优先使用原始嫌疑人全名列表，而非投票分布中的（可能被截断的）名字
        vote_distribution = vote_result.get("vote_distribution", {})
        
        # 从投票分布提取名字，但用原始嫌疑人列表做补全
        tree_suspects = list(vote_distribution.keys())
        if suspect_name_list:
            # 用原始嫌疑人列表替换投票分布中截断的名字
            completed_suspects = []
            for ts in tree_suspects:
                matched = False
                for sn in suspect_name_list:
                    if normalize_name(ts) == normalize_name(sn) or normalize_name(ts) in normalize_name(sn):
                        completed_suspects.append(sn)
                        matched = True
                        break
                if not matched:
                    completed_suspects.append(ts)
            tree_suspects = completed_suspects

        # 如果投票没有嫌疑人(全投"无法确定")，直接用原始嫌疑人列表
        if not tree_suspects:
            tree_suspects = list(suspect_name_list)

        # 构建搜索结果汇总
        search_results_summary = {
            "motive_ranking": search_results.get("motive", {}).get("data", {}).get("ranking", []),
            "opportunity_ranking": search_results.get("opportunity", {}).get("data", {}).get("ranking", []),
            "capability_ranking": search_results.get("capability", {}).get("data", {}).get("ranking", []),
            "temporal_insight": search_results.get("temporal", {}).get("data", {}).get("key_insight", ""),
            "contradiction_ranking": contradiction_data.get("ranking", []),
            "contradiction_insight": contradiction_data.get("key_insight", ""),
        }

        try:
            tree_result = self.reasoning_tree.search(
                case_text=case_text,
                suspects=tree_suspects,
                structured_knowledge=structured_knowledge,
                search_results=search_results_summary,
                expert_analyses=[
                    {
                        "data": {
                            "perspective": r.get("data", r).get("perspective", "?"),
                            "culprit": r.get("data", r).get("culprit", "?"),
                            "confidence": r.get("data", r).get("confidence", 0),
                            "reasoning": r.get("data", r).get("reasoning", "")[:300],
                        }
                    }
                    for r in all_expert_results
                ],
                vote_result=vote_result,
            )

            stage35_time = time.time() - stage35_start
            tree_culprit = tree_result.get("tree_culprit", "?")
            tree_confidence = tree_result.get("tree_confidence", 0)
            is_different = tree_result.get("is_different_from_vote", False)

            self.logger.info(f"🌳 Stage 3.5 完成 ({stage35_time:.1f}s): "
                             f"推理树结论={tree_culprit} ({tree_confidence:.3f})")
            if is_different:
                self.logger.info(f"  ⚡ 推理树结论与投票不同! 投票={vote_result.get('winner', '?')} → 推理树={tree_culprit}")

            # 记录推理树统计
            tree_stats = tree_result.get("stats", {})
            self.logger.info(f"  📊 推理树统计: 节点={tree_stats.get('nodes_created', 0)}, "
                             f"剪枝={tree_stats.get('nodes_pruned', 0)}, "
                             f"回退={tree_stats.get('backtracks', 0)}, "
                             f"LLM调用={tree_stats.get('llm_calls', 0)}次")

        except Exception as e:
            self.logger.warning(f"  ⚠️ 推理树异常，跳过: {e}")
            tree_result = {"track": "ReasoningTree", "tree_culprit": "?", "tree_confidence": 0,
                           "is_different_from_vote": False, "error": str(e)}
            stage35_time = time.time() - stage35_start

        # ==========================================
        # Stage 4: 裁判Agent (v2新增)
        # ==========================================
        self.logger.info("⚖️ Stage 4: 裁判Agent最终裁决")
        stage4_start = time.time()

        adjudicator_input = {
            "expert_results": all_expert_results,
            "vote_result": vote_result,
            "contradiction_data": contradiction_data,
            "structured_knowledge": structured_knowledge,
            "suspects": suspects,
            "tree_result": tree_result,  # 🆕 v8: 推理树结果
            "investigation_vote": investigation_vote,    # 🆕 v9: 调查层投票
            "trial_vote": trial_vote,                    # 🆕 v9: 审判层投票
            "elimination_result": elimination_result,    # 🆕 v12.0: 反向排除结果
            "divergence_analysis": divergence_analysis,   # 🆕 v12.0: 分歧焦点
        }

        _, adjudicator_result, adj_error = self._run_agent_safe(
            self.adjudicator, adjudicator_input, "adjudicator"
        )

        if adj_error or adjudicator_result is None:
            all_errors.append(f"adjudicator: {adj_error}")
            self.logger.warning(f"  ⚠️ 裁判Agent异常")
            # 🆕 v8: 裁判异常时，优先使用推理树结论（如果有）
            tree_is_valid = tree_result.get("tree_confidence", 0) > 0.2
            if tree_is_valid and tree_result.get("is_different_from_vote"):
                final_conclusion = {
                    "culprit": tree_result["tree_culprit"],
                    "confidence": tree_result["tree_confidence"],
                    "consensus_level": "推理树验证",
                    "source": "reasoning_tree_fallback",
                    "overturned": True,
                    "vote_winner": vote_result["winner"],
                    "reasoning": f"推理树通过多维度证据链验证，发现{vote_result['winner']}存在矛盾，"
                                 f"转而指向{tree_result['tree_culprit']}",
                }
                self.logger.info(f"  🌳 推理树接管(裁判异常): {vote_result['winner']} → {tree_result['tree_culprit']}")
            else:
                final_conclusion = {
                    "culprit": vote_result["winner"],
                    "confidence": vote_result["confidence"],
                    "consensus_level": vote_result["consensus_level"],
                    "source": "vote_fallback",
                }
        else:
            adj_data = adjudicator_result.get("data", adjudicator_result)
            detail = adj_data.get("detail", {})
            overturned = detail.get("overturned", False)
            is_multi_culprit = adj_data.get("is_multi_culprit", False)
            culprit_roles = adj_data.get("culprit_roles", {})

            # 🆕 v9.2: 推翻保护机制 (Overturn Safeguard)
            # 当调查层、审判层、联合投票三层一致时，裁判不能轻易推翻
            adj_culprit = adj_data.get("culprit", vote_result["winner"])
            vote_winner = vote_result["winner"]
            inv_winner = investigation_vote.get("winner", "?")
            trial_winner = trial_vote.get("winner", "?")

            # 三层一致检查
            three_layer_unanimous = (
                inv_winner == trial_winner == vote_winner
                and adj_culprit != vote_winner
            )
            # 高置信度投票检查 (>80%)
            high_confidence_vote = vote_result.get("confidence", 0) >= 0.8
            # 投票一致性: 第一名和第二名差距
            vote_dist = vote_result.get("vote_distribution", {})
            sorted_votes = sorted(vote_dist.values(), reverse=True) if vote_dist else []
            vote_gap = (sorted_votes[0] - sorted_votes[1]) / 100.0 if len(sorted_votes) >= 2 else 1.0

            should_reject_overturn = False
            reject_reason = ""

            # 🆕 v12.1: 反向排除盲点发现时，降低推翻保护门槛
            # 如果反向排除验证发现了盲点，裁判有权推翻（即使三层一致）
            blind_spot_detected = (
                elimination_result 
                and elimination_result.get("blind_spot_found", False) 
                and elimination_result.get("recommendation", "") != "维持原判"
            )

            if blind_spot_detected:
                # 盲点发现时，只有极强共识才保护（>85%且三层一致且大差距）
                if three_layer_unanimous and high_confidence_vote and vote_gap > 0.4:
                    should_reject_overturn = True
                    reject_reason = f"极强共识+盲点不足以推翻({inv_winner}/{trial_winner}/{vote_winner}, {vote_result['confidence']:.0%}, gap={vote_gap:.1%})"
                # 否则允许裁判推翻
            elif three_layer_unanimous and high_confidence_vote:
                should_reject_overturn = True
                reject_reason = f"三层一致({inv_winner}/{trial_winner}/{vote_winner}) + 高置信度({vote_result['confidence']:.0%})"
            elif three_layer_unanimous and vote_gap > 0.3:
                should_reject_overturn = True
                reject_reason = f"三层一致 + 投票大差距({vote_gap:.1%})"

            if should_reject_overturn and overturned:
                self.logger.warning(f"  🛡️ v9.2 推翻保护激活! 拒绝裁判推翻: {adj_culprit} → 恢复 {vote_winner}")
                self.logger.warning(f"  🛡️ 保护原因: {reject_reason}")
                # 恢复投票结果
                final_conclusion = {
                    "culprit": vote_winner,
                    "confidence": vote_result["confidence"],
                    "consensus_level": vote_result["consensus_level"],
                    "source": "adjudicator_overruled_safeguard",
                    "overturned": False,
                    "vote_winner": vote_winner,
                    "reasoning": f"[推翻保护] {reject_reason}。裁判原裁决={adj_culprit}被拒绝。",
                    "overlooked_evidence": detail.get("overlooked_evidence", []),
                    "warning": detail.get("warning", ""),
                    "is_multi_culprit": False,
                    "culprit_roles": {},
                }
            else:
                final_conclusion = {
                    "culprit": adj_culprit,
                    "confidence": adj_data.get("confidence", vote_result["confidence"]),
                    "consensus_level": detail.get("verdict_type", vote_result["consensus_level"]),
                    "source": "adjudicator",
                    "overturned": overturned,
                    "vote_winner": vote_winner,
                    "reasoning": adj_data.get("reasoning", ""),
                    "overlooked_evidence": detail.get("overlooked_evidence", []),
                    "warning": detail.get("warning", ""),
                    "is_multi_culprit": is_multi_culprit,
                    "culprit_roles": culprit_roles,
                }
                if overturned:
                    self.logger.info(f"  ⚡ 裁判推翻投票! {vote_winner} → {final_conclusion['culprit']}")
                if is_multi_culprit:
                    self.logger.info(f"  🔗 裁判检测到多罪犯合谋: {final_conclusion['culprit']}")

        stage4_time = time.time() - stage4_start
        self.logger.info(f"⚖️ Stage 4 完成 ({stage4_time:.1f}s)")
        self._emit("stage_done", {"stage": "stage_4_adjudicator", "timing": round(stage4_time, 1)}, stage="stage_4_adjudicator")
        self._emit("conclusion", {
            "culprit": final_conclusion.get("culprit", "?"),
            "confidence": final_conclusion.get("confidence", 0),
            "overturned": final_conclusion.get("overturned", False),
        }, stage="conclusion")

        total_time = time.time() - start_time

        self.logger.info("=" * 60)
        self.logger.info(f"🚀 ASMR v9 三层架构流水线完成! 总耗时: {total_time:.1f}s")
        self.logger.info(f"调查层共识: {investigation_vote['winner']} ({investigation_vote['confidence']:.2%})")
        self.logger.info(f"审判层投票: {trial_vote['winner']} ({trial_vote['confidence']:.2%})")
        self.logger.info(f"联合投票: {vote_result['winner']} ({vote_result['confidence']:.2%})")
        if tree_result.get("is_different_from_vote"):
            self.logger.info(f"🌳 推理树: {tree_result.get('tree_culprit', '?')} "
                             f"({tree_result.get('tree_confidence', 0):.3f}) "
                             f"[与投票不同!]")
        self.logger.info(f"最终结论: {final_conclusion['culprit']} ({final_conclusion['confidence']:.2%})")
        if final_conclusion.get("overturned"):
            self.logger.info(f"⚡ 结论被推翻! 原投票={final_conclusion.get('vote_winner', '?')}")
        self.logger.info("=" * 60)

        return {
            "track": "ASMR",
            "conclusion": final_conclusion,
            "vote_result": vote_result,
            "vote_report": vote_report,
            "adjudicator_result": adjudicator_result,
            "contradiction_data": contradiction_data,
            "structured_knowledge": structured_knowledge,
            "search_results": {
                "motive_ranking": search_results.get("motive", {}).get("data", {}).get("ranking", []),
                "opportunity_ranking": search_results.get("opportunity", {}).get("data", {}).get("ranking", []),
                "capability_ranking": search_results.get("capability", {}).get("data", {}).get("ranking", []),
                "temporal_insight": search_results.get("temporal", {}).get("data", {}).get("key_insight", ""),
                "contradiction_ranking": contradiction_data.get("ranking", []),
                "contradiction_insight": contradiction_data.get("key_insight", ""),
            },
            "expert_analyses": [
                {
                    "perspective": r.get("data", r).get("perspective", "?"),
                    "culprit": r.get("data", r).get("culprit", "?"),
                    "confidence": r.get("data", r).get("confidence", 0),
                    "layer": "investigation" if i < len(investigation_results) else "trial",
                }
                for i, r in enumerate(all_expert_results)
            ],
            "timing": {
                "stage0_vision": round(image_analysis_time, 1),
                "stage1_readers": round(stage1_time, 1),
                "stage2_searchers": round(stage2_time, 1),
                "stage31_investigation": round(stage31_time, 1),
                "stage32_trial": round(stage32_time, 1),
                "stage33_elimination": round(stage33_time, 1),
                "stage3_total": round(stage3_time, 1),
                "stage35_reasoning_tree": round(stage35_time, 1),
                "stage4_adjudicator": round(stage4_time, 1),
                "total": round(total_time, 1),
            },
            "images_analysis": {
                "total_images": len(images),
                "analyzed": len(image_descriptions),
                "descriptions": image_descriptions,
            },
            "evidence_graph": evidence_graph_data,  # 🆕 v13: 图片证据图谱
            "errors": all_errors,
            # 🆕 v8: 推理树结果
            "reasoning_tree_result": tree_result,
            # 🧠 v4新增: 记忆系统统计
            "memory_stats": {
                "total_memories": sum(len(self.memory_store.load(t)) for t in MemoryStore.EXPERT_TYPES),
                "total_skills": len(self.skill_registry.get_all_skills()),
                "total_patterns": len(self.pattern_library.get_all_patterns()),
            },
            # 🆕 v5: 动态领域专家信息
            "domain_experts": {
                "activated": len(domain_expert_results),
                "experts": [
                    {
                        "name": r.get("data", r).get("expert_meta", {}).get("name", "?"),
                        "domain": r.get("data", r).get("expert_meta", {}).get("domain", "?"),
                        "culprit": r.get("data", r).get("culprit", "?"),
                        "confidence": r.get("data", r).get("confidence", 0),
                    }
                    for r in domain_expert_results
                ],
            },
            # 🆕 v9: 三层架构信息
            "three_layer": {
                "investigation_vote": investigation_vote,
                "trial_vote": trial_vote,
                "investigation_count": len(investigation_results),
                "trial_count": len(trial_results),
            },
        }

    # ══════════════════════════════════════════════════════════════
    # v4新增: 记忆与技能学习
    # ══════════════════════════════════════════════════════════════

    def learn_from_result(self, case_id: str, case_data: Dict,
                          asmr_result: Dict, actual_culprit: str) -> Dict[str, Any]:
        """
        从案件结果中学习
        
        在得到实际答案后调用，系统自动:
        1. 记录各专家的正确/错误记忆
        2. 使用LLM提炼推理技能
        3. 发现犯罪模式
        
        Args:
            case_id: 案件ID
            case_data: 案件原始数据
            asmr_result: run()的返回结果
            actual_culprit: 实际凶手
        
        Returns:
            学习结果统计
        """
        self.logger.info(f"🧠 开始从案件 {case_id} 中学习 (实际凶手={actual_culprit})")

        # 构建expert_results映射
        expert_results = {}
        for analysis in asmr_result.get("expert_analyses", []):
            perspective = analysis.get("perspective", "")
            expert_results[perspective] = {"data": analysis}

        voting_result = asmr_result.get("vote_result", {})
        adjudicator_result = asmr_result.get("adjudicator_result")

        learning_result = self.skill_learner.learn_from_case(
            case_id=case_id,
            case_data=case_data,
            expert_results=expert_results,
            voting_result=voting_result,
            actual_culprit=actual_culprit,
            adjudicator_result=adjudicator_result,
        )

        self.logger.info(f"🧠 学习完成: {learning_result['new_skills']}新技能, "
                         f"{learning_result['new_patterns']}新模式")

        return learning_result

    def get_memory_report(self) -> Dict[str, Any]:
        """获取记忆系统报告"""
        return {
            "memory_stats": self.memory_store.get_stats(),
            "skill_stats": self.skill_registry.get_stats(),
            "pattern_count": len(self.pattern_library.get_all_patterns()),
        }

    # ══════════════════════════════════════════════════════════════
    # v3新增: 时序侦破推理
    # ══════════════════════════════════════════════════════════════

    def run_staged(
        self,
        case_text: str,
        suspects: List[Dict] = None,
        evidence: List[Any] = None,
        timeline: List[Dict] = None,
        evidence_stages: Optional[Dict] = None,
        run_asmr_after: bool = True,
    ) -> Dict[str, Any]:
        """
        运行时序侦破推理 + ASMR综合分析

        Args:
            case_text: 案件描述
            suspects: 嫌疑人列表
            evidence: 证据列表
            timeline: 时间线
            evidence_stages: 可选的预设阶段划分
            run_asmr_after: 是否在时序推理后运行ASMR流水线

        Returns:
            包含 stage_investigation 和 asmr_result 的综合结果
        """
        start_time = time.time()
        suspects = suspects or []
        evidence = evidence or []
        timeline = timeline or []

        self.logger.info("=" * 60)
        self.logger.info("🕵️ ASMR v3 时序侦破模式启动")
        self.logger.info(f"案件文本: {len(case_text)}字, 嫌疑人: {len(suspects)}人, "
                         f"证据: {len(evidence)}条")
        self.logger.info("=" * 60)

        # ==========================================
        # Part 1: 时序侦破推理
        # ==========================================
        self.logger.info("📋 Part 1: 时序侦破推理引擎")
        stage_start = time.time()

        investigation_stages = self.stage_engine.run_investigation(
            case_text=case_text,
            suspects=suspects,
            all_evidence=evidence,
            timeline=timeline,
            evidence_stages=evidence_stages,
        )

        stage_time = time.time() - stage_start
        self.logger.info(f"📋 时序推理完成 ({stage_time:.1f}s): {len(investigation_stages)} 个阶段")

        # 序列化阶段数据
        stages_data = [s.to_dict() for s in investigation_stages]

        # 提取最后阶段的结论
        last_stage = investigation_stages[-1] if investigation_stages else None
        stage_conclusion = {
            "culprit": last_stage.suspect_ranking[0].get("name", "未知") if last_stage and last_stage.suspect_ranking else "未知",
            "confidence": last_stage.confidence if last_stage else 0,
            "reasoning": last_stage.reasoning if last_stage else "",
            "suspect_ranking": last_stage.suspect_ranking if last_stage else [],
        }

        # ==========================================
        # Part 2: ASMR流水线（可选）
        # ==========================================
        asmr_result = None
        asmr_time = 0

        if run_asmr_after:
            self.logger.info("🔬 Part 2: ASMR v2 四阶段流水线")
            asmr_start = time.time()

            # 构建增强的case_text — 加入阶段推理结论作为上下文
            enhanced_case_text = case_text
            if last_stage and last_stage.reasoning:
                enhanced_case_text += f"\n\n[时序推理参考结论]\n{last_stage.reasoning[:500]}"

            try:
                asmr_result = self.run(
                    case_text=enhanced_case_text,
                    suspects=suspects,
                )
            except Exception as e:
                self.logger.error(f"ASMR流水线异常: {e}")
                asmr_result = {"error": str(e), "conclusion": {"culprit": "未知", "confidence": 0}}

            asmr_time = time.time() - asmr_start
            self.logger.info(f"🔬 ASMR完成 ({asmr_time:.1f}s)")

        total_time = time.time() - start_time

        self.logger.info("=" * 60)
        self.logger.info(f"🕵️ 时序侦破模式完成! 总耗时: {total_time:.1f}s")
        self.logger.info(f"时序推理结论: {stage_conclusion['culprit']} ({stage_conclusion['confidence']:.0%})")
        if asmr_result and "conclusion" in asmr_result:
            self.logger.info(f"ASMR结论: {asmr_result['conclusion'].get('culprit', '?')} "
                             f"({asmr_result['conclusion'].get('confidence', 0):.0%})")
        self.logger.info("=" * 60)

        return {
            "track": "Staged_ASMR",
            "stage_investigation": {
                "stages": stages_data,
                "stage_count": len(investigation_stages),
                "conclusion": stage_conclusion,
                "timing": round(stage_time, 1),
            },
            "asmr_result": asmr_result,
            "total_time": round(total_time, 1),
            "timing": {
                "stage_engine": round(stage_time, 1),
                "asmr_pipeline": round(asmr_time, 1),
                "total": round(total_time, 1),
            },
        }
