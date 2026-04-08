"""
WebUI v14.0 — 🧵 忒修斯之线 (Ariadne's Thread) · 赛博侦探 Cyber-Sleuth UI
v14.0: 多轮推理对话 — 每个专家4轮深度推理(初步→审视→深入→整合)，实时展示推理演进
v10.0: 推理事件总线 + 专家推理矩阵实时展示 + 动态投票分布
v4.0: 全面赛博朋克化 + 思维殿堂(Brain Palace)布局 + 四主角系统
"""
import gradio as gr
from typing import Dict, List
import yaml, os, sys, json, time, threading
from concurrent.futures import ThreadPoolExecutor, Future

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api.llm_client import LLMClient
from loguru import logger
from ui.reasoning_log import ReasoningLog
from ui.format_helpers import txt, names, fmt_trad, fmt_asmr, fmt_fusion, fmt_graph, load_test_results
from ui.graph_renderer import render_force_graph
from ui.stage_visualization import render_stage_timeline, render_suspect_evolution_chart
from ui.brain_palace import BrainPalace, stage_evidence_extraction, stage_traditional_reasoning, stage_asmr_search, stage_voting, stage_verdict, stage_fusion
from ui.expert_card_renderer import ExpertCardRenderer
from ui.expert_chat_renderer import ExpertChatRenderer
from ui.reasoning_event_bus import ReasoningEventBus, reset_event_bus
from ui.conversation_store import get_conversation_store, reset_conversation_store

# Load CSS
_css_path = os.path.join(os.path.dirname(__file__), "_css.txt")
with open(_css_path) as f:
    POLICE_CSS = f.read()


class DetectiveWebUI:
    def __init__(self, config_path="./config/config.yaml"):
        self.logger = logger.bind(module="webui")
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        self.llm_client = LLMClient()
        self.chat_renderer = ExpertChatRenderer()
        self._current_case_text = ""  # 🆕 v14.0.1: 存储当前案件文本

    # ================================================================
    #  核心: 流式推理 (generator) — 双线并行 + 实时更新 + 专家推理矩阵
    # ================================================================
    def stream_analyze(self, case_text):
        """每步yield实时更新 — 11个输出: live_text, live_graph, brain_palace, expert_matrix, trad, asmr, fusion, graph, json, status, chat"""
        self._current_case_text = case_text  # 🆕 存储供专家群聊使用
        E = '<div style="color:#55556A;text-align:center;padding:30px;font-family:JetBrains Mono,monospace">等待中…</div>'
        bp = BrainPalace()
        ec = ExpertCardRenderer()
        chat = self.chat_renderer
        BP_EMPTY = bp.render_empty()
        EC_EMPTY = ec.render_empty()
        CHAT_EMPTY = '<div style="text-align:center;color:#55556A;padding:30px;font-family:JetBrains Mono,monospace;font-size:11px">等待推理启动…</div>'
        if not case_text.strip():
            yield ("❌ 请输入案件描述", "", BP_EMPTY, EC_EMPTY, E, E, E, E, "{}", "❌ 无输入", CHAT_EMPTY); return

        log = ReasoningLog()

        # 清除图谱布局缓存 — 新案件，旧节点位置全部失效
        from ui.graph_renderer_svg import reset_layout_cache
        reset_layout_cache()

        try:
            from agents.clue_extractor import ClueExtractorAgent
            from agents.suspect_analyzer import SuspectAnalyzerAgent
            from agents.evidence_connector import EvidenceConnectorAgent
            from agents.graph_builder import GraphBuilderAgent
            from agents.graph_reasoner import GraphReasonerAgent
            from agents.reasoning_generator import ReasoningGeneratorAgent
            from agents.asmr.orchestrator import ASMROrchestrator
            from agents.asmr.dual_track_fusion import DualTrackFusionEngine
        except Exception as e:
            log.stage("❌", "初始化失败", "error"); log.thought(f"无法导入: {e}", "danger")
            yield (log.render(), log.render_graph(), BP_EMPTY, EC_EMPTY, E, E, E, "{}", "❌ 失败", CHAT_EMPTY); return

        llm = self.llm_client
        total_t0 = time.time()

        # ---- Shared state between threads ----
        trad_result = {
            "done": False, "error": None,
            "culprit": "未知", "confidence": 0, "time": 0,
            "full": {}, "md": E, "graph_html": "", "graph_json": "{}"
        }
        asmr_result = {
            "done": False, "error": None,
            "culprit": "未知", "confidence": 0, "time": 0,
            "full": {}, "md": E
        }
        bp_state = {"needs_render": False}  # flag for main thread to re-render BP
        log_lock = threading.Lock()

        # 🆕 v10.0: 事件总线 — ASMR线路的推理过程实时暴露
        event_bus = reset_event_bus()
        
        # 🆕 v14.0: 对话持久化存储
        conv_store = reset_conversation_store()
        conv_store.start_case("webui_case", case_text[:100])

        # ---- Traditional Line (runs in thread) ----
        def _extract_entities_from_clues(clues):
            """从线索中提取实体节点和初始关系边"""
            nodes = []
            edges = []
            # clues可能是 dict(含key_clues) 或 list
            clue_list = clues
            if isinstance(clues, dict):
                clue_list = clues.get("key_clues", clues.get("data", {}).get("key_clues", []))
                if not isinstance(clue_list, list):
                    clue_list = [clue_list]
                # 提取时间线事件
                timeline = clues.get("timeline", clues.get("data", {}).get("timeline", []))
                if isinstance(timeline, list):
                    for t in timeline[:8]:
                        evt_text = t.get("event", str(t))[:20] if isinstance(t, dict) else str(t)[:20]
                        person = t.get("involves", "") if isinstance(t, dict) else ""
                        if evt_text:
                            nodes.append({"id": f"evt_{evt_text}", "label": evt_text, "type": "event"})
                            if person:
                                edges.append({"source": f"evt_{evt_text}", "target": person, "label": "涉及", "weight": 0.4})

            if isinstance(clue_list, list):
                for i, c in enumerate(clue_list[:12]):
                    if isinstance(c, dict):
                        desc = c.get("clue", c.get("description", ""))
                        ctype = c.get("type", "")
                        # 物理证据→evidence节点
                        if ctype in ("物理证据", "物证", "数字证据"):
                            label = desc[:14] if desc else f"证据{i}"
                            nodes.append({"id": f"evi_{i}_{label}", "label": label, "type": "evidence"})
                        # 从涉及人物中提取人名（简单的中文姓名模式）
                        if desc:
                            import re
                            # 匹配中文姓名(2-3字,后面跟着常见动词,但排除非人名的组合)
                            persons = re.findall(r'[\u4e00-\u9fff]{2,3}(?=的|说|称|曾|有|是|与|被|将|去|来|把)', desc)
                            for p in set(persons[:3]):
                                nodes.append({"id": p, "label": p, "type": "person"})
                    elif isinstance(c, str) and len(c) > 2:
                        nodes.append({"id": f"clue_{i}", "label": c[:14], "type": "evidence"})

            # 去重
            seen = set()
            unique = []
            for n in nodes:
                if n["id"] not in seen:
                    seen.add(n["id"])
                    unique.append(n)
            return unique, edges

        def _extract_suspect_edges(suspects):
            """从嫌疑人分析中提取关系边 — 优先提取嫌疑人之间的语义关系"""
            edges = []
            suspect_list = []
            if isinstance(suspects, list):
                suspect_list = suspects
            elif isinstance(suspects, dict):
                data = suspects.get("data", suspects)
                # 可能是单个嫌疑人 dict 或包含 suspects 列表
                if "suspects" in data:
                    suspect_list = data["suspects"]
                else:
                    suspect_list = [data]

            for s in suspect_list:
                if not isinstance(s, dict):
                    continue
                name = s.get("name", s.get("data", {}).get("name", ""))
                if not name:
                    continue
                data = s.get("data", s)

                # 动机 → has_motive
                motive = data.get("motive", {})
                if isinstance(motive, dict):
                    mtype = motive.get("type", "")
                    mdesc = motive.get("description", "")
                    target = motive.get("target", motive.get("victim", ""))
                    if mtype:
                        edge_label = mtype if len(mtype) <= 14 else mtype[:14]
                        edges.append({"source": name, "target": target or "案件", "label": edge_label, "weight": 0.7})

                # 机会 → has_opportunity
                opportunity = data.get("opportunity", data.get("access", {}))
                if isinstance(opportunity, dict):
                    odesc = opportunity.get("description", opportunity.get("type", ""))
                    if odesc:
                        edges.append({"source": name, "target": "案件", "label": "有机会", "weight": 0.6})

                # 嫌疑人间关系 (如 共谋/认识/敌对)
                relationships = data.get("relationships", data.get("connections", []))
                if isinstance(relationships, list):
                    for rel in relationships[:4]:
                        if isinstance(rel, dict):
                            other = rel.get("with", rel.get("target", rel.get("person", "")))
                            rtype = rel.get("type", rel.get("relationship", ""))
                            if other and rtype:
                                edges.append({"source": name, "target": other, "label": rtype[:14], "weight": 0.6})

                # 关联证据
                linked_evidence = data.get("linked_evidence", data.get("evidence", []))
                if isinstance(linked_evidence, list):
                    for ev in linked_evidence[:3]:
                        if isinstance(ev, dict):
                            evname = ev.get("name", ev.get("description", ""))
                            evtype = ev.get("type", "关联")
                            if evname:
                                edges.append({"source": name, "target": evname[:14], "label": evtype[:10] if evtype else "关联", "weight": 0.6})
            return edges

        def _extract_evidence_edges(evidence, suspects):
            """从证据关联中提取证据-嫌疑人关联边"""
            edges = []
            sn = names(suspects) if isinstance(suspects, (list, dict)) else []
            if isinstance(evidence, dict):
                ev_data = evidence.get("data", evidence)
                connections = ev_data.get("connections", ev_data.get("evidence_links", []))
                if isinstance(connections, list):
                    for c in connections[:8]:
                        if isinstance(c, dict):
                            src = c.get("from", c.get("source", ""))
                            tgt = c.get("to", c.get("target", ""))
                            rel = c.get("relationship", c.get("type", "关联"))
                            if src and tgt:
                                edges.append({"source": src, "target": tgt, "label": rel[:10], "weight": 0.5})
                # 如果没有结构化的connections，尝试从suspicious_evidence提取
                se = ev_data.get("suspicious_evidence", [])
                if isinstance(se, list):
                    for i, s in enumerate(se[:4]):
                        if isinstance(s, dict):
                            person = s.get("person", s.get("suspect", ""))
                            evi = s.get("evidence", s.get("description", f"证据{i}"))
                            if person:
                                edges.append({"source": evi[:14], "target": person, "label": "指向", "weight": 0.7})
            elif isinstance(evidence, list):
                for i, ev in enumerate(evidence[:6]):
                    if isinstance(ev, dict):
                        desc = ev.get("description", ev.get("clue", ""))
                        for person in sn:
                            if person and desc and person in str(desc):
                                edges.append({"source": f"evi_{i}", "target": person, "label": "关联", "weight": 0.5})
            return edges

        def run_traditional():
            t0 = time.time()
            try:
                with log_lock:
                    log.stage("🔍", "🔵 传统线路 — 启动", "running")
                    log.thought("解析案件文本，识别关键实体与事件要素…")
                    stage_evidence_extraction(bp)
                    bp.add_fragment("puppeteer", "线索提取中…")
                    bp_state["needs_render"] = True

                # 1. Clues → 立即提取实体节点渲染图谱
                clues = ClueExtractorAgent(llm_client=llm).process({"case_text": case_text})
                logger.info(f"[图谱增量] clues type={type(clues).__name__}, keys={list(clues.keys()) if isinstance(clues, dict) else 'N/A'}")
                clue_data = clues.get("data", clues) if isinstance(clues, dict) else clues
                nc = len(clue_data.get("key_clues", [])) if isinstance(clue_data, dict) else 1
                clue_nodes, clue_edges = _extract_entities_from_clues(clues)
                logger.info(f"[图谱增量] Stage1: {len(clue_nodes)} nodes, {len(clue_edges)} edges from {nc} clues")
                with log_lock:
                    log.finish("Stage 1/6 — 线索提取", "done", f"✅ {nc} 组线索")
                    if clue_nodes:
                        log.nodes(clue_nodes)
                    if clue_edges:
                        log.edges(clue_edges)
                    bp.update_agent("puppeteer", 35, "ANALYZING")
                    bp.add_fragment("puppeteer", f"提取 {nc} 组线索完成，发现 {len(clue_nodes)} 个实体")
                    bp.add_timeline_event(f"✅ 线索提取: {nc} 组, {len(clue_nodes)} 实体", "success")
                    bp_state["needs_render"] = True

                # 2. Suspects → 添加嫌疑人节点 + 关系边
                suspects = SuspectAnalyzerAgent(llm_client=llm).process({"suspect_info": case_text, "case_clues": clues})
                sn = names(suspects)
                logger.info(f"[图谱增量] Stage2: suspects={sn}, type={type(suspects).__name__}")
                suspect_edges = _extract_suspect_edges(suspects)
                with log_lock:
                    log.set_suspect_names(sn)  # 缓存嫌疑人用于图谱高亮
                    log.finish("Stage 2/6 — 嫌疑人分析", "done", f"✅ {len(sn)} 名嫌疑人: {', '.join(sn[:6])}")
                    # 嫌疑人节点标为 suspect 类型(更醒目)
                    for nm in sn[:10]:
                        log.nodes([{"id": nm, "label": nm, "type": "suspect"}])
                    # 从嫌疑人数据中提取动机边 (跳过"案件中心"辐射连接)
                    if suspect_edges:
                        log.edges(suspect_edges)
                    bp.update_agent("puppeteer", 50, "ANALYZING")
                    bp.add_fragment("puppeteer", f"识别嫌疑人: {', '.join(sn[:4])}")
                    bp.add_timeline_event(f"👤 嫌疑人: {', '.join(sn[:4])}", "info")
                    bp_state["needs_render"] = True

                # 3. Evidence → 添加证据节点 + 关联边
                evidence = EvidenceConnectorAgent(llm_client=llm).process({"evidence_list": clues, "suspects": suspects, "case_clues": clues})
                evi_edges = _extract_evidence_edges(evidence, suspects)
                with log_lock:
                    log.finish("Stage 3/6 — 证据关联", "done", "✅ 证据链完成")
                    if evi_edges:
                        log.edges(evi_edges)
                    bp.update_agent("puppeteer", 60, "ANALYZING")
                    bp.add_fragment("puppeteer", "证据链构建完成")
                    bp.add_timeline_event("🔗 证据链完成", "success")
                    bp_state["needs_render"] = True

                # 4. Graph (完整图谱会覆盖增量节点)
                graph_raw = GraphBuilderAgent(llm_client=llm).process({"case_clues": clues, "suspects": suspects, "evidence": evidence})
                # 解包 format_output 包装: {"agent":..., "data": {"nodes":..., "edges":...}}
                graph = graph_raw.get("data", graph_raw) if isinstance(graph_raw, dict) else graph_raw
                gn = graph.get("nodes", []) if isinstance(graph, dict) else []
                ge = graph.get("edges", []) if isinstance(graph, dict) else []
                logger.info(f"[图谱增量] Stage4: graph_raw keys={list(graph_raw.keys()) if isinstance(graph_raw, dict) else 'N/A'}, nodes={len(gn)}, edges={len(ge)}")
                with log_lock:
                    log.nodes(gn[:15]); log.edges(ge[:12])
                    log.finish("Stage 4/6 — 知识图谱", "done", f"✅ {len(gn)}节点 {len(ge)}关系")
                    bp.update_agent("chronos", 45, "ANALYZING")
                    bp.add_fragment("chronos", f"图谱: {len(gn)}节点 {len(ge)}关系")
                    bp.add_timeline_event(f"🕸️ 图谱构建: {len(gn)}N/{len(ge)}E", "info")
                    bp_state["needs_render"] = True

                # 5. Graph reasoning
                gr_res = GraphReasonerAgent(llm_client=llm).process({"graph": graph, "case_clues": clues, "suspect_analyses": suspects, "evidence_connections": evidence})
                with log_lock:
                    log.finish("Stage 5/6 — 图谱推理", "done")
                    bp.update_agent("chronos", 60, "ANALYZING")
                    bp_state["needs_render"] = True

                # 6. Report
                reasoning = ReasoningGeneratorAgent(llm_client=llm).process({"case_clues": clues, "suspect_analyses": suspects, "evidence_connections": evidence, "graph": graph, "graph_reasoning": gr_res})
                elapsed = time.time() - t0
                # 解包 format_output 包装: {"agent":..., "data": {...}}
                rdata = reasoning.get("data", reasoning) if isinstance(reasoning, dict) else reasoning
                parsed = rdata if isinstance(rdata, dict) else reasoning

                culprit = "未知"
                confidence = 0
                if isinstance(parsed, dict):
                    # 优先从 final_conclusion 取（ReasoningGeneratorAgent 的结构）
                    fc = parsed.get("final_conclusion", {})
                    if isinstance(fc, dict):
                        culprit = fc.get("top_suspect", fc.get("culprit", fc.get("suspect", "")))
                        confidence = fc.get("confidence", parsed.get("confidence_score", 0))
                    if not culprit or culprit == "None":
                        culprit = parsed.get("culprit", parsed.get("suspect", parsed.get("final_suspect", "未知"))) or "未知"
                    if not confidence:
                        confidence = parsed.get("confidence", parsed.get("overall_confidence", parsed.get("final_confidence", parsed.get("confidence_score", 0)))) or 0
                elif isinstance(parsed, str):
                    culprit = parsed[:20] if parsed else "未知"
                if not culprit or culprit == "None":
                    culprit = "未知"

                full = {"clues": clues, "suspects": suspects, "evidence": evidence,
                        "graph": graph, "graph_reasoning": gr_res, "reasoning": reasoning,
                        "culprit": culprit, "confidence": confidence, "time": elapsed}

                with log_lock:
                    log.finish("Stage 6/6 — 推理报告", "done", f"✅ {culprit} ({confidence:.1%})")
                    log.conclusion(f"🔵传统: {culprit}, {confidence:.1%}, {elapsed:.1f}s")
                    log.finish("🔵 传统线路 — 启动", "done")
                    bp.update_agent("puppeteer", 80, "DONE")
                    bp.add_fragment("puppeteer", f"传统结论: {culprit} ({confidence:.0%})")
                    bp.add_timeline_event(f"🔵 传统完成: {culprit} ({confidence:.0%})", "success")
                    bp_state["needs_render"] = True

                trad_result.update({
                    "done": True, "culprit": culprit, "confidence": confidence, "time": elapsed,
                    "full": full, "md": fmt_trad(full, elapsed),
                    "graph_html": fmt_graph(graph, gr_res), "graph_json": json.dumps(graph, ensure_ascii=False) if isinstance(graph, dict) else "{}"
                })
            except Exception as e:
                elapsed = time.time() - t0
                with log_lock:
                    log.finish("传统线路", "error", f"❌ {e}")
                    bp.update_agent("puppeteer", 100, "DONE")
                    bp.add_timeline_event(f"❌ 传统错误: {str(e)[:40]}", "danger")
                    bp_state["needs_render"] = True
                trad_result.update({
                    "done": True, "error": str(e), "time": elapsed,
                    "md": fmt_trad({"error": str(e), "culprit": "未知", "confidence": 0}, elapsed)
                })

        # ---- ASMR Line (runs in thread) ----
        def run_asmr():
            t0 = time.time()
            try:
                with log_lock:
                    log.stage("🔬", "🟢 ASMR线路 — 启动", "running")
                    log.thought("四阶段: Readers → Searchers → Experts → Adjudicator")
                    bp.update_agent("mirror", 20, "ANALYZING")
                    bp.add_timeline_event("🟢 ASMR线路启动", "info")
                    bp.add_fragment("mirror", "ASMR矛盾搜索开始…")
                    bp_state["needs_render"] = True

                orch = ASMROrchestrator(llm_client=llm, event_bus=event_bus, conversation_store=conv_store)
                asmr_raw = orch.run(case_text=case_text, suspects=[])
                elapsed = time.time() - t0

                conc = asmr_raw.get("conclusion", {})
                culprit = conc.get("culprit", asmr_raw.get("final_conclusion", "未知"))
                confidence = conc.get("confidence", asmr_raw.get("confidence", 0))

                vw = asmr_raw.get("vote_result", {}).get("winner", "?")
                ot = conc.get("overturned", False)

                with log_lock:
                    for e in asmr_raw.get("expert_analyses", []):
                        log.vote(e.get("perspective", "?"), e.get("culprit", "?"), e.get("confidence", 0))
                    cd = asmr_raw.get("contradiction_data", {})
                    contradictions_list = cd.get("contradictions", [])
                    for c in contradictions_list:
                        log.contra(c.get("significance", "?"), c.get("person", "?"), c.get("description", "?"))
                    bp.update_agent("mirror", 60, "ANALYZING")
                    if contradictions_list:
                        for c in contradictions_list[:3]:
                            desc = c.get("description", str(c))[:40]
                            bp.add_fragment("mirror", f"矛盾: {desc}")
                    if ot:
                        log.thought(f"⚡ 裁判推翻投票! {vw}→{culprit}", "danger")
                        log.conclusion(f"🟢ASMR裁判推翻! {vw}→{culprit} ({confidence:.1%})")
                        bp.add_fragment("mirror", f"⚠️ 推翻! {vw}→{culprit}")
                        bp.add_timeline_event(f"⚠️ 裁判推翻! {vw}→{culprit}", "danger")
                    else:
                        log.conclusion(f"🟢ASMR: {culprit} ({confidence:.1%}, {elapsed:.1f}s)")
                        bp.add_fragment("mirror", f"ASMR结论: {culprit} ({confidence:.0%})")
                        bp.add_timeline_event(f"🟢 ASMR完成: {culprit} ({confidence:.0%})", "success")
                    bp.update_agent("mirror", 80, "DONE")
                    log.finish("🟢 ASMR线路 — 启动", "done")
                    bp_state["needs_render"] = True

                    # 🆕 v15.2: 注入 ASMR 证据图谱到 WebUI 可视化
                    eg = asmr_raw.get("evidence_graph", {})
                    eg_nodes = eg.get("nodes", [])
                    eg_edges = eg.get("edges", [])
                    if eg_nodes:
                        for n in eg_nodes[:30]:
                            log.nodes([{"id": n.get("id", ""), "label": n.get("name", n.get("id", "")), "type": n.get("type", "default")}])
                        for e in eg_edges[:30]:
                            log.edges([{
                                "source": e.get("src", e.get("source", "")),
                                "target": e.get("tgt", e.get("target", "")),
                                "label": e.get("rel", e.get("label", "")),
                                "weight": float(e.get("weight", 0.5)),
                            }])
                        bp.add_timeline_event(f"🕸️ 图谱注入: {len(eg_nodes)}节点 {len(eg_edges)}关系", "info")

                asmr_result.update({
                    "done": True, "culprit": culprit, "confidence": confidence, "time": elapsed,
                    "full": asmr_raw, "md": fmt_asmr(asmr_raw, elapsed)
                })
            except Exception as e:
                elapsed = time.time() - t0
                with log_lock:
                    log.finish("ASMR线路", "error", f"❌ {e}")
                    bp.update_agent("mirror", 100, "DONE")
                    bp.add_timeline_event(f"❌ ASMR错误: {str(e)[:40]}", "danger")
                    bp_state["needs_render"] = True
                asmr_result.update({
                    "done": True, "error": str(e), "time": elapsed,
                    "md": fmt_asmr({"error": str(e)}, elapsed)
                })

        # ---- Launch both threads ----
        log.stage("🚀", "双线并行推理 — 启动", "running")
        log.thought("🔵传统 + 🟢ASMR 同时启动，竞争分析")
        bp.add_timeline_event("🚀 双线并行启动", "info")
        yield (log.render(), log.render_graph(), bp.render(), EC_EMPTY, E, E, E, E, "{}", "🚀 双线并行启动…", CHAT_EMPTY)

        with ThreadPoolExecutor(max_workers=2) as pool:
            trad_future = pool.submit(run_traditional)
            asmr_future = pool.submit(run_asmr)

            # Poll and yield UI updates
            poll_interval = 1.5
            last_trad_done = False
            last_asmr_done = False
            _prev_graph_nc = 0   # 上次图谱渲染时的节点数
            _prev_graph_ec = 0   # 上次图谱渲染时的边数
            _cached_graph_html = ""
            while not (trad_result["done"] and asmr_result["done"]):
                time.sleep(poll_interval)

                # Build status
                parts = []
                if trad_result["done"] and not last_trad_done:
                    last_trad_done = True
                    parts.append(f"🔵传统✅({trad_result['time']:.0f}s)")
                elif not trad_result["done"]:
                    parts.append("🔵传统…")
                if asmr_result["done"] and not last_asmr_done:
                    last_asmr_done = True
                    parts.append(f"🟢ASMR✅({asmr_result['time']:.0f}s)")
                elif not asmr_result["done"]:
                    parts.append("🟢ASMR…")
                status = " | ".join(parts)

                elapsed_so_far = time.time() - total_t0
                with log_lock:
                    rendered = log.render()
                    # 🆕 图谱只在节点/边数变化时才重新渲染（避免每1.5秒整体重绘）
                    cur_nc, cur_ec = log.node_count, log.edge_count
                    if cur_nc != _prev_graph_nc or cur_ec != _prev_graph_ec:
                        rendered_graph = log.render_graph()
                        _cached_graph_html = rendered_graph
                        _prev_graph_nc, _prev_graph_ec = cur_nc, cur_ec
                    else:
                        rendered_graph = _cached_graph_html
                    bp_html = bp.render() if bp_state["needs_render"] else None
                    bp_state["needs_render"] = False

                if bp_html is None:
                    bp_html = bp.render()

                # 🆕 v10.0: 从事件总线渲染专家推理矩阵
                bus_state = event_bus.get_state()
                expert_matrix_html = ec.render_expert_panel(
                    agent_results=bus_state["agent_results"],
                    vote_history=bus_state["vote_history"],
                    stage_progress=bus_state["stage_progress"],
                    stage_timings=bus_state["stage_timings"],
                    agent_round_log=bus_state.get("agent_round_log", {}),  # 🆕 v14: 多轮推理日志
                )

                # 🆕 v14.0: 渲染专家群聊
                chat_html = chat.render_chat_panel(
                    agent_results=bus_state["agent_results"],
                    agent_round_log=bus_state.get("agent_round_log", {}),
                    stage_progress=bus_state["stage_progress"],
                )

                # 🆕 关联图谱实时刷新: 优先用增量图谱(log累积所有节点/边)
                if trad_result.get("graph_html"):
                    graph_html = rendered_graph if rendered_graph else trad_result["graph_html"]
                else:
                    graph_html = rendered_graph

                yield (rendered, graph_html, bp_html, expert_matrix_html,
                       trad_result["md"], asmr_result["md"], E,
                       graph_html,
                       "{}", f"⏱ {elapsed_so_far:.0f}s — {status}", chat_html)

        # ---- Fusion ----
        log.stage("🔀", "双线融合决策", "running")
        t_trad = trad_result["culprit"]
        t_conf = trad_result["confidence"]
        a_trad = asmr_result["culprit"]
        a_conf = asmr_result["confidence"]
        log.thought(f"🔵传统={t_trad}({t_conf:.0%}) vs 🟢ASMR={a_trad}({a_conf:.0%})")

        # Brain Palace: voting stage
        stage_voting(bp, t_trad, t_conf, a_trad, a_conf)

        fusion = {}
        try:
            fuser = DualTrackFusionEngine()
            fusion = fuser.fuse(
                traditional_result=trad_result["full"],
                asmr_result=asmr_result["full"])
        except Exception as e:
            fusion = {"error": str(e), "culprit": "融合失败", "confidence": 0}

        agr = "🟢一致" if trad_result["culprit"] == asmr_result["culprit"] else "🔴不一致"
        # 融合引擎返回嵌套结构: {"conclusion": {"culprit": ..., "confidence": ...}, ...}
        fusion_conc = fusion.get("conclusion", fusion)
        fusion_c_name = fusion_conc.get("culprit", fusion.get("culprit", "?"))
        fusion_c_conf = fusion_conc.get("confidence", fusion.get("confidence", 0))
        log.conclusion(f"🔀融合: {fusion_c_name} ({fusion_c_conf:.1%}) | {agr}")
        log.finish("双线融合决策", "done")
        log.finish("🚀 双线并行推理 — 启动", "done")

        # Brain Palace: final verdict
        fusion_conc = fusion.get("conclusion", fusion)
        fusion_culprit = fusion_conc.get("culprit", trad_result["culprit"])
        fusion_conf = fusion_conc.get("confidence", trad_result["confidence"])
        overridden = fusion_conc.get("overridden", fusion.get("overridden", False))
        stage_verdict(bp, fusion_culprit, fusion_conf, overridden=overridden, shield=True)
        stage_fusion(bp, fusion)

        fusion_md = fmt_fusion(fusion, trad_result, asmr_result)
        full_json = json.dumps({
            "traditional": {"culprit": trad_result["culprit"], "confidence": trad_result["confidence"], "time": trad_result["time"]},
            "asmr": {"culprit": asmr_result["culprit"], "confidence": asmr_result["confidence"], "time": asmr_result["time"]},
            "fusion": fusion
        }, ensure_ascii=False, indent=2)
        total = time.time() - total_t0
        # 🆕 v14.0: 保存最终融合结果到对话存储
        conv_store.save_fusion(fusion)
        conv_store.finish_case({
            "traditional_culprit": trad_result["culprit"],
            "asmr_culprit": asmr_result["culprit"],
            "fusion_culprit": fusion_culprit,
            "fusion_confidence": fusion_conf,
            "total_time": total,
        })

        log.conclusion(f"⏱ 总计 {total:.1f}s (并行: max({trad_result['time']:.0f}, {asmr_result['time']:.0f}) + 融合)")

        # 🆕 v10.0: 最终专家矩阵 (含完整数据)
        bus_state = event_bus.get_state()
        final_expert_html = ec.render_expert_panel(
            agent_results=bus_state["agent_results"],
            vote_history=bus_state["vote_history"],
            stage_progress=bus_state["stage_progress"],
            stage_timings=bus_state["stage_timings"],
            agent_round_log=bus_state.get("agent_round_log", {}),  # 🆕 v14: 多轮推理日志
        )

        # 🆕 v14.0: 最终群聊渲染
        final_chat_html = chat.render_chat_panel(
            agent_results=bus_state["agent_results"],
            agent_round_log=bus_state.get("agent_round_log", {}),
            stage_progress=bus_state["stage_progress"],
        )

        # 最终图谱: 用log累积的全部节点(含增量构建的中间节点)，比纯graph_builder更丰富
        final_graph_html = log.render_graph() or trad_result["graph_html"]
        yield (log.render(), final_graph_html, bp.render(), final_expert_html,
               trad_result["md"], asmr_result["md"], fusion_md,
               final_graph_html, full_json,
               f"✅ 完成 ({total:.0f}s) 🔵{trad_result['time']:.0f}s 🟢{asmr_result['time']:.0f}s",
               final_chat_html)

    # ================================================================
    #  时序侦破推理 (generator) — 逐步展示推理过程
    # ================================================================
    def stream_staged_analyze(self, case_text):
        """时序侦破推理 — 证据逐步释放，阶段推理，实时展示"""
        self._current_case_text = case_text  # 🆕 存储供专家群聊使用
        EMPTY = '<div style="color:#55556A;text-align:center;padding:40px;font-family:JetBrains Mono,monospace;font-size:12px">等待时序侦破启动…</div>'
        if not case_text.strip():
            yield (EMPTY, EMPTY, "❌ 请输入案件描述", "⏳ 等待输入…"); return

        try:
            from agents.asmr.stage_engine import StageEngine, create_stage_llm_client
            from tests.test_detective_cases import get_all_test_cases
        except Exception as e:
            yield (EMPTY, EMPTY, f"❌ 初始化失败: {e}", "❌ 失败"); return

        # 获取案件数据（嫌疑人、证据、时间线）
        suspects, evidence, timeline = [], [], []
        try:
            cases = get_all_test_cases()
            for c in cases:
                if c.case_text.strip() == case_text.strip() or c.case_text[:100] in case_text:
                    suspects = c.suspects
                    evidence = c.evidence if isinstance(c.evidence, list) else []
                    timeline = c.timeline if hasattr(c, 'timeline') and c.timeline else []
                    break
        except:
            pass

        yield (EMPTY, EMPTY, "🕵️ 时序引擎初始化中…", "🔄 初始化中…")

        t0 = time.time()
        try:
            llm = create_stage_llm_client()
            engine = StageEngine(llm)

            # 运行完整时序侦破
            stages = engine.run_investigation(
                case_text=case_text,
                suspects=suspects,
                evidence=evidence,
                timeline=timeline,
            )

            elapsed = time.time() - t0

            # 渲染结果
            stages_data = [s.to_dict() for s in stages]
            timeline_html = render_stage_timeline(stages_data)
            chart_html = render_suspect_evolution_chart(stages_data)

            # Markdown总结
            summary_md = self._fmt_staged_summary(stages, elapsed)

            total_time = time.time() - t0
            status = f"✅ 时序侦破完成 ({total_time:.0f}s, {len(stages)} 阶段)"
            yield (timeline_html, chart_html, summary_md, status)

        except Exception as e:
            elapsed = time.time() - t0
            error_md = f"## ❌ 时序侦破失败\n\n```\n{e}\n```\n\n⏱ {elapsed:.1f}s"
            yield (EMPTY, EMPTY, error_md, f"❌ 失败 ({elapsed:.0f}s)")

    def _fmt_staged_summary(self, stages, elapsed):
        """格式化时序侦破的Markdown总结"""
        if not stages:
            return "暂无结果"

        md = f"## 🕵️ 时序侦破推理总结\n**⏱ {elapsed:.1f}s** | **{len(stages)} 阶段**\n\n"

        for stage in stages:
            sid = stage.stage_id
            name = stage.stage_name
            conf = stage.confidence or 0
            ranking = stage.suspect_ranking or []
            tp = stage.key_turning_point

            md += f"### 阶段 {sid}: {name}\n"
            md += f"- 置信度: {conf:.0%}\n"
            if tp:
                md += f"- ⚡ 关键转折: {tp}\n"

            if ranking:
                md += "\n| 排名 | 嫌疑人 | 嫌疑度 | 理由 |\n|------|--------|--------|------|\n"
                for i, s in enumerate(ranking[:4], 1):
                    sc = s.get("suspicion_score", 0)
                    md += f"| {i} | {s.get('name','?')} | {sc:.0%} | {s.get('reason','')[:40]} |\n"

            if stage.investigation_advice:
                md += "\n**调查建议**: "
                md += " | ".join([f"{'🔴' if a.get('priority')=='高' else '🟡'} {a.get('direction','?')}" for a in stage.investigation_advice[:3]])
                md += "\n"

            if stage.hypotheses:
                md += "\n**假设预测**: "
                md += " | ".join([f"🔮 {h.get('condition','?')[:30]}→{h.get('then','?')[:20]}" for h in stage.hypotheses[:2]])
                md += "\n"

            md += "\n---\n\n"

        # 最终结论
        last = stages[-1]
        ranking = last.suspect_ranking or []
        if ranking:
            top = ranking[0]
            md += f"## 🏁 最终结论\n"
            md += f"**最大嫌疑人**: `{top.get('name', '?')}` ({top.get('suspicion_score', 0):.0%})\n"
            md += f"**最终置信度**: {last.confidence:.0%}\n"

            turning_points = [s for s in stages if s.key_turning_point]
            if turning_points:
                md += f"\n**关键转折**: {len(turning_points)}次\n"
                for tp in turning_points:
                    md += f"- ⚡ 阶段{tp.stage_id}: {tp.key_turning_point}\n"

        return md

    # ================================================================
    #  Test cases
    # ================================================================
    def load_test_case(self, case_idx):
        try:
            from tests.test_detective_cases import get_all_test_cases
            cases = get_all_test_cases()
            if 0 <= case_idx < len(cases):
                return cases[case_idx].case_text
            return "❌ 索引越界"
        except Exception as e:
            return f"❌ {e}"

    def get_test_case_list(self):
        try:
            from tests.test_detective_cases import get_all_test_cases
            cases = get_all_test_cases()
            return [f"CASE-{i+1:03d}: {c.case_type} (难度{'★'*c.difficulty})" for i, c in enumerate(cases)]
        except:
            return ["⚠️ 无法加载"]

    # ================================================================
    #  UI Builder
    # ================================================================
    def load_case_resources(self, case_idx):
        """加载案件原始资料: 返回 (case_text, raw_docs_html, raw_data_html)"""
        EMPTY = '<div style="color:#55556A;text-align:center;padding:20px;font-family:JetBrains Mono,monospace;font-size:11px">请先加载案件</div>'
        try:
            from tests.test_detective_cases import get_all_test_cases
            cases = get_all_test_cases()
            if not (0 <= case_idx < len(cases)):
                return "", EMPTY, EMPTY
            c = cases[case_idx]
            case_text = c.case_text

            # === Tab1: 案卷文档 === 原始案卷文本全文 + 基础信息
            doc_html = f'''<div style="font-family:'Noto Serif SC',serif;color:#C8C8D0;line-height:1.8;font-size:13px">
              <div style="background:rgba(0,255,65,0.04);border:1px solid rgba(0,255,65,0.12);border-radius:6px;padding:12px 16px;margin-bottom:16px;display:flex;gap:20px;flex-wrap:wrap;font-family:JetBrains Mono,monospace;font-size:11px">
                <span>📂 案件编号: <b style="color:#00FF41">{c.case_id}</b></span>
                <span>🏷️ 类型: <b style="color:#FFD700">{c.case_type}</b></span>
                <span>⭐ 难度: <b style="color:#FF6B6B">{'★'*c.difficulty}{'☆'*(5-c.difficulty)}</b></span>
              </div>
              <div style="background:rgba(20,20,28,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:6px;padding:16px 20px;white-space:pre-wrap;word-break:break-all">{c.case_text}</div>
            </div>'''

            # === Tab2: 案件附件 === 原始证据/嫌疑人/时间线/线索等资料清单 + 图片展示
            files_html = '<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:#B0B0C0">'

            # 🖼️ 图片证据展示 — 前5张，横排布局，充分利用宽度
            if c.images:
                imgs = c.images[:5]
                img_count = len(imgs)
                # 根据图片数量动态调整宽度
                if img_count == 1:
                    col_w = "100%"
                elif img_count == 2:
                    col_w = "49%"
                elif img_count == 3:
                    col_w = "32%"
                elif img_count == 4:
                    col_w = "24%"
                else:
                    col_w = "19%"

                files_html += f'''<div style="background:rgba(20,20,28,0.6);border:1px solid rgba(0,255,65,0.08);border-radius:6px;margin-bottom:12px;overflow:hidden">
                  <div style="background:rgba(0,245,255,0.06);padding:8px 14px;border-bottom:1px solid rgba(255,255,255,0.04);display:flex;justify-content:space-between;align-items:center">
                    <span style="color:#00F5FF;font-weight:700;font-size:12px">🖼️ 图片证据</span>
                    <span style="color:#666;font-size:10px">展示前{img_count}张 / 共{len(c.images)}张</span>
                  </div>
                  <div style="padding:10px;display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-start">'''
                
                import base64
                for i, img in enumerate(imgs):
                    img_path = img.get("path", "")
                    caption = img.get("caption", f"图片{i+1}")
                    
                    # 尝试加载图片为base64
                    img_src = ""
                    if os.path.exists(img_path):
                        try:
                            ext = img_path.rsplit('.', 1)[-1].lower()
                            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")
                            with open(img_path, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode()
                            img_src = f'data:{mime};base64,{b64}'
                        except:
                            img_src = ""
                    
                    if img_src:
                        files_html += f'''<div style="width:{col_w};min-width:120px;background:rgba(6,6,8,0.6);border:1px solid rgba(0,245,255,0.12);border-radius:4px;overflow:hidden;cursor:pointer" title="{caption}">
                          <img src="{img_src}" style="width:100%;height:140px;object-fit:cover;display:block;border-bottom:1px solid rgba(255,255,255,0.04)" alt="{caption}"/>
                          <div style="padding:5px 8px;font-size:9px;color:#8A8A9A;line-height:1.3;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{caption}</div>
                        </div>'''
                    else:
                        files_html += f'''<div style="width:{col_w};min-width:120px;background:rgba(6,6,8,0.6);border:1px solid rgba(0,245,255,0.12);border-radius:4px;overflow:hidden">
                          <div style="width:100%;height:140px;display:flex;align-items:center;justify-content:center;background:rgba(20,20,28,0.8);color:#555;font-size:11px;border-bottom:1px solid rgba(255,255,255,0.04)">🖼️ {caption[:20]}</div>
                          <div style="padding:5px 8px;font-size:9px;color:#8A8A9A;line-height:1.3;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{caption}</div>
                        </div>'''

                files_html += '</div></div>'

            # 文本类资源
            text_items = []

            # 证据材料
            if c.evidence:
                evi_text = '\n'.join([f"  {i+1}. {e}" for i, e in enumerate(c.evidence)])
                text_items.append(("📄 证据材料清单", evi_text, len(c.evidence)))

            # 时间线记录
            if c.timeline:
                tl_text = '\n'.join([f"  {t.get('time','?')}  {t.get('event',str(t))}" if isinstance(t, dict) else f"  {t}" for t in c.timeline])
                text_items.append(("📄 事件时间线记录", tl_text, len(c.timeline)))

            # 关键线索
            if c.key_clues:
                cl_text = '\n'.join([f"  {i+1}. {k}" for i, k in enumerate(c.key_clues)])
                text_items.append(("📄 关键线索记录", cl_text, len(c.key_clues)))

            # 红鲱鱼(误导线索)
            if c.red_herrings:
                rh_text = '\n'.join([f"  {i+1}. {r}" for i, r in enumerate(c.red_herrings)])
                text_items.append(("📄 干扰信息", rh_text, len(c.red_herrings)))

            for title, content, count in text_items:
                files_html += f'''<div style="background:rgba(20,20,28,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:6px;margin-bottom:12px;overflow:hidden">
                  <div style="background:rgba(0,255,65,0.06);padding:8px 14px;border-bottom:1px solid rgba(255,255,255,0.04);display:flex;justify-content:space-between;align-items:center">
                    <span style="color:#00FF41;font-weight:700;font-size:12px">{title}</span>
                    <span style="color:#666;font-size:10px">{count} 条</span>
                  </div>
                  <div style="padding:10px 14px;white-space:pre-wrap;line-height:1.7;font-size:11px">{content}</div>
                </div>'''

            # 嫌疑人档案
            if c.suspects:
                files_html += '<div style="background:rgba(20,20,28,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:6px;margin-bottom:12px;overflow:hidden">'
                files_html += f'<div style="background:rgba(0,255,65,0.06);padding:8px 14px;border-bottom:1px solid rgba(255,255,255,0.04);display:flex;justify-content:space-between;align-items:center"><span style="color:#00FF41;font-weight:700;font-size:12px">📄 嫌疑人档案</span><span style="color:#666;font-size:10px">{len(c.suspects)} 人</span></div>'
                files_html += '<div style="padding:10px 14px">'
                for s in c.suspects:
                    name = s.get("name", "?")
                    files_html += f'''<div style="border-left:3px solid rgba(0,255,65,0.3);padding:8px 12px;margin-bottom:10px;background:rgba(0,255,65,0.02)">
                      <div style="color:#00FF41;font-weight:700;font-size:13px;margin-bottom:4px">👤 {name}</div>
                      <div style="font-size:11px;line-height:1.8;color:#A0A0B0">
                        <div>动机: {s.get('motive','—')}</div>
                        <div>机会: {s.get('opportunity','—')}</div>
                        <div>能力: {s.get('capability','—')}</div>
                        <div>不在场证明: {s.get('alibi','—')}</div>
                      </div>
                    </div>'''
                files_html += '</div></div>'

            files_html += '</div>'

            self._current_case_text = case_text  # 🆕 存储当前案件
            return case_text, doc_html, files_html
        except Exception as e:
            return f"❌ {e}", EMPTY, EMPTY

    # 🆕 v14.0.1: 专家群聊交互 — 支持@提及、多专家、上下文记忆
    def chat_with_experts(self, user_message: str) -> str:
        """用户向专家群发送消息并获取回复（支持@提及特定专家）"""
        if not user_message or not user_message.strip():
            return self.chat_renderer.render_chat_panel({}, {})

        # 0. 添加用户消息（带@标记）
        clean_msg = user_message.strip()
        self.chat_renderer.add_user_message(clean_msg)

        # 1. 解析@提及 → 确定目标专家
        EXPERT_TAGS = {
            "法医": "forensic", "🔬": "forensic",
            "犯罪": "criminal", "🔫": "criminal",
            "心理": "profiler", "🧠": "profiler",
            "技术": "tech", "💻": "tech",
            "金融": "financial", "💰": "financial",
            "审讯": "interrogation", "🗣️": "interrogation",
            "情报": "intelligence", "🕵️": "intelligence",
            "福尔摩斯": "sherlock", "🔍": "sherlock",
            "波洛": "poirot", "🎩": "poirot",
            "宋慈": "song_ci", "📜": "song_ci",
            "李昌钰": "henry_lee", "检察官": "prosecution", "🔴": "prosecution",
            "律师": "defense", "🔵": "defense",
        }
        mentioned = []
        for tag, agent_id in EXPERT_TAGS.items():
            if tag in clean_msg:
                mentioned.append(agent_id)
        # 默认 @all → 所有专家都回复（选前3个关键专家）
        if "@all" in clean_msg or not mentioned:
            mentioned = ["sherlock", "forensic", "profiler"]

        # 2. 构建对话历史上下文
        history_lines = []
        for m in self.chat_renderer._messages[-10:-1]:  # 排除刚加的用户消息
            role = "用户" if m["type"] == "user" else m["agent_id"]
            history_lines.append(f"{role}: {m['message']}")
        history_ctx = "\n".join(history_lines)

        # 3. 案件上下文
        case_ctx = self._current_case_text or "（当前无案件上下文，请基于对话历史分析）"

        # 4. 每个@提及专家调用LLM
        for agent_id in mentioned:
            agent_prompts = {
                "forensic": ("🔬 法医专家", "你是一位资深法医，从医学、毒理学、病理学的专业视角分析案件证据和死因，给出专业判断。"),
                "criminal": ("🔫 犯罪分析师", "你是一位资深犯罪分析师，从犯罪手法、作案工具、行为模式的角度分析案件。"),
                "profiler": ("🧠 心理画像师", "你是一位资深犯罪心理画像师，从心理学、行为科学、动机分析的角度解读嫌疑人。"),
                "tech": ("💻 技术调查员", "你是一位技术调查专家，从电子证据、网络痕迹、数据分析的角度提供线索。"),
                "financial": ("💰 金融调查员", "你是一位金融调查专家，从资金流向、经济犯罪、账目异常的角度分析案件。"),
                "interrogation": ("🗣️ 审讯分析师", "你是一位审讯策略专家，从微表情、供词矛盾、审讯技巧的角度提供建议。"),
                "intelligence": ("🕵️ 情报分析师", "你是一位情报分析专家，从背景调查、人际关系、情报关联的角度分析案件。"),
                "sherlock": ("🔍 福尔摩斯", "你是名侦探福尔摩斯，擅长观察细节、逻辑推理，从蛛丝马迹中还原真相。"),
                "poirot": ("🎩 波洛", "你是侦探波洛，擅长心理分析和灰色脑细胞，从人性弱点切入案件。"),
                "song_ci": ("📜 宋慈", "你是古代法医宋慈，从验尸结论、死亡时间、死因推定的角度分析案件。"),
                "henry_lee": ("🔬 李昌钰", "你是李昌钰博士，享誉国际的鉴识专家，从物证鉴识、实验室分析角度提供专业意见。"),
                "prosecution": ("🔴 检察官", "你是一位资深检察官，从控方立场分析证据链完备性和指控逻辑。"),
                "defense": ("🔵 辩护律师", "你是一位资深辩护律师，从辩方立场审视证据漏洞和合理怀疑。"),
            }
            agent_name, agent_persona = agent_prompts.get(agent_id, ("🤖 专家", "你是一位刑侦专家。"))

            # 从用户消息中提取实际提问（去掉@标签）
            actual_question = clean_msg
            for tag in EXPERT_TAGS:
                actual_question = actual_question.replace(f"@{tag}", "").replace(tag, "")
            actual_question = actual_question.strip()
            if not actual_question:
                actual_question = "基于以上案件信息，请发表你的专业见解。"

            system_prompt = (
                f"角色: {agent_persona}\n\n"
                f"【当前案件】\n{case_ctx}\n\n"
                f"【对话历史（最近10轮）】\n{history_ctx}\n\n"
                f"【当前问题】{actual_question}\n\n"
                f"请以{agent_name}的身份，基于案件信息和对话历史，给出专业、详实但简洁的回答。"
                f"回复格式: 先给出核心观点（1-2句），再简述推理依据。"
                f"注意: 如果案件信息不足，请基于对话上下文合理推断，不要简单回答'信息不足'。"
            )

            try:
                resp = self.llm_client.chat_completion(
                    messages=[{"role": "user", "content": system_prompt}],
                    max_tokens=8192,
                    temperature=0.4,
                    timeout=120,
                )
                # chat_completion returns str, not dict
                reply = resp if isinstance(resp, str) else resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not reply:
                    reply = "（专家暂时无法回答，请稍后重试）"
            except Exception as e:
                reply = f"⚠️ {agent_name}连接异常: {str(e)[:80]}"

            self.chat_renderer.add_agent_response(agent_id, reply.strip())

        return self.chat_renderer.render_chat_panel({}, {})

    def create_interface(self):
        with gr.Blocks(title="🧵 忒修斯之线 — 双线刑侦推理系统", css=POLICE_CSS) as demo:
            # v14.1: Cinematic Splash Screen — 启动片头
            from ui.boot_animation import get_boot_animation_html
            gr.HTML(get_boot_animation_html())

            # JS: 消失动画（立即 + 延迟两种）
            # 点击"进入系统" → 消失动画
            _dismiss_now = """(function(){
              var el = document.getElementById('splash-overlay');
              if (!el) return;
              el.classList.add('splash-out');
              setTimeout(function(){
                el.style.display='none';
                el.style.visibility='hidden';
                // 隐藏进入系统按钮（Gradio生成，class=splash-enter-btn）
                var btn = document.querySelector('.splash-enter-btn');
                if (btn){ btn.style.display='none'; btn.style.visibility='hidden'; }
              }, 920);
            })()"""

            # 进入系统按钮（Gradio原生，渲染在splash上方，z-index自动覆盖）
            enter_btn = gr.Button("🔮 进入系统", elem_classes=["splash-enter-btn"], visible=True)
            enter_btn.click(fn=lambda: None, js=_dismiss_now)

            # Top bar — Cyber-Sleuth theme
            gr.HTML("""<div class="top-bar"><div>
                <h1>🧵 忒修斯之线</h1>
                <span class="subtitle">双线并行推理 · 13专家×4轮深度推理 + 4名侦探 + 动态领域 · NEXUS图谱 · 融合决策</span>
            </div><div><span class="badge">v14.0</span><span class="badge">在线</span></div></div>""")

            # === 案件综合信息区 ===
            with gr.Column(elem_classes=["main-col"]):
                # 案件输入 + 操作按钮
                with gr.Column(elem_classes=["case-card"]):
                    case_input = gr.Textbox(label="📝 案件综合信息",
                        placeholder="输入案件详情：人物、时间线、证据…", lines=5)
                    with gr.Row():
                        analyze_btn = gr.Button("🔍 双线推理", variant="primary", size="lg")
                        staged_btn = gr.Button("🕵️ 时序侦破", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑 清空", variant="secondary")

                # 案件档案 + 运行状态（原侧边栏内容移到这里）
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Column(elem_classes=["panel-flat"]):
                            gr.HTML("<h3 style='color:#00FF41;font-family:Orbitron,monospace;font-size:12px;font-weight:700;margin:0 0 10px 0'>📁 案件档案</h3>")
                            case_list = gr.Dropdown(label="选择案件", choices=self.get_test_case_list(), type="index")
                            load_btn = gr.Button("📂 加载案件", variant="secondary", size="sm")
                    with gr.Column(scale=2):
                        with gr.Column(elem_classes=["panel-flat"]):
                            gr.HTML("<h3 style='color:#00FF41;font-family:Orbitron,monospace;font-size:12px;font-weight:700;margin:0 0 10px 0'>📊 运行状态</h3>")
                            status_text = gr.Textbox(value="⏳ 等待输入…", interactive=False, lines=1, show_label=False)

                # 案件资料展示 — 原始案卷文件
                with gr.Accordion("📂 案件原始资料", open=True):
                    with gr.Tabs():
                        with gr.Tab("📄 案卷文档"):
                            raw_doc_display = gr.HTML(
                                value='<div style="color:#55556A;text-align:center;padding:20px;font-family:JetBrains Mono,monospace;font-size:11px">请先加载案件</div>')
                        with gr.Tab("📁 案件附件"):
                            raw_files_display = gr.HTML(
                                value='<div style="color:#55556A;text-align:center;padding:20px;font-family:JetBrains Mono,monospace;font-size:11px">请先加载案件</div>')

                # === Live reasoning panel ===
                with gr.Row(elem_classes=["force-row"]):
                    with gr.Column(scale=1, min_width=400):
                        live_output = gr.HTML(
                            value='<div class="live-panel"><div class="live-header">🧵 推理线程 <span style="font-size:11px;font-weight:400;color:#55556A">待命</span></div><div class="live-body"><div style="color:#55556A;text-align:center;padding:40px;font-family:JetBrains Mono,monospace;font-size:12px">等待双线推理协议启动…</div></div></div>')
                    with gr.Column(scale=1, min_width=400):
                        brain_palace_output = gr.HTML(
                            value='<div style="background:rgba(6,6,10,0.9);border:1px solid rgba(0,255,65,0.15);border-radius:8px;padding:40px 20px;text-align:center;font-family:JetBrains Mono,monospace;backdrop-filter:blur(12px)"><div style="font-size:28px;margin-bottom:12px">🧠</div><div style="color:#00FF41;font-family:Orbitron,monospace;font-size:14px;letter-spacing:3px;margin-bottom:8px">思维殿堂</div><div style="color:#55556A;font-size:11px">等待推理协议启动…</div></div>')

                # 关系图谱 + 群聊对话框 — 并排展示
                with gr.Row():
                    with gr.Column(scale=3, min_width=500):
                        live_graph = gr.HTML(
                            value='<div style="background:rgba(10,10,14,0.85);border:1px solid rgba(0,255,65,0.2);border-radius:6px;overflow:hidden;backdrop-filter:blur(12px)"><div style="background:linear-gradient(135deg,rgba(0,255,65,0.08),rgba(0,255,65,0.02));color:#00FF41;padding:10px 18px;font-family:Orbitron,monospace;font-weight:700;font-size:13px;letter-spacing:2px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(0,255,65,0.15)">🧵 关联图谱 <span style="font-size:11px;font-weight:400;color:#55556A;font-family:JetBrains Mono,monospace">待命…</span></div><div style="padding:60px;color:#55556A;text-align:center;background:rgba(6,6,8,0.6);font-size:12px;font-family:JetBrains Mono,monospace">图谱将在此处渲染</div></div>')
                    with gr.Column(scale=2, min_width=350):
                        chat_output = gr.HTML(
                            value='<div style="background:rgba(10,10,14,0.85);border:1px solid rgba(0,245,255,0.15);border-radius:6px;overflow:hidden;backdrop-filter:blur(12px)"><div style="background:linear-gradient(135deg,rgba(0,245,255,0.08),rgba(0,245,255,0.02));color:#00F5FF;padding:10px 18px;font-family:Orbitron,monospace;font-weight:700;font-size:13px;letter-spacing:2px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(0,245,255,0.15)">💬 专家群聊 <span style="font-size:11px;font-weight:400;color:#55556A;font-family:JetBrains Mono,monospace">实时对话</span></div><div style="text-align:center;color:#55556A;padding:30px;font-family:JetBrains Mono,monospace;font-size:11px">等待推理启动…</div></div>')
                        # 🆕 v14.0.1: 群聊输入框 + 发送按钮
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="向专家群提问…",
                                lines=1,
                                show_label=False,
                                container=False,
                                elem_id="chat-input",
                                scale=4,
                            )
                            chat_send_btn = gr.Button("📤", variant="primary", scale=1, min_width=40)

                # === Result tabs ===
                with gr.Tabs():
                    with gr.Tab("📋 融合"):
                        fusion_output = gr.Markdown(
                            value='<div style="color:#55556A;text-align:center;padding:30px;font-family:JetBrains Mono,monospace">等待推理完成</div>',
                            elem_classes=["fusion-panel"])
                    with gr.Tab("🃏 专家"):
                        expert_matrix_output = gr.HTML(
                            value='<div style="background:rgba(6,6,10,0.9);border:1px solid rgba(0,245,255,0.15);border-radius:8px;padding:40px 20px;text-align:center;font-family:JetBrains Mono,monospace"><div style="font-size:28px;margin-bottom:12px">🃏</div><div style="color:#00F5FF;font-family:Orbitron,monospace;font-size:14px;letter-spacing:3px;margin-bottom:8px">专家推理矩阵</div><div style="color:#55556A;font-size:11px">等待Agent部署…</div></div>')
                    with gr.Tab("🔍 传统"):
                        trad_output = gr.Markdown(
                            value='<div style="color:#55556A;text-align:center;padding:30px;font-family:JetBrains Mono,monospace">等待中</div>')
                    with gr.Tab("🔬 ASMR"):
                        asmr_output = gr.Markdown(
                            value='<div style="color:#55556A;text-align:center;padding:30px;font-family:JetBrains Mono,monospace">等待中</div>')
                    with gr.Tab("🧵 关联"):
                        graph_output = gr.HTML(
                            value='<div style="color:#55556A;text-align:center;padding:40px;font-family:JetBrains Mono,monospace">等待推理完成</div>')
                    with gr.Tab("📊 结果"):
                        load_results_btn = gr.Button("🔄 加载结果", variant="secondary")
                        results_output = gr.Markdown()
                    with gr.Tab("📋 JSON"):
                        json_output = gr.Code(language="json", value="{}")
                    with gr.Tab("🕵️ 时序"):
                        gr.HTML("<div style='color:#55556A;padding:8px;font-size:11px;font-family:JetBrains Mono,monospace'>时序侦破: 证据释放 → 推理 → 调查建议 → 假设预测</div>")
                        staged_timeline = gr.HTML(
                            value='<div style="color:#55556A;text-align:center;padding:40px;font-family:JetBrains Mono,monospace;font-size:12px">等待时序侦破启动…</div>')
                        staged_chart = gr.HTML(
                            value='<div style="color:#55556A;text-align:center;padding:40px;font-family:JetBrains Mono,monospace;font-size:12px">嫌疑人演化图…</div>')
                        staged_summary = gr.Markdown(
                            value='<div style="color:#55556A;text-align:center;padding:30px;font-family:JetBrains Mono,monospace">等待扫描</div>')
                        staged_status = gr.Textbox(value="⏳ 等待输入…", interactive=False, lines=1, show_label=False)

            # Examples
            with gr.Accordion("📚 案例库", open=False):
                gr.Examples(
                    examples=[
                        ["2026年3月15日晚，某别墅发生命案。受害者张某，男，45岁，企业家。案发时别墅内有四人：妻子李某、合作伙伴王某、秘书赵某、管家王福来。现场发现凶器为水果刀，刀柄上有模糊指纹。监控显示王某曾在案发时间段进出别墅。管家王福来声称一直在厨房准备晚餐，但厨房刀具少了一把。"],
                    ],
                    inputs=case_input, label="快速输入")

            # === Events ===
            # 加载案件: 同时更新文本 + 原始案卷资料
            load_btn.click(
                fn=self.load_case_resources,
                inputs=[case_list],
                outputs=[case_input, raw_doc_display, raw_files_display],
            )

            analyze_btn.click(
                fn=lambda: ("⏳ 分析中…",),
                outputs=[status_text],
            ).then(
                fn=self.stream_analyze,
                inputs=[case_input],
                outputs=[live_output, live_graph, brain_palace_output, expert_matrix_output, trad_output, asmr_output, fusion_output, graph_output, json_output, status_text, chat_output],
            )

            clear_btn.click(
                fn=lambda: (
                    "",
                    '<div class="live-panel"><div class="live-header">🧵 推理线程 <span style="font-size:11px;font-weight:400;color:#55556A">待命</span></div><div class="live-body"><div style="color:#55556A;text-align:center;padding:40px;font-family:JetBrains Mono,monospace;font-size:12px">等待双线推理协议启动…</div></div></div>',
                    '<div style="background:rgba(6,6,10,0.9);border:1px solid rgba(0,255,65,0.15);border-radius:8px;padding:40px 20px;text-align:center;font-family:JetBrains Mono,monospace;backdrop-filter:blur(12px)"><div style="font-size:28px;margin-bottom:12px">🧠</div><div style="color:#00FF41;font-family:Orbitron,monospace;font-size:14px;letter-spacing:3px;margin-bottom:8px">思维殿堂</div><div style="color:#55556A;font-size:11px">等待推理协议启动…</div></div>',
                    '<div style="background:rgba(6,6,10,0.9);border:1px solid rgba(0,245,255,0.15);border-radius:8px;padding:40px 20px;text-align:center;font-family:JetBrains Mono,monospace"><div style="font-size:28px;margin-bottom:12px">🃏</div><div style="color:#00F5FF;font-family:Orbitron,monospace;font-size:14px;letter-spacing:3px;margin-bottom:8px">专家推理矩阵</div><div style="color:#55556A;font-size:11px">等待Agent部署…</div></div>',
                    '<div style="color:#55556A;text-align:center;padding:30px;font-family:JetBrains Mono,monospace">等待中</div>',
                    '<div style="color:#55556A;text-align:center;padding:30px;font-family:JetBrains Mono,monospace">等待中</div>',
                    '<div style="color:#55556A;text-align:center;padding:30px;font-family:JetBrains Mono,monospace">等待推理完成</div>',
                    '<div style="color:#55556A;text-align:center;padding:40px;font-family:JetBrains Mono,monospace">等待推理完成</div>',
                    "{}",
                    "⏳ 等待输入…",
                    '<div style="text-align:center;color:#55556A;padding:30px;font-family:JetBrains Mono,monospace;font-size:11px">等待推理启动…</div>',
                    '<div style="color:#55556A;text-align:center;padding:40px;font-family:JetBrains Mono,monospace;font-size:12px">等待时序侦破启动…</div>',
                    '<div style="color:#55556A;text-align:center;padding:40px;font-family:JetBrains Mono,monospace;font-size:12px">嫌疑人演化图…</div>',
                    '<div style="color:#55556A;text-align:center;padding:30px;font-family:JetBrains Mono,monospace">等待扫描</div>',
                    "⏳ 等待输入…",
                ),
                outputs=[case_input, live_output, live_graph, brain_palace_output, expert_matrix_output, trad_output, asmr_output, fusion_output, graph_output, json_output, status_text, chat_output, staged_timeline, staged_chart, staged_summary, staged_status],
            )

            staged_btn.click(
                fn=lambda: ("🔄 时序扫描中…",),
                outputs=[staged_status],
            ).then(
                fn=self.stream_staged_analyze,
                inputs=[case_input],
                outputs=[staged_timeline, staged_chart, staged_summary, staged_status],
            )

            load_results_btn.click(fn=load_test_results, outputs=[results_output])

            # 🆕 v14.0.1: 专家群聊发送按钮 — 回车或点击都触发
            chat_send_btn.click(
                fn=self.chat_with_experts,
                inputs=[chat_input],
                outputs=[chat_output],
            )
            # 回车发送（Shift+Enter换行）
            chat_input.submit(
                fn=self.chat_with_experts,
                inputs=[chat_input],
                outputs=[chat_output],
            )

        return demo

    def launch(self):
        demo = self.create_interface()
        host = self.config.get("webui", {}).get("host", "0.0.0.0")
        port = self.config.get("webui", {}).get("port", 7860)
        self.logger.info(f"WebUI启动: http://{host}:{port}")
        # Boot animation JS — DISABLED: Gradio 6 head= param causes black screen
        # from ui.boot_animation import get_boot_animation_js
        # boot_js = get_boot_animation_js()
        demo.launch(server_name=host, server_port=port, share=False, show_error=True,
                    css=POLICE_CSS)


if __name__ == "__main__":
    webui = DetectiveWebUI()
    webui.launch()
