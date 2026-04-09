#!/usr/bin/env python3
"""多案件双线测试 - 8个案件全部跑一遍"""
import sys, os, time, json, traceback
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
os.environ.setdefault("ZHIPUAI_API_KEY", "")  # 从环境变量读取，请勿硬编码

from api.llm_client import LLMClient
from agents.clue_extractor import ClueExtractorAgent
from agents.suspect_analyzer import SuspectAnalyzerAgent
from agents.evidence_connector import EvidenceConnectorAgent
from agents.graph_builder import GraphBuilderAgent
from agents.graph_reasoner import GraphReasonerAgent
from agents.reasoning_generator import ReasoningGeneratorAgent
from agents.asmr.orchestrator import ASMROrchestrator
from agents.asmr.dual_track_fusion import DualTrackFusionEngine
from tests.test_detective_cases import get_all_test_cases


def run_traditional(case, llm):
    """传统线路: 6步Pipeline"""
    t0 = time.time()

    # Step 1: Clue extraction
    clue_agent = ClueExtractorAgent(llm_client=llm)
    clues = clue_agent.process({
        "case_text": case.case_text,
        "images": [],
        "case_type": case.case_type
    })
    clue_data = clues.get("data", clues)
    print(f"    Step 1 线索提取: {time.time()-t0:.1f}s")

    # Step 2: Suspect analysis (one at a time)
    suspect_agent = SuspectAnalyzerAgent(llm_client=llm)
    suspect_analyses = []
    for s in case.suspects:
        a = suspect_agent.process({
            "suspect_info": s,
            "case_clues": clue_data,
            "evidence": case.evidence
        })
        suspect_analyses.append(a)
    print(f"    Step 2 嫌疑人分析 ({len(suspect_analyses)}人): {time.time()-t0:.1f}s")

    # Step 3: Evidence connection
    evidence_agent = EvidenceConnectorAgent(llm_client=llm)
    evidence = evidence_agent.process({
        "evidence_list": case.evidence,
        "suspects": case.suspects,
        "case_clues": clue_data
    })
    evidence_data = evidence.get("data", evidence)
    print(f"    Step 3 证据关联: {time.time()-t0:.1f}s")

    # Step 4: Graph building
    graph_agent = GraphBuilderAgent(llm_client=llm)
    graph = graph_agent.process({
        "case_clues": clue_data,
        "suspects": case.suspects,
        "evidence": case.evidence,
    })
    graph_data = graph.get("data", graph)
    print(f"    Step 4 图谱构建: {time.time()-t0:.1f}s")

    # Step 4.5: Graph reasoning
    graph_reasoner = GraphReasonerAgent(llm_client=llm)
    graph_reasoning = graph_reasoner.process({
        "graph": graph_data,
        "case_clues": clue_data,
        "suspect_analyses": suspect_analyses,
        "evidence_connections": evidence_data
    })
    graph_reasoning_data = graph_reasoning.get("data", graph_reasoning)
    print(f"    Step 4.5 图谱推理: {time.time()-t0:.1f}s")

    # Step 5: Reasoning (with graph reasoning input)
    reasoning_agent = ReasoningGeneratorAgent(llm_client=llm)
    reasoning = reasoning_agent.process({
        "case_clues": clue_data,
        "suspect_analyses": suspect_analyses,
        "evidence_connections": evidence_data,
        "graph_reasoning": graph_reasoning_data
    })
    reasoning_data = reasoning.get("data", reasoning)
    print(f"    Step 5 推理: {time.time()-t0:.1f}s")

    # Extract culprit from reasoning
    culprit = "未知"
    confidence = 0

    # Try final_conclusion first
    final = reasoning_data.get("final_conclusion", {})
    if isinstance(final, dict):
        culprit = final.get("top_suspect", final.get("culprit", "未知"))
        confidence = final.get("confidence", 0)

    # Fallback: check suspect_ranking
    if culprit == "未知" and "suspect_ranking" in final:
        ranking = final["suspect_ranking"]
        if ranking and isinstance(ranking, list) and len(ranking) > 0:
            top = ranking[0]
            culprit = top.get("name", "未知")
            confidence = top.get("score", 0)

    # Fallback: confidence_score
    if confidence == 0:
        confidence = reasoning_data.get("confidence_score", 0)

    trad_time = time.time() - t0
    return {
        "culprit": culprit,
        "confidence": confidence,
        "time": round(trad_time, 1),
    }


def run_asmr(case, llm):
    """ASMR线路"""
    t0 = time.time()
    orchestrator = ASMROrchestrator(llm_client=llm)
    result = orchestrator.run(
        case_text=case.case_text,
        suspects=[{"name": s["name"], "motive": s.get("motive",""),
                   "opportunity": s.get("opportunity",""), "capability": s.get("capability","")}
                  for s in case.suspects],
        case_type=case.case_type,
    )
    asmr_time = time.time() - t0

    conclusion = result.get("conclusion", {})
    culprit = conclusion.get("culprit", "未知")
    confidence = conclusion.get("confidence", 0)
    consensus = conclusion.get("consensus_level", "无")

    return {
        "culprit": culprit,
        "confidence": confidence,
        "consensus": consensus,
        "time": round(asmr_time, 1),
    }


def main():
    cases = get_all_test_cases()
    llm = LLMClient()
    fusion = DualTrackFusionEngine()

    # Verify LLM
    print("验证LLM连接...")
    resp = llm.simple_chat("请回复OK", temperature=0.1)
    print(f"  LLM响应: {resp[:50]}")

    results = []

    for i, case in enumerate(cases):
        print(f"\n{'='*60}")
        print(f"案件 {i+1}/{len(cases)}: {case.case_id} - {case.case_type} (难度{'★'*case.difficulty})")
        print(f"真凶: {case.expected_result.get('culprit', '?')}")
        print(f"{'='*60}")

        expected = case.expected_result.get("culprit", "?")

        # Run traditional
        print("  🔵 传统线路启动...")
        try:
            trad = run_traditional(case, llm)
            trad_ok = "ok"
        except Exception as e:
            trad = {"culprit": "ERROR", "confidence": 0, "time": 0}
            trad_ok = str(e)[:200]
            traceback.print_exc()

        # Run ASMR
        print("  🟢 ASMR线路启动...")
        try:
            asmr = run_asmr(case, llm)
            asmr_ok = "ok"
        except Exception as e:
            asmr = {"culprit": "ERROR", "confidence": 0, "consensus": "error", "time": 0}
            asmr_ok = str(e)[:200]
            traceback.print_exc()

        # Fusion
        try:
            fusion_result = fusion.fuse(
                traditional_result={
                    "culprit": trad["culprit"], "confidence": trad["confidence"]
                },
                asmr_result={
                    "culprit": asmr["culprit"], "confidence": asmr["confidence"],
                    "consensus_level": asmr.get("consensus", "无"),
                }
            )
            fusion_culprit = fusion_result.get("conclusion", {}).get("culprit", "?")
            fusion_conf = fusion_result.get("conclusion", {}).get("confidence", 0)
        except Exception as e:
            fusion_culprit = "ERROR"
            fusion_conf = 0
            fusion_result = {"conclusion": {"culprit": "ERROR", "confidence": 0}}

        # Check match — split expected by "+" for multi-culprit cases
        expected_names = [n.strip() for n in expected.split("+")]
        trad_match = any(any(n in trad["culprit"] for n in expected_names) for _ in [1]) and trad["culprit"] != "ERROR"
        asmr_match = any(any(n in asmr["culprit"] for n in expected_names) for _ in [1]) and asmr["culprit"] != "ERROR"
        fusion_match = any(any(n in fusion_culprit for n in expected_names) for _ in [1]) and fusion_culprit != "ERROR"

        # Also check if at least one expected name appears
        if not trad_match:
            trad_match = any(n in trad["culprit"] for n in expected_names) and trad["culprit"] != "未知"
        if not asmr_match:
            asmr_match = any(n in asmr["culprit"] for n in expected_names) and asmr["culprit"] != "未知"
        if not fusion_match:
            fusion_match = any(n in fusion_culprit for n in expected_names) and fusion_culprit != "未知"

        r = {
            "case_id": case.case_id,
            "type": case.case_type,
            "difficulty": case.difficulty,
            "expected": expected,
            "traditional": {
                "culprit": trad["culprit"], "confidence": round(trad["confidence"], 3),
                "time": trad["time"], "match": trad_match, "error": trad_ok
            },
            "asmr": {
                "culprit": asmr["culprit"], "confidence": round(asmr["confidence"], 3),
                "time": asmr["time"], "match": asmr_match, "error": asmr_ok,
                "consensus": asmr.get("consensus", ""),
            },
            "fusion": {
                "culprit": fusion_culprit, "confidence": round(fusion_conf, 3),
                "match": fusion_match,
            },
        }
        results.append(r)

        print(f"\n  📊 结果:")
        print(f"    传统: {trad['culprit']} (c={trad['confidence']:.3f}, t={trad['time']:.0f}s) {'✅' if trad_match else '❌'}")
        print(f"    ASMR: {asmr['culprit']} (c={asmr['confidence']:.3f}, t={asmr['time']:.0f}s, {asmr.get('consensus','')}) {'✅' if asmr_match else '❌'}")
        print(f"    融合: {fusion_culprit} (c={fusion_conf:.3f}) {'✅' if fusion_match else '❌'}")

        # Save intermediate after each case
        save_path = os.path.join(PROJECT_ROOT, "data", "test_results", "multi_case_results.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    trad_wins = sum(1 for r in results if r["traditional"]["match"])
    asmr_wins = sum(1 for r in results if r["asmr"]["match"])
    fusion_wins = sum(1 for r in results if r["fusion"]["match"])

    total = len(results)
    trad_times = [r["traditional"]["time"] for r in results if r["traditional"]["time"] > 0]
    asmr_times = [r["asmr"]["time"] for r in results if r["asmr"]["time"] > 0]

    summary = {
        "total": total,
        "trad_wins": trad_wins,
        "asmr_wins": asmr_wins,
        "fusion_wins": fusion_wins,
        "avg_trad_time": round(sum(trad_times)/len(trad_times), 1) if trad_times else 0,
        "avg_asmr_time": round(sum(asmr_times)/len(asmr_times), 1) if asmr_times else 0,
    }

    print(f"\n{'='*60}")
    print(f"📊 总计: {total}个案件")
    print(f"  传统线路: {trad_wins}/{total} 正确 ({trad_wins/total*100:.0f}%)")
    print(f"  ASMR线路: {asmr_wins}/{total} 正确 ({asmr_wins/total*100:.0f}%)")
    print(f"  融合结果: {fusion_wins}/{total} 正确 ({fusion_wins/total*100:.0f}%)")
    if trad_times:
        print(f"  平均耗时: 传统={summary['avg_trad_time']}s, ASMR={summary['avg_asmr_time']}s")
    print(f"{'='*60}")

    # Per-case detail
    print("\n详细结果:")
    for r in results:
        t = "✅" if r["traditional"]["match"] else "❌"
        a = "✅" if r["asmr"]["match"] else "❌"
        f = "✅" if r["fusion"]["match"] else "❌"
        print(f"  {r['case_id']} ({r['type']}, D{r['difficulty']}): "
              f"传统{t} ASMR{a} 融合{f} | "
              f"真凶: {r['expected'][:20]}")

    # Save final with summary
    save_path = os.path.join(PROJECT_ROOT, "data", "test_results", "multi_case_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n📁 结果已保存: {save_path}")


if __name__ == "__main__":
    main()
