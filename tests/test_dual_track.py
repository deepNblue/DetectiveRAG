"""
双线融合测试 — 传统RAG + ASMR并行，融合出最终结论
"""

import sys
import os
import time
import json
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# os.environ["ZHIPUAI_API_KEY"] — 使用环境变量，不再硬编码

from api.llm_client import LLMClient
from agents.asmr.orchestrator import ASMROrchestrator
from agents.asmr.dual_track_fusion import DualTrackFusionEngine
from agents.clue_extractor import ClueExtractorAgent
from agents.suspect_analyzer import SuspectAnalyzerAgent
from agents.evidence_connector import EvidenceConnectorAgent
from agents.reasoning_generator import ReasoningGeneratorAgent
from agents.graph_builder import GraphBuilderAgent
from agents.graph_reasoner import GraphReasonerAgent
from tests.test_detective_cases import get_all_test_cases

from concurrent.futures import ThreadPoolExecutor


def run_traditional_track(llm, case):
    """运行传统RAG线路（5步流水线）"""
    print("  🔵 传统线路启动...")
    t0 = time.time()
    results = {}

    # Step 1: 提取线索
    extractor = ClueExtractorAgent(llm_client=llm)
    clues = extractor.process({
        "case_text": case.case_text,
        "images": [],
        "case_type": case.case_type
    })
    results["clues"] = clues
    print(f"    Step 1 线索提取: {time.time()-t0:.1f}s")

    # Step 2: 分析嫌疑人
    analyzer = SuspectAnalyzerAgent(llm_client=llm)
    analyses = []
    for s in case.suspects:
        a = analyzer.process({
            "suspect_info": s,
            "case_clues": clues["data"],
            "evidence": case.evidence
        })
        analyses.append(a)
    results["suspect_analyses"] = analyses
    print(f"    Step 2 嫌疑人分析: {time.time()-t0:.1f}s")

    # Step 3: 证据关联
    connector = EvidenceConnectorAgent(llm_client=llm)
    evidence_result = connector.process({
        "evidence_list": case.evidence,
        "suspects": case.suspects,
        "case_clues": clues["data"]
    })
    results["evidence_connections"] = evidence_result
    print(f"    Step 3 证据关联: {time.time()-t0:.1f}s")

    # Step 4: 图谱构建（提前到推理之前）
    builder = GraphBuilderAgent(llm_client=llm)
    graph = builder.process({
        "case_clues": clues["data"],
        "suspects": case.suspects,
        "evidence": case.evidence,
    })
    results["graph"] = graph
    print(f"    Step 4 图谱构建: {time.time()-t0:.1f}s")

    # Step 4.5: 图谱推理（新增！基于图谱结构做关系推理）
    reasoner = GraphReasonerAgent(llm_client=llm)
    graph_reasoning = reasoner.process({
        "graph": graph.get("data", graph),
        "case_clues": clues["data"],
        "suspect_analyses": analyses,
        "evidence_connections": evidence_result.get("data", {}),
    })
    results["graph_reasoning"] = graph_reasoning
    print(f"    Step 4.5 图谱推理: {time.time()-t0:.1f}s")

    # Step 5: 图谱增强推理（使用图谱推理结果作为额外上下文）
    generator = ReasoningGeneratorAgent(llm_client=llm)
    reasoning = generator.process({
        "case_clues": clues["data"],
        "suspect_analyses": analyses,
        "evidence_connections": evidence_result["data"],
        "graph_reasoning": graph_reasoning.get("data", {}),
    })
    results["reasoning"] = reasoning
    results["track_time"] = round(time.time() - t0, 1)
    print(f"    Step 5 图谱增强推理: {results['track_time']}s")

    return results


def run_asmr_track(llm, case):
    """运行ASMR线路"""
    print("  🟢 ASMR线路启动...")
    t0 = time.time()
    orchestrator = ASMROrchestrator(llm_client=llm)
    result = orchestrator.run(
        case_text=case.case_text,
        suspects=case.suspects,
        case_type=case.case_type,
    )
    result["track_time"] = result["timing"]["total"]
    print(f"    ASMR完成: {result['track_time']}s")
    return result


def main():
    print("=" * 60)
    print("  🔀 双线融合测试 — 传统RAG + ASMR")
    print("=" * 60)

    # 初始化LLM
    print("\n[1] 初始化LLM...")
    llm = LLMClient()
    resp = llm.simple_chat("请回复OK", temperature=0.1)
    print(f"  ✅ LLM: {resp[:30]}")

    # 加载案件
    print("\n[2] 加载案件...")
    cases = get_all_test_cases()
    case = cases[0]
    print(f"  案件: {case.case_type}, 真凶: {case.expected_result['culprit']}")

    # 双线并行运行
    print("\n[3] 双线并行启动!")
    print("-" * 60)
    dual_start = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        trad_future = executor.submit(run_traditional_track, llm, case)
        asmr_future = executor.submit(run_asmr_track, llm, case)
        trad_result = trad_future.result()
        asmr_result = asmr_future.result()

    dual_time = time.time() - dual_start
    print(f"\n  ⏱️ 双线并行总耗时: {dual_time:.1f}s (传统{trad_result['track_time']}s | ASMR{asmr_result['track_time']}s)")

    # 双线融合
    print("\n[4] 双线融合...")
    fusion_engine = DualTrackFusionEngine()
    fusion_result = fusion_engine.fuse(
        traditional_result=trad_result,
        asmr_result=asmr_result,
    )
    report = fusion_engine.generate_report(fusion_result, asmr_result)

    # 输出结果
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # 对比评估
    trad_culprit = fusion_result["traditional_contribution"]["culprit"]
    asmr_culprit = fusion_result["asmr_contribution"]["culprit"]
    final_culprit = fusion_result["conclusion"]["culprit"]
    expected = case.expected_result["culprit"]

    trad_match = any(k in trad_culprit for k in ["王福来", "管家"]) if trad_culprit != "未知" else False
    asmr_match = any(k in asmr_culprit for k in ["王福来", "管家"])
    final_match = any(k in final_culprit for k in ["王福来", "管家"])

    print(f"\n📊 对比评估:")
    print(f"  传统RAG: {trad_culprit} {'✅' if trad_match else '❌'}")
    print(f"  ASMR:    {asmr_culprit} {'✅' if asmr_match else '❌'} (置信度{asmr_result['conclusion']['confidence']:.2%})")
    print(f"  融合:    {final_culprit} {'✅' if final_match else '❌'} (置信度{fusion_result['conclusion']['confidence']:.2%})")
    print(f"  一致性:  {fusion_result['agreement']['status']}")

    # 保存结果
    save = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "test_type": "dual_track_fusion",
        "case_id": case.case_id,
        "expected_culprit": expected,
        "traditional": {
            "culprit": trad_culprit,
            "confidence": fusion_result["traditional_contribution"]["confidence"],
            "time": trad_result["track_time"],
            "match": trad_match,
        },
        "asmr": {
            "culprit": asmr_culprit,
            "confidence": asmr_result["conclusion"]["confidence"],
            "consensus": asmr_result["conclusion"]["consensus_level"],
            "time": asmr_result["track_time"],
            "expert_analyses": asmr_result["expert_analyses"],
            "match": asmr_match,
        },
        "fusion": {
            "culprit": final_culprit,
            "confidence": fusion_result["conclusion"]["confidence"],
            "certainty": fusion_result["conclusion"]["certainty"],
            "agreement": fusion_result["agreement"]["status"],
            "suspect_ranking": fusion_result["fused_suspects"]["ranking"],
        },
        "timing": {
            "dual_parallel": round(dual_time, 1),
            "traditional": trad_result["track_time"],
            "asmr": asmr_result["track_time"],
            "fusion": fusion_result["timing"]["fusion_time"],
        },
        "match": final_match,
    }

    report_path = os.path.join(PROJECT_ROOT, "data", "test_results", "dual_track_result.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    print(f"\n  📁 结果已保存: {report_path}")

    return final_match


if __name__ == "__main__":
    try:
        success = main()
        print(f"\n{'='*60}")
        print(f"  双线融合测试: {'✅ 通过' if success else '⚠️ 需检查'}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()
