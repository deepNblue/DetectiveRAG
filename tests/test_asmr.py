"""
ASMR多智能体线路测试
测试密室杀人案的完整ASMR三阶段流水线
"""

import sys
import os
import time
import json
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

os.environ["ZHIPUAI_API_KEY"] = "REDACTED_ZHIPUAI_KEY"

from api.llm_client import LLMClient
from agents.asmr.orchestrator import ASMROrchestrator
from agents.asmr.dual_track_fusion import DualTrackFusionEngine
from tests.test_detective_cases import get_all_test_cases


def main():
    print("=" * 60)
    print("  🧪 ASMR多智能体线路测试 - 密室杀人案")
    print("=" * 60)
    
    # 初始化LLM
    print("\n[1] 初始化LLM客户端...")
    llm = LLMClient()
    resp = llm.simple_chat("请回复OK", temperature=0.1)
    print(f"  ✅ LLM连接正常: {resp[:50]}")
    
    # 加载案件
    print("\n[2] 加载测试案件...")
    cases = get_all_test_cases()
    case = cases[0]  # 密室杀人案
    print(f"  案件: {case.case_type} (难度{'★'*case.difficulty})")
    print(f"  真凶: {case.expected_result['culprit']}")
    print(f"  嫌疑人: {len(case.suspects)}人")
    
    # 运行ASMR流水线
    print("\n[3] 启动ASMR三阶段流水线...")
    print("-" * 60)
    
    orchestrator = ASMROrchestrator(llm_client=llm)
    
    total_start = time.time()
    result = orchestrator.run(
        case_text=case.case_text,
        suspects=case.suspects,
        case_type=case.case_type,
    )
    total_time = time.time() - total_start
    
    # 输出结果
    print("\n" + "=" * 60)
    print("  📊 ASMR测试结果")
    print("=" * 60)
    
    conclusion = result["conclusion"]
    print(f"\n  🎯 ASMR结论: 真凶={conclusion['culprit']}, 置信度={conclusion['confidence']:.2%}")
    print(f"  🤝 共识程度: {conclusion['consensus_level']}")
    
    expected_culprit = case.expected_result["culprit"]
    match = any(k in conclusion["culprit"] for k in ["王福来", "管家"])
    print(f"  ✅ 与期望结果对比: {'匹配' if match else '未匹配'} (期望: {expected_culprit})")
    
    # Stage耗时
    timing = result["timing"]
    print(f"\n  ⏱️ 耗时统计:")
    print(f"     Stage 1 (Readers):    {timing['stage1_readers']}s")
    print(f"     Stage 2 (Searchers):  {timing['stage2_searchers']}s")
    print(f"     Stage 3 (Experts):    {timing['stage3_experts']}s")
    print(f"     总计:                  {timing['total']}s")
    
    # 专家分析摘要
    print(f"\n  🧠 专家分析:")
    for ea in result["expert_analyses"]:
        print(f"     {ea['perspective']}: 真凶={ea['culprit']}, 置信度={ea['confidence']:.2f}")
    
    # 投票报告
    print(f"\n  🗳️ 投票报告:")
    print(result["vote_report"])
    
    # 搜索结果摘要
    print(f"\n  🔍 搜索排名:")
    sr = result["search_results"]
    print(f"     动机排名: {sr['motive_ranking']}")
    print(f"     机会排名: {sr['opportunity_ranking']}")
    print(f"     能力排名: {sr['capability_ranking']}")
    print(f"     时间洞察: {sr['temporal_insight'][:100] if sr['temporal_insight'] else '无'}...")
    
    # 错误检查
    if result["errors"]:
        print(f"\n  ⚠️ 错误: {result['errors']}")
    else:
        print(f"\n  ✅ 无错误")
    
    # 保存结果
    report_dir = os.path.join(PROJECT_ROOT, "data", "test_results")
    os.makedirs(report_dir, exist_ok=True)
    
    # 简化保存（去除嵌套data中的大结构）
    save_result = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "test_type": "ASMR_only",
        "case_id": case.case_id,
        "case_type": case.case_type,
        "expected_culprit": expected_culprit,
        "conclusion": conclusion,
        "expert_analyses": result["expert_analyses"],
        "vote_result": {
            "winner": result["vote_result"]["winner"],
            "confidence": result["vote_result"]["confidence"],
            "consensus_level": result["vote_result"]["consensus_level"],
            "vote_distribution": result["vote_result"]["vote_distribution"],
        },
        "search_results": sr,
        "timing": timing,
        "errors": result["errors"],
        "match": match,
        "total_time": total_time,
    }
    
    report_path = os.path.join(report_dir, "asmr_test_result.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(save_result, f, ensure_ascii=False, indent=2)
    print(f"\n  📁 结果已保存: {report_path}")
    
    return match


if __name__ == "__main__":
    try:
        success = main()
        print(f"\n{'='*60}")
        print(f"  测试结果: {'✅ 通过' if success else '⚠️ 需检查'}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()
