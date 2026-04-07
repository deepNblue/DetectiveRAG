#!/usr/bin/env python3
"""
CASE-002 端到端测试 — v14.0 多轮推理(10轮上限) + 对话持久化 + 图谱着色
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tests.test_detective_cases import get_all_test_cases
from agents.asmr.orchestrator import ASMROrchestrator
from api.llm_client import LLMClient
from ui.conversation_store import ConversationStore
from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

def main():
    cases = get_all_test_cases()
    case = cases[1]  # CASE-002
    print(f"\n{'='*60}")
    print(f"🎯 CASE-002 端到端测试 — v14.0")
    print(f"   类型: {case.case_type} | 难度: {'★'*case.difficulty}")
    print(f"   预期: {case.expected_result.get('culprit', '?')}")
    print(f"   嫌疑人: {len(case.suspects)}人")
    print(f"{'='*60}\n")
    
    # 初始化
    llm = LLMClient()
    conv_store = ConversationStore()
    conv_store.start_case(case.case_id, case.case_text[:100])
    
    orch = ASMROrchestrator(
        llm_client=llm,
        max_workers=4,
        conversation_store=conv_store,
    )
    
    # 运行
    t0 = time.time()
    result = orch.run(
        case_text=case.case_text,
        suspects=case.suspects,
        case_type=case.case_type,
        images=case.images,
    )
    elapsed = time.time() - t0
    
    # 结果
    conclusion = result.get("conclusion", {})
    culprit = conclusion.get("culprit", "?")
    confidence = conclusion.get("confidence", 0)
    timing = result.get("timing", {})
    expert_analyses = result.get("expert_analyses", [])
    three_layer = result.get("three_layer", {})
    
    # 保存到对话存储
    conv_store.save_asmr_line(result)
    conv_store.finish_case({
        "predicted": culprit,
        "expected": case.expected_result.get("culprit", "?"),
        "confidence": confidence,
        "timing": timing,
    })
    
    # 输出
    print(f"\n{'='*60}")
    print(f"🏁 CASE-002 测试完成!")
    print(f"{'='*60}")
    print(f"⏱️ 总耗时: {elapsed:.1f}s")
    print(f"   Stage 0 (Vision): {timing.get('stage0_vision', 0):.1f}s")
    print(f"   Stage 1 (Readers): {timing.get('stage1_readers', 0):.1f}s")
    print(f"   Stage 2 (Searchers): {timing.get('stage2_searchers', 0):.1f}s")
    print(f"   Stage 3.1 (Investigation): {timing.get('stage31_investigation', 0):.1f}s")
    print(f"   Stage 3.2 (Trial): {timing.get('stage32_trial', 0):.1f}s")
    print(f"   Stage 3.3 (Elimination): {timing.get('stage33_elimination', 0):.1f}s")
    print(f"   Stage 3.5 (Tree): {timing.get('stage35_reasoning_tree', 0):.1f}s")
    print(f"   Stage 4 (Adjudicator): {timing.get('stage4_adjudicator', 0):.1f}s")
    
    print(f"\n🎯 预测: {culprit} ({confidence:.1%})")
    print(f"   预期: {case.expected_result.get('culprit', '?')}")
    
    # 检查多轮推理
    mr_count = 0
    max_rounds_seen = 0
    for ea in expert_analyses:
        mr = ea.get("multi_round", {})
        if isinstance(mr, dict) and mr.get("total_rounds", 0) > 1:
            mr_count += 1
            max_rounds_seen = max(max_rounds_seen, mr["total_rounds"])
    print(f"\n🔄 多轮推理: {mr_count}个专家使用, 最高{max_rounds_seen}轮 (上限=10)")
    
    # 三层架构
    inv = three_layer.get("investigation_vote", {})
    trial = three_layer.get("trial_vote", {})
    print(f"\n📊 三层架构:")
    print(f"   调查层: {inv.get('winner', '?')} ({inv.get('confidence', 0):.1%})")
    print(f"   审判层: {trial.get('winner', '?')} ({trial.get('confidence', 0):.1%})")
    
    # 推翻检查
    if conclusion.get("overturned"):
        print(f"   ⚡ 裁判推翻! {conclusion.get('vote_winner', '?')} → {culprit}")
    
    # 专家投票分布
    print(f"\n🗳️ 专家投票 (共{len(expert_analyses)}个):")
    from collections import Counter
    votes = Counter()
    for ea in expert_analyses:
        c = ea.get("culprit", "?")
        votes[c] += 1
    for name, count in votes.most_common():
        print(f"   {name}: {count}票 ({count/len(expert_analyses)*100:.0f}%)")
    
    # 对话存储检查
    conv_dir = conv_store.current_dir
    if conv_dir:
        print(f"\n💾 对话存储: {conv_dir}")
        for f in os.listdir(conv_dir):
            fp = os.path.join(conv_dir, f)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                print(f"   📄 {f} ({size}B)")
            elif os.path.isdir(fp):
                files = os.listdir(fp)
                print(f"   📁 {f}/ ({len(files)}文件)")
    
    # 结果评估
    expected = case.expected_result.get("culprit", "")
    # 宽松匹配: 预测名出现在预期中即可
    hit = False
    if culprit in expected:
        hit = True
    else:
        for name in culprit.replace("（", "(").replace("）", ")").split("+"):
            name = name.strip()
            if name and name in expected:
                hit = True
                break
    for name in expected.replace("（", "(").replace("）", ")").split("+"):
        name = name.strip()
        if name and name in culprit:
            hit = True
            break
    
    status = "✅ 正确!" if hit else "❌ 错误"
    print(f"\n{'='*60}")
    print(f"🏁 最终判断: {status}")
    print(f"{'='*60}")
    
    return {
        "case_id": case.case_id,
        "predicted": culprit,
        "expected": expected,
        "correct": hit,
        "confidence": confidence,
        "elapsed": elapsed,
        "timing": timing,
        "multi_round_max": max_rounds_seen,
    }

if __name__ == "__main__":
    result = main()
    # 保存结果
    ts = time.strftime("%Y%m%d_%H%M%S")
    outpath = f"data/test_results/v140_case002_{ts}.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 结果已保存: {outpath}")
