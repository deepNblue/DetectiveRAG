#!/usr/bin/env python3
"""
v15.2 端到端测试 — 多轮推理精简版验证
跑 3 个改编案 + 1 个福尔摩斯案，验证:
  1. 多轮推理上限 3 轮（vs 旧版 10 轮）
  2. 准确率对比 v12.0 (50%)
  3. 耗时优化（预期 < 2min/案）
"""
import json, time, sys, os
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<5} | {message}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.llm_client import LLMClient
from agents.asmr.orchestrator import ASMROrchestrator
from agents.asmr.name_utils import normalize_name

# ── 选案 ──────────────────────────────────────────
# 3 个改编案 (v12.0 正确率 5/7=71.4%)
# 1 个福尔摩斯案 (v12.0 正确率 0/3=0%)
TEST_INDICES = [0, 3, 11]  # 影后的秘密, 密室疑云, 毒影重重(王刚)
# 加一个 sherlock 案
SHERLOCK_CASE = None  # 后面单独加载

with open("data/real_cases/rewritten/rewritten_cases.json") as f:
    all_cases = json.load(f)

# 加福尔摩斯案
try:
    with open("data/real_cases/sherlock/sherlock_cases.json") as f:
        sherlock_cases = json.load(f)
    # 选 SH-008 (真凶: 张律师) — 之前名字截断bug案
    for sc in sherlock_cases:
        if sc.get("case_id") == "SH-008":
            SHERLOCK_CASE = sc
            break
except Exception as e:
    logger.warning(f"⚠️ 无法加载福尔摩斯案: {e}")

def build_case_text(case):
    """构建案件文本"""
    parts = [f"案件: {case['title']}"]
    parts.append(f"案件类型: {case.get('case_type', '未知')}")
    if case.get('victim'):
        v = case['victim']
        if isinstance(v, dict):
            parts.append(f"\n受害人: {v.get('name', '?')}, {v.get('age', '?')}岁, {v.get('occupation', '?')}")
    parts.append("\n嫌疑人:")
    for s in case.get('suspects', []):
        parts.append(f"  - {s['name']}({s.get('relationship', s.get('role', '?'))}): {s.get('motive', s.get('description', '?'))}")
    parts.append(f"\n案件经过:\n{case.get('case_text', case.get('description', ''))}")
    if case.get('evidence'):
        parts.append("\n关键证据:")
        for e in case['evidence']:
            if isinstance(e, dict):
                parts.append(f"  - [{e.get('type','?')}] {e.get('description', e.get('content', str(e)))}")
            else:
                parts.append(f"  - {e}")
    if case.get('timeline'):
        parts.append("\n时间线:")
        for t in case['timeline']:
            if isinstance(t, dict):
                parts.append(f"  {t.get('time', t.get('timestamp', '?'))}: {t.get('event', t.get('description', '?'))}")
            else:
                parts.append(f"  {t}")
    return "\n".join(parts)


def run_single(orchestrator, case, case_label):
    """跑单个案件"""
    case_text = build_case_text(case)
    suspects = [{"name": s["name"], "relation": s.get("relationship", s.get("role", ""))} for s in case.get("suspects", [])]
    actual = case.get("solution", {}).get("criminal", "?")
    
    logger.info(f"{'='*60}")
    logger.info(f"🔍 案件: {case_label} — {case['title']}")
    logger.info(f"   嫌疑人: {[s['name'] for s in suspects]}")
    logger.info(f"   真凶: {actual}")
    logger.info(f"{'='*60}")
    
    t0 = time.time()
    result = orchestrator.run(
        case_text=case_text,
        suspects=suspects,
        case_type=case.get('case_type', '未知'),
    )
    elapsed = time.time() - t0
    
    predicted = result.get("culprit", "?")
    predicted_norm = normalize_name(predicted)
    actual_norm = normalize_name(actual)
    correct = predicted_norm == actual_norm or actual_norm in predicted_norm
    
    logger.info(f"{'='*60}")
    logger.info(f"{'✅' if correct else '❌'} 结果: 预测={predicted} | 真凶={actual} | 耗时={elapsed:.1f}s")
    logger.info(f"   置信度: {result.get('confidence', '?')}")
    logger.info(f"{'='*60}")
    
    # learn from result
    try:
        orchestrator.learn_from_result(
            case_id=case_label,
            case_text=case_text,
            result=result,
            actual_culprit=actual,
        )
        logger.info(f"📚 learn_from_result 完成 ({case_label})")
    except Exception as e:
        logger.warning(f"⚠️ learn_from_result 失败: {e}")
    
    return {
        "case_id": case_label,
        "title": case['title'],
        "predicted": predicted,
        "actual": actual,
        "correct": correct,
        "confidence": result.get("confidence", "?"),
        "elapsed": round(elapsed, 1),
        "vote_distribution": result.get("vote_distribution", {}),
        "stage_times": result.get("stage_times", {}),
    }


def main():
    logger.info("🔍 v15.2 端到端测试 — 多轮推理精简版验证")
    logger.info(f"   测试案数: {len(TEST_INDICES)} 改编案" + (f" + 1 福尔摩斯案" if SHERLOCK_CASE else ""))
    logger.info(f"   时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化
    llm = LLMClient()
    orchestrator = ASMROrchestrator(llm_client=llm)
    
    results = []
    
    # 跑改编案
    for idx in TEST_INDICES:
        case = all_cases[idx]
        case_label = f"rw_{idx}"
        try:
            r = run_single(orchestrator, case, case_label)
            results.append(r)
        except Exception as e:
            logger.error(f"❌ {case_label} 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({"case_id": case_label, "title": case['title'], "error": str(e), "correct": False})
    
    # 跑福尔摩斯案
    if SHERLOCK_CASE:
        try:
            r = run_single(orchestrator, SHERLOCK_CASE, "SH-008")
            results.append(r)
        except Exception as e:
            logger.error(f"❌ SH-008 失败: {e}")
            results.append({"case_id": "SH-008", "title": SHERLOCK_CASE.get('title','?'), "error": str(e), "correct": False})
    
    # 汇总
    logger.info("\n" + "="*60)
    logger.info("📊 v15.2 端到端测试汇总")
    logger.info("="*60)
    
    correct_count = sum(1 for r in results if r.get("correct"))
    total = len(results)
    
    for r in results:
        if "error" in r:
            logger.info(f"  ❌ {r['case_id']}: {r['title']} — 错误: {r['error'][:50]}")
        else:
            icon = "✅" if r['correct'] else "❌"
            logger.info(f"  {icon} {r['case_id']}: 预测={r['predicted']} | 真凶={r['actual']} | {r['elapsed']}s")
    
    logger.info(f"\n  准确率: {correct_count}/{total} = {correct_count/total*100:.1f}%")
    logger.info(f"  对比 v12.0: 50% (5/10)")
    
    # 保存结果
    report = {
        "version": "v15.2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "accuracy": f"{correct_count}/{total} = {correct_count/total*100:.1f}%",
    }
    
    report_path = f"data/test_results/v150_e2e_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("data/test_results", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\n💾 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
