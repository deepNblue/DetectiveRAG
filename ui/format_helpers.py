"""Format helpers for webui"""
import json

def txt(data, n=200):
    if isinstance(data, str): return data[:n]
    if isinstance(data, (dict, list)): return json.dumps(data, ensure_ascii=False)[:n]
    return str(data)[:n]

def names(suspects):
    ns = []
    if isinstance(suspects, list):
        for s in suspects:
            if isinstance(s, dict): ns.append(s.get("name", s.get("suspect","?")))
            else: ns.append(str(s))
    elif isinstance(suspects, dict):
        ns.append(suspects.get("name", suspects.get("suspect","?")))
    return ns

def fmt_trad(r, elapsed):
    if "error" in r and not r.get("culprit"):
        return f"## ❌ 传统线路失败\n\n```\n{r['error']}\n```\n\n⏱ {elapsed:.1f}s"
    c = r.get("culprit","?"); conf = r.get("confidence",0)
    md = f"## 传统线路报告\n**⏱ {elapsed:.1f}s**\n\n| 项目 | 结果 |\n|------|------|\n| 嫌疑人 | `{c}` |\n| 置信度 | {conf:.1%} |\n"
    gr_data = r.get("graph_reasoning",{})
    # 解包 format_output 包装
    if isinstance(gr_data, dict) and "data" in gr_data:
        gr_data = gr_data["data"]
    if isinstance(gr_data, dict):
        s = gr_data.get("reasoning_summary", gr_data.get("parsed_result",{}))
        # parsed_result 可能也是 format_output 包装
        if isinstance(s, dict) and "data" in s:
            s = s["data"]
        md += f"\n### 图谱推理\n```\n{txt(s,600)}\n```\n"
    g = r.get("graph",{})
    if isinstance(g, dict):
        md += f"\n### 图谱统计\n- 节点: {len(g.get('nodes',[]))}\n- 边: {len(g.get('edges',[]))}\n"
    return md

def fmt_asmr(r, elapsed):
    if "error" in r and not r.get("final_conclusion"):
        return f"## ❌ ASMR失败\n\n```\n{r['error']}\n```\n\n⏱ {elapsed:.1f}s"
    conc = r.get("conclusion",{})
    c = conc.get("culprit", r.get("final_conclusion","?"))
    conf = conc.get("confidence", r.get("confidence",0))
    cons = conc.get("consensus_level", r.get("consensus","无"))
    ot = conc.get("overturned", False)
    vw = r.get("vote_result",{}).get("winner","?")
    md = f"## ASMR v3 报告\n**⏱ {elapsed:.1f}s**\n\n| 项目 | 结果 |\n|------|------|\n| 嫌疑人 | `{c}` |\n| 置信度 | {conf:.1%} |\n| 共识度 | {cons} |\n"
    if ot:
        md += f"\n> ⚡ 裁判推翻! 投票={vw} → 裁判={c} ({conf:.1%})\n"
    _DETECTIVE_NAMES = {
        'sherlock_analysis': '🔍 福尔摩斯(演绎推理)',
        'henry_lee_analysis': '🔬 李昌钰(鉴识科学)',
        'song_ci_analysis': '⚖️ 宋慈(法医鉴识)',
        'poirot_analysis': '🧠 波洛(心理侦探)',
    }
    for e in r.get("expert_analyses",[]):
        p = e.get('perspective','?')
        display_name = _DETECTIVE_NAMES.get(p, p)
        marker = ' 🔮' if p in _DETECTIVE_NAMES else ''
        md += f"\n- **{display_name}**: `{e.get('culprit','?')}` ({e.get('confidence',0):.0%}){marker}"
    # 🆕 v5: 动态领域专家
    de = r.get("domain_experts", {})
    de_list = de.get("experts", []) if isinstance(de, dict) else []
    if de_list:
        md += f"\n\n### 🔬 动态领域专家 ({de.get('activated', len(de_list))}个激活)\n"
        for d in de_list:
            md += f"- **{d.get('name','?')}** ({d.get('domain','?')}): `{d.get('culprit','?')}` ({d.get('confidence',0):.0%})\n"
    cd = r.get("contradiction_data",{})
    cc = cd.get("contradictions",[])
    if cc:
        md += f"\n\n### 矛盾发现 ({len(cc)}个)\n"
        for ci in cc:
            ic = "🔴" if ci.get("significance")=="high" else "🟡"
            md += f"- {ic} {ci.get('person','?')}: {ci.get('description','?')[:80]}\n"
    timing = r.get("timing",{})
    if timing:
        md += "\n### 四阶段\n| 阶段 | 耗时 |\n|------|------|\n"
        md += f"| S1 Readers | {timing.get('stage1_readers','?')}s |\n"
        md += f"| S2 Searchers | {timing.get('stage2_searchers','?')}s |\n"
        md += f"| S3 Experts | {timing.get('stage3_experts','?')}s |\n"
        md += f"| S4 Adjudicator | {timing.get('stage4_adjudicator','?')}s |\n"
        md += f"| **总计** | **{timing.get('total','?')}s** |\n"
    return md

def fmt_fusion(fusion, trad, asmr):
    if "error" in fusion:
        return f"## ❌ 融合失败\n\n```\n{fusion.get('error','?')}\n```"
    # 融合引擎返回嵌套结构: {"conclusion": {"culprit": ..., "confidence": ...}, ...}
    conc = fusion.get("conclusion", fusion)
    c = conc.get("culprit", fusion.get("culprit","?"))
    conf = conc.get("confidence", fusion.get("confidence",0))
    cert = conc.get("certainty", fusion.get("certainty","未知"))
    agreement = fusion.get("agreement", {})
    agr_status = agreement.get("status", "")
    if "一致" in agr_status and "不" not in agr_status:
        agr = f"🟢 {agr_status}"
    elif "冲突" in agr_status:
        agr = f"🔴 {agr_status}"
    else:
        agr = f"🟡 {agr_status}" if agr_status else ("🟢 完全一致" if trad.get("culprit","?")==asmr.get("culprit","?") else "🔴 不一致")
    tc = trad.get("culprit","?"); ac = asmr.get("culprit","?")
    md = f"""## 双线融合报告

### 最终结论

| 项目 | 结果 |
|------|------|
| **嫌疑人** | **`{c}`** |
| **融合置信度** | **{conf:.1%}** |
| **确定性** | {cert} |
| **双线一致** | {agr} |

### 双线对比

| 维度 | 传统线路 | ASMR线路 |
|------|---------|----------|
| 嫌疑人 | `{tc}` | `{ac}` |
| 置信度 | {trad.get('confidence',0):.1%} | {asmr.get('confidence',0):.1%} |
| 耗时 | {trad.get('time',0):.1f}s | {asmr.get('time',0):.1f}s |
"""
    return md

def fmt_graph(graph, graph_reasoning):
    """使用纯 SVG 渲染知识图谱（Gradio不执行<script>，不能用D3.js）"""
    from ui.graph_renderer_svg import render_force_graph
    nodes = graph.get("nodes",[]) if isinstance(graph,dict) else []
    edges = graph.get("edges",[]) if isinstance(graph,dict) else []
    if not nodes:
        return '<p style="color:#55556A;text-align:center;padding:40px;font-family:JetBrains Mono,monospace">图谱数据为空</p>'

    # Try to extract suspect names for highlighting
    suspect_names = []
    for n in nodes:
        if n.get("type") == "person":
            suspect_names.append(n.get("label", n.get("id", "")))

    graph_html = render_force_graph(nodes, edges, suspect_names=suspect_names)

    rtxt = ""
    grp = graph_reasoning.get("parsed_result", graph_reasoning)
    if isinstance(grp, dict):
        rtxt = grp.get("reasoning_summary", "")
    elif isinstance(grp, str):
        rtxt = grp[:300]

    reasoning_section = ""
    if rtxt:
        reasoning_section = f'<div style="margin-top:12px;color:#8A8A9A;font-size:13px;white-space:pre-wrap;line-height:1.6;font-family:JetBrains Mono,monospace">{rtxt[:500]}</div>'

    return f"""<div style="margin-top:8px">
<h3 style="color:#00FF41;font-size:16px;margin:0 0 12px 0;padding-bottom:8px;border-bottom:2px solid rgba(0,255,65,0.3);font-family:JetBrains Mono,monospace">知识图谱 ({len(nodes)}节点/{len(edges)}关系)</h3>
{graph_html}
{reasoning_section}
</div>"""

def load_test_results():
    import os
    v2p = "data/test_results/v2_multi_case_results.json"
    v1p = "data/test_results/multi_case_results.json"
    data = None; is_v2 = False
    if os.path.exists(v2p):
        try:
            with open(v2p) as f: data=json.load(f)
            is_v2 = True
        except: pass
    if not data and os.path.exists(v1p):
        try:
            with open(v1p) as f: data=json.load(f)
        except: return "⚠️ 无法加载"
    if not data: return "⚠️ 无测试数据"
    s = data.get("summary",{}); results = data.get("results",[]); total = s.get("total",len(results))
    ver = "**ASMR v2**" if is_v2 else "v1"
    md = f"### 测试结果 ({ver})\n**总计**: {total}案\n- 传统: **{s.get('trad_wins',0)}**/{total}\n- ASMR: **{s.get('asmr_wins',0)}**/{total}\n- 融合: **{s.get('fusion_wins',0)}**/{total}\n"
    if is_v2: md += f"- 裁判推翻: **{s.get('overturned_count',0)}**次\n"
    md += "\n| 案件 | 类型 | 真凶 | 传统 | ASMR | 融合 | v2 |\n|------|------|------|------|------|------|----|\n"
    for r in results:
        ti = "✅" if r.get("traditional",{}).get("match") else "❌"
        ai = "✅" if r.get("asmr",{}).get("match") else "❌"
        fi = "✅" if r.get("fusion",{}).get("match") else "❌"
        v2f = "⚡推翻" if r.get("asmr",{}).get("overturned") else ""
        md += f"| {r.get('case_id','')} | {r.get('type','')} | {r.get('expected','?')[:15]} | {ti} | {ai} | {fi} | {v2f} |\n"
    return md
