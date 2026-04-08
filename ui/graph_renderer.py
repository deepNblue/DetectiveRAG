"""
动态知识图谱渲染器 — D3.js v7 force-directed graph
用于在 Gradio HTML 组件中实时渲染可交互的知识图谱

特性:
  - 每次推理步骤更新时，图谱动态增加节点和边
  - 新增节点有入场动画（从0透明度渐入+缩放）
  - 节点可拖拽，画布可缩放
  - 不同类型节点有不同颜色
  - 边的粗细和透明度表示权重
  - 高亮嫌疑人的节点（更大、有光晕效果）
"""
import json
import time as _time
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# D3.js CDN
# ---------------------------------------------------------------------------
D3_CDN = "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"

# ---------------------------------------------------------------------------
# 颜色映射 — 赛博侦探(Cyber-Sleuth)主题
# ---------------------------------------------------------------------------
NODE_COLORS = {
    "person":   "#00FF41",     # 霓虹绿 — 人员/嫌疑人
    "evidence": "#FFB800",     # 琥珀 — 证据
    "location": "#00F5FF",     # 电子青 — 地点
    "time":     "#8A8A9A",     # 灰 — 时间
    "event":    "#B37FEB",     # 紫 — 事件
    "default":  "#8A8A9A",     # 灰 — 默认
}

NODE_BG = {
    "person":   "rgba(0,255,65,0.12)",
    "evidence": "rgba(255,184,0,0.12)",
    "location": "rgba(0,245,255,0.12)",
    "time":     "rgba(138,138,154,0.1)",
    "event":    "rgba(179,127,235,0.12)",
    "default":  "rgba(138,138,154,0.1)",
}


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _safe_id(raw: str) -> str:
    """Convert arbitrary string to a JS/D3-safe identifier."""
    return (
        raw.replace("-", "_")
           .replace(" ", "_")
           .replace("(", "")
           .replace(")", "")
           .replace("（", "")
           .replace("）", "")
           .replace('"', "")
           .replace("'", "")
    )


def _empty_graph_html(msg: str) -> str:
    return (
        '<div style="background:rgba(10,10,14,0.85);border:1px solid rgba(0,255,65,0.2);border-radius:6px;overflow:hidden;backdrop-filter:blur(12px)">'
        '<div style="background:linear-gradient(135deg,rgba(0,255,65,0.08),rgba(0,255,65,0.02));color:#00FF41;'
        'padding:10px 18px;font-family:Orbitron,monospace;font-weight:700;font-size:13px;letter-spacing:2px;display:flex;align-items:center;'
        'justify-content:space-between;border-bottom:1px solid rgba(0,255,65,0.15)">'
        '🧵 关联图谱 <span style="font-size:11px;font-weight:400;color:#8A8A9A;font-family:JetBrains Mono,monospace">待命…</span></div>'
        '<div style="padding:40px;color:#55556A;text-align:center;background:rgba(6,6,8,0.6);'
        f'font-size:12px;font-family:JetBrains Mono,monospace">{msg}</div></div>'
    )


# ---------------------------------------------------------------------------
# 主渲染函数
# ---------------------------------------------------------------------------
def render_force_graph(
    nodes: List[Dict],
    edges: List[Dict],
    suspect_names: List[str] | None = None,
    width: int = 600,
    height: int = 450,
) -> str:
    """
    渲染一个可交互的 D3.js 力导向图。

    Args:
        nodes: [{"id":"...", "label":"...", "type":"person|evidence|location|time|event"}]
        edges: [{"source":"...", "target":"...", "label":"...", "weight":0.8}]
        suspect_names: 嫌疑人名称列表，用于高亮
        width/height: SVG 尺寸

    Returns:
        HTML string with embedded D3.js visualization
    """
    if not nodes:
        return _empty_graph_html("推理开始后，图谱将在此逐步构建")

    suspect_names = [s for s in (suspect_names or []) if s]

    # ---- Build sanitized data ----
    nodes_data: List[Dict] = []
    seen_nids: set = set()
    for n in nodes[:60]:
        nid = _safe_id(n.get("id", "?"))
        if nid in seen_nids:
            continue
        seen_nids.add(nid)
        label = n.get("label", n.get("id", "?"))
        # A node is a suspect if its label matches a suspect name, or
        # we detect it by type+context.
        is_suspect = label in suspect_names
        nodes_data.append({
            "id": nid,
            "label": label[:14],
            "type": n.get("type", "default"),
            "isSuspect": is_suspect,
        })

    edges_data: List[Dict] = []
    seen_ekeys: set = set()
    for e in edges[:60]:
        src = _safe_id(e.get("source", "?"))
        tgt = _safe_id(e.get("target", "?"))
        key = f"{src}|{tgt}"
        if key in seen_ekeys:
            continue
        seen_ekeys.add(key)
        edges_data.append({
            "source": src,
            "target": tgt,
            "label": e.get("label", "")[:12],
            "weight": float(e.get("weight", 0.5)),
        })

    nodes_json = json.dumps(nodes_data, ensure_ascii=False)
    edges_json = json.dumps(edges_data, ensure_ascii=False)
    colors_json = json.dumps(NODE_COLORS, ensure_ascii=False)
    bg_json = json.dumps(NODE_BG, ensure_ascii=False)

    # Unique container id — prevents conflicts when Gradio re-renders
    container_id = f"kg_{int(_time.time()*1000)}_{len(nodes_data)}"

    # ---- Full HTML with inline D3.js ----
    html = f'''<div id="{container_id}" class="kg-container" style="background:rgba(10,10,14,0.85);border:1px solid rgba(0,255,65,0.2);border-radius:6px;overflow:hidden;box-shadow:0 4px 24px rgba(0,255,65,0.04),0 0 40px rgba(0,255,65,0.02);backdrop-filter:blur(12px)">
<div style="background:linear-gradient(135deg,rgba(0,255,65,0.08),rgba(0,255,65,0.02));color:#00FF41;padding:10px 18px;font-family:Orbitron,monospace;font-weight:700;font-size:13px;letter-spacing:2px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(0,255,65,0.15);text-shadow:0 0 10px rgba(0,255,65,0.3)">
🧵 关联图谱 <span style="font-size:11px;font-weight:400;color:#8A8A9A;font-family:JetBrains Mono,monospace">{len(nodes_data)} 节点 · {len(edges_data)} 关系</span>
</div>
<div class="kg-canvas" style="position:relative;width:{width}px;height:{height}px;background:rgba(6,6,8,0.8)">
<script src="{D3_CDN}"></script>
<script>
(function() {{
    "use strict";
    var cid = "{container_id}";
    var el = document.getElementById(cid);
    if (!el) return;

    /* ---- Remove any previously appended SVG (re-render guard) ---- */
    var oldSvg = el.querySelector("svg.kg-svg");
    if (oldSvg) oldSvg.remove();

    var nodes = {nodes_json};
    var links = {edges_json};
    var colors = {colors_json};
    var bgColors = {bg_json};
    var W = {width}, H = {height};

    /* ---- SVG setup ---- */
    var svg = d3.select(el).select(".kg-canvas")
        .append("svg")
        .classed("kg-svg", true)
        .attr("width", W)
        .attr("height", H)
        .style("background", "#08080A");

    /* ---- Zoom / Pan ---- */
    var g = svg.append("g");
    svg.call(
        d3.zoom()
          .scaleExtent([0.2, 4])
          .on("zoom", function(ev) {{ g.attr("transform", ev.transform); }})
    );

    /* ---- Defs: glow filter ---- */
    var defs = svg.append("defs");
    var glowFilter = defs.append("filter")
        .attr("id", "kg-glow")
        .attr("x", "-50%").attr("y", "-50%")
        .attr("width", "200%").attr("height", "200%");
    glowFilter.append("feGaussianBlur")
        .attr("stdDeviation", "4")
        .attr("result", "blur");
    var feMerge = glowFilter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "blur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    /* ---- Defs: arrow marker ---- */
    var markerColors = {{
        "person": "#00FF41", "evidence": "#FFB800",
        "location": "#00F5FF", "time": "#8A8A9A",
        "event": "#B37FEB", "default": "#8A8A9A"
    }};
    Object.keys(markerColors).forEach(function(t) {{
        defs.append("marker")
            .attr("id", "arrow-" + t)
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 28).attr("refY", 0)
            .attr("markerWidth", 6).attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", markerColors[t]);
    }});

    /* ---- Force simulation ---- */
    var simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(function(d) {{ return d.id; }}).distance(110))
        .force("charge", d3.forceManyBody().strength(-350))
        .force("center", d3.forceCenter(W / 2, H / 2))
        .force("collision", d3.forceCollide().radius(38))
        .force("x", d3.forceX(W / 2).strength(0.04))
        .force("y", d3.forceY(H / 2).strength(0.04));

    /* ---- Draw links ---- */
    var link = g.append("g").selectAll("line")
        .data(links)
        .join("line")
        .attr("stroke", function(d) {{
            return d3.interpolateRgb("#333340", "#00FF41")(d.weight);
        }})
        .attr("stroke-width", function(d) {{ return Math.max(1, d.weight * 5); }})
        .attr("stroke-opacity", function(d) {{ return 0.25 + d.weight * 0.55; }})
        .attr("marker-end", function(d) {{
            /* Try to find type from source node */
            var srcNode = nodes.find(function(n) {{ return n.id === (d.source.id || d.source); }});
            var t = srcNode ? (srcNode.type || "default") : "default";
            return "url(#arrow-" + t + ")";
        }});

    /* ---- Link labels ---- */
    var linkLabel = g.append("g").selectAll("text")
        .data(links)
        .join("text")
        .text(function(d) {{ return d.label; }})
        .attr("font-size", "9px")
        .attr("fill", "#55556A")
        .attr("text-anchor", "middle")
        .attr("dy", -6);

    /* ---- Node groups ---- */
    var node = g.append("g").selectAll("g")
        .data(nodes)
        .join("g")
        .style("cursor", "grab")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    /* ---- Outer glow ring for suspects ---- */
    node.filter(function(d) {{ return d.isSuspect; }})
        .append("circle")
        .attr("r", 24)
        .attr("fill", "none")
        .attr("stroke", function(d) {{ return colors[d.type] || colors["default"]; }})
        .attr("stroke-width", 2)
        .attr("stroke-opacity", 0.25)
        .attr("filter", "url(#kg-glow)");

    /* ---- Main circles ---- */
    node.append("circle")
        .attr("r", 0)   /* start from 0 for animation */
        .attr("fill", function(d) {{ return bgColors[d.type] || bgColors["default"]; }})
        .attr("stroke", function(d) {{ return colors[d.type] || colors["default"]; }})
        .attr("stroke-width", function(d) {{ return d.isSuspect ? 3 : 1.5; }})
        .attr("filter", function(d) {{ return d.isSuspect ? "url(#kg-glow)" : "none"; }})
        .transition()
        .duration(650)
        .delay(function(d, i) {{ return i * 60; }})
        .ease(d3.easeBackOut.overshoot(1.2))
        .attr("r", function(d) {{ return d.isSuspect ? 18 : 12; }});

    /* ---- Node icon (type emoji) ---- */
    var typeIcon = {{
        "person": "👤", "evidence": "📌", "location": "📍",
        "time": "🕐", "event": "⚡", "default": "•"
    }};
    node.append("text")
        .text(function(d) {{ return typeIcon[d.type] || "•"; }})
        .attr("text-anchor", "middle")
        .attr("dy", 1)
        .attr("font-size", function(d) {{ return d.isSuspect ? "14px" : "11px"; }})
        .style("pointer-events", "none")
        .style("opacity", 0)
        .transition()
        .duration(500)
        .delay(function(d, i) {{ return i * 60 + 200; }})
        .style("opacity", 1);

    /* ---- Node labels ---- */
    node.append("text")
        .text(function(d) {{ return d.label; }})
        .attr("font-size", function(d) {{ return d.isSuspect ? "11px" : "10px"; }})
        .attr("font-weight", function(d) {{ return d.isSuspect ? "700" : "500"; }})
        .attr("fill", function(d) {{ return colors[d.type] || colors["default"]; }})
        .attr("text-anchor", "middle")
        .attr("dy", function(d) {{ return d.isSuspect ? 32 : 24; }})
        .style("pointer-events", "none")
        .style("text-shadow", function(d) {{ return "0 0 6px " + (colors[d.type] || colors["default"]); }})
        .style("opacity", 0)
        .transition()
        .duration(500)
        .delay(function(d, i) {{ return i * 60 + 100; }})
        .style("opacity", 1);

    /* ---- Tooltip ---- */
    node.append("title")
        .text(function(d) {{
            return d.label + " [" + d.type + "]" + (d.isSuspect ? " ⚠嫌疑人" : "");
        }});

    /* ---- Tick ---- */
    simulation.on("tick", function() {{
        link.attr("x1", function(d) {{ return d.source.x; }})
            .attr("y1", function(d) {{ return d.source.y; }})
            .attr("x2", function(d) {{ return d.target.x; }})
            .attr("y2", function(d) {{ return d.target.y; }});
        linkLabel
            .attr("x", function(d) {{ return (d.source.x + d.target.x) / 2; }})
            .attr("y", function(d) {{ return (d.source.y + d.target.y) / 2; }});
        node.attr("transform", function(d) {{
            return "translate(" + d.x + "," + d.y + ")";
        }});
    }});

    /* ---- Drag helpers ---- */
    function dragstarted(event) {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }}
    function dragged(event) {{
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }}
    function dragended(event) {{
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }}
}})();
</script>
</div></div>'''
    return html


# ---------------------------------------------------------------------------
# Incremental variant (currently renders full graph each time,
# but marks this as the API for future incremental updates)
# ---------------------------------------------------------------------------
def render_incremental_graph(
    all_nodes: List[Dict],
    all_edges: List[Dict],
    suspect_names: List[str] | None = None,
    new_nodes: List[Dict] | None = None,
    new_edges: List[Dict] | None = None,
) -> str:
    """
    渲染增量图谱 — 新节点有特殊入场动画。
    目前仍渲染完整图谱，后续可优化为真正的增量更新。
    """
    return render_force_graph(all_nodes, all_edges, suspect_names)
