"""
动态知识图谱渲染器 v2 — D3.js v7 分层力导向图
用于在 Gradio HTML 组件中实时渲染可交互的知识图谱

v2 优化:
  - 12种节点类型 (对齐 evidence_graph 实体分类)
  - 12种关系类型着色 (动机/手段/矛盾/证明/时间线...)
  - 重要性分层布局 (嫌疑人内圈→证据中圈→时间地点外圈)
  - 权重感知力导向 (高权重边更短=关系更紧密)
  - 智能标签 (只显示高权重边标签，避免重叠)
  - 低价值边折叠 (located_at 等弱关系默认淡化)
"""
import json
import time as _time
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# D3.js CDN
# ---------------------------------------------------------------------------
D3_CDN = "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"

# ---------------------------------------------------------------------------
# 颜色映射 — 12种节点类型
# ---------------------------------------------------------------------------
NODE_COLORS = {
    "person":       "#00FF41",     # 霓虹绿 — 嫌疑人/人物
    "suspect":      "#FF3B5C",     # 赤红 — 核心嫌疑人
    "victim":       "#9B59B6",     # 紫 — 受害者
    "evidence":     "#FFB800",     # 琥珀 — 物证
    "weapon":       "#FF6B35",     # 橙红 — 凶器
    "substance":    "#E74C3C",     # 红 — 毒物/药物
    "location":     "#00F5FF",     # 电子青 — 地点
    "time":         "#8A8A9A",     # 灰 — 时间
    "document":     "#F39C12",     # 金 — 文件/遗嘱
    "communication":"#3498DB",     # 蓝 — 通讯/短信
    "financial":    "#2ECC71",     # 翠绿 — 财务
    "surveillance": "#1ABC9C",     # 青 — 监控
    "biometric":    "#E91E63",     # 粉 — DNA/指纹
    "motive":       "#FF3B5C",     # 赤红 — 动机
    "event":        "#B37FEB",     # 紫 — 事件
    "default":      "#8A8A9A",     # 灰 — 默认
}

NODE_BG = {
    "person":       "rgba(0,255,65,0.10)",
    "suspect":      "rgba(255,59,92,0.14)",
    "victim":       "rgba(155,89,182,0.12)",
    "evidence":     "rgba(255,184,0,0.10)",
    "weapon":       "rgba(255,107,53,0.12)",
    "substance":    "rgba(231,76,60,0.12)",
    "location":     "rgba(0,245,255,0.10)",
    "time":         "rgba(138,138,154,0.08)",
    "document":     "rgba(243,156,18,0.10)",
    "communication":"rgba(52,152,219,0.10)",
    "financial":    "rgba(46,204,113,0.10)",
    "surveillance": "rgba(26,188,156,0.10)",
    "biometric":    "rgba(233,30,99,0.10)",
    "motive":       "rgba(255,59,92,0.10)",
    "event":        "rgba(179,127,235,0.10)",
    "default":      "rgba(138,138,154,0.08)",
}

# 节点层级 — 决定初始位置和引力强度
# Tier 1 = 核心内圈(嫌疑人/受害者), Tier 2 = 关键中圈(证据/凶器/毒物), Tier 3 = 外圈(时间/地点)
NODE_TIER = {
    "suspect": 1, "victim": 1, "person": 1, "motive": 1,
    "weapon": 2, "substance": 2, "evidence": 2, "biometric": 2, "document": 2,
    "financial": 2, "communication": 2, "surveillance": 2,
    "location": 3, "time": 3, "event": 3,
    "default": 3,
}

# ---------------------------------------------------------------------------
# 边颜色 — 按关系类型着色
# ---------------------------------------------------------------------------
EDGE_COLORS = {
    "has_motive":       "#FF3B5C",   # 赤红 — 动机
    "has_opportunity":  "#FF9500",   # 橙 — 机会
    "has_means":        "#FF6B35",   # 橙红 — 手段
    "contradicts":      "#FFD700",   # 金黄 — 矛盾
    "proves":           "#00FF41",   # 绿 — 证明
    "implies":          "#3498DB",   # 蓝 — 暗示
    "alibis":           "#2ECC71",   # 翠绿 — 不在场证明
    "tampered":         "#E74C3C",   # 红 — 篡改
    "hides":            "#E91E63",   # 粉 — 隐藏
    "suspicious_of":    "#B37FEB",   # 紫 — 可疑
    "threatens":        "#FF3B5C",   # 赤红 — 威胁
    "disputes_with":    "#FF9500",   # 橙 — 纠纷
    "witnessed":        "#1ABC9C",   # 青 — 目击
    "knows":            "#3498DB",   # 蓝 — 知道
    "owns":             "#F39C12",   # 金 — 拥有
    "contacted_with":   "#00F5FF",   # 电子青 — 接触
    "located_at":       "#555566",   # 暗灰 — 位置(弱关系)
    "happened_at":      "#555566",   # 暗灰 — 发生于(弱关系)
    "belongs_to":       "#555566",   # 暗灰 — 归属(弱关系)
    "same_as":          "#8A8A9A",   # 灰 — 同一
}

# 关系类型中文标签
EDGE_LABELS_ZH = {
    "has_motive": "🎯动机", "has_opportunity": "🕐机会", "has_means": "🔧手段",
    "contradicts": "⚡矛盾", "proves": "✅证明", "implies": "🔍暗示",
    "alibis": "🛡️不在场", "tampered": "🔧篡改", "hides": "🔒隐藏",
    "suspicious_of": "❓可疑", "threatens": "⚠️威胁", "disputes_with": "⚔️纠纷",
    "witnessed": "👁️目击", "knows": "🔑知道", "owns": "💰拥有",
    "contacted_with": "📞接触", "located_at": "📍位于", "happened_at": "📍发生",
    "belongs_to": "📎归属", "same_as": "🔄同一",
}

# 弱关系类型 — 默认淡化显示
WEAK_RELATIONS = {"located_at", "happened_at", "belongs_to", "same_as"}

# 节点图标
NODE_ICONS = {
    "person": "👤", "suspect": "🔴", "victim": "💀", "evidence": "📌",
    "weapon": "🔪", "substance": "⚗️", "location": "📍", "time": "🕐",
    "document": "📋", "communication": "📞", "financial": "💰",
    "surveillance": "📹", "biometric": "🧬", "motive": "🎯",
    "event": "⚡", "default": "•",
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


# 中文关键词 → 关系类型映射 (用于传统路线的中文标签)
_ZH_KEYWORD_MAP = {
    # 动机类
    "动机": "has_motive", "经济动机": "has_motive", "报复": "has_motive",
    "遗产纠纷": "disputes_with", "纠纷": "disputes_with", "商业纠纷": "disputes_with",
    "争执": "disputes_with", "矛盾": "disputes_with", "冲突": "disputes_with",
    "威胁": "threatens", "恐吓": "threatens",
    # 手段类
    "手段": "has_means", "有机会接触": "has_means", "接触": "has_means",
    "有机会": "has_opportunity", "机会": "has_opportunity",
    "篡改": "tampered", "破坏": "tampered", "更换": "tampered",
    "隐藏": "hides", "隐瞒": "hides", "掩盖": "hides",
    # 证明/证据类
    "证明": "proves", "证实": "proves", "确认": "proves",
    "暗示": "implies", "推测": "implies", "疑似": "implies",
    "不在场": "alibis", "不在场证明": "alibis",
    "矛盾": "contradicts", "冲突": "contradicts", "不一致": "contradicts",
    # 关系类
    "知道": "knows", "了解": "knows", "掌握": "knows", "认识": "knows",
    "拥有": "owns", "持有": "owns", "控制": "owns",
    "目击": "witnessed", "看到": "witnessed", "发现": "witnessed",
    "接触": "contacted_with", "通讯": "contacted_with", "通话": "contacted_with",
    # 可疑行为
    "可疑": "suspicious_of", "疑点": "suspicious_of", "异常": "suspicious_of",
    "指向": "suspicious_of", "涉及": "suspicious_of",
    # 关联(通用)
    "关联": "implies", "相关": "implies", "联系": "implies",
    # 身份类
    "嫌疑人": "has_motive", "受害者": "has_motive",
    "致死原因": "has_means", "死因": "has_means",
}


def _normalize_edge_label(label: str) -> str:
    """将边标签标准化为关系类型 key — 支持英文key/中文关键词/模糊匹配"""
    if not label:
        return ""
    label = label.strip()
    # 1. 精确匹配英文 key
    if label in EDGE_COLORS:
        return label
    # 2. 小写匹配
    low = label.lower().replace(" ", "_")
    if low in EDGE_COLORS:
        return low
    # 3. 尝试中文 emoji 标签匹配 (如 "🎯动机")
    for key, zh in EDGE_LABELS_ZH.items():
        if zh in label or label in zh:
            return key
    # 4. 中文关键词匹配 — 最长优先
    best_match = ""
    best_key = ""
    for kw, key in _ZH_KEYWORD_MAP.items():
        if kw in label and len(kw) > len(best_match):
            best_match = kw
            best_key = key
    if best_key:
        return best_key
    return ""


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
    渲染一个可交互的 D3.js v2 分层力导向图。

    v2 特性:
      - 节点按重要性分层 (嫌疑人内圈/证据中圈/时间外圈)
      - 边按关系类型着色 (动机=红/手段=橙/矛盾=黄...)
      - 权重感知距离 (高权重=短边=关系紧密)
      - 弱关系自动淡化 (located_at 等低透明度)
      - 智能标签 (只显示重要边的标签)

    Args:
        nodes: [{"id":"...", "label":"...", "type":"..."}]
        edges: [{"source":"...", "target":"...", "label":"...", "weight":0.8}]
        suspect_names: 嫌疑人名称列表，用于高亮
        width/height: SVG 尺寸
    """
    if not nodes:
        return _empty_graph_html("推理开始后，图谱将在此逐步构建")

    suspect_names = [s for s in (suspect_names or []) if s]

    # ---- 节点数过多时自适应尺寸 ----
    n_count = min(len(nodes), 80)
    if n_count > 40:
        width = max(width, 800)
        height = max(height, 600)

    # ---- Build sanitized data ----
    nodes_data: List[Dict] = []
    seen_nids: set = set()

    # 先算每个节点的度(连接数)，用于决定大小
    node_degree: Dict[str, int] = {}
    for e in edges:
        for key in ("source", "target"):
            nid_raw = e.get(key, "?")
            nid = _safe_id(nid_raw)
            node_degree[nid] = node_degree.get(nid, 0) + 1

    for n in nodes[:80]:
        nid = _safe_id(n.get("id", "?"))
        if nid in seen_nids:
            continue
        seen_nids.add(nid)
        label = n.get("label", n.get("id", "?"))
        ntype = n.get("type", "default")
        is_suspect = label in suspect_names or ntype == "suspect"
        tier = NODE_TIER.get(ntype, 3)
        degree = node_degree.get(nid, 0)

        nodes_data.append({
            "id": nid,
            "label": label[:16],
            "type": ntype,
            "isSuspect": is_suspect,
            "tier": tier,
            "degree": degree,
        })

    edges_data: List[Dict] = []
    seen_ekeys: set = set()
    weak_count = 0
    for e in edges[:100]:
        src = _safe_id(e.get("source", "?"))
        tgt = _safe_id(e.get("target", "?"))
        key = f"{src}|{tgt}"
        if key in seen_ekeys:
            continue
        # 跳过两端都不在 nodes 里的边
        if src not in seen_nids or tgt not in seen_nids:
            continue
        seen_ekeys.add(key)
        raw_label = e.get("label", "")
        rel_type = _normalize_edge_label(raw_label)
        weight = float(e.get("weight", 0.5))
        is_weak = rel_type in WEAK_RELATIONS

        # 限制弱关系数量 — 最多显示 15 条
        if is_weak:
            weak_count += 1
            if weak_count > 15:
                continue

        edges_data.append({
            "source": src,
            "target": tgt,
            "label": raw_label[:14],
            "weight": weight,
            "relType": rel_type,
            "isWeak": is_weak,
        })

    nodes_json = json.dumps(nodes_data, ensure_ascii=False)
    edges_json = json.dumps(edges_data, ensure_ascii=False)
    colors_json = json.dumps(NODE_COLORS, ensure_ascii=False)
    bg_json = json.dumps(NODE_BG, ensure_ascii=False)
    edge_colors_json = json.dumps(EDGE_COLORS, ensure_ascii=False)
    icons_json = json.dumps(NODE_ICONS, ensure_ascii=False)

    container_id = f"kg_{int(_time.time()*1000)}_{len(nodes_data)}"

    # ---- 统计信息 ----
    strong_edges = [e for e in edges_data if not e["isWeak"]]
    weak_edges = [e for e in edges_data if e["isWeak"]]
    stats_text = f"{len(nodes_data)} 节点 · {len(strong_edges)} 关键关系 · {len(weak_edges)} 辅助关系"

    # ---- Full HTML with inline D3.js v2 ----
    html = f'''<div id="{container_id}" class="kg-container" style="background:rgba(10,10,14,0.85);border:1px solid rgba(0,255,65,0.2);border-radius:6px;overflow:hidden;box-shadow:0 4px 24px rgba(0,255,65,0.04),0 0 40px rgba(0,255,65,0.02);backdrop-filter:blur(12px)">
<div style="background:linear-gradient(135deg,rgba(0,255,65,0.08),rgba(0,255,65,0.02));color:#00FF41;padding:10px 18px;font-family:Orbitron,monospace;font-weight:700;font-size:13px;letter-spacing:2px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(0,255,65,0.15);text-shadow:0 0 10px rgba(0,255,65,0.3)">
🧵 关联图谱 <span style="font-size:11px;font-weight:400;color:#8A8A9A;font-family:JetBrains Mono,monospace">{stats_text}</span>
</div>
<div class="kg-canvas" style="position:relative;width:{width}px;height:{height}px;background:rgba(6,6,8,0.8)">
<script src="{D3_CDN}"></script>
<script>
(function() {{
    "use strict";
    var cid = "{container_id}";
    var el = document.getElementById(cid);
    if (!el) return;

    var oldSvg = el.querySelector("svg.kg-svg");
    if (oldSvg) oldSvg.remove();

    var nodes = {nodes_json};
    var links = {edges_json};
    var colors = {colors_json};
    var bgColors = {bg_json};
    var edgeColors = {edge_colors_json};
    var icons = {icons_json};
    var W = {width}, H = {height};
    var CX = W / 2, CY = H / 2;

    /* ---- SVG ---- */
    var svg = d3.select(el).select(".kg-canvas")
        .append("svg")
        .classed("kg-svg", true)
        .attr("width", W)
        .attr("height", H)
        .style("background", "#08080A");

    var g = svg.append("g");
    svg.call(
        d3.zoom()
          .scaleExtent([0.2, 5])
          .on("zoom", function(ev) {{ g.attr("transform", ev.transform); }})
    );

    /* ---- Defs: glow ---- */
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

    /* ---- Defs: arrow markers per edge color ---- */
    var uniqueEdgeColors = new Set();
    links.forEach(function(l) {{
        var ec = edgeColors[l.relType] || "#555566";
        uniqueEdgeColors.add(ec);
    }});
    uniqueEdgeColors.forEach(function(c) {{
        var safeC = c.replace("#", "c");
        defs.append("marker")
            .attr("id", "arrow-" + safeC)
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 26).attr("refY", 0)
            .attr("markerWidth", 5).attr("markerHeight", 5)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-4L8,0L0,4")
            .attr("fill", c)
            .attr("fill-opacity", 0.7);
    }});

    /* ---- 初始化节点位置: 分层同心环 ---- */
    var tierRadii = {{ 1: Math.min(W, H) * 0.12, 2: Math.min(W, H) * 0.28, 3: Math.min(W, H) * 0.42 }};
    var tierCounts = {{ 1: 0, 2: 0, 3: 0 }};
    nodes.forEach(function(n) {{ tierCounts[n.tier || 3]++; }});

    var tierIdx = {{ 1: 0, 2: 0, 3: 0 }};
    nodes.forEach(function(n) {{
        var t = n.tier || 3;
        var r = tierRadii[t];
        var count = tierCounts[t];
        var angle = (tierIdx[t] / count) * 2 * Math.PI - Math.PI / 2;
        tierIdx[t]++;
        n.x = CX + r * Math.cos(angle);
        n.y = CY + r * Math.sin(angle);
    }});

    /* ---- Force simulation v2: 分层引力 ---- */
    var simulation = d3.forceSimulation(nodes)
        /* 连接力: 高权重=短距离(紧密) */
        .force("link", d3.forceLink(links)
            .id(function(d) {{ return d.id; }})
            .distance(function(d) {{
                if (d.isWeak) return 140;
                return Math.max(50, 130 - d.weight * 40);
            }})
            .strength(function(d) {{
                return d.isWeak ? 0.15 : 0.4 + d.weight * 0.2;
            }})
        )
        /* 排斥力: 按节点大小缩放 */
        .force("charge", d3.forceManyBody()
            .strength(function(d) {{
                if (d.isSuspect) return -500;
                return -250 - d.degree * 15;
            }})
        )
        /* 中心引力: 按 tier 分层 */
        .force("center", d3.forceCenter(CX, CY).strength(0.05))
        /* Tier 引力: 核心(tier1)被拉向中心, 外圈(tier3)可以远离 */
        .force("tierX", d3.forceX(function(d) {{
            return CX + (d.tier - 2) * W * 0.02;
        }}).strength(function(d) {{
            return d.tier === 1 ? 0.08 : (d.tier === 2 ? 0.03 : 0.01);
        }}))
        .force("tierY", d3.forceY(function(d) {{
            return CY + (d.tier - 2) * H * 0.02;
        }}).strength(function(d) {{
            return d.tier === 1 ? 0.08 : (d.tier === 2 ? 0.03 : 0.01);
        }}))
        /* 碰撞 */
        .force("collision", d3.forceCollide().radius(function(d) {{
            if (d.isSuspect) return 42;
            if (d.degree >= 5) return 32;
            return 24;
        }}))
        .alphaDecay(0.02)
        .velocityDecay(0.3);

    /* ---- Draw links (弱关系底层, 强关系上层) ---- */
    var weakLinks = links.filter(function(d) {{ return d.isWeak; }});
    var strongLinks = links.filter(function(d) {{ return !d.isWeak; }});

    /* 弱关系: 细灰虚线 */
    var weakLink = g.append("g").selectAll("line.weak")
        .data(weakLinks)
        .join("line")
        .classed("weak", true)
        .attr("stroke", "#333340")
        .attr("stroke-width", 0.8)
        .attr("stroke-opacity", 0.15)
        .attr("stroke-dasharray", "4,4");

    /* 强关系: 着色实线 */
    var strongLink = g.append("g").selectAll("line.strong")
        .data(strongLinks)
        .join("line")
        .classed("strong", true)
        .attr("stroke", function(d) {{
            return edgeColors[d.relType] || "#555566";
        }})
        .attr("stroke-width", function(d) {{ return Math.max(1.2, d.weight * 3.5); }})
        .attr("stroke-opacity", function(d) {{ return 0.3 + d.weight * 0.5; }})
        .attr("marker-end", function(d) {{
            var ec = edgeColors[d.relType] || "#555566";
            return "url(#arrow-" + ec.replace("#", "c") + ")";
        }});

    /* ---- Link labels: 只显示强关系标签 ---- */
    var linkLabel = g.append("g").selectAll("text")
        .data(strongLinks.filter(function(d) {{ return d.weight >= 0.5 && d.label; }}))
        .join("text")
        .text(function(d) {{ return d.label; }})
        .attr("font-size", "8px")
        .attr("fill", function(d) {{
            return edgeColors[d.relType] || "#555566";
        }})
        .attr("fill-opacity", 0.6)
        .attr("text-anchor", "middle")
        .attr("dy", -5)
        .style("pointer-events", "none");

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
        .attr("r", 26)
        .attr("fill", "none")
        .attr("stroke", function(d) {{ return colors[d.type] || colors["default"]; }})
        .attr("stroke-width", 2.5)
        .attr("stroke-opacity", 0.3)
        .attr("filter", "url(#kg-glow)");

    /* ---- Main circles: 大小根据度数 ---- */
    node.append("circle")
        .attr("r", 0)
        .attr("fill", function(d) {{ return bgColors[d.type] || bgColors["default"]; }})
        .attr("stroke", function(d) {{ return colors[d.type] || colors["default"]; }})
        .attr("stroke-width", function(d) {{ return d.isSuspect ? 3 : 1.5; }})
        .attr("filter", function(d) {{ return d.isSuspect ? "url(#kg-glow)" : "none"; }})
        .transition()
        .duration(600)
        .delay(function(d, i) {{ return i * 40; }})
        .ease(d3.easeBackOut.overshoot(1.1))
        .attr("r", function(d) {{
            if (d.isSuspect) return 20;
            if (d.degree >= 6) return 16;
            if (d.degree >= 3) return 13;
            return 10;
        }});

    /* ---- Node icon ---- */
    node.append("text")
        .text(function(d) {{ return icons[d.type] || "•"; }})
        .attr("text-anchor", "middle")
        .attr("dy", 1)
        .attr("font-size", function(d) {{
            if (d.isSuspect) return "14px";
            if (d.degree >= 3) return "11px";
            return "9px";
        }})
        .style("pointer-events", "none")
        .style("opacity", 0)
        .transition()
        .duration(400)
        .delay(function(d, i) {{ return i * 40 + 150; }})
        .style("opacity", 1);

    /* ---- Node labels ---- */
    node.append("text")
        .text(function(d) {{ return d.label; }})
        .attr("font-size", function(d) {{
            if (d.isSuspect) return "11px";
            if (d.degree >= 3) return "10px";
            return "9px";
        }})
        .attr("font-weight", function(d) {{ return d.isSuspect ? "700" : "500"; }})
        .attr("fill", function(d) {{ return colors[d.type] || colors["default"]; }})
        .attr("text-anchor", "middle")
        .attr("dy", function(d) {{
            var r = d.isSuspect ? 20 : (d.degree >= 6 ? 16 : (d.degree >= 3 ? 13 : 10));
            return r + 12;
        }})
        .style("pointer-events", "none")
        .style("text-shadow", function(d) {{ return "0 0 6px " + (colors[d.type] || colors["default"]); }})
        .style("opacity", 0)
        .transition()
        .duration(400)
        .delay(function(d, i) {{ return i * 40 + 80; }})
        .style("opacity", 1);

    /* ---- Tooltip ---- */
    node.append("title")
        .text(function(d) {{
            var typeLabel = d.type;
            var info = d.label + " [" + typeLabel + "]";
            if (d.isSuspect) info += " ⚠嫌疑人";
            info += " (连接数:" + d.degree + ")";
            return info;
        }});

    /* ---- Tick ---- */
    simulation.on("tick", function() {{
        weakLink
            .attr("x1", function(d) {{ return d.source.x; }})
            .attr("y1", function(d) {{ return d.source.y; }})
            .attr("x2", function(d) {{ return d.target.x; }})
            .attr("y2", function(d) {{ return d.target.y; }});
        strongLink
            .attr("x1", function(d) {{ return d.source.x; }})
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

    /* ---- Drag ---- */
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
# Incremental variant
# ---------------------------------------------------------------------------
def render_incremental_graph(
    all_nodes: List[Dict],
    all_edges: List[Dict],
    suspect_names: List[str] | None = None,
    new_nodes: List[Dict] | None = None,
    new_edges: List[Dict] | None = None,
) -> str:
    """渲染增量图谱 — 新节点有特殊入场动画。"""
    return render_force_graph(all_nodes, all_edges, suspect_names)
