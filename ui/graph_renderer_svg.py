"""
静态SVG知识图谱渲染器 — 纯SVG+CSS，不依赖JavaScript
用于在 Gradio HTML 组件中渲染知识图谱（Gradio不执行<script>标签）

特性:
  - 增量布局: 已有节点位置固定，新节点从已有邻居附近生长
  - Fruchterman-Reingold 弹簧布局
  - 纯 SVG+CSS，Gradio 100% 可渲染
"""
import math
import random
from typing import Dict, List, Any, Optional, Tuple

# ---------------------------------------------------------------------------
# 颜色映射 — 赛博侦探(Cyber-Sleuth)主题
# ---------------------------------------------------------------------------
NODE_COLORS = {
    "person":   "#00FF41",
    "evidence": "#FFB800",
    "location": "#00F5FF",
    "time":     "#8A8A9A",
    "event":    "#B37FEB",
    "default":  "#8A8A9A",
}

# 🆕 v14: 边关系类型颜色映射 — 按关系语义分类着色
EDGE_COLORS = {
    # 嫌疑/指向 (红色系 — 最醒目)
    "嫌疑人":    "#FF003C",
    "指向":      "#FF003C",
    "关联":      "#FF4466",
    "涉及":      "#FF6B6B",
    "动机":      "#FF2D55",
    
    # 证据 (金色系)
    "证据":      "#FFB800",
    "物证":      "#FFD700",
    "证明":      "#FFCC33",
    "提供":      "#FFAA00",
    
    # 时空 (蓝色系)
    "时间":      "#00F5FF",
    "地点":      "#00CFFF",
    "出现在":    "#0099FF",
    "目击":      "#0088FF",
    
    # 社交关系 (紫色系)
    "认识":      "#B37FEB",
    "亲属":      "#9B59B6",
    "朋友":      "#8E44AD",
    "合作":      "#7D3C98",
    "同事":      "#6C3483",
    
    # 对立/矛盾 (橙红系)
    "矛盾":      "#FF6600",
    "冲突":      "#FF4500",
    "矛盾点":    "#FF3300",
}

def _get_edge_color(label: str) -> str:
    """根据边标签获取对应颜色，模糊匹配"""
    if not label:
        return "#3A3A45"
    # 精确匹配
    if label in EDGE_COLORS:
        return EDGE_COLORS[label]
    # 模糊匹配: 标签包含关键词
    for key, color in EDGE_COLORS.items():
        if key in label:
            return color
    # 默认暗灰色
    return "#3A3A45"

NODE_BG = {
    "person":   "rgba(0,255,65,0.12)",
    "evidence": "rgba(255,184,0,0.12)",
    "location": "rgba(0,245,255,0.12)",
    "time":     "rgba(138,138,154,0.1)",
    "event":    "rgba(179,127,235,0.12)",
    "default":  "rgba(138,138,154,0.1)",
}

TYPE_ICONS = {
    "person": "👤", "evidence": "📌", "location": "📍",
    "time": "🕐", "event": "⚡", "default": "•",
}

# ---------------------------------------------------------------------------
# 全局布局缓存 — 跨次 render 调用保持已有节点位置
# ---------------------------------------------------------------------------
_prev_positions: Dict[str, Tuple[float, float]] = {}
_prev_node_keys: set = set()  # 用于检测哪些节点是"旧的"
_prev_edge_keys: set = set()  # 用于检测哪些边是"旧的"


def _safe_id(raw: str) -> str:
    return (raw.replace("-", "_").replace(" ", "_").replace("(", "")
              .replace(")", "").replace("（", "").replace("）", "")
              .replace('"', "").replace("'", "").replace("/", "_"))


def _empty_graph_html(msg: str) -> str:
    return (
        '<div style="background:rgba(10,10,14,0.85);border:1px solid rgba(0,255,65,0.2);'
        'border-radius:6px;overflow:hidden;backdrop-filter:blur(12px)">'
        '<div style="background:linear-gradient(135deg,rgba(0,255,65,0.08),rgba(0,255,65,0.02));'
        'color:#00FF41;padding:10px 18px;font-family:Orbitron,monospace;font-weight:700;font-size:13px;'
        'letter-spacing:2px;display:flex;align-items:center;justify-content:space-between;'
        'border-bottom:1px solid rgba(0,255,65,0.15)">'
        '🧵 关联图谱 <span style="font-size:11px;font-weight:400;color:#8A8A9A;'
        'font-family:JetBrains Mono,monospace">待命…</span></div>'
        '<div style="padding:40px;color:#55556A;text-align:center;background:rgba(6,6,8,0.6);'
        f'font-size:12px;font-family:JetBrains Mono,monospace">{msg}</div></div>'
    )


def reset_layout_cache():
    """清除布局缓存（新案件开始时调用）"""
    global _prev_positions, _prev_node_keys, _prev_edge_keys
    _prev_positions.clear()
    _prev_node_keys.clear()
    _prev_edge_keys.clear()


def _incremental_layout(
    nodes: List[Dict],
    edges: List[Dict],
    width: int = 960,
    height: int = 520,
    iterations: int = 60,
) -> Dict[str, Tuple[float, float]]:
    """
    增量弹簧布局:
    - 已有节点: 位置固定不动
    - 新节点: 放在已有关联邻居的附近，然后只对**新节点**做弹簧迭代
    """
    global _prev_positions, _prev_node_keys, _prev_edge_keys

    if not nodes:
        return {}

    n = len(nodes)
    node_ids = [_safe_id(nd.get("id", f"n{i}")) for i, nd in enumerate(nodes)]
    current_node_keys = set(node_ids)

    # 当前边的 key 集合
    current_edge_keys = set()
    for e in edges:
        src = _safe_id(e.get("source", "?"))
        tgt = _safe_id(e.get("target", "?"))
        current_edge_keys.add(f"{src}|{tgt}")

    # 构建 adjacency (用 safe id)
    adj = {nid: set() for nid in node_ids}
    for e in edges:
        src = _safe_id(e.get("source", "?"))
        tgt = _safe_id(e.get("target", "?"))
        if src in adj and tgt in adj:
            adj[src].add(tgt)
            adj[tgt].add(src)

    # ---- 分离旧节点和新节点 ----
    old_ids = [nid for nid in node_ids if nid in _prev_positions]
    new_ids = [nid for nid in node_ids if nid not in _prev_positions]

    pos = {}

    # 旧节点: 保持位置不变
    for nid in old_ids:
        pos[nid] = _prev_positions[nid]

    # 新节点: 放置在已有邻居附近
    if new_ids:
        for nid in new_ids:
            # 找到已有关联的邻居（位置已确定）
            placed_neighbors = [nb for nb in adj.get(nid, set()) if nb in pos]
            if placed_neighbors:
                # 取第一个邻居的位置，加随机偏移
                bx, by = pos[placed_neighbors[0]]
                pos[nid] = (bx + random.uniform(-50, 50),
                            by + random.uniform(-50, 50))
            else:
                # 无邻居，随机放在画布上
                pos[nid] = (random.uniform(40, width - 40),
                            random.uniform(40, height - 40))

        # ---- 只对新节点做弹簧迭代（旧节点不动）----
        area = width * height
        k = math.sqrt(area / max(n, 1)) * 0.8
        temperature = width / 6.0

        for iteration in range(iterations):
            disp = {nid: [0.0, 0.0] for nid in new_ids}

            # 排斥力: 新节点之间
            for i in range(len(new_ids)):
                for j in range(i + 1, len(new_ids)):
                    dx = pos[new_ids[i]][0] - pos[new_ids[j]][0]
                    dy = pos[new_ids[i]][1] - pos[new_ids[j]][1]
                    dist = max(math.sqrt(dx * dx + dy * dy), 0.1)
                    force = (k * k) / dist
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    disp[new_ids[i]][0] += fx
                    disp[new_ids[i]][1] += fy
                    disp[new_ids[j]][0] -= fx
                    disp[new_ids[j]][1] -= fy

            # 排斥力: 新节点 vs 旧节点（旧不动）
            for nid in new_ids:
                for oid in old_ids:
                    dx = pos[nid][0] - pos[oid][0]
                    dy = pos[nid][1] - pos[oid][1]
                    dist = max(math.sqrt(dx * dx + dy * dy), 0.1)
                    force = (k * k) / dist
                    disp[nid][0] += (dx / dist) * force
                    disp[nid][1] += (dy / dist) * force

            # 吸引力: 新节点的所有边
            for e in edges:
                src = _safe_id(e.get("source", "?"))
                tgt = _safe_id(e.get("target", "?"))
                if src in pos and tgt in pos:
                    # 只处理至少一端是新节点的边
                    if src not in new_ids and tgt not in new_ids:
                        continue
                    dx = pos[src][0] - pos[tgt][0]
                    dy = pos[src][1] - pos[tgt][1]
                    dist = max(math.sqrt(dx * dx + dy * dy), 0.1)
                    force = (dist * dist) / k * 0.4
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    if src in new_ids:
                        disp[src][0] -= fx
                        disp[src][1] -= fy
                    if tgt in new_ids:
                        disp[tgt][0] += fx
                        disp[tgt][1] += fy

            # 应用力到新节点
            for nid in new_ids:
                dx, dy = disp[nid]
                dist = max(math.sqrt(dx * dx + dy * dy), 0.1)
                limited = min(dist, temperature)
                pos[nid] = (
                    max(25, min(width - 25, pos[nid][0] + (dx / dist) * limited)),
                    max(25, min(height - 25, pos[nid][1] + (dy / dist) * limited)),
                )
            temperature *= 0.90

    elif not old_ids:
        # 全新图谱（没有任何旧节点）— 完整布局
        pos = _full_layout(node_ids, edges, adj, width, height, iterations)

    # 缓存
    _prev_positions.update(pos)
    _prev_node_keys = current_node_keys
    _prev_edge_keys = current_edge_keys

    return pos


def _full_layout(
    node_ids: List[str],
    edges: List[Dict],
    adj: Dict[str, set],
    width: int,
    height: int,
    iterations: int = 80,
) -> Dict[str, Tuple[float, float]]:
    """完整 Fruchterman-Reingold 布局（首次渲染时使用）— v15.2: 重要性感知布局"""
    n = len(node_ids)
    
    # 🆕 计算节点重要性评分（基于连接度和类型）
    node_importance = {}
    for nid in node_ids:
        # 基础分数
        score = 0.5
        
        # 类型加分: 嫌疑人/证据 > 更重要
        # 从adj中获取节点信息（adj是nid到节点数据的映射）
        # 注意：这里的adj实际上是节点ID到邻居集合的映射
        # 我们需要从原始nodes列表中获取类型信息
        # 但在这个函数中，我们没有原始nodes列表，只有node_ids
        # 所以我们暂时只基于连接度评分
        
        # 连接度加分
        deg = len(adj.get(nid, set()))
        score += min(2.0, deg / 3.0)  # 最多加2.0
        
        node_importance[nid] = score
    
    # 🆕 按重要性分层放置
    # Tier 1 (核心): importance >= 1.5 → 内圈 (距中心 50-100px)
    # Tier 2 (重要): 1.0 <= importance < 1.5 → 中圈 (距中心 150-200px)
    # Tier 3 (普通): importance < 1.0 → 外圈 (距中心 250-300px)
    cx, cy = width / 2, height / 2
    pos = {}
    
    for nid in node_ids:
        importance = node_importance[nid]
        
        if importance >= 1.5:
            # 核心节点 → 内圈
            r = random.uniform(50, 100)
            angle = random.uniform(0, 2 * math.pi)
            pos[nid] = (cx + r * math.cos(angle), cy + r * math.sin(angle))
        elif importance >= 1.0:
            # 重要节点 → 中圈
            r = random.uniform(150, 200)
            angle = random.uniform(0, 2 * math.pi)
            pos[nid] = (cx + r * math.cos(angle), cy + r * math.sin(angle))
        else:
            # 普通节点 → 外圈
            r = random.uniform(250, 300)
            angle = random.uniform(0, 2 * math.pi)
            pos[nid] = (cx + r * math.cos(angle), cy + r * math.sin(angle))

    area = width * height
    k = math.sqrt(area / max(n, 1)) * 0.8
    temperature = width / 6.0  # 🔽 降低初始温度，减少节点移动

    for iteration in range(iterations):
        disp = {nid: [0.0, 0.0] for nid in node_ids}

        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[node_ids[i]][0] - pos[node_ids[j]][0]
                dy = pos[node_ids[i]][1] - pos[node_ids[j]][1]
                dist = max(math.sqrt(dx * dx + dy * dy), 0.1)
                force = (k * k) / dist
                fx = (dx / dist) * force
                fy = (dy / dist) * force
                disp[node_ids[i]][0] += fx
                disp[node_ids[i]][1] += fy
                disp[node_ids[j]][0] -= fx
                disp[node_ids[j]][1] -= fy

        for e in edges:
            src = _safe_id(e.get("source", "?"))
            tgt = _safe_id(e.get("target", "?"))
            if src in pos and tgt in pos:
                dx = pos[src][0] - pos[tgt][0]
                dy = pos[src][1] - pos[tgt][1]
                dist = max(math.sqrt(dx * dx + dy * dy), 0.1)
                force = (dist * dist) / k
                fx = (dx / dist) * force
                fy = (dy / dist) * force
                disp[src][0] -= fx * 0.3
                disp[src][1] -= fy * 0.3
                disp[tgt][0] += fx * 0.3
                disp[tgt][1] += fy * 0.3

        # 🆕 向心力: 核心节点被拉向中心
        for nid in node_ids:
            dx = pos[nid][0] - cx
            dy = pos[nid][1] - cy
            dist_to_center = math.sqrt(dx * dx + dy * dy)
            if dist_to_center > 0:
                importance = node_importance[nid]
                # 向心力强度与重要性成正比
                center_force = importance * 0.8
                disp[nid][0] -= (dx / dist_to_center) * center_force
                disp[nid][1] -= (dy / dist_to_center) * center_force

        for nid in node_ids:
            dx, dy = disp[nid]
            dist = max(math.sqrt(dx * dx + dy * dy), 0.1)
            limited = min(dist, temperature)
            pos[nid] = (
                max(25, min(width - 25, pos[nid][0] + (dx / dist) * limited)),
                max(25, min(height - 25, pos[nid][1] + (dy / dist) * limited)),
            )
        temperature *= 0.90  # 🔽 降低温度衰减率，让布局更稳定

    return pos


def render_force_graph(
    nodes: List[Dict],
    edges: List[Dict],
    suspect_names: Optional[List[str]] = None,
    width: int = 960,
    height: int = 520,
) -> str:
    """
    渲染纯 SVG 知识图谱 — 增量布局，旧节点不动。

    Args:
        nodes: [{"id":"...", "label":"...", "type":"person|evidence|location|time|event"}]
        edges: [{"source":"...", "target":"...", "label":"...", "weight":0.8}]
        suspect_names: 嫌疑人名称列表，用于高亮
        width/height: SVG 尺寸

    Returns:
        HTML string with inline SVG visualization
    """
    if not nodes:
        return _empty_graph_html("推理开始后，图谱将在此逐步构建")

    suspect_names = [s for s in (suspect_names or []) if s]

    # ---- Deduplicate ----
    seen_nids = set()
    unique_nodes = []
    for n in nodes[:80]:
        nid = _safe_id(n.get("id", "?"))
        if nid in seen_nids:
            continue
        seen_nids.add(nid)
        label = n.get("label", n.get("id", "?"))
        is_suspect = label in suspect_names
        unique_nodes.append({
            "id": nid,
            "label": label[:14],
            "type": n.get("type", "default"),
            "isSuspect": is_suspect,
        })

    seen_ekeys = set()
    unique_edges = []
    for e in edges[:80]:
        src = _safe_id(e.get("source", "?"))
        tgt = _safe_id(e.get("target", "?"))
        key = f"{src}|{tgt}"
        if key in seen_ekeys:
            continue
        seen_ekeys.add(key)
        unique_edges.append({
            "source": src,
            "target": tgt,
            "label": e.get("label", "")[:10],
            "weight": float(e.get("weight", 0.5)),
        })

    # ---- 增量布局 ----
    canvas_h = max(400, min(580, height))
    pos = _incremental_layout(unique_nodes, unique_edges, width, canvas_h)

    # ---- 判断新旧节点（用于渲染动画） ----
    new_node_ids = set()
    for nd in unique_nodes:
        if nd["id"] not in _prev_node_keys:
            new_node_ids.add(nd["id"])

    # ---- Build SVG ----
    svg_parts = []

    # Defs
    svg_parts.append('<defs>')
    svg_parts.append(
        '<filter id="kg-glow" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="3" result="blur"/>'
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
        '</filter>'
    )
    # 新节点入场动画
    svg_parts.append(
        '<style>'
        '@keyframes nodeAppear { from { opacity: 0; transform: scale(0.5); } to { opacity: 1; transform: scale(1); } }'
        '.node-new { animation: nodeAppear 0.6s ease-out; }'
        '</style>'
    )
    for ntype, color in NODE_COLORS.items():
        svg_parts.append(
            f'<marker id="arrow-{ntype}" viewBox="0 -5 10 10" refX="26" refY="0" '
            f'markerWidth="5" markerHeight="5" orient="auto">'
            f'<path d="M0,-5L10,0L0,5" fill="{color}" opacity="0.6"/></marker>'
        )
    svg_parts.append('</defs>')

    # ---- Edges ----
    for e in unique_edges:
        src = e["source"]
        tgt = e["target"]
        if src not in pos or tgt not in pos:
            continue
        x1, y1 = pos[src]
        x2, y2 = pos[tgt]
        weight = e.get("weight", 0.5)
        opacity = 0.3 + weight * 0.5
        stroke_w = max(1.0, weight * 1.5)

        # 🆕 v14: 按关系类型着色
        edge_label = e.get("label", "")
        edge_color = _get_edge_color(edge_color if False else edge_label)

        src_node = next((n for n in unique_nodes if n["id"] == src), None)
        src_type = src_node["type"] if src_node else "default"

        # 新边的动画 class
        is_new_edge = src in new_node_ids or tgt in new_node_ids
        edge_style = f'stroke-opacity:{opacity:.2f}'
        if is_new_edge:
            edge_style += ';stroke-dasharray:6,3'

        svg_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{edge_color}" stroke-width="{stroke_w:.1f}" '
            f'style="{edge_style}" '
            f'marker-end="url(#arrow-{src_type})"/>'
        )

        if e.get("label"):
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            svg_parts.append(
                f'<text x="{mx:.1f}" y="{my:.1f}" '
                f'font-size="9" fill="{edge_color}" text-anchor="middle" '
                f'font-family="JetBrains Mono,monospace" dy="-4" '
                f'opacity="0.85">'
                f'{e["label"]}</text>'
            )

    # ---- Nodes ----
    for nd in unique_nodes:
        if nd["id"] not in pos:
            continue
        x, y = pos[nd["id"]]
        color = NODE_COLORS.get(nd["type"], NODE_COLORS["default"])
        bg = NODE_BG.get(nd["type"], NODE_BG["default"])
        is_sus = nd.get("isSuspect", False)
        is_new = nd["id"] in new_node_ids
        r = 24 if is_sus else 16
        sw = 3 if is_sus else 1.5
        icon = TYPE_ICONS.get(nd["type"], "•")
        anim_cls = ' class="node-new"' if is_new else ""

        if is_sus:
            svg_parts.append(
                f'<g{anim_cls}>'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 10}" '
                f'fill="none" stroke="{color}" stroke-width="2" '
                f'stroke-opacity="0.25" filter="url(#kg-glow)"/>'
            )

        glow_attr = ' filter="url(#kg-glow)"' if is_sus else ""
        svg_parts.append(
            f'<g{anim_cls if not is_sus else ""}>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" '
            f'fill="{bg}" stroke="{color}" stroke-width="{sw}"'
            f'{glow_attr}/>'
        )

        # Type icon
        svg_parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" '
            f'font-size="{18 if is_sus else 14}" text-anchor="middle" '
            f'dominant-baseline="central" style="pointer-events:none">'
            f'{icon}</text>'
        )

        # Label
        font_size = 13 if is_sus else 11
        label_y = y + r + 14
        svg_parts.append(
            f'<text x="{x:.1f}" y="{label_y:.1f}" '
            f'font-size="{font_size}" fill="{color}" text-anchor="middle" '
            f'font-weight="{700 if is_sus else 500}" '
            f'font-family="JetBrains Mono,monospace" '
            f'style="text-shadow:0 0 6px {color};pointer-events:none">'
            f'{nd["label"]}</text>'
        )

        if is_sus:
            svg_parts.append(
                f'<text x="{x + r - 2:.1f}" y="{y - r + 4:.1f}" '
                f'font-size="8" fill="#FF003C" font-weight="700">⚠</text>'
            )

        svg_parts.append('</g>')
        if is_sus:
            svg_parts.append('</g>')

    svg_content = "\n".join(svg_parts)

    # ---- HTML wrapper ----
    html = (
        f'<div style="background:rgba(10,10,14,0.85);border:1px solid rgba(0,255,65,0.2);'
        f'border-radius:6px;overflow:hidden;box-shadow:0 4px 24px rgba(0,255,65,0.04);'
        f'backdrop-filter:blur(12px)">'
        f'<div style="background:linear-gradient(135deg,rgba(0,255,65,0.08),rgba(0,255,65,0.02));'
        f'color:#00FF41;padding:10px 18px;font-family:Orbitron,monospace;font-weight:700;'
        f'font-size:13px;letter-spacing:2px;display:flex;align-items:center;'
        f'justify-content:space-between;border-bottom:1px solid rgba(0,255,65,0.15);'
        f'text-shadow:0 0 10px rgba(0,255,65,0.3)">'
        f'🧵 关联图谱 <span style="font-size:11px;font-weight:400;color:#8A8A9A;'
        f'font-family:JetBrains Mono,monospace">'
        f'{len(unique_nodes)} 节点 · {len(unique_edges)} 关系</span></div>'
        f'<div style="background:rgba(6,6,8,0.8)">'
        f'<svg width="{width}" height="{canvas_h}" viewBox="0 0 {width} {canvas_h}" '
        f'style="background:#08080A;display:block;max-width:100%">'
        f'{svg_content}'
        f'</svg></div></div>'
    )
    return html


def render_incremental_graph(
    all_nodes: List[Dict],
    all_edges: List[Dict],
    suspect_names: Optional[List[str]] = None,
    new_nodes: Optional[List[Dict]] = None,
    new_edges: Optional[List[Dict]] = None,
) -> str:
    return render_force_graph(all_nodes, all_edges, suspect_names)
