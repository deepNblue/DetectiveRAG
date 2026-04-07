"""实时推理日志构建器"""
from datetime import datetime
# 使用纯SVG渲染器（Gradio不执行<script>，D3.js方案不可用）
from ui.graph_renderer_svg import render_force_graph

class ReasoningLog:
    def __init__(self):
        self.steps = []
        self.node_count = 0
        self.edge_count = 0
        self._suspect_names: list = []  # 缓存嫌疑人名称用于图谱高亮

    def ts(self):
        return datetime.now().strftime("%H:%M:%S")

    def stage(self, icon, title, status="running"):
        self.steps.append({"type":"stage","icon":icon,"title":title,
                          "status":status,"ts":self.ts(),"details":[]})

    def finish(self, title, status="done", msg=""):
        for s in reversed(self.steps):
            if s["title"]==title:
                s["status"]=status
                if msg:
                    s["details"].append({"type":"text","c":msg,"ts":self.ts()})
                break

    def thought(self, text, style="normal"):
        if self.steps:
            self.steps[-1]["details"].append({"type":"thought","c":text,"s":style,"ts":self.ts()})

    def nodes(self, items):
        self.node_count += len(items)
        if self.steps:
            self.steps[-1]["details"].append({"type":"nodes","items":items,"ts":self.ts()})

    def edges(self, items):
        self.edge_count += len(items)
        if self.steps:
            self.steps[-1]["details"].append({"type":"edges","items":items,"ts":self.ts()})

    def vote(self, expert, culprit, conf):
        if self.steps:
            self.steps[-1]["details"].append({"type":"vote","expert":expert,
                "culprit":culprit,"conf":conf,"ts":self.ts()})

    def contra(self, level, person, desc):
        if self.steps:
            self.steps[-1]["details"].append({"type":"contra","level":level,
                "person":person,"desc":desc,"ts":self.ts()})

    def conclusion(self, text):
        if self.steps:
            self.steps[-1]["details"].append({"type":"conc","c":text,"ts":self.ts()})

    def _safe_id(self, raw_id):
        """Convert node id to Mermaid-safe identifier"""
        return raw_id.replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "").replace("（", "").replace("）", "")

    def set_suspect_names(self, names_list: list):
        """缓存嫌疑人名称，用于图谱高亮"""
        self._suspect_names = list(names_list)

    def render_graph(self):
        """Render accumulated nodes/edges as an interactive D3.js force-directed graph"""
        # Collect all nodes and edges from all steps
        all_nodes = []
        all_edges = []
        for s in self.steps:
            for d in s["details"]:
                if d["type"] == "nodes":
                    all_nodes.extend(d["items"])
                elif d["type"] == "edges":
                    all_edges.extend(d["items"])

        # Deduplicate nodes
        seen = set()
        unique_nodes = []
        for n in all_nodes:
            nid = n.get("id", "?")
            if nid not in seen:
                seen.add(nid)
                unique_nodes.append(n)

        # Deduplicate edges
        seen_e = set()
        unique_edges = []
        for e in all_edges:
            key = f"{e.get('source','?')}|{e.get('target','?')}"
            if key not in seen_e:
                seen_e.add(key)
                unique_edges.append(e)

        return render_force_graph(
            nodes=unique_nodes,
            edges=unique_edges,
            suspect_names=self._suspect_names,
        )

    def _ntag(self, n):
        tc = {"person":"ntag-p","evidence":"ntag-e","location":"ntag-l","event":"ntag-v"}
        lbl = n.get("label", n.get("id","?"))
        if len(lbl)>8: lbl=lbl[:8]+".."
        cls = tc.get(n.get("type",""),"ntag-t")
        return f'<span class="ntag {cls}">{lbl}</span>'

    def render(self):
        si = {"running":"🔵","done":"✅","error":"❌","pending":"⏳"}
        sc = {"running":"step-active","done":"step-done","error":"step-err","pending":""}
        h = '<div class="live-panel">'
        h += f'<div class="live-header">🧵 推理线程 <span style="font-size:11px;font-weight:400;color:#55556A;font-family:JetBrains Mono,monospace">节点:{self.node_count} 关系:{self.edge_count}</span></div>'
        h += '<div class="live-body">'
        for s in self.steps:
            ico = si.get(s["status"],"⏳")
            cls = sc.get(s["status"],"")
            h += f'<div class="step-box {cls}">'
            h += f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">'
            h += f'<span>{ico}</span><strong style="color:#E0E0E8;font-size:12px;font-family:Orbitron,monospace;letter-spacing:1px">{s["icon"]} {s["title"]}</strong>'
            h += f'<span style="margin-left:auto;font-size:10px;color:#55556A;font-family:JetBrains Mono,monospace">{s["ts"]}</span></div>'
            for d in s["details"]:
                if d["type"]=="text":
                    h += f'<div style="margin:3px 0 3px 20px;font-size:11px;color:#8A8A9A">{d["c"]}</div>'
                elif d["type"]=="thought":
                    tc = "thought"
                    if d["s"]=="warn": tc="thought-warn"
                    elif d["s"]=="danger": tc="thought-danger"
                    h += f'<div class="{tc}">💭 {d["c"]}</div>'
                elif d["type"]=="nodes":
                    tags = " ".join(self._ntag(n) for n in d["items"][:20])
                    h += f'<div style="margin:3px 0 3px 20px">🏷 {tags}</div>'
                elif d["type"]=="edges":
                    for e in d["items"][:12]:
                        src=e.get("source","?")[:6]; tgt=e.get("target","?")[:6]
                        rel=e.get("label","")[:10]
                        h += f'<div style="margin:2px 0 2px 20px;font-size:10px;color:#8A8A9A"><span class="ntag ntag-p">{src}</span> <span class="rarr">—[{rel}]→</span> <span class="ntag ntag-p">{tgt}</span></div>'
                elif d["type"]=="vote":
                    pct=d["conf"]*100
                    bc="#00FF41" if pct>70 else "#FFB800" if pct>40 else "#FF003C"
                    h += f'<div style="margin:3px 0 3px 20px;font-size:11px">🧑‍⚖️ <b>{d["expert"]}</b> → <code>{d["culprit"]}</code> <span class="vote-bar"><span style="display:inline-block;width:{pct}%;height:100%;background:{bc};border-radius:2px;box-shadow:0 0 6px {bc}"></span></span> {pct:.0f}%</div>'
                elif d["type"]=="contra":
                    ci="🔴" if d["level"]=="high" else "🟡" if d["level"]=="medium" else "🟢"
                    h += f'<div style="margin:3px 0 3px 20px;font-size:11px;color:#FF003C">{ci} [{d["level"]}] {d["person"]}: {d["desc"][:80]}</div>'
                elif d["type"]=="conc":
                    h += f'<div style="margin:4px 0 3px 20px;padding:6px 10px;background:rgba(0,255,65,0.06);border-radius:4px;border:1px solid rgba(0,255,65,0.2);font-size:11px;color:#00FF41;font-weight:600;font-family:JetBrains Mono,monospace;text-shadow:0 0 6px rgba(0,255,65,0.3)">📌 {d["c"]}</div>'
            h += '</div>'
        h += '</div></div>'
        return h
