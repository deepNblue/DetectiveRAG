"""
🧠 Brain Palace — 思维殿堂可视化渲染器
忒修斯之线 v10.0 · 赛博侦探 UI

四主角系统 + 逻辑碎片碰撞 + 逻辑漩涡投票 + 裁决栏
"""

import time
import html as html_lib
from typing import Dict, List, Optional


class BrainPalace:
    """思维殿堂可视化渲染器 — 管理推理过程的实时可视化状态"""

    # ── 角色定义 ──
    AGENT_DEFS = {
        "puppeteer": {"name": "Puppeteer",  "cn": "提线木偶", "icon": "🧑‍💻", "color": "#00FF41", "role": "调查控制"},
        "mirror":    {"name": "Mirror",     "cn": "碎镜人",   "icon": "🧠", "color": "#00F5FF", "role": "矛盾搜索"},
        "chronos":   {"name": "Chronos",    "cn": "计时员",   "icon": "🔬", "color": "#FFD700", "role": "时间线"},
        "compass":   {"name": "Compass",    "cn": "指南针",   "icon": "🧭", "color": "#FF003C", "role": "方向投票"},
    }

    STATUS_ICONS = {
        "IDLE":      "⏳",
        "ANALYZING": "⚡",
        "VOTING":    "🗳️",
        "DONE":      "✅",
    }

    def __init__(self):
        self.agents: Dict[str, dict] = {}
        for aid, defs in self.AGENT_DEFS.items():
            self.agents[aid] = {
                **defs,
                "stress": 0,
                "status": "IDLE",
            }
        self.fragments: List[dict] = []
        self.votes: List[dict] = []
        self.verdict: Optional[dict] = None
        self.sync_rate: int = 0
        self.shield_active: bool = False
        self.timeline_events: List[dict] = []
        self._render_id: int = 0  # for unique CSS animation IDs

    # ── 状态更新 API ──

    def update_agent(self, agent_id: str, stress: int, status: str):
        """更新角色状态 (stress 0-100, status: IDLE/ANALYZING/VOTING/DONE)"""
        if agent_id in self.agents:
            self.agents[agent_id]["stress"] = max(0, min(100, stress))
            self.agents[agent_id]["status"] = status

    def add_fragment(self, agent_id: str, content: str):
        """添加逻辑碎片"""
        color = self.agents.get(agent_id, {}).get("color", "#888")
        name = self.agents.get(agent_id, {}).get("name", "?")
        self.fragments.append({
            "id": len(self.fragments),
            "agent_id": agent_id,
            "agent_name": name,
            "content": content,
            "color": color,
            "timestamp": time.time(),
        })

    def add_vote(self, agent_id: str, suspect: str, confidence: float):
        """记录投票"""
        color = self.agents.get(agent_id, {}).get("color", "#888")
        name = self.agents.get(agent_id, {}).get("name", "?")
        self.votes.append({
            "agent_id": agent_id,
            "agent_name": name,
            "suspect": suspect,
            "confidence": confidence,
            "color": color,
        })

    def add_timeline_event(self, text: str, event_type: str = "info"):
        """添加时间线事件"""
        ts = time.strftime("%H:%M:%S")
        self.timeline_events.append({
            "text": text,
            "type": event_type,  # info, warning, danger, success
            "timestamp": ts,
        })

    def set_verdict(self, suspect: str, confidence: float,
                    overridden: bool = False, shield: bool = False):
        """设置最终裁决"""
        self.verdict = {
            "suspect": suspect,
            "confidence": confidence,
            "overridden": overridden,
            "shield": shield,
        }
        self.shield_active = shield

    def set_sync_rate(self, rate: int):
        """设置同步率 (0-100)"""
        self.sync_rate = max(0, min(100, rate))

    # ── HTML 渲染 ──

    def render(self) -> str:
        """渲染完整思维殿堂 HTML (CSS + HTML + JS)"""
        self._render_id += 1
        rid = self._render_id

        css = self._render_css(rid)
        left = self._render_agents_panel()
        center = self._render_palace_center(rid)
        right = self._render_timeline_panel()
        bottom = self._render_verdict_bar()

        return f"""<div class="bp-root-{rid}">
{css}
<div class="bp-container">
  <div class="bp-layout">
    <div class="bp-left">{left}</div>
    <div class="bp-center">{center}</div>
    <div class="bp-right">{right}</div>
  </div>
  <div class="bp-bottom">{bottom}</div>
</div>
</div>"""

    def render_empty(self) -> str:
        """渲染空状态（等待推理）"""
        return """<div style="
    background: rgba(6,6,10,0.9);
    border: 1px solid rgba(0,255,65,0.15);
    border-radius: 8px;
    padding: 40px 20px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    backdrop-filter: blur(12px);
">
  <div style="font-size:28px;margin-bottom:12px">🧠</div>
  <div style="color:#00FF41;font-family:Orbitron,monospace;font-size:14px;letter-spacing:3px;margin-bottom:8px">思维殿堂</div>
  <div style="color:#55556A;font-size:11px">等待推理协议启动…</div>
</div>"""

    # ── 内部渲染方法 ──

    def _render_css(self, rid: int) -> str:
        """生成作用域内联 CSS"""
        return f"""<style>
.bp-root-{rid} {{
  font-family: 'JetBrains Mono', monospace;
  color: #c8c8d0;
  --c-puppeteer: #00FF41;
  --c-mirror: #00F5FF;
  --c-chronos: #FFD700;
  --c-compass: #FF003C;
  --bg-card: rgba(10,10,18,0.85);
  --border-glow: rgba(0,255,65,0.18);
}}
.bp-root-{rid} * {{ box-sizing: border-box; }}
.bp-container {{
  background: rgba(6,6,10,0.92);
  border: 1px solid var(--border-glow);
  border-radius: 8px;
  overflow: hidden;
  backdrop-filter: blur(16px);
}}
.bp-layout {{
  display: flex;
  min-height: 380px;
}}
/* ── Left: Agent Cards ── */
.bp-left {{
  width: 170px;
  min-width: 150px;
  border-right: 1px solid rgba(0,255,65,0.1);
  padding: 10px 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow-y: auto;
}}
.bp-agent-card {{
  background: var(--bg-card);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 6px;
  padding: 10px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s;
}}
.bp-agent-card:hover {{
  border-color: rgba(255,255,255,0.15);
}}
.bp-agent-card .agent-header {{
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}}
.bp-agent-card .agent-icon {{
  font-size: 18px;
}}
.bp-agent-card .agent-name {{
  font-family: 'Orbitron', monospace;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
}}
.bp-agent-card .agent-role {{
  font-size: 9px;
  color: #666;
  margin-bottom: 6px;
}}
.bp-agent-card .agent-status {{
  font-size: 9px;
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: 6px;
}}
.bp-agent-card .agent-status .status-dot {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
  display: inline-block;
}}
.bp-agent-card .agent-status .status-dot.idle {{ background: #555; }}
.bp-agent-card .agent-status .status-dot.analyzing {{ background: #FFD700; animation: bp-pulse-{rid} 0.8s infinite; }}
.bp-agent-card .agent-status .status-dot.voting {{ background: #00F5FF; animation: bp-pulse-{rid} 0.6s infinite; }}
.bp-agent-card .agent-status .status-dot.done {{ background: #00FF41; }}
.bp-stress-bar {{
  height: 4px;
  background: rgba(255,255,255,0.06);
  border-radius: 2px;
  overflow: hidden;
}}
.bp-stress-bar .fill {{
  height: 100%;
  border-radius: 2px;
  transition: width 0.5s ease, background 0.3s;
}}

/* ── Center: Palace ── */
.bp-center {{
  flex: 1;
  padding: 10px;
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
}}
.bp-center-header {{
  text-align: center;
  font-family: 'Orbitron', monospace;
  font-size: 12px;
  letter-spacing: 3px;
  color: #00FF41;
  margin-bottom: 10px;
  opacity: 0.8;
}}
.bp-fragments-zone {{
  flex: 1;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-content: flex-start;
  overflow-y: auto;
  padding: 4px;
  min-height: 80px;
}}
.bp-fragment {{
  background: var(--bg-card);
  border-left: 3px solid;
  border-radius: 4px;
  padding: 6px 10px;
  font-size: 11px;
  max-width: 200px;
  animation: bp-frag-fly-{rid} 0.6s ease-out;
  position: relative;
}}
.bp-fragment .frag-source {{
  font-size: 8px;
  opacity: 0.5;
  margin-bottom: 2px;
}}
.bp-fragment .frag-text {{
  color: #d0d0d8;
  line-height: 1.4;
}}

/* Vortex */
.bp-vortex-zone {{
  height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  border-top: 1px solid rgba(0,255,65,0.08);
  padding-top: 10px;
  margin-top: 8px;
  position: relative;
}}
.bp-vortex {{
  width: 50px;
  height: 50px;
  border-radius: 50%;
  border: 2px solid rgba(0,255,65,0.3);
  border-top-color: #00FF41;
  border-right-color: #00F5FF;
  border-bottom-color: #FFD700;
  border-left-color: #FF003C;
  animation: bp-spin-{rid} 2s linear infinite;
  flex-shrink: 0;
}}
.bp-votes-list {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}}
.bp-vote-chip {{
  font-size: 10px;
  padding: 3px 8px;
  border-radius: 10px;
  background: var(--bg-card);
  border: 1px solid;
  display: flex;
  align-items: center;
  gap: 4px;
}}
.bp-vote-chip .vote-agent {{
  opacity: 0.6;
}}
.bp-vote-chip .vote-suspect {{
  font-weight: 700;
}}

/* ── Right: Timeline ── */
.bp-right {{
  width: 180px;
  min-width: 160px;
  border-left: 1px solid rgba(0,255,65,0.1);
  padding: 10px 8px;
  display: flex;
  flex-direction: column;
}}
.bp-right-header {{
  font-family: 'Orbitron', monospace;
  font-size: 10px;
  letter-spacing: 2px;
  color: #55556A;
  text-align: center;
  margin-bottom: 8px;
}}
.bp-timeline {{
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 6px;
  overflow-y: auto;
}}
.bp-tl-event {{
  font-size: 10px;
  padding: 5px 8px;
  border-radius: 4px;
  background: var(--bg-card);
  border-left: 2px solid;
  line-height: 1.3;
}}
.bp-tl-event .tl-time {{
  font-size: 8px;
  opacity: 0.4;
  margin-bottom: 2px;
}}
.bp-tl-event.info {{ border-left-color: #555; }}
.bp-tl-event.success {{ border-left-color: #00FF41; }}
.bp-tl-event.warning {{ border-left-color: #FFD700; }}
.bp-tl-event.danger {{ border-left-color: #FF003C; }}

/* ── Bottom: Verdict ── */
.bp-bottom {{
  border-top: 1px solid rgba(0,255,65,0.15);
  padding: 10px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 42px;
}}
.bp-verdict {{
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: 'Orbitron', monospace;
  font-size: 12px;
  letter-spacing: 1px;
}}
.bp-verdict .verdict-icon {{
  font-size: 16px;
}}
.bp-verdict .verdict-suspect {{
  font-weight: 700;
}}
.bp-verdict .verdict-conf {{
  opacity: 0.7;
  font-size: 11px;
}}
.bp-shield {{
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
}}
.bp-shield-active {{
  color: #00FF41;
  text-shadow: 0 0 8px rgba(0,255,65,0.5);
  animation: bp-glow-green-{rid} 1.5s ease-in-out infinite;
}}
.bp-shield-inactive {{
  color: #555;
}}
.bp-verdict-alert {{
  animation: bp-alert-flash-{rid} 0.5s ease-in-out infinite;
  color: #FF003C;
  font-weight: 700;
}}

/* ── Animations ── */
@keyframes bp-pulse-{rid} {{
  0%, 100% {{ opacity: 1; transform: scale(1); }}
  50% {{ opacity: 0.4; transform: scale(0.7); }}
}}
@keyframes bp-spin-{rid} {{
  from {{ transform: rotate(0deg); }}
  to {{ transform: rotate(360deg); }}
}}
@keyframes bp-frag-fly-{rid} {{
  0% {{ opacity: 0; transform: translateY(-10px) scale(0.8); }}
  100% {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
@keyframes bp-glow-green-{rid} {{
  0%, 100% {{ text-shadow: 0 0 6px rgba(0,255,65,0.3); }}
  50% {{ text-shadow: 0 0 14px rgba(0,255,65,0.7); }}
}}
@keyframes bp-alert-flash-{rid} {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.3; }}
}}
</style>"""

    def _render_agents_panel(self) -> str:
        """渲染左侧角色面板"""
        cards = []
        for aid, agent in self.agents.items():
            color = agent["color"]
            stress = agent["stress"]
            status = agent["status"]
            status_cls = status.lower()

            # Stress bar color gradient
            if stress < 40:
                bar_color = color
            elif stress < 70:
                bar_color = "#FFD700"
            else:
                bar_color = "#FF003C"

            status_icon = self.STATUS_ICONS.get(status, "?")
            active = status in ("ANALYZING", "VOTING")

            cards.append(f"""<div class="bp-agent-card" style="border-left:2px solid {color}{'55' if not active else 'ff'}">
  <div class="agent-header">
    <span class="agent-icon">{agent['icon']}</span>
    <span class="agent-name" style="color:{color}">{agent['name']}</span>
  </div>
  <div class="agent-role">{agent['cn']} · {agent['role']}</div>
  <div class="agent-status">
    <span class="status-dot {status_cls}" {'style="background:'+color+'"' if active else ''}></span>
    <span style="color:{'#999' if status=='IDLE' else color}">{status_icon} {status}</span>
  </div>
  <div class="bp-stress-bar">
    <div class="fill" style="width:{stress}%;background:{bar_color}"></div>
  </div>
  <div style="font-size:8px;color:#666;margin-top:2px;text-align:right">负载 {stress}%</div>
</div>""")

        header = '<div style="font-family:Orbitron,monospace;font-size:10px;letter-spacing:2px;color:#55556A;text-align:center;margin-bottom:4px">🎭 角色</div>'
        return header + "\n".join(cards)

    def _render_palace_center(self, rid: int) -> str:
        """渲染中央: 逻辑碎片区 + 逻辑漩涡"""
        # Fragments
        frag_htmls = []
        for f in self.fragments:
            safe_content = html_lib.escape(f["content"])
            frag_htmls.append(f"""<div class="bp-fragment" style="border-left-color:{f['color']}">
  <div class="frag-source" style="color:{f['color']}">{html_lib.escape(f['agent_name'])}</div>
  <div class="frag-text">{safe_content}</div>
</div>""")

        fragments_section = f"""<div class="bp-fragments-zone">
{"".join(frag_htmls) if frag_htmls else '<div style="color:#55556A;text-align:center;width:100%;padding:20px;font-size:11px">等待逻辑碎片…</div>'}
</div>"""

        # Vortex + Votes
        vote_chips = []
        for v in self.votes:
            vote_chips.append(f"""<div class="bp-vote-chip" style="border-color:{v['color']}">
  <span class="vote-agent" style="color:{v['color']}">{html_lib.escape(v['agent_name'])}</span>
  →
  <span class="vote-suspect">{html_lib.escape(v['suspect'])}</span>
  <span style="opacity:0.5">{v['confidence']:.0%}</span>
</div>""")

        has_votes = len(self.votes) > 0
        vortex_spinning = any(a["status"] == "VOTING" for a in self.agents.values())

        vortex_section = f"""<div class="bp-vortex-zone">
  <div class="bp-vortex" style="{'animation-play-state:running' if (vortex_spinning or has_votes) else 'animation-play-state:paused;opacity:0.3'}"></div>
  <div class="bp-votes-list">
    {"".join(vote_chips) if vote_chips else '<div style="color:#555;font-size:10px">🗳️ 等待投票…</div>'}
  </div>
</div>"""

        return f"""<div class="bp-center-header">⚡ 思维殿堂</div>
{fragments_section}
{vortex_section}"""

    def _render_timeline_panel(self) -> str:
        """渲染右侧时间线"""
        events_html = []
        for ev in self.timeline_events:
            safe_text = html_lib.escape(ev["text"])
            events_html.append(f"""<div class="bp-tl-event {ev['type']}">
  <div class="tl-time">{ev['timestamp']}</div>
  <div>{safe_text}</div>
</div>""")

        return f"""<div class="bp-right-header">📋 时间线</div>
<div class="bp-timeline">
{"".join(events_html) if events_html else '<div style="color:#555;font-size:10px;text-align:center;padding:20px">暂无事件</div>'}
</div>"""

    def _render_verdict_bar(self) -> str:
        """渲染底部裁决栏"""
        if self.verdict is None:
            return """<div class="bp-verdict">
  <span class="verdict-icon">⚖️</span>
  <span style="color:#555">裁决: 等待中…</span>
</div>
<div class="bp-shield bp-shield-inactive">🛡️ 护盾: 关闭</div>"""

        v = self.verdict
        suspect = html_lib.escape(v["suspect"])
        conf = v["confidence"]
        overridden = v["overridden"]
        shield = v["shield"]

        if overridden:
            verdict_html = f"""<div class="bp-verdict">
  <span class="verdict-icon">⚖️</span>
  <span class="bp-verdict-alert">⚠️ 系统警报 — 已推翻</span>
  <span class="verdict-suspect" style="color:#FF003C">{suspect}</span>
  <span class="verdict-conf">({conf:.0%})</span>
</div>"""
        else:
            verdict_html = f"""<div class="bp-verdict">
  <span class="verdict-icon">⚖️</span>
  <span style="color:#999">裁决:</span>
  <span class="verdict-suspect" style="color:#00FF41">{suspect}</span>
  <span class="verdict-conf">({conf:.0%})</span>
</div>"""

        shield_cls = "bp-shield-active" if shield else "bp-shield-inactive"
        shield_label = "激活" if shield else "关闭"
        shield_html = f"""<div class="bp-shield {shield_cls}">
  🛡️ 护盾: {shield_label}
</div>"""

        return verdict_html + shield_html


# ── 便捷函数: 生成推理流程中各阶段的 BrainPalace 状态快照 ──

def create_brain_palace() -> BrainPalace:
    """创建一个新的 BrainPalace 实例"""
    return BrainPalace()


def stage_evidence_extraction(bp: BrainPalace, case_desc: str = ""):
    """Stage 1: 证据提取 — Puppeteer 启动"""
    bp.update_agent("puppeteer", stress=30, status="ANALYZING")
    bp.set_sync_rate(25)
    bp.add_timeline_event("🔍 证据提取启动", "info")
    bp.add_timeline_event("解析案件文本…", "info")


def stage_traditional_reasoning(bp: BrainPalace, culprit: str = "?", 
                                  confidence: float = 0, suspects: list = None):
    """Stage 2: 传统推理 — Puppeteer 深入分析"""
    bp.update_agent("puppeteer", stress=60, status="ANALYZING")
    bp.set_sync_rate(45)
    bp.add_timeline_event(f"🔵 传统推理: {culprit} ({confidence:.0%})", "info")
    if suspects:
        bp.add_fragment("puppeteer", f"嫌疑人识别: {', '.join(suspects[:5])}")
    if culprit and culprit != "?":
        bp.add_fragment("puppeteer", f"传统推理指向: {culprit}")


def stage_asmr_search(bp: BrainPalace, contradictions: list = None,
                       culprit: str = "?", confidence: float = 0):
    """Stage 3: ASMR矛盾搜索 — Mirror 启动"""
    bp.update_agent("mirror", stress=50, status="ANALYZING")
    bp.update_agent("chronos", stress=40, status="ANALYZING")
    bp.set_sync_rate(65)
    bp.add_timeline_event(f"🟢 ASMR搜索: {culprit} ({confidence:.0%})", "info")
    if contradictions:
        for c in contradictions[:3]:
            text = c if isinstance(c, str) else c.get("description", str(c))
            bp.add_fragment("mirror", f"矛盾发现: {text[:50]}")
            bp.add_timeline_event(f"⚠️ 矛盾: {text[:30]}", "warning")


def stage_voting(bp: BrainPalace, trad_culprit: str = "?", trad_conf: float = 0,
                  asmr_culprit: str = "?", asmr_conf: float = 0):
    """Stage 4: 投票阶段 — 所有角色进入 VOTING"""
    bp.update_agent("puppeteer", stress=70, status="VOTING")
    bp.update_agent("mirror", stress=70, status="VOTING")
    bp.update_agent("chronos", stress=75, status="VOTING")
    bp.update_agent("compass", stress=80, status="VOTING")
    bp.set_sync_rate(85)

    bp.add_vote("puppeteer", trad_culprit, trad_conf)
    bp.add_vote("mirror", asmr_culprit, asmr_conf)
    bp.add_vote("chronos", trad_culprit, trad_conf * 0.9)
    bp.add_vote("compass", asmr_culprit if asmr_conf > trad_conf else trad_culprit,
                max(trad_conf, asmr_conf))

    bp.add_timeline_event(f"🗳️ 投票: 🔵{trad_culprit} vs 🟢{asmr_culprit}", "warning")
    agree = trad_culprit == asmr_culprit
    if agree:
        bp.add_timeline_event(f"✅ 一致裁决: {trad_culprit}", "success")
    else:
        bp.add_timeline_event(f"🔴 分歧: 需要仲裁", "danger")


def stage_verdict(bp: BrainPalace, suspect: str, confidence: float,
                   overridden: bool = False, shield: bool = True):
    """Stage 5: 最终裁决"""
    bp.update_agent("puppeteer", stress=90, status="DONE")
    bp.update_agent("mirror", stress=85, status="DONE")
    bp.update_agent("chronos", stress=80, status="DONE")
    bp.update_agent("compass", stress=95, status="DONE")
    bp.set_sync_rate(100)

    bp.set_verdict(suspect, confidence, overridden, shield)
    bp.add_timeline_event(f"⚖️ 裁决: {suspect} ({confidence:.0%})", 
                          "danger" if overridden else "success")
    bp.add_fragment("compass", f"最终裁决: {suspect} ({confidence:.0%})")


def stage_fusion(bp: BrainPalace, fusion_result: dict):
    """Stage 6: 融合完成 — 所有角色休息"""
    conc = fusion_result.get("conclusion", fusion_result)
    culprit = conc.get("culprit", fusion_result.get("culprit", "?"))
    conf = conc.get("confidence", fusion_result.get("confidence", 0))

    bp.update_agent("puppeteer", stress=100, status="DONE")
    bp.update_agent("mirror", stress=100, status="DONE")
    bp.update_agent("chronos", stress=100, status="DONE")
    bp.update_agent("compass", stress=100, status="DONE")
    bp.set_sync_rate(100)

    bp.set_verdict(culprit, conf, overridden=False, shield=True)
    bp.add_fragment("puppeteer", f"融合结果: {culprit} ({conf:.0%})")
    bp.add_timeline_event(f"🔀 融合完成: {culprit} ({conf:.0%})", "success")
