"""
🃏 ExpertCardRenderer — 专家推理卡片渲染器
v10.0: 将每个Agent的推理过程渲染为动态卡片
v14.0: 多轮推理展示 — 每个专家卡片显示R1→R2→R3→R4推理过程

每个专家卡片包含:
  - 头像/图标 + 名称 + 层级标识(调查层/审判层)
  - 状态: 等待/分析中/完成
  - 多轮推理指示器 (🔄 ×N, 结论是否改变)
  - 推理摘要(完成时显示)
  - 投票: 嫌疑人 + 置信度条
"""

import time
import html as html_lib
from typing import Dict, List, Optional, Any


# ── 专家元数据 ──
EXPERT_META = {
    # === 调查层 ===
    "forensic":      {"name": "法医专家",     "icon": "🔬", "layer": "investigation", "color": "#00FF41"},
    "criminal":      {"name": "刑侦专家",     "icon": "🔍", "layer": "investigation", "color": "#00F5FF"},
    "profiler":      {"name": "心理画像师",   "icon": "🧠", "layer": "investigation", "color": "#FFD700"},
    "tech":          {"name": "技侦专家",     "icon": "💻", "layer": "investigation", "color": "#FF69B4"},
    "financial":     {"name": "经侦专家",     "icon": "💰", "layer": "investigation", "color": "#FFA500"},
    "interrogation": {"name": "审讯分析师",   "icon": "🎤", "layer": "investigation", "color": "#9370DB"},
    "intelligence":  {"name": "情报分析师",   "icon": "📡", "layer": "investigation", "color": "#20B2AA"},
    "sherlock":      {"name": "福尔摩斯",     "icon": "🦌", "layer": "investigation", "color": "#C0C0C0"},
    "henry_lee":     {"name": "李昌钰",       "icon": "🔬", "layer": "investigation", "color": "#4169E1"},
    "song_ci":       {"name": "宋慈",         "icon": "📜", "layer": "investigation", "color": "#8B4513"},
    "poirot":        {"name": "波洛",         "icon": "🎩", "layer": "investigation", "color": "#9B59B6"},
    "logic_verifier":{"name": "逻辑验证官",   "icon": "⚖️", "layer": "verifier",      "color": "#FF003C"},
    # === 审判层 ===
    "prosecution":   {"name": "检察官",       "icon": "🔴", "layer": "trial",         "color": "#FF4444"},
    "defense":       {"name": "辩护律师",     "icon": "🔵", "layer": "trial",         "color": "#4444FF"},
    "judge":         {"name": "法官",         "icon": "🧑‍⚖️", "layer": "trial",       "color": "#FFD700"},
    "juror":         {"name": "陪审员",       "icon": "👥", "layer": "trial",         "color": "#888888"},
    # === 裁判 ===
    "adjudicator":   {"name": "裁判Agent",    "icon": "🏆", "layer": "adjudicator",   "color": "#FF003C"},
}

LAYER_META = {
    "investigation": {"name": "调查层", "icon": "🔬", "color": "#00F5FF", "bg": "rgba(0,245,255,0.05)"},
    "verifier":      {"name": "验证层", "icon": "⚖️", "color": "#FF003C", "bg": "rgba(255,0,60,0.05)"},
    "trial":         {"name": "审判层", "icon": "🧑‍⚖️", "color": "#FFD700", "bg": "rgba(255,215,0,0.05)"},
    "adjudicator":   {"name": "裁判",   "icon": "🏆", "color": "#FF003C", "bg": "rgba(255,0,60,0.05)"},
}


class ExpertCardRenderer:
    """专家推理卡片渲染器"""

    def __init__(self):
        self._rid = 0

    def render_expert_panel(self, agent_results: Dict[str, Dict],
                            vote_history: List[Dict],
                            stage_progress: Dict[str, float],
                            stage_timings: Dict[str, float],
                            agent_round_log: Dict[str, List[Dict]] = None) -> str:
        """
        渲染完整的专家面板

        Args:
            agent_results: {agent_id: {culprit, confidence, reasoning, ...}}
            vote_history:  [{expert, culprit, confidence, weight}, ...]
            stage_progress: {stage: 0.0-1.0}
            stage_timings: {stage: seconds}
            agent_round_log: {agent_id: [{round, phase, culprit, confidence, changed}...]}  # 🆕 多轮推理日志
        """
        self._rid += 1
        rid = self._rid
        agent_round_log = agent_round_log or {}

        css = self._render_css(rid)
        layers_html = self._render_layers(agent_results, stage_progress, stage_timings, agent_round_log)
        vote_html = self._render_vote_distribution(vote_history)
        timeline_html = self._render_stage_timeline(stage_timings)

        return f"""<div class="ec-root-{rid}">
{css}
<div class="ec-container">
  <div class="ec-header">
    <span class="ec-title">🃏 专家推理矩阵</span>
    <span class="ec-subtitle">{len(agent_results)} 个Agent已完成</span>
  </div>
  <div class="ec-body">
    <div class="ec-layers">{layers_html}</div>
    <div class="ec-sidebar">
      <div class="ec-vote-section">{vote_html}</div>
      <div class="ec-timeline-section">{timeline_html}</div>
    </div>
  </div>
</div>
</div>"""

    def render_empty(self) -> str:
        return """<div style="
    background: rgba(6,6,10,0.9);
    border: 1px solid rgba(0,245,255,0.15);
    border-radius: 8px;
    padding: 40px 20px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
">
  <div style="font-size:28px;margin-bottom:12px">🃏</div>
  <div style="color:#00F5FF;font-family:Orbitron,monospace;font-size:14px;letter-spacing:3px;margin-bottom:8px">专家推理矩阵</div>
  <div style="color:#55556A;font-size:11px">等待Agent部署…</div>
</div>"""

    def _render_css(self, rid: int) -> str:
        return f"""<style>
.ec-root-{rid} {{
  font-family: 'JetBrains Mono', monospace;
  color: #c8c8d0;
}}
.ec-root-{rid} * {{ box-sizing: border-box; }}
.ec-container {{
  background: rgba(6,6,10,0.92);
  border: 1px solid rgba(0,245,255,0.15);
  border-radius: 8px;
  overflow: hidden;
}}
.ec-header {{
  background: linear-gradient(135deg, rgba(0,245,255,0.08), rgba(0,245,255,0.02));
  padding: 10px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid rgba(0,245,255,0.1);
}}
.ec-title {{
  font-family: 'Orbitron', monospace;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 2px;
  color: #00F5FF;
}}
.ec-subtitle {{
  font-size: 10px;
  color: #55556A;
}}
.ec-body {{
  display: flex;
  min-height: 360px;
}}
.ec-layers {{
  flex: 1;
  padding: 10px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.ec-sidebar {{
  width: 220px;
  min-width: 200px;
  border-left: 1px solid rgba(0,245,255,0.08);
  display: flex;
  flex-direction: column;
}}

/* Layer section */
.ec-layer {{
  border-radius: 6px;
  padding: 8px;
  border: 1px solid rgba(255,255,255,0.04);
}}
.ec-layer-header {{
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
  font-size: 11px;
  font-weight: 700;
  font-family: 'Orbitron', monospace;
  letter-spacing: 1px;
}}
.ec-layer-badge {{
  font-size: 8px;
  padding: 2px 6px;
  border-radius: 8px;
  opacity: 0.7;
}}
.ec-cards {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}}

/* Expert card */
.ec-card {{
  background: rgba(10,10,18,0.85);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 6px;
  padding: 8px 10px;
  min-width: 140px;
  max-width: 180px;
  flex: 1;
  position: relative;
  transition: border-color 0.3s, box-shadow 0.3s;
  animation: ec-fade-in-{rid} 0.5s ease-out;
}}
.ec-card:hover {{
  border-color: rgba(255,255,255,0.15);
}}
.ec-card.running {{
  border-color: rgba(255,215,0,0.3);
  box-shadow: 0 0 8px rgba(255,215,0,0.1);
}}
.ec-card.done {{
  border-color: rgba(0,255,65,0.2);
}}
.ec-card-header {{
  display: flex;
  align-items: center;
  gap: 5px;
  margin-bottom: 4px;
}}
.ec-card-icon {{
  font-size: 14px;
}}
.ec-card-name {{
  font-size: 10px;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.ec-card-status {{
  font-size: 8px;
  margin-left: auto;
}}
.ec-card-status.idle {{ color: #555; }}
.ec-card-status.running {{ color: #FFD700; animation: ec-pulse-{rid} 1s infinite; }}
.ec-card-status.done {{ color: #00FF41; }}

.ec-card-vote {{
  margin-top: 4px;
  display: flex;
  align-items: center;
  gap: 4px;
}}
.ec-card-suspect {{
  font-size: 11px;
  font-weight: 700;
  color: #E0E0E8;
}}
.ec-card-conf-bar {{
  flex: 1;
  height: 3px;
  background: rgba(255,255,255,0.06);
  border-radius: 2px;
  overflow: hidden;
}}
.ec-card-conf-fill {{
  height: 100%;
  border-radius: 2px;
  transition: width 0.5s ease;
}}
.ec-card-conf-text {{
  font-size: 9px;
  color: #888;
  min-width: 28px;
  text-align: right;
}}
.ec-card-reasoning {{
  font-size: 9px;
  color: #777;
  margin-top: 4px;
  line-height: 1.3;
  max-height: 36px;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
}}

/* Vote distribution */
.ec-vote-section {{
  padding: 10px;
  border-bottom: 1px solid rgba(0,245,255,0.06);
  flex: 1;
  overflow-y: auto;
}}
.ec-vote-title {{
  font-family: 'Orbitron', monospace;
  font-size: 10px;
  letter-spacing: 2px;
  color: #55556A;
  margin-bottom: 8px;
  text-align: center;
}}
.ec-vote-bar-row {{
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 5px;
}}
.ec-vote-name {{
  font-size: 10px;
  min-width: 50px;
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.ec-vote-bar {{
  flex: 1;
  height: 14px;
  background: rgba(255,255,255,0.04);
  border-radius: 3px;
  overflow: hidden;
  position: relative;
}}
.ec-vote-bar-fill {{
  height: 100%;
  border-radius: 3px;
  transition: width 0.8s ease;
  min-width: 2px;
}}
.ec-vote-count {{
  font-size: 9px;
  color: #888;
  min-width: 24px;
}}

/* Timeline */
.ec-timeline-section {{
  padding: 10px;
  max-height: 140px;
  overflow-y: auto;
}}
.ec-tl-title {{
  font-family: 'Orbitron', monospace;
  font-size: 10px;
  letter-spacing: 2px;
  color: #55556A;
  margin-bottom: 6px;
  text-align: center;
}}
.ec-tl-item {{
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 10px;
  padding: 3px 0;
  border-bottom: 1px solid rgba(255,255,255,0.02);
}}
.ec-tl-icon {{
  font-size: 10px;
}}
.ec-tl-stage {{
  color: #888;
  flex: 1;
}}
.ec-tl-time {{
  color: #00FF41;
  font-size: 9px;
  min-width: 36px;
  text-align: right;
}}

/* Animations */
@keyframes ec-fade-in-{rid} {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes ec-pulse-{rid} {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.4; }}
}}
</style>"""

    def _render_layers(self, agent_results: Dict, stage_progress: Dict, stage_timings: Dict, agent_round_log: Dict = None) -> str:
        """按层渲染专家卡片"""
        agent_round_log = agent_round_log or {}
        # 按层分组
        layers = {
            "investigation": [],
            "verifier": [],
            "trial": [],
            "adjudicator": [],
        }

        for agent_id, result in agent_results.items():
            meta = EXPERT_META.get(agent_id, {"name": agent_id, "icon": "❓", "layer": "investigation", "color": "#888"})
            layers.setdefault(meta["layer"], []).append((agent_id, result, meta))

        # 也加入尚未完成的专家（显示为等待状态）
        all_known_ids = set(agent_results.keys())
        for agent_id, meta in EXPERT_META.items():
            if agent_id not in all_known_ids:
                layers.setdefault(meta["layer"], []).append((agent_id, None, meta))

        html_parts = []
        for layer_id, layer_meta in LAYER_META.items():
            agents = layers.get(layer_id, [])
            if not agents:
                continue

            timing = stage_timings.get(f"stage_{layer_id}", stage_timings.get(f"stage_31_{layer_id}", 0))
            timing_str = f"{timing:.1f}s" if timing else "…"

            cards_html = ""
            for agent_id, result, meta in agents:
                round_log = agent_round_log.get(agent_id, [])
                cards_html += self._render_expert_card(agent_id, result, meta, round_log)

            html_parts.append(f"""
<div class="ec-layer" style="background:{layer_meta['bg']}">
  <div class="ec-layer-header">
    <span>{layer_meta['icon']}</span>
    <span style="color:{layer_meta['color']}">{layer_meta['name']}</span>
    <span class="ec-layer-badge" style="background:{layer_meta['color']}22;color:{layer_meta['color']}">{len(agents)} 位专家</span>
    <span style="margin-left:auto;font-size:9px;color:#555">⏱ {timing_str}</span>
  </div>
  <div class="ec-cards">{cards_html}</div>
</div>""")

        return "".join(html_parts)

    def _render_expert_card(self, agent_id: str, result: Optional[Dict], meta: Dict, round_log: List[Dict] = None) -> str:
        """渲染单个专家卡片（含多轮推理信息）"""
        icon = meta.get("icon", "❓")
        name = meta.get("name", agent_id)
        color = meta.get("color", "#888")
        round_log = round_log or []

        if result is None:
            # 等待状态
            return f"""
<div class="ec-card">
  <div class="ec-card-header">
    <span class="ec-card-icon">{icon}</span>
    <span class="ec-card-name" style="color:{color}">{name}</span>
    <span class="ec-card-status idle">⏳</span>
  </div>
</div>"""

        culprit = result.get("culprit", "?")
        confidence = result.get("confidence", 0)
        reasoning = result.get("reasoning", "")
        
        # 🆕 多轮推理信息
        mr_info = result.get("multi_round", {})
        if not mr_info and isinstance(result.get("detail"), dict):
            mr_info = result.get("detail", {}).get("multi_round", {})
        
        total_rounds = mr_info.get("total_rounds", len(round_log)) if isinstance(mr_info, dict) else len(round_log)
        early_stop = mr_info.get("early_stop", False) if isinstance(mr_info, dict) else False
        rounds_detail = mr_info.get("rounds", round_log) if isinstance(mr_info, dict) else round_log
        
        # 检测结论是否在各轮之间改变
        conclusion_changed = False
        if round_log and len(round_log) >= 2:
            first_culprit = round_log[0].get("culprit", culprit)
            for r in round_log[1:]:
                if r.get("culprit") != first_culprit:
                    conclusion_changed = True
                    break

        status = "done"
        status_icon = "✅"

        # 置信度颜色
        if confidence >= 0.8:
            conf_color = "#00FF41"
        elif confidence >= 0.6:
            conf_color = "#FFD700"
        else:
            conf_color = "#FF003C"

        conf_pct = confidence * 100

        # 截断reasoning
        reasoning_short = reasoning[:80] + "…" if len(reasoning) > 80 else reasoning

        # 🆕 多轮推理指示器
        round_badge = ""
        if total_rounds > 1:
            round_badge = f'<span style="font-size:8px;padding:1px 4px;border-radius:6px;background:rgba(0,245,255,0.15);color:#00F5FF;margin-left:3px" title="多轮推理: {total_rounds}轮">🔄×{total_rounds}</span>'
        elif total_rounds == 1 and early_stop:
            round_badge = '<span style="font-size:8px;padding:1px 4px;border-radius:6px;background:rgba(0,255,65,0.12);color:#00FF41;margin-left:3px" title="高置信度,提前终止">⚡</span>'
        
        # 结论改变标记
        change_badge = ""
        if conclusion_changed:
            change_badge = '<span style="font-size:8px;padding:1px 4px;border-radius:6px;background:rgba(255,215,0,0.15);color:#FFD700;margin-left:2px" title="多轮推理后改变了结论">🔀</span>'

        # 🆕 多轮推理演进轨迹 (只在有多轮数据时显示)
        round_evolution_html = ""
        if round_log and len(round_log) >= 2:
            evolution_parts = []
            for r in round_log[:4]:
                r_num = r.get("round", "?")
                r_phase = r.get("phase", "")[:6]
                r_culprit = str(r.get("culprit", "?"))
                r_conf = r.get("confidence", 0)
                r_changed = r.get("changed", False)
                
                # 缩短人名
                short_name = r_culprit[:3] if len(r_culprit) > 3 else r_culprit
                
                if r_changed:
                    evolution_parts.append(f'<span style="color:#FFD700">R{r_num}:{html_lib.escape(short_name)}</span>')
                else:
                    evolution_parts.append(f'<span style="color:#666">R{r_num}:{html_lib.escape(short_name)}</span>')
            
            arrow_join = " → "
            round_evolution_html = f'<div style="font-size:8px;margin-top:3px;line-height:1.2;color:#555;font-family:JetBrains Mono,monospace">{arrow_join.join(evolution_parts)}</div>'

        return f"""
<div class="ec-card {status}">
  <div class="ec-card-header">
    <span class="ec-card-icon">{icon}</span>
    <span class="ec-card-name" style="color:{color}">{name}</span>
    {round_badge}{change_badge}
    <span class="ec-card-status {status}">{status_icon}</span>
  </div>
  <div class="ec-card-vote">
    <span class="ec-card-suspect">{html_lib.escape(str(culprit))}</span>
    <div class="ec-card-conf-bar">
      <div class="ec-card-conf-fill" style="width:{conf_pct}%;background:{conf_color};box-shadow:0 0 6px {conf_color}66"></div>
    </div>
    <span class="ec-card-conf-text">{conf_pct:.0f}%</span>
  </div>
  {round_evolution_html}
  <div class="ec-card-reasoning">{html_lib.escape(reasoning_short)}</div>
</div>"""

    def _render_vote_distribution(self, vote_history: List[Dict]) -> str:
        """渲染投票分布条形图"""
        if not vote_history:
            return """<div class="ec-vote-section">
  <div class="ec-vote-title">🗳️ 投票分布</div>
  <div style="color:#555;font-size:10px;text-align:center;padding:20px 0">暂无投票</div>
</div>"""

        # 汇总投票
        tally = {}
        for v in vote_history:
            culprit = v.get("culprit", "未知")
            weight = v.get("weight", 1.0)
            tally[culprit] = tally.get(culprit, 0) + weight

        if not tally:
            return '<div style="color:#555;font-size:10px;text-align:center;padding:20px">无数据</div>'

        max_val = max(tally.values())
        sorted_votes = sorted(tally.items(), key=lambda x: -x[1])

        # 颜色分配
        colors = ["#00FF41", "#00F5FF", "#FFD700", "#FF003C", "#9B59B6", "#FF69B4", "#FFA500"]

        rows = []
        for i, (name, score) in enumerate(sorted_votes[:8]):
            pct = (score / max_val * 100) if max_val > 0 else 0
            color = colors[i % len(colors)]
            rows.append(f"""
<div class="ec-vote-bar-row">
  <span class="ec-vote-name">{html_lib.escape(name)}</span>
  <div class="ec-vote-bar">
    <div class="ec-vote-bar-fill" style="width:{pct}%;background:{color};box-shadow:0 0 8px {color}44"></div>
  </div>
  <span class="ec-vote-count">{score:.1f}</span>
</div>""")

        return f"""<div class="ec-vote-section">
  <div class="ec-vote-title">🗳️ 投票分布</div>
  {"".join(rows)}
</div>"""

    def _render_stage_timeline(self, stage_timings: Dict[str, float]) -> str:
        """渲染阶段时间线"""
        stage_order = [
            ("stage_1_readers",     "📚 读者层",     "📚"),
            ("stage_2_searchers",   "🔍 搜索层",   "🔍"),
            ("stage_31_investigation","🔬 调查层",     "🔬"),
            ("stage_32_trial",      "⚖️ 审判层",       "⚖️"),
            ("stage_35_tree",       "🌳 推理树",       "🌳"),
            ("stage_4_adjudicator", "🗳️ 裁判",         "🗳️"),
        ]

        items = []
        for key, label, icon in stage_order:
            t = stage_timings.get(key)
            if t:
                time_str = f"{t:.1f}s"
                items.append(f"""
<div class="ec-tl-item">
  <span class="ec-tl-icon">{icon}</span>
  <span class="ec-tl-stage">{label}</span>
  <span class="ec-tl-time">{time_str}</span>
</div>""")
            else:
                items.append(f"""
<div class="ec-tl-item">
  <span class="ec-tl-icon" style="opacity:0.3">{icon}</span>
  <span class="ec-tl-stage" style="opacity:0.3">{label}</span>
  <span class="ec-tl-time" style="opacity:0.3">—</span>
</div>""")

        if not any(stage_timings.get(k) for k, _, _ in stage_order):
            return """<div class="ec-timeline-section">
  <div class="ec-tl-title">⏱️ 阶段耗时</div>
  <div style="color:#555;font-size:10px;text-align:center;padding:10px 0">等待中</div>
</div>"""

        total = sum(v for v in stage_timings.values() if isinstance(v, (int, float)))
        total_line = f"""<div class="ec-tl-item" style="border-top:1px solid rgba(0,255,65,0.1);padding-top:5px;margin-top:2px">
  <span class="ec-tl-icon">⏱️</span>
  <span class="ec-tl-stage" style="color:#00FF41;font-weight:700">总计</span>
  <span class="ec-tl-time" style="color:#00FF41">{total:.1f}s</span>
</div>"""

        return f"""<div class="ec-timeline-section">
  <div class="ec-tl-title">⏱️ 阶段耗时</div>
  {"".join(items)}
  {total_line}
</div>"""
