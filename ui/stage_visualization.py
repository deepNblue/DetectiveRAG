#!/usr/bin/env python3
"""
时序侦破可视化 — HTML+CSS+JS 动态渲染
生成时序推理的HTML可视化页面，嵌入到 WebUI 中。

特性:
  - 时间轴展示各阶段
  - 每阶段显示: 证据数量、结论、嫌疑人排名变化
  - 关键转折点高亮
  - 调查建议卡片
  - 假设预测的if-then展示
  - 纯HTML+CSS+JS，不依赖npm
"""

import json
import time as _time
from typing import Dict, List, Any, Optional


# ═══════════════════════════════════════════════════════════════
# 颜色配置
# ═══════════════════════════════════════════════════════════════

STAGE_COLORS = [
    "#00FF41",  # 霓虹绿 — 报案
    "#00FF41",  # 霓虹绿 — 初步
    "#FFB800",  # 琥珀 — 深入
    "#B37FEB",  # 紫 — 综合
    "#FF003C",  # 深渊红 — 终局
]

PRIORITY_STYLES = {
    "高": "background:rgba(255,0,60,0.15);color:#FF003C;border-color:#FF003C;",
    "中": "background:rgba(255,184,0,0.15);color:#FFB800;border-color:#FFB800;",
    "低": "background:rgba(0,255,65,0.15);color:#00FF41;border-color:#00FF41;",
}


# ═══════════════════════════════════════════════════════════════
# 主渲染函数
# ═══════════════════════════════════════════════════════════════

def render_stage_timeline(stages: List[Dict]) -> str:
    """
    渲染时序侦破过程的时间轴可视化

    Args:
        stages: 阶段数据列表 (InvestigationStage.to_dict())

    Returns:
        HTML字符串
    """
    if not stages:
        return _empty_html("暂无侦破阶段数据")

    container_id = f"stage_tl_{int(_time.time()*1000)}"

    # 序列化阶段数据
    stages_json = json.dumps(stages, ensure_ascii=False, default=str)

    # 计算嫌疑人排名变化
    all_suspect_names = []
    for stage in stages:
        for sr in stage.get("suspect_ranking", []):
            name = sr.get("name", "")
            if name and name not in all_suspect_names:
                all_suspect_names.append(name)

    suspect_names_json = json.dumps(all_suspect_names, ensure_ascii=False)

    html = f'''<div id="{container_id}" class="stage-timeline-container">
<style>
  #{container_id} {{
    font-family: "JetBrains Mono", "Fira Code", "SF Mono", monospace;
    background: #0A0A0C;
    border: 2px solid rgba(0,255,65,0.3);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,255,65,.12);
  }}
  #{container_id} .stl-header {{
    background: linear-gradient(135deg, rgba(0,255,65,0.08), rgba(0,255,65,0.25));
    color: #00FF41;
    padding: 14px 20px;
    font-weight: 700;
    font-size: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  #{container_id} .stl-body {{
    padding: 20px;
    background: #0A0A0C;
  }}
  #{container_id} .stl-timeline {{
    position: relative;
    padding-left: 40px;
  }}
  #{container_id} .stl-timeline::before {{
    content: '';
    position: absolute;
    left: 18px;
    top: 0;
    bottom: 0;
    width: 3px;
    background: linear-gradient(to bottom, #00FF41, #B37FEB, #FFB800);
    border-radius: 2px;
  }}
  #{container_id} .stl-stage {{
    position: relative;
    margin-bottom: 24px;
    background: rgba(10,10,14,0.85);
    border-radius: 10px;
    border: 1px solid rgba(0,255,65,0.2);
    overflow: hidden;
    transition: box-shadow 0.2s;
  }}
  #{container_id} .stl-stage:hover {{
    box-shadow: 0 4px 16px rgba(0,255,65,.1);
  }}
  #{container_id} .stl-stage-dot {{
    position: absolute;
    left: -32px;
    top: 16px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 3px solid #0A0A0C;
    box-shadow: 0 0 0 2px #00FF41;
  }}
  #{container_id} .stl-stage-header {{
    padding: 12px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    border-bottom: 1px solid #F2F3F5;
  }}
  #{container_id} .stl-stage-header:hover {{
    background: #F7F8FA;
  }}
  #{container_id} .stl-stage-title {{
    font-weight: 700;
    font-size: 14px;
    color: #1D2129;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  #{container_id} .stl-stage-meta {{
    font-size: 12px;
    color: #8A8A9A;
    display: flex;
    gap: 12px;
  }}
  #{container_id} .stl-stage-body {{
    padding: 16px;
    display: none;
  }}
  #{container_id} .stl-stage.expanded .stl-stage-body {{
    display: block;
  }}
  #{container_id} .stl-section {{
    margin-bottom: 16px;
  }}
  #{container_id} .stl-section-title {{
    font-size: 12px;
    font-weight: 700;
    color: #8A8A9A;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }}
  #{container_id} .stl-reasoning {{
    background: #F7F8FA;
    border-radius: 8px;
    padding: 12px;
    font-size: 13px;
    line-height: 1.7;
    color: #4E5969;
    white-space: pre-wrap;
  }}
  #{container_id} .stl-ranking {{
    width: 100%;
  }}
  #{container_id} .stl-ranking-row {{
    display: flex;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #F2F3F5;
    gap: 8px;
  }}
  #{container_id} .stl-ranking-row:last-child {{
    border-bottom: none;
  }}
  #{container_id} .stl-rank-num {{
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #E8F0FF;
    color: #00FF41;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    flex-shrink: 0;
  }}
  #{container_id} .stl-rank-num.top {{
    background: #FFB800;
    color: #FFF;
  }}
  #{container_id} .stl-rank-name {{
    font-size: 13px;
    font-weight: 600;
    color: #1D2129;
    width: 100px;
    flex-shrink: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  #{container_id} .stl-rank-bar-bg {{
    flex: 1;
    height: 8px;
    background: #F2F3F5;
    border-radius: 4px;
    overflow: hidden;
  }}
  #{container_id} .stl-rank-bar {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
  }}
  #{container_id} .stl-rank-score {{
    font-size: 12px;
    font-weight: 700;
    width: 40px;
    text-align: right;
    flex-shrink: 0;
  }}
  #{container_id} .stl-rank-reason {{
    font-size: 11px;
    color: #8A8A9A;
    margin-left: 4px;
    flex-shrink: 0;
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  #{container_id} .stl-advice {{
    background: #F7F8FA;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 6px;
    border-left: 3px solid #00FF41;
  }}
  #{container_id} .stl-advice-title {{
    font-size: 13px;
    font-weight: 600;
    color: #1D2129;
    margin-bottom: 4px;
  }}
  #{container_id} .stl-advice-reason {{
    font-size: 12px;
    color: #8A8A9A;
  }}
  #{container_id} .stl-advice-priority {{
    display: inline-block;
    font-size: 11px;
    padding: 1px 8px;
    border-radius: 4px;
    font-weight: 600;
    border: 1px solid;
  }}
  #{container_id} .stl-hypothesis {{
    background: #FFF7E8;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 6px;
    border-left: 3px solid #FFB800;
  }}
  #{container_id} .stl-hypothesis-cond {{
    font-size: 13px;
    font-weight: 600;
    color: #FFB800;
  }}
  #{container_id} .stl-hypothesis-then {{
    font-size: 12px;
    color: #4E5969;
    margin-top: 2px;
  }}
  #{container_id} .stl-hypothesis-impact {{
    font-size: 11px;
    color: #8A8A9A;
    margin-top: 2px;
  }}
  #{container_id} .stl-turning-point {{
    background: #FFF0F0;
    border: 1px solid #CB2634;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 12px;
    font-size: 13px;
    color: #CB2634;
    font-weight: 600;
  }}
  #{container_id} .stl-confidence {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
  }}
  #{container_id} .stl-confidence-bar-bg {{
    flex: 1;
    height: 10px;
    background: #F2F3F5;
    border-radius: 5px;
    overflow: hidden;
  }}
  #{container_id} .stl-confidence-bar {{
    height: 100%;
    border-radius: 5px;
    transition: width 0.5s ease;
  }}
  #{container_id} .stl-evidence-tag {{
    display: inline-block;
    background: #E8F0FF;
    color: #00FF41;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px;
  }}
  #{container_id} .stl-evidence-tag.new {{
    background: #FFF0F0;
    color: #CB2634;
  }}
  #{container_id} .stl-arrow {{
    font-size: 12px;
    transition: transform 0.2s;
  }}
  #{container_id} .stl-stage.expanded .stl-arrow {{
    transform: rotate(90deg);
  }}
  #{container_id} .stl-summary {{
    background: linear-gradient(135deg, rgba(0,255,65,0.08), #00FF41);
    color: #FFF;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 16px;
  }}
  #{container_id} .stl-summary h4 {{
    margin: 0 0 8px 0;
    font-size: 14px;
  }}
  #{container_id} .stl-summary-table {{
    width: 100%;
    font-size: 12px;
  }}
  #{container_id} .stl-summary-table td {{
    padding: 3px 8px;
  }}
</style>

<div class="stl-header">
  <span>🕵️ 时序侦破推理过程</span>
  <span style="font-size:12px;font-weight:400">{len(stages)} 阶段</span>
</div>

<div class="stl-body">
  <div class="stl-timeline" id="{container_id}_timeline"></div>
  <div id="{container_id}_summary"></div>
</div>

<script>
(function() {{
  "use strict";
  var cid = "{container_id}";
  var stages = {stages_json};
  var suspectNames = {suspect_names_json};
  var colors = {json.dumps(STAGE_COLORS)};
  var priorityStyles = {json.dumps(PRIORITY_STYLES)};

  // 嫌疑人颜色映射
  var suspectColors = {{}};
  var sColors = ["#00FF41","#FFB800","#00FF41","#B37FEB","#FF003C","#8A8A9A","#00F5FF","#FF6B35"];
  suspectNames.forEach(function(n, i) {{ suspectColors[n] = sColors[i % sColors.length]; }});

  var timeline = document.getElementById(cid + "_timeline");
  if (!timeline) return;

  function escHtml(s) {{
    return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }}

  function renderStages() {{
    var html = "";
    stages.forEach(function(stage, idx) {{
      var color = colors[idx % colors.length];
      var sid = stage.stage_id || (idx + 1);
      var name = stage.stage_name || ("阶段" + sid);
      var desc = stage.stage_description || "";
      var reasoning = stage.reasoning || "";
      var ranking = stage.suspect_ranking || [];
      var advice = stage.investigation_advice || [];
      var hypotheses = stage.hypotheses || [];
      var confidence = stage.confidence || 0;
      var turningPoint = stage.key_turning_point;
      var newEv = stage.new_evidence || [];
      var allEv = stage.available_evidence || [];

      html += '<div class="stl-stage" id="' + cid + '_stage_' + idx + '">';

      // Dot
      html += '<div class="stl-stage-dot" style="background:' + color + ';box-shadow:0 0 0 2px ' + color + '"></div>';

      // Header
      html += '<div class="stl-stage-header" onclick="document.getElementById(\\'' + cid + '_stage_' + idx + '\\').classList.toggle(\\'expanded\\')">';
      html += '<div class="stl-stage-title">';
      html += '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + color + '"></span>';
      html += '阶段' + sid + ': ' + escHtml(name);
      if (turningPoint) html += ' <span style="color:#CB2634;font-size:11px">⚡转折</span>';
      html += '</div>';
      html += '<div class="stl-stage-meta">';
      html += '<span>📊 证据 ' + allEv.length + ' (🆕' + newEv.length + ')</span>';
      html += '<span>🎯 置信度 ' + Math.round(confidence * 100) + '%</span>';
      if (ranking.length > 0) html += '<span>👤 ' + escHtml(ranking[0].name) + ' (' + Math.round((ranking[0].suspicion_score||0)*100) + '%)</span>';
      html += '<span class="stl-arrow">▶</span>';
      html += '</div>';
      html += '</div>';

      // Body
      html += '<div class="stl-stage-body">';

      // Turning point
      if (turningPoint) {{
        html += '<div class="stl-turning-point">⚡ 关键转折: ' + escHtml(turningPoint) + '</div>';
      }}

      // Evidence
      html += '<div class="stl-section">';
      html += '<div class="stl-section-title">📋 证据</div>';
      html += '<div style="margin-bottom:8px">';
      newEv.forEach(function(ev) {{
        html += '<span class="stl-evidence-tag new">🆕 ' + escHtml(ev).substring(0, 40) + '</span> ';
      }});
      var oldEv = allEv.filter(function(e) {{ return newEv.indexOf(e) < 0; }});
      oldEv.forEach(function(ev) {{
        html += '<span class="stl-evidence-tag">' + escHtml(ev).substring(0, 40) + '</span> ';
      }});
      html += '</div></div>';

      // Reasoning
      html += '<div class="stl-section">';
      html += '<div class="stl-section-title">🧠 推理过程</div>';
      html += '<div class="stl-reasoning">' + escHtml(reasoning) + '</div>';
      html += '</div>';

      // Suspect ranking
      if (ranking.length > 0) {{
        html += '<div class="stl-section">';
        html += '<div class="stl-section-title">👤 嫌疑人排名</div>';
        html += '<div class="stl-ranking">';
        ranking.forEach(function(sr, ri) {{
          var sc = sr.suspicion_score || 0;
          var name = sr.name || "?";
          var reason = sr.reason || "";
          var sColor = suspectColors[name] || "#4E5969";
          html += '<div class="stl-ranking-row">';
          html += '<div class="stl-rank-num ' + (ri === 0 ? 'top' : '') + '">' + (ri+1) + '</div>';
          html += '<div class="stl-rank-name" title="' + escHtml(name) + '">' + escHtml(name) + '</div>';
          html += '<div class="stl-rank-bar-bg"><div class="stl-rank-bar" style="width:' + (sc*100) + '%;background:' + sColor + '"></div></div>';
          html += '<div class="stl-rank-score" style="color:' + sColor + '">' + Math.round(sc*100) + '%</div>';
          html += '<div class="stl-rank-reason" title="' + escHtml(reason) + '">' + escHtml(reason.substring(0,30)) + '</div>';
          html += '</div>';
        }});
        html += '</div></div>';
      }}

      // Confidence
      html += '<div class="stl-section">';
      html += '<div class="stl-section-title">🎯 整体置信度</div>';
      html += '<div class="stl-confidence">';
      html += '<div class="stl-confidence-bar-bg"><div class="stl-confidence-bar" style="width:' + (confidence*100) + '%;background:' + color + '"></div></div>';
      html += '<span style="font-size:14px;font-weight:700;color:' + color + '">' + Math.round(confidence*100) + '%</span>';
      html += '</div></div>';

      // Advice
      if (advice.length > 0) {{
        html += '<div class="stl-section">';
        html += '<div class="stl-section-title">🎯 调查建议</div>';
        advice.forEach(function(a) {{
          var pStyle = priorityStyles[a.priority] || priorityStyles["中"];
          html += '<div class="stl-advice">';
          html += '<div class="stl-advice-title">' + escHtml(a.direction);
          html += ' <span class="stl-advice-priority" style="' + pStyle + '">' + escHtml(a.priority) + '</span>';
          html += '</div>';
          html += '<div class="stl-advice-reason">理由: ' + escHtml(a.reason) + '</div>';
          if (a.expected_finding) html += '<div class="stl-advice-reason">预期: ' + escHtml(a.expected_finding) + '</div>';
          html += '</div>';
        }});
        html += '</div>';
      }}

      // Hypotheses
      if (hypotheses.length > 0) {{
        html += '<div class="stl-section">';
        html += '<div class="stl-section-title">🔮 假设预测</div>';
        hypotheses.forEach(function(h) {{
          html += '<div class="stl-hypothesis">';
          html += '<div class="stl-hypothesis-cond">如果: ' + escHtml(h.condition) + '</div>';
          html += '<div class="stl-hypothesis-then">→ 则: ' + escHtml(h.then) + '</div>';
          if (h.impact) html += '<div class="stl-hypothesis-impact">📈 影响: ' + escHtml(h.impact) + '</div>';
          html += '</div>';
        }});
        html += '</div>';
      }}

      html += '</div>'; // stl-stage-body
      html += '</div>'; // stl-stage
    }});

    timeline.innerHTML = html;

    // 默认展开第一个和最后一个
    if (stages.length > 0) {{
      var first = document.getElementById(cid + '_stage_0');
      var last = document.getElementById(cid + '_stage_' + (stages.length - 1));
      if (first) first.classList.add('expanded');
      if (last) last.classList.add('expanded');
    }}
  }}

  // Summary
  function renderSummary() {{
    if (stages.length === 0) return;
    var last = stages[stages.length - 1];
    var ranking = last.suspect_ranking || [];
    var confidence = last.confidence || 0;
    var turningPoints = stages.filter(function(s) {{ return s.key_turning_point; }});

    var html = '<div class="stl-summary">';
    html += '<h4>🏁 最终结论</h4>';
    html += '<table class="stl-summary-table"><tr><td>最终置信度</td><td><b>' + Math.round(confidence*100) + '%</b></td></tr>';
    if (ranking.length > 0) {{
      html += '<tr><td>最大嫌疑人</td><td><b>' + escHtml(ranking[0].name) + '</b> (' + Math.round((ranking[0].suspicion_score||0)*100) + '%)</td></tr>';
    }}
    html += '<tr><td>侦破阶段</td><td>' + stages.length + ' 阶段</td></tr>';
    html += '<tr><td>关键转折</td><td>' + turningPoints.length + ' 次</td></tr>';
    html += '</table>';

    if (turningPoints.length > 0) {{
      html += '<div style="margin-top:8px;font-size:12px">';
      turningPoints.forEach(function(tp) {{
        html += '⚡ ' + escHtml(tp.key_turning_point).substring(0, 60) + '<br/>';
      }});
      html += '</div>';
    }}

    html += '</div>';

    document.getElementById(cid + "_summary").innerHTML = html;
  }}

  renderStages();
  renderSummary();
}})();
</script>
</div>'''

    return html


# ═══════════════════════════════════════════════════════════════
# 排名变化对比图（嫌疑人在各阶段的变化）
# ═══════════════════════════════════════════════════════════════

def render_suspect_evolution_chart(stages: List[Dict]) -> str:
    """
    渲染嫌疑人在各阶段的嫌疑度变化折线图

    Args:
        stages: 阶段数据列表

    Returns:
        HTML字符串
    """
    if not stages:
        return _empty_html("暂无数据")

    container_id = f"suspect_evo_{int(_time.time()*1000)}"

    # 收集所有嫌疑人名称
    all_names = []
    for stage in stages:
        for sr in stage.get("suspect_ranking", []):
            name = sr.get("name", "")
            if name and name not in all_names:
                all_names.append(name)

    # 构建数据
    chart_data = {}
    for name in all_names:
        scores = []
        for stage in stages:
            found = 0
            for sr in stage.get("suspect_ranking", []):
                if sr.get("name") == name:
                    found = sr.get("suspicion_score", 0)
                    break
            scores.append(found)
        chart_data[name] = scores

    labels = [f"S{stage.get('stage_id', i+1)}" for i, stage in enumerate(stages)]
    labels_json = json.dumps(labels, ensure_ascii=False)
    data_json = json.dumps(chart_data, ensure_ascii=False)
    names_json = json.dumps(all_names, ensure_ascii=False)

    s_colors = ["#00FF41","#FFB800","#00FF41","#B37FEB","#FF003C","#8A8A9A","#00F5FF","#FF6B35"]

    html = f'''<div id="{container_id}" style="background:#FFF;border:2px solid #00FF41;border-radius:12px;overflow:hidden;box-shadow:0 4px 20px rgba(22,93,255,.12)">
<div style="background:linear-gradient(135deg,rgba(0,255,65,0.08),#00FF41);color:#FFF;padding:12px 18px;font-weight:700;font-size:14px">
📈 嫌疑人嫌疑度变化趋势
</div>
<div style="padding:16px;background:rgba(6,6,8,0.6)">
<canvas id="{container_id}_canvas" width="560" height="280"></canvas>
<div id="{container_id}_legend" style="margin-top:8px;display:flex;flex-wrap:wrap;gap:8px"></div>
</div>
<script>
(function() {{
  var cid = "{container_id}";
  var canvas = document.getElementById(cid + "_canvas");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  var labels = {labels_json};
  var data = {data_json};
  var names = {names_json};
  var colors = {json.dumps(s_colors)};

  var W = canvas.width, H = canvas.height;
  var padL = 50, padR = 20, padT = 20, padB = 40;
  var plotW = W - padL - padR, plotH = H - padT - padB;

  // Background
  ctx.fillStyle = "rgba(6,6,8,0.6)";
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = "#E5E6EB";
  ctx.lineWidth = 1;
  for (var i = 0; i <= 5; i++) {{
    var y = padT + plotH * (1 - i/5);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    ctx.fillStyle = "#8A8A9A";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText((i*20) + "%", padL - 8, y + 4);
  }}

  // X labels
  ctx.textAlign = "center";
  ctx.fillStyle = "#4E5969";
  ctx.font = "12px sans-serif";
  labels.forEach(function(l, i) {{
    var x = padL + plotW * i / (labels.length - 1 || 1);
    ctx.fillText(l, x, H - 10);
  }});

  // Lines
  names.forEach(function(name, ni) {{
    var scores = data[name] || [];
    var color = colors[ni % colors.length];

    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    scores.forEach(function(s, si) {{
      var x = padL + plotW * si / (labels.length - 1 || 1);
      var y = padT + plotH * (1 - s);
      if (si === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }});
    ctx.stroke();

    // Dots
    scores.forEach(function(s, si) {{
      var x = padL + plotW * si / (labels.length - 1 || 1);
      var y = padT + plotH * (1 - s);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#FFF";
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
    }});
  }});

  // Legend
  var legend = document.getElementById(cid + "_legend");
  if (legend) {{
    names.forEach(function(name, ni) {{
      var color = colors[ni % colors.length];
      var span = document.createElement("span");
      span.style.cssText = "display:inline-flex;align-items:center;gap:4px;font-size:12px;color:#4E5969";
      span.innerHTML = '<span style="display:inline-block;width:12px;height:3px;background:' + color + ';border-radius:2px"></span>' + name;
      legend.appendChild(span);
    }});
  }}
}})();
</script>
</div>'''

    return html


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _empty_html(msg: str) -> str:
    """空状态占位"""
    return (
        '<div style="background:#FFF;border:2px solid #E5E6EB;border-radius:12px;overflow:hidden">'
        '<div style="background:linear-gradient(135deg,rgba(0,255,65,0.08),#00FF41);color:#FFF;'
        'padding:12px 18px;font-weight:700;font-size:15px">🕵️ 时序侦破推理</div>'
        '<div style="padding:40px;color:#8A8A9A;text-align:center;background:rgba(6,6,8,0.6);'
        f'font-size:13px">{msg}</div></div>'
    )


def render_stage_markdown(stages: List[Dict]) -> str:
    """
    将阶段数据渲染为Markdown格式（用于非HTML场景）

    Args:
        stages: 阶段数据列表

    Returns:
        Markdown字符串
    """
    if not stages:
        return "暂无侦破阶段数据"

    md = "## 🕵️ 时序侦破推理过程\n\n"

    for stage in stages:
        sid = stage.get("stage_id", "?")
        name = stage.get("stage_name", f"阶段{sid}")
        confidence = stage.get("confidence", 0)
        reasoning = stage.get("reasoning", "")
        ranking = stage.get("suspect_ranking", [])
        advice = stage.get("investigation_advice", [])
        hypotheses = stage.get("hypotheses", [])
        turning_point = stage.get("key_turning_point")
        new_ev = stage.get("new_evidence", [])

        md += f"### 阶段 {sid}: {name}\n\n"
        md += f"- **置信度**: {confidence:.0%}\n"
        md += f"- **新增证据**: {len(new_ev)} 条\n"

        if turning_point:
            md += f"- **⚡ 关键转折**: {turning_point}\n"

        md += f"\n**推理过程**:\n> {reasoning[:500]}\n"

        if ranking:
            md += "\n**嫌疑人排名**:\n"
            for i, sr in enumerate(ranking[:5], 1):
                sc = sr.get("suspicion_score", 0)
                md += f"  {i}. **{sr.get('name', '?')}**: {sc:.0%} — {sr.get('reason', '')[:50]}\n"

        if advice:
            md += "\n**调查建议**:\n"
            for a in advice[:3]:
                md += f"- [{a.get('priority', '?')}] {a.get('direction', '?')} — {a.get('reason', '')[:50]}\n"

        if hypotheses:
            md += "\n**假设预测**:\n"
            for h in hypotheses[:3]:
                md += f"- 🔮 如果{h.get('condition', '?')} → 则{h.get('then', '?')[:50]}\n"

        md += "\n---\n\n"

    # 最终结论
    if stages:
        last = stages[-1]
        ranking = last.get("suspect_ranking", [])
        if ranking:
            md += "### 🏁 最终结论\n\n"
            md += f"**最大嫌疑人**: {ranking[0].get('name', '?')} "
            md += f"({ranking[0].get('suspicion_score', 0):.0%})\n"
            md += f"**最终置信度**: {last.get('confidence', 0):.0%}\n"

    return md
