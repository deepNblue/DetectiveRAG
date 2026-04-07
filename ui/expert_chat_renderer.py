"""
💬 ExpertChatRenderer — 专家群聊消息渲染器
v14.0: 将推理事件总线中的Agent对话实时渲染为群聊界面
"""
from typing import Dict, List, Any, Optional
import time


# Agent显示名映射
AGENT_NAMES = {
    "forensic": "🔬 法医专家",
    "criminal": "🔫 犯罪分析师",
    "profiler": "🧠 心理画像师",
    "tech": "💻 技术调查员",
    "financial": "💰 金融调查员",
    "interrogation": "🗣️ 审讯分析师",
    "intelligence": "🕵️ 情报分析师",
    "sherlock": "🔍 福尔摩斯",
    "henry_lee": "🔬 李昌钰",
    "song_ci": "📜 宋慈",
    "poirot": "🎩 波洛",
    "prosecution": "🔴 检察官",
    "defense": "🔵 辩护律师",
    "judge": "⚖️ 法官",
    "juror": "👥 陪审员",
    "logic_verifier": "✅ 逻辑验证",
    "adjudicator": "👨‍⚖️ 裁判",
    "timeline": "📅 时间线Reader",
    "person_relation": "👥 人物关系Reader",
    "evidence": "📋 证据Reader",
    "motive": "🎯 动机搜索",
    "opportunity": "⏰ 机会搜索",
    "capability": "💪 能力搜索",
    "temporal": "🕐 时序搜索",
    "contradiction": "⚡ 矛盾搜索",
    "user": "👤 指挥官",
    "system": "⚙️ 系统",
}

# Agent颜色映射
AGENT_COLORS = {
    "forensic": "#00FF41",
    "criminal": "#FF6B6B",
    "profiler": "#B37FEB",
    "tech": "#00F5FF",
    "financial": "#FFD700",
    "interrogation": "#FF8C00",
    "intelligence": "#0088FF",
    "sherlock": "#FF003C",
    "henry_lee": "#00FF41",
    "song_ci": "#FFB800",
    "poirot": "#9B59B6",
    "prosecution": "#FF4444",
    "defense": "#4488FF",
    "judge": "#FFD700",
    "juror": "#888888",
    "logic_verifier": "#00F5FF",
    "adjudicator": "#FFD700",
    "user": "#00FF41",
    "system": "#55556A",
}


def _get_agent_display(agent_id: str) -> tuple:
    """获取Agent的显示名和颜色"""
    name = AGENT_NAMES.get(agent_id, f"🤖 {agent_id[:10]}")
    color = AGENT_COLORS.get(agent_id, "#8A8A9A")
    return name, color


def render_chat_message(agent_id: str, message: str, msg_type: str = "text",
                         timestamp: str = "", round_info: str = "") -> str:
    """渲染单条聊天消息"""
    name, color = _get_agent_display(agent_id)
    ts_display = f'<span style="color:#55556A;font-size:9px">{timestamp}</span>' if timestamp else ''
    round_display = f'<span style="color:#FFD700;font-size:9px;margin-left:4px">{round_info}</span>' if round_info else ''
    
    # 截断长消息
    display_msg = message[:300] + "…" if len(message) > 300 else message
    
    if agent_id == "user":
        # 用户消息靠右
        return f'''<div style="margin:6px 0;text-align:right">
          <div style="display:inline-block;text-align:left;max-width:85%;background:rgba(0,255,65,0.08);border:1px solid rgba(0,255,65,0.15);border-radius:8px;padding:6px 10px">
            <div style="font-size:9px;color:#00FF41;margin-bottom:2px">{name} {ts_display}{round_display}</div>
            <div style="font-size:11px;color:#B0B0C0;line-height:1.5">{display_msg}</div>
          </div>
        </div>'''
    elif agent_id == "system":
        return f'''<div style="margin:4px 0;text-align:center">
          <span style="background:rgba(85,85,106,0.15);color:#55556A;font-size:10px;padding:2px 8px;border-radius:4px">{display_msg}</span>
        </div>'''
    else:
        # Agent消息靠左
        return f'''<div style="margin:6px 0">
          <div style="display:inline-block;text-align:left;max-width:90%;background:rgba({_hex_to_rgb_str(color)},0.06);border:1px solid rgba({_hex_to_rgb_str(color)},0.12);border-radius:8px;padding:6px 10px">
            <div style="font-size:9px;color:{color};margin-bottom:2px">{name} {ts_display}{round_display}</div>
            <div style="font-size:11px;color:#B0B0C0;line-height:1.5">{display_msg}</div>
          </div>
        </div>'''


def _hex_to_rgb_str(hex_color: str) -> str:
    """#00FF41 → 0,255,65"""
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return "138,138,154"
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


class ExpertChatRenderer:
    """专家群聊渲染器 — 从事件总线获取状态，渲染聊天界面"""
    
    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages
        self._messages: List[Dict] = []
    
    def add_user_message(self, message: str) -> None:
        """添加用户消息到对话历史"""
        import time
        self._messages.append({
            "agent_id": "user",
            "message": message,
            "type": "user",
            "timestamp": time.time(),
        })
        # 限制总数
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
    
    def add_agent_response(self, agent_id: str, message: str) -> None:
        """添加Agent回复到对话历史"""
        import time
        self._messages.append({
            "agent_id": agent_id,
            "message": message,
            "type": "agent",
            "timestamp": time.time(),
        })
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
    
    def render_chat_panel(self, agent_results: Dict, agent_round_log: Dict = None,
                          stage_progress: Dict = None) -> str:
        """
        渲染完整群聊面板
        
        Args:
            agent_results: 事件总线中的agent结果 {agent_id: {culprit, confidence, reasoning, ...}}
            agent_round_log: 多轮推理日志 {agent_id: [{round, phase, culprit, ...}]}
            stage_progress: 阶段进度
        """
        messages = []
        agent_round_log = agent_round_log or {}
        
        # 🆕 v14.0.1: 混入用户和Agent历史消息
        for m in self._messages:
            messages.append({
                "agent_id": m["agent_id"],
                "message": m["message"],
                "type": m["type"],
            })
        
        # 系统消息: 阶段进度
        if stage_progress:
            for stage, status in stage_progress.items():
                if status == "done":
                    stage_name = {
                        "stage_0_vision": "🖼️ 图片分析",
                        "stage_1_readers": "📚 情报摄取",
                        "stage_2_searchers": "🔍 矛盾搜索",
                        "stage_31_investigation": "🔬 调查层分析",
                        "stage_32_trial": "⚖️ 审判层辩论",
                        "stage_4_adjudicator": "👨‍⚖️ 裁判裁决",
                    }.get(stage, stage)
                    messages.append({
                        "agent_id": "system",
                        "message": f"✅ {stage_name} 完成",
                        "type": "system",
                    })
        
        # Agent完成消息
        for agent_id, result in agent_results.items():
            if not isinstance(result, dict):
                continue
            
            error = result.get("error")
            if error:
                _, color = _get_agent_display(agent_id)
                messages.append({
                    "agent_id": agent_id,
                    "message": f"⚠️ 分析异常: {error[:100]}",
                    "type": "error",
                    "timestamp": "",
                })
                continue
            
            culprit = result.get("culprit", "?")
            confidence = result.get("confidence", 0)
            reasoning = result.get("reasoning", "")
            
            # 获取多轮推理信息
            mr = result.get("multi_round", {})
            total_rounds = mr.get("total_rounds", 0) if isinstance(mr, dict) else 0
            
            # 主结论消息
            conf_str = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else str(confidence)
            msg = f"结论: {culprit} (置信度={conf_str})"
            if reasoning:
                msg += f"\n💡 {reasoning[:200]}"
            
            round_info = f"R{total_rounds}轮" if total_rounds > 1 else ""
            
            messages.append({
                "agent_id": agent_id,
                "message": msg,
                "type": "result",
                "round_info": round_info,
            })
            
            # 多轮推理过程消息
            if agent_id in agent_round_log:
                rounds = agent_round_log[agent_id]
                if isinstance(rounds, list):
                    for r in rounds[-3:]:  # 只展示最近3轮
                        phase = r.get("phase", "")
                        r_culprit = r.get("culprit", "?")
                        r_conf = r.get("confidence", 0)
                        changed = r.get("changed", r.get("culprit_changed", False))
                        
                        if changed:
                            r_msg = f"🔄 结论改变! → {r_culprit} ({r_conf:.0%})"
                        else:
                            r_msg = f"维持: {r_culprit} ({r_conf:.0%})"
                        
                        messages.append({
                            "agent_id": agent_id,
                            "message": f"[{phase}] {r_msg}",
                            "type": "round",
                            "round_info": f"R{r.get('round', '?')}",
                        })
        
        # 限制消息数
        messages = messages[-self.max_messages:]
        
        if not messages:
            return '<div style="text-align:center;color:#55556A;padding:30px;font-family:JetBrains Mono,monospace;font-size:11px">等待推理启动…</div>'
        
        # 渲染
        parts = []
        for msg in messages:
            parts.append(render_chat_message(
                agent_id=msg["agent_id"],
                message=msg["message"],
                msg_type=msg.get("type", "text"),
                timestamp=msg.get("timestamp", ""),
                round_info=msg.get("round_info", ""),
            ))
        
        # 包裹在滚动容器中
        return f'''<div style="height:320px;overflow-y:auto;font-family:JetBrains Mono,monospace;font-size:11px;padding:8px">
          {"".join(parts)}
        </div>'''
