"""
🧵 ReasoningEventBus — 推理事件总线
v10.0: 让orchestrator内部推理过程实时暴露给UI

核心思路:
  orchestrator 每完成一个Agent → push_event()
  WebUI 轮询 → pop_events() → 渲染

事件类型:
  - stage_start:   阶段开始 (Stage 1/2/3.1/3.2/3.5/4)
  - stage_done:    阶段完成
  - agent_start:   单个Agent开始分析
  - agent_done:    单个Agent完成 (含推理摘要)
  - vote_cast:     投票 (每个专家的投票)
  - contradiction: 矛盾发现
  - overturn:      裁判推翻
  - conclusion:    最终结论
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class EventType(str, Enum):
    STAGE_START = "stage_start"
    STAGE_DONE = "stage_done"
    AGENT_START = "agent_start"
    AGENT_DONE = "agent_done"
    AGENT_ROUND_START = "agent_round_start"   # 🆕 多轮推理: 单轮开始
    AGENT_ROUND_DONE = "agent_round_done"     # 🆕 多轮推理: 单轮完成
    VOTE_CAST = "vote_cast"
    CONTRADICTION = "contradiction"
    OVERTURN = "overturn"
    CONCLUSION = "conclusion"
    MEMORY_RETRIEVED = "memory_retrieved"
    SKILL_APPLIED = "skill_applied"
    PROGRESS = "progress"


@dataclass
class ReasoningEvent:
    """单个推理事件"""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    stage: str = ""           # stage_1, stage_2, stage_31, stage_32, stage_35, stage_4
    agent_id: str = ""        # expert标识: forensic, criminal, sherlock, ...
    agent_name: str = ""      # 中文名: 法医专家, 刑侦专家, 福尔摩斯, ...
    agent_role: str = ""      # 层级: investigation, trial, verifier, adjudicator
    data: Dict[str, Any] = field(default_factory=dict)
    # data内容根据event_type不同:
    # agent_done: {"culprit": str, "confidence": float, "reasoning": str, "perspective": str,
    #              "multi_round": {"total_rounds": int, "rounds": [...], "early_stop": bool, "conclusion_changed": bool}}
    # agent_round_done: {"round": int, "phase": str, "culprit": str, "confidence": float, "changed": bool}
    # vote_cast:  {"expert": str, "culprit": str, "confidence": float, "weight": float}
    # stage_done: {"timing": float, "summary": str, "stats": dict}

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "data": self.data,
        }


class ReasoningEventBus:
    """
    线程安全的推理事件总线

    使用方式:
      bus = ReasoningEventBus()
      # orchestrator侧
      bus.push(ReasoningEvent(EventType.AGENT_DONE, agent_id="forensic", data={...}))
      # UI侧
      events = bus.pop_all()  # 获取并清空所有未消费事件
    """

    def __init__(self):
        self._events: List[ReasoningEvent] = []
        self._lock = threading.Lock()
        self._consumed_index: int = 0  # 已消费到的位置
        self._callbacks: List[Callable[[ReasoningEvent], None]] = []
        self._stage_timings: Dict[str, float] = {}
        self._agent_results: Dict[str, Dict] = {}  # agent_id -> latest result
        self._vote_history: List[Dict] = []  # 所有投票记录
        self._stage_progress: Dict[str, float] = {}  # stage -> progress 0-1
        self._agent_round_log: Dict[str, List[Dict]] = {}  # 🆕 agent_id -> [{round, phase, culprit, confidence}...]

    def push(self, event: ReasoningEvent):
        """推送事件（orchestrator侧调用）"""
        with self._lock:
            self._events.append(event)
            # 更新内部状态追踪
            if event.event_type == EventType.AGENT_DONE:
                self._agent_results[event.agent_id] = event.data
            elif event.event_type == EventType.AGENT_ROUND_DONE:
                # 🆕 追踪每个专家的多轮推理过程
                if event.agent_id not in self._agent_round_log:
                    self._agent_round_log[event.agent_id] = []
                self._agent_round_log[event.agent_id].append({
                    "round": event.data.get("round", 0),
                    "phase": event.data.get("phase", ""),
                    "culprit": event.data.get("culprit", "?"),
                    "confidence": event.data.get("confidence", 0),
                    "changed": event.data.get("changed", False),
                })
            elif event.event_type == EventType.VOTE_CAST:
                self._vote_history.append(event.data)
            elif event.event_type == EventType.STAGE_DONE:
                self._stage_timings[event.stage] = event.data.get("timing", event.data.get("time", 0))

        # 触发回调
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.warning(f"EventBus callback error: {e}")

    def pop_new(self) -> List[ReasoningEvent]:
        """获取所有未消费的事件（UI侧调用）"""
        with self._lock:
            new_events = self._events[self._consumed_index:]
            self._consumed_index = len(self._events)
            return new_events

    def pop_all(self) -> List[ReasoningEvent]:
        """获取并重置所有事件"""
        with self._lock:
            all_events = list(self._events)
            self._consumed_index = len(self._events)
            return all_events

    def get_state(self) -> Dict[str, Any]:
        """获取当前推理状态快照（用于UI渲染）"""
        with self._lock:
            return {
                "total_events": len(self._events),
                "stage_timings": dict(self._stage_timings),
                "agent_results": dict(self._agent_results),
                "vote_history": list(self._vote_history),
                "stage_progress": dict(self._stage_progress),
                "agent_round_log": dict(self._agent_round_log),  # 🆕 {agent_id: [{round, phase, culprit, confidence}...]}
            }

    def on_event(self, callback: Callable[[ReasoningEvent], None]):
        """注册事件回调"""
        self._callbacks.append(callback)

    def reset(self):
        """重置总线（新一轮推理前调用）"""
        with self._lock:
            self._events.clear()
            self._consumed_index = 0
            self._stage_timings.clear()
            self._agent_results.clear()
            self._vote_history.clear()
            self._stage_progress.clear()
            self._agent_round_log.clear()


# ── 全局单例 ──
_global_bus: Optional[ReasoningEventBus] = None

def get_event_bus() -> ReasoningEventBus:
    """获取全局事件总线单例"""
    global _global_bus
    if _global_bus is None:
        _global_bus = ReasoningEventBus()
    return _global_bus

def reset_event_bus() -> ReasoningEventBus:
    """重置并返回全局事件总线"""
    global _global_bus
    _global_bus = ReasoningEventBus()
    return _global_bus
