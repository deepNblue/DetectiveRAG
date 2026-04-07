"""
💾 ConversationStore — 推理对话持久化存储
v14.0: 保存推理过程中所有对话、分析结果，以时间戳+案件ID区分

存储结构:
  data/conversations/
    ├── {case_id}_{timestamp}/
    │   ├── meta.json          — 案件元数据
    │   ├── trad_line.json     — 传统线路完整对话
    │   ├── asmr_line.json     — ASMR线路完整对话(含多轮推理)
    │   ├── expert_rounds/     — 每个专家的多轮推理详情
    │   │   ├── forensic.json
    │   │   ├── criminal.json
    │   │   └── ...
    │   ├── fusion.json        — 融合决策
    │   └── events.jsonl       — 事件总线日志(每行一个事件)
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
from loguru import logger


class ConversationStore:
    """推理对话持久化存储"""
    
    def __init__(self, base_dir: str = "data/conversations"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._current_dir: Optional[str] = None
        self._case_id: Optional[str] = None
        self._events_buffer: List[Dict] = []
        self._rounds_buffer: Dict[str, List[Dict]] = {}  # agent_id -> rounds
    
    def start_case(self, case_id: str, case_text: str = "") -> str:
        """开始新案件，创建存储目录"""
        ts = time.strftime("%Y%m%d_%H%M%S")
        dir_name = f"{case_id}_{ts}"
        self._current_dir = os.path.join(self.base_dir, dir_name)
        os.makedirs(self._current_dir, exist_ok=True)
        os.makedirs(os.path.join(self._current_dir, "expert_rounds"), exist_ok=True)
        self._case_id = case_id
        self._events_buffer = []
        self._rounds_buffer = {}
        
        # 写入元数据
        meta = {
            "case_id": case_id,
            "timestamp": ts,
            "start_time": time.time(),
            "case_text_preview": case_text[:200] if case_text else "",
        }
        self._write_json("meta.json", meta)
        logger.info(f"💾 [ConversationStore] 案件 {case_id} 开始记录 → {dir_name}")
        return dir_name
    
    def save_trad_line(self, data: Dict):
        """保存传统线路完整数据"""
        if not self._current_dir:
            return
        self._write_json("trad_line.json", data)
    
    def save_asmr_line(self, data: Dict):
        """保存ASMR线路完整数据"""
        if not self._current_dir:
            return
        self._write_json("asmr_line.json", data)
    
    def save_fusion(self, data: Dict):
        """保存融合决策数据"""
        if not self._current_dir:
            return
        self._write_json("fusion.json", data)
    
    def save_event(self, event_type: str, agent_id: str, data: Dict):
        """追加保存单个事件"""
        if not self._current_dir:
            return
        event = {
            "ts": time.time(),
            "time": time.strftime("%H:%M:%S"),
            "type": event_type,
            "agent": agent_id,
            "data": data,
        }
        self._events_buffer.append(event)
        
        # 每20个事件flush一次
        if len(self._events_buffer) >= 20:
            self._flush_events()
    
    def save_agent_round(self, agent_id: str, round_data: Dict):
        """保存单个专家的一轮推理"""
        if not self._current_dir:
            return
        if agent_id not in self._rounds_buffer:
            self._rounds_buffer[agent_id] = []
        self._rounds_buffer[agent_id].append(round_data)
        
        # 每轮都flush到文件
        rounds_dir = os.path.join(self._current_dir, "expert_rounds")
        safe_id = agent_id.replace("/", "_").replace(" ", "_")
        self._write_json(os.path.join("expert_rounds", f"{safe_id}.json"),
                         self._rounds_buffer[agent_id])
    
    def save_agent_llm_call(self, agent_id: str, prompt: str, response: str,
                            round_num: int = 0, phase: str = ""):
        """保存Agent的单次LLM调用"""
        if not self._current_dir:
            return
        call_data = {
            "ts": time.time(),
            "time": time.strftime("%H:%M:%S"),
            "agent": agent_id,
            "round": round_num,
            "phase": phase,
            "prompt": prompt[:3000],  # 截断避免过大
            "response": response[:3000],
        }
        # 追加到llm_calls.jsonl
        llm_file = os.path.join(self._current_dir, "llm_calls.jsonl")
        with open(llm_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(call_data, ensure_ascii=False) + "\n")
    
    def finish_case(self, summary: Dict = None):
        """案件结束，flush所有数据"""
        if not self._current_dir:
            return
        
        # Flush剩余事件
        self._flush_events()
        
        # 写入摘要
        if summary:
            summary["end_time"] = time.time()
            summary["total_events"] = len(self._events_buffer)
            summary["agents_with_rounds"] = list(self._rounds_buffer.keys())
            self._write_json("summary.json", summary)
        
        logger.info(f"💾 [ConversationStore] 案件 {self._case_id} 记录完成, "
                     f"events={len(self._events_buffer)}, agents={len(self._rounds_buffer)}")
        
        self._current_dir = None
        self._case_id = None
    
    def _flush_events(self):
        """将事件缓冲区写入文件"""
        if not self._events_buffer or not self._current_dir:
            return
        events_file = os.path.join(self._current_dir, "events.jsonl")
        with open(events_file, "a", encoding="utf-8") as f:
            for event in self._events_buffer:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._events_buffer = []
    
    def _write_json(self, filename: str, data: Any):
        """写入JSON文件"""
        if not self._current_dir:
            return
        filepath = os.path.join(self._current_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    @property
    def current_dir(self) -> Optional[str]:
        return self._current_dir


# 全局单例
_store: Optional[ConversationStore] = None

def get_conversation_store() -> ConversationStore:
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store

def reset_conversation_store() -> ConversationStore:
    global _store
    _store = ConversationStore()
    return _store
