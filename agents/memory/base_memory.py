"""
记忆存储基类
每个专家Agent的案件经验记忆 — 记录每案的分析结果、对错、关键洞察

记忆结构:
  - case_id: 案件ID
  - expert_type: 专家类型(forensic/criminal/psychological/logic)
  - conclusion: 该专家的结论(凶手+置信度+推理)
  - actual_culprit: 实际凶手(事后反馈)
  - correct: 结论是否正确
  - key_insights: 该案中发现的洞察
  - missed_evidence: 遗漏的证据
  - reasoning_patterns: 使用的推理模式
  - case_type: 案件类型标签
  - timestamp: 记忆时间
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from loguru import logger
from collections import defaultdict


@dataclass
class CaseMemory:
    """单个案件经验记忆"""
    case_id: str
    expert_type: str           # forensic / criminal / psychological / logic
    conclusion: Dict[str, Any]  # {culprit, confidence, reasoning}
    actual_culprit: str = ""
    correct: Optional[bool] = None
    key_insights: List[str] = field(default_factory=list)
    missed_evidence: List[str] = field(default_factory=list)
    reasoning_patterns: List[str] = field(default_factory=list)
    case_type: str = ""         # 投毒案/密室杀人/盗窃...
    difficulty: str = ""        # 简单/中等/困难
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'CaseMemory':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MemoryStore:
    """
    记忆存储引擎
    按专家类型分目录存储, 支持按案件类型/对错检索
    
    存储结构:
      data/memory/
        forensic.json       # 法医专家的案件记忆
        criminal.json       # 刑侦专家的案件记忆
        psychological.json  # 心理画像专家的案件记忆
        logic.json          # 逻辑验证专家的案件记忆
        adjudicator.json    # 裁判Agent的案件记忆
    """

    EXPERT_TYPES = ["forensic", "criminal", "psychological", "logic", "adjudicator", "tech", "defense",
                    "financial", "interrogation", "prosecution", "intelligence",
                    "sherlock", "henry_lee", "song_ci", "poirot"]

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'memory'
            )
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.logger = logger.bind(module="MemoryStore")
        self._cache: Dict[str, List[CaseMemory]] = {}

    def _path(self, expert_type: str) -> str:
        return os.path.join(self.base_dir, f"{expert_type}.json")

    def load(self, expert_type: str) -> List[CaseMemory]:
        """加载某专家的所有记忆"""
        if expert_type in self._cache:
            return self._cache[expert_type]

        path = self._path(expert_type)
        if not os.path.exists(path):
            self._cache[expert_type] = []
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            memories = [CaseMemory.from_dict(d) for d in data]
            self._cache[expert_type] = memories
            self.logger.info(f"加载 {expert_type} 记忆: {len(memories)}条")
            return memories
        except Exception as e:
            self.logger.error(f"加载记忆失败 {expert_type}: {e}")
            self._cache[expert_type] = []
            return []

    def save(self, expert_type: str, memories: List[CaseMemory] = None):
        """保存记忆到磁盘"""
        if memories is None:
            memories = self._cache.get(expert_type, [])

        path = self._path(expert_type)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump([m.to_dict() for m in memories], f, ensure_ascii=False, indent=2)
            self._cache[expert_type] = memories
            self.logger.info(f"保存 {expert_type} 记忆: {len(memories)}条")
        except Exception as e:
            self.logger.error(f"保存记忆失败 {expert_type}: {e}")

    def add_memory(self, memory: CaseMemory):
        """添加一条案件记忆"""
        memories = self.load(memory.expert_type)
        
        # 去重: 同一专家同一案件只保留最新
        memories = [m for m in memories if m.case_id != memory.case_id]
        memories.append(memory)
        
        self.save(memory.expert_type, memories)
        self.logger.info(f"添加记忆: {memory.expert_type}/{memory.case_id} "
                         f"correct={memory.correct}")

    def add_batch(self, memories: List[CaseMemory]):
        """批量添加记忆"""
        by_expert = defaultdict(list)
        for m in memories:
            by_expert[m.expert_type].append(m)
        
        for expert_type, new_memories in by_expert.items():
            existing = self.load(expert_type)
            existing_ids = {m.case_id for m in existing}
            
            for m in new_memories:
                if m.case_id in existing_ids:
                    existing = [e for e in existing if e.case_id != m.case_id]
                existing.append(m)
            
            self.save(expert_type, existing)

    def query(self, expert_type: str = None,
              case_type: str = None,
              correct_only: bool = False,
              incorrect_only: bool = False,
              limit: int = 20) -> List[CaseMemory]:
        """
        查询记忆
        
        Args:
            expert_type: 专家类型(None=全部)
            case_type: 案件类型过滤
            correct_only: 只看正确案例
            incorrect_only: 只看错误案例
            limit: 返回数量限制
        """
        if expert_type:
            types = [expert_type]
        else:
            types = self.EXPERT_TYPES

        results = []
        for t in types:
            for m in self.load(t):
                if case_type and m.case_type != case_type:
                    continue
                if correct_only and m.correct is not True:
                    continue
                if incorrect_only and m.correct is not False:
                    continue
                results.append(m)

        # 按时间倒序
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        stats = {}
        for t in self.EXPERT_TYPES:
            memories = self.load(t)
            total = len(memories)
            correct = sum(1 for m in memories if m.correct is True)
            incorrect = sum(1 for m in memories if m.correct is False)
            unknown = total - correct - incorrect
            
            # 按案件类型统计
            case_types = defaultdict(int)
            for m in memories:
                if m.case_type:
                    case_types[m.case_type] += 1

            stats[t] = {
                "total": total,
                "correct": correct,
                "incorrect": incorrect,
                "unknown": unknown,
                "accuracy": round(correct / max(correct + incorrect, 1), 3),
                "case_types": dict(case_types),
            }
        return stats

    def get_insights(self, expert_type: str, case_type: str = None, limit: int = 10) -> List[str]:
        """获取某专家积累的关键洞察"""
        memories = self.load(expert_type)
        
        if case_type:
            memories = [m for m in memories if m.case_type == case_type]
        
        # 优先返回正确案例的洞察
        correct = [m for m in memories if m.correct is True]
        incorrect = [m for m in memories if m.correct is False]
        
        insights = []
        for m in correct + incorrect:
            insights.extend(m.key_insights)
            if m.correct is False and m.missed_evidence:
                insights.append(f"[教训] 遗漏了: {', '.join(m.missed_evidence)}")
        
        # 去重
        seen = set()
        unique = []
        for ins in insights:
            if ins not in seen:
                seen.add(ins)
                unique.append(ins)
        
        return unique[:limit]
