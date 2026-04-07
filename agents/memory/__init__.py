"""
侦探技能与记忆系统 (Detective Skill & Memory System)
每个专家Agent在分析案件过程中积累经验和技能，可复用。

架构:
  Memory层: 存储每个Agent的案件经验 (成功/失败教训)
  Skill层: 从经验中提炼的可复用推理技巧
  Pattern层: 案件类型/证据模式的模式库
  Learner层: 自动从案件结果中学习新技能
  Retriever层: 分析新案件时自动检索相关技能和记忆
"""

from .base_memory import MemoryStore, CaseMemory
from .skill_registry import SkillRegistry, DetectiveSkill
from .pattern_library import PatternLibrary, CrimePattern
from .skill_learner import SkillLearner
from .memory_retriever import MemoryRetriever

__all__ = [
    "MemoryStore", "CaseMemory",
    "SkillRegistry", "DetectiveSkill",
    "PatternLibrary", "CrimePattern",
    "SkillLearner",
    "MemoryRetriever",
]
