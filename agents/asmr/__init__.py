"""
ASMR (Agentic Search and Memory Retrieval) 模块
参考 Supermemory 团队技术：用多智能体取代传统向量检索RAG

三阶段架构:
  Stage 1 - 多Reader并行摄取: 从原始文本主动提取结构化知识
  Stage 2 - 多Searcher并行检索: 从不同维度主动推理搜索
  Stage 3 - 多Expert并行投票推理: 专家Agent投票得出最终结论
"""

from .readers import TimelineReader, PersonRelationReader, EvidenceReader
from .searchers import MotiveSearcher, OpportunitySearcher, CapabilitySearcher, TemporalSearcher
from .experts import ForensicExpert, CriminalExpert, PsychologicalProfiler, LogicVerifier
from .voting import ExpertVotingEngine
from .orchestrator import ASMROrchestrator

__all__ = [
    "TimelineReader",
    "PersonRelationReader",
    "EvidenceReader",
    "MotiveSearcher",
    "OpportunitySearcher",
    "CapabilitySearcher",
    "TemporalSearcher",
    "ForensicExpert",
    "CriminalExpert",
    "PsychologicalProfiler",
    "LogicVerifier",
    "ExpertVotingEngine",
    "ASMROrchestrator",
]
