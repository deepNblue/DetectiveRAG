"""
记忆增强的Expert混入类 (Memory-Enhanced Expert Mixin)
为Expert Agent提供记忆和技能检索能力

使用方式: 在Expert的process()中调用self.retrieve_memory_context()
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import Dict, List, Any
from loguru import logger

from agents.memory import MemoryRetriever, MemoryStore, SkillRegistry, PatternLibrary


class MemoryEnhancedMixin:
    """
    记忆增强混入类
    为任何BaseAgent子类提供记忆检索能力
    
    使用方法:
        class MyExpert(MemoryEnhancedMixin, BaseAgent):
            def __init__(self, ...):
                super().__init__(...)
                self._init_memory(self.name)
            
            def process(self, input_data):
                # 获取记忆增强上下文
                memory_ctx = self.retrieve_memory_context(
                    case_type=input_data.get("case_type"),
                    case_data=input_data,
                )
                # 注入到prompt
                prompt = base_prompt + memory_ctx["context_text"]
                ...
    """

    # 类级别的共享实例(所有Expert共享同一个检索器)
    _shared_retriever: MemoryRetriever = None
    _shared_memory_store: MemoryStore = None
    _shared_skill_registry: SkillRegistry = None
    _shared_pattern_library: PatternLibrary = None

    def _init_memory(self, agent_name: str):
        """初始化记忆系统(延迟加载, 所有Expert共享)"""
        if MemoryEnhancedMixin._shared_retriever is None:
            MemoryEnhancedMixin._shared_memory_store = MemoryStore()
            MemoryEnhancedMixin._shared_skill_registry = SkillRegistry()
            MemoryEnhancedMixin._shared_pattern_library = PatternLibrary()
            MemoryEnhancedMixin._shared_retriever = MemoryRetriever(
                memory_store=MemoryEnhancedMixin._shared_memory_store,
                skill_registry=MemoryEnhancedMixin._shared_skill_registry,
                pattern_library=MemoryEnhancedMixin._shared_pattern_library,
            )
            logger.info("🧠 记忆系统初始化完成(全局共享)")

        self._retriever = MemoryEnhancedMixin._shared_retriever
        self._memory_store = MemoryEnhancedMixin._shared_memory_store
        self._skill_registry = MemoryEnhancedMixin._shared_skill_registry

        # 映射agent name → expert type
        self._expert_type = self._map_expert_type(agent_name)
        logger.debug(f"记忆增强: {agent_name} → expert_type={self._expert_type}")

    def _map_expert_type(self, name: str) -> str:
        """将Agent名称映射到记忆系统中的expert_type"""
        mapping = {
            "ASMR-ForensicExpert": "forensic",
            "ASMR-CriminalExpert": "criminal",
            "ASMR-PsychologicalProfiler": "psychological",
            "ASMR-LogicVerifier": "logic",
            "ASMR-Adjudicator": "adjudicator",
            "ASMR-TechInvestigator": "tech",
            "ASMR-DefenseAttorney": "defense",
        }
        return mapping.get(name, name.lower().replace("asmr-", ""))

    def retrieve_memory_context(self, case_type: str = None,
                                 case_data: Dict = None,
                                 keywords: List[str] = None) -> Dict[str, Any]:
        """
        检索与当前案件相关的记忆和技能
        
        Returns:
            {
                "context_text": str,     # 可注入prompt的文本
                "skills": list,          # 相关技能
                "patterns": list,        # 相关模式
                "used_skill_ids": list,  # 使用的技能ID
            }
        """
        if not hasattr(self, '_retriever') or self._retriever is None:
            self._init_memory(getattr(self, 'name', 'unknown'))

        return self._retriever.retrieve_context(
            expert_type=self._expert_type,
            case_type=case_type,
            case_data=case_data,
            keywords=keywords,
        )

    def get_memory_enhanced_prompt(self, base_prompt: str,
                                    case_type: str = None,
                                    case_data: Dict = None) -> str:
        """
        获取记忆增强后的prompt
        
        在base_prompt末尾追加相关的技能/经验/模式
        """
        ctx = self.retrieve_memory_context(
            case_type=case_type,
            case_data=case_data,
        )
        
        if ctx["context_text"]:
            return base_prompt + "\n" + ctx["context_text"]
        return base_prompt
