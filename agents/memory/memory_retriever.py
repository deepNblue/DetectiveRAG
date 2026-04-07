"""
记忆检索器 (Memory Retriever)
分析新案件时，自动检索相关技能、记忆和模式，注入到Expert的prompt中

这是记忆系统的"读取端"，与SkillLearner的"写入端"配合使用。
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
from loguru import logger

from .base_memory import MemoryStore, CaseMemory
from .skill_registry import SkillRegistry, DetectiveSkill
from .pattern_library import PatternLibrary, CrimePattern


class MemoryRetriever:
    """
    记忆检索器
    在分析新案件前，检索相关技能和经验，生成增强的prompt上下文
    """

    def __init__(self, memory_store: MemoryStore = None,
                 skill_registry: SkillRegistry = None,
                 pattern_library: PatternLibrary = None):
        self.memory = memory_store or MemoryStore()
        self.skills = skill_registry or SkillRegistry()
        self.patterns = pattern_library or PatternLibrary()
        self.logger = logger.bind(module="MemoryRetriever")

    def retrieve_context(self, expert_type: str,
                         case_type: str = None,
                         case_data: Dict = None,
                         keywords: List[str] = None,
                         max_skills: int = 3,
                         max_memories: int = 3,
                         max_patterns: int = 2) -> Dict[str, Any]:
        """
        检索与当前案件相关的所有知识
        
        Returns:
            {
                "skills": [DetectiveSkill...],           # 相关技能
                "memories": [CaseMemory...],             # 相关历史记忆
                "patterns": [CrimePattern...],           # 相关犯罪模式
                "red_flags": [str...],                   # 红旗信号
                "insights": [str...],                    # 积累的洞察
                "context_text": str,                     # 格式化的上下文文本(可直接注入prompt)
                "used_skill_ids": [str...],              # 使用的技能ID(用于后续效果追踪)
            }
        """
        # 提取关键词
        if keywords is None and case_data:
            keywords = self._extract_keywords(case_data)

        # 1. 检索相关技能
        relevant_skills = self.skills.find_relevant(
            expert_type=expert_type,
            case_type=case_type,
            keywords=keywords,
            limit=max_skills,
        )

        # 2. 检索相关记忆
        relevant_memories = self._find_similar_memories(
            expert_type=expert_type,
            case_type=case_type,
            limit=max_memories,
        )

        # 3. 检索相关模式
        matching_patterns = self.patterns.find_matching(
            case_type=case_type,
            keywords=keywords,
            limit=max_patterns,
        )

        # 4. 收集红旗信号
        red_flags = self.patterns.get_red_flags_for(case_type) if case_type else []

        # 5. 收集积累的洞察
        insights = self.memory.get_insights(expert_type, case_type=case_type, limit=5)

        # 6. 生成上下文文本
        context_text = self._format_context(
            relevant_skills, relevant_memories, matching_patterns,
            red_flags, insights
        )

        result = {
            "skills": relevant_skills,
            "memories": relevant_memories,
            "patterns": matching_patterns,
            "red_flags": red_flags[:5],
            "insights": insights,
            "context_text": context_text,
            "used_skill_ids": [s.skill_id for s in relevant_skills],
        }

        self.logger.info(f"检索完成: {expert_type}/{case_type} → "
                         f"{len(relevant_skills)}技能, "
                         f"{len(relevant_memories)}记忆, "
                         f"{len(matching_patterns)}模式")

        return result

    def get_enhanced_prompt_section(self, expert_type: str,
                                    case_type: str = None,
                                    case_data: Dict = None,
                                    keywords: List[str] = None) -> str:
        """
        生成可注入到Expert prompt中的增强文本
        
        用法: 在构建expert prompt时，将此文本追加到prompt末尾
        
        Example:
            prompt = base_prompt + retriever.get_enhanced_prompt_section(
                expert_type="forensic",
                case_type="投毒案",
                case_data=case_data,
            )
        """
        ctx = self.retrieve_context(
            expert_type=expert_type,
            case_type=case_type,
            case_data=case_data,
            keywords=keywords,
        )
        return ctx["context_text"]

    def _find_similar_memories(self, expert_type: str,
                                case_type: str = None,
                                limit: int = 3) -> List[CaseMemory]:
        """查找相似案件的历史记忆"""
        # 优先查找同类型案件的正确经验
        memories = self.memory.query(
            expert_type=expert_type,
            case_type=case_type,
            correct_only=True,
            limit=limit,
        )

        # 如果太少，补充错误经验(教训也有价值)
        if len(memories) < limit:
            wrong_memories = self.memory.query(
                expert_type=expert_type,
                case_type=case_type,
                incorrect_only=True,
                limit=limit - len(memories),
            )
            memories.extend(wrong_memories)

        # 如果还太少，不限案件类型
        if len(memories) < limit:
            any_memories = self.memory.query(
                expert_type=expert_type,
                limit=limit - len(memories),
            )
            # 去重
            existing_ids = {m.case_id for m in memories}
            for m in any_memories:
                if m.case_id not in existing_ids:
                    memories.append(m)

        return memories[:limit]

    def _extract_keywords(self, case_data: Dict) -> List[str]:
        """从案件数据中提取关键词"""
        keywords = []

        # 案件类型
        case_type = case_data.get("case_type", "")
        if case_type:
            keywords.append(case_type)

        # 证据类型
        for e in case_data.get("evidence", []):
            if isinstance(e, dict):
                etype = e.get("type", "")
                if etype:
                    keywords.append(etype)

        # 特殊关键词检测
        text = json.dumps(case_data, ensure_ascii=False)
        special_kw = ["密室", "投毒", "嫁祸", "合谋", "不在场证明", "伪造",
                       "盗窃", "诈骗", "绑架", "纵火", "网络犯罪"]
        for kw in special_kw:
            if kw in text:
                keywords.append(kw)

        return list(set(keywords))

    def _format_context(self, skills: List[DetectiveSkill],
                        memories: List[CaseMemory],
                        patterns: List[CrimePattern],
                        red_flags: List[str],
                        insights: List[str]) -> str:
        """格式化为可注入prompt的文本"""
        sections = []

        # 技能
        if skills:
            lines = ["\n📚 你积累的推理技能:"]
            for s in skills:
                status = "✅可靠" if s.is_reliable else "🧪实验性"
                lines.append(f"  [{status}] {s.name}: {s.knowledge}")
            sections.append("\n".join(lines))

        # 历史经验
        if memories:
            lines = ["\n📖 相关历史经验:"]
            for m in memories:
                if m.correct is True:
                    lines.append(f"  ✅ {m.case_id}({m.case_type}): "
                                 f"正确识别{m.actual_culprit}")
                    for ins in m.key_insights[:2]:
                        lines.append(f"     洞察: {ins[:80]}")
                elif m.correct is False:
                    lines.append(f"  ❌ {m.case_id}({m.case_type}): "
                                 f"误判为{m.conclusion.get('culprit','?')}, "
                                 f"实际是{m.actual_culprit}")
                    for me in m.missed_evidence[:2]:
                        lines.append(f"     教训: {me[:80]}")
            sections.append("\n".join(lines))

        # 犯罪模式
        if patterns:
            lines = ["\n🔍 相关犯罪模式:"]
            for p in patterns:
                lines.append(f"  ⚡ {p.name}: {p.description}")
                if p.red_flags:
                    lines.append(f"     红旗信号: {'; '.join(p.red_flags[:3])}")
            sections.append("\n".join(lines))

        # 积累洞察
        if insights:
            lines = ["\n💡 积累的洞察:"]
            for ins in insights[:5]:
                lines.append(f"  • {ins[:80]}")
            sections.append("\n".join(lines))

        if not sections:
            return ""

        return "\n".join(sections) + "\n\n请参考以上经验进行分析，但不要被历史经验束缚，每个案件都有其独特之处。"
