"""
动态业务专家系统 (Dynamic Domain Expert System)

核心理念:
  不是所有案件都需要相同的专家。经济案需要会计和金融专家，
  医疗案需要法医+临床医生，家暴案需要心理社工。
  
  系统在分析案件前，先通过LLM判断案件涉及的领域，
  然后动态创建/加载对应的业务专家Agent参与推理。

架构:
  DomainExpertFactory — 根据案件特征动态创建专家
  DomainKnowledgeBase — 每个领域的专业知识库
  ExpertRegistry      — 已注册的领域专家模板

使用流程:
  1. 案件进入 → CaseAnalyzer判断涉及的领域
  2. DomainExpertFactory创建对应的专家Agent
  3. DomainKnowledgeBase注入领域知识到prompt
  4. 动态专家参与并行推理
  5. 结果汇总 → 裁判

v1: 2026-04-04 初始版本
"""

from .expert_factory import DomainExpertFactory
from .domain_knowledge_base import DomainKnowledgeBase, KnowledgeEntry
from .expert_registry import ExpertRegistry, ExpertTemplate, get_default_registry

__all__ = [
    "DomainExpertFactory",
    "DomainKnowledgeBase",
    "KnowledgeEntry",
    "ExpertRegistry",
    "ExpertTemplate",
    "get_default_registry",
]