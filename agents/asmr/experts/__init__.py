"""ASMR Stage 3 - 多Expert并行投票推理

v9: 三层架构重构 — 调查层 + 审判层 + 裁判层

=== 调查层 (Stage 3.1) — 搜集证据、构建推理链、提出假设 ===
  技术专家: ForensicExpert + CriminalExpert + TechInvestigator + FinancialInvestigator
  行为专家: PsychologicalProfiler + InterrogationAnalyst + IntelligenceAnalyst
  名侦探: SherlockAnalyst + HenryLeeAnalyst + SongCiAnalyst + PoirotAnalyst
  验证: LogicVerifier

=== 审判层 (Stage 3.2) — 控辩对抗 + 中立裁判 + 常识判断 ===
  🔴 检察官 ProsecutionReviewer: 有罪推定，构建控诉证据链
  🔵 辩护律师 DefenseAttorney: 无罪推定，寻找合理怀疑
  🧑‍⚖️ 法官 Judge: 中立裁判，独立心证
  👥 陪审员 Juror: 常识判断，朴素正义

=== 裁判层 (Stage 4) — 综合裁决 ===
  ⚖️ 裁判官 Adjudicator: 综合投票+推理树+审判团，最终裁决
"""

from .forensic_expert import ForensicExpert
from .criminal_expert import CriminalExpert
from .psychological_profiler import PsychologicalProfiler
from .logic_verifier import LogicVerifier
from .adjudicator import Adjudicator
from .tech_investigator import TechInvestigator
from .defense_attorney import DefenseAttorney
from .financial_investigator import FinancialInvestigator          # Phase 2
from .interrogation_analyst import InterrogationAnalyst            # Phase 2
from .prosecution_reviewer import ProsecutionReviewer              # Phase 3 → 审判层
from .intelligence_analyst import IntelligenceAnalyst              # Phase 3
from .sherlock_analyst import SherlockAnalyst                      # Phase 4: 名侦探
from .henry_lee_analyst import HenryLeeAnalyst                     # Phase 4: 名侦探
from .song_ci_analyst import SongCiAnalyst                         # Phase 4: 名侦探
from .poirot_analyst import PoirotAnalyst                          # Phase 4: 名侦探
# 🆕 v9: 审判层角色
from .judge import Judge                                           # 🆕 法官
from .juror import Juror                                           # 🆕 陪审员

__all__ = [
    # 调查层
    "ForensicExpert",
    "CriminalExpert",
    "PsychologicalProfiler",
    "LogicVerifier",
    "TechInvestigator",
    "FinancialInvestigator",
    "InterrogationAnalyst",
    "IntelligenceAnalyst",
    # 名侦探
    "SherlockAnalyst",
    "HenryLeeAnalyst",
    "SongCiAnalyst",
    "PoirotAnalyst",
    # 审判层
    "DefenseAttorney",
    "ProsecutionReviewer",
    "Judge",
    "Juror",
    # 裁判层
    "Adjudicator",
]
