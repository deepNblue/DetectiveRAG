"""
技能注册表 (Skill Registry)
管理专家Agent积累的可复用推理技能

技能生命周期:
  1. Learner从案件结果中提取 → 生成候选技能
  2. Registry校验并注册 → 存储到磁盘
  3. 分析新案件时 → Retriever匹配相关技能 → 注入Expert的prompt
  4. 使用后更新效果统计 → 自动淘汰低效技能

技能类型:
  - 推理技巧: "在投毒案中先检查受害者的社交圈"
  - 证据分析: "微量物证需要关注时间衰减"
  - 模式识别: "密室杀人的三种常见手法"
  - 反欺诈: "注意嫁祸和伪造证据的信号"
"""

import json
import os
import time
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from loguru import logger
from collections import defaultdict


@dataclass
class DetectiveSkill:
    """侦探技能"""
    skill_id: str
    name: str                      # 技能名称
    expert_type: str               # 所属专家类型
    category: str                  # 推理技巧/证据分析/模式识别/反欺诈/犯罪重建
    description: str               # 技能描述
    knowledge: str                 # 具体知识内容(会被注入prompt)
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
        # 触发条件:
        #   case_type: ["投毒案", "投放危险物质案"]
        #   evidence_types: ["毒物", "呕吐物"]
        #   keywords: ["密室", "不在场证明"]
        #   suspect_count_range: [3, 6]
    
    source_cases: List[str] = field(default_factory=list)   # 来源案件ID
    effectiveness: float = 0.0     # 成功率
    times_used: int = 0            # 使用次数
    times_correct: int = 0         # 使用后正确的次数
    times_incorrect: int = 0       # 使用后错误的次数
    confidence: float = 0.5        # 技能置信度
    learned_at: float = 0.0
    last_used: float = 0.0
    last_updated: float = 0.0
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.learned_at == 0.0:
            self.learned_at = time.time()
        if self.last_updated == 0.0:
            self.last_updated = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'DetectiveSkill':
        # 兼容旧数据
        valid_fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def update_effectiveness(self, correct: bool):
        """使用后更新效果统计"""
        self.times_used += 1
        if correct:
            self.times_correct += 1
        else:
            self.times_incorrect += 1
        self.effectiveness = self.times_correct / max(self.times_used, 1)
        self.last_used = time.time()
        self.last_updated = time.time()

    @property
    def is_reliable(self) -> bool:
        """技能是否可靠(使用>=3次且成功率>=50%)"""
        return self.times_used >= 3 and self.effectiveness >= 0.5

    @property
    def is_experimental(self) -> bool:
        """实验性技能(使用<3次)"""
        return self.times_used < 3


class SkillRegistry:
    """
    技能注册表
    管理所有专家的技能, 支持注册/查询/更新/淘汰
    """

    SKILL_CATEGORIES = [
        "reasoning_technique",    # 推理技巧
        "evidence_analysis",      # 证据分析
        "pattern_recognition",    # 模式识别
        "anti_deception",         # 反欺诈(嫁祸/伪造)
        "crime_reconstruction",   # 犯罪重建
        "timeline_analysis",      # 时间线分析
        "suspect_profiling",      # 嫌疑人画像
        "cross_validation",       # 交叉验证
    ]

    def __init__(self, path: str = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'memory', 'skills.json'
            )
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.logger = logger.bind(module="SkillRegistry")
        self._skills: Dict[str, DetectiveSkill] = {}
        self._load()

    def _load(self):
        """从磁盘加载技能"""
        if not os.path.exists(self.path):
            self._skills = {}
            return

        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._skills = {d["skill_id"]: DetectiveSkill.from_dict(d) for d in data}
            self.logger.info(f"加载 {len(self._skills)} 个技能")
        except Exception as e:
            self.logger.error(f"加载技能失败: {e}")
            self._skills = {}

    def _save(self):
        """保存技能到磁盘"""
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump([s.to_dict() for s in self._skills.values()],
                          f, ensure_ascii=False, indent=2)
            self.logger.info(f"保存 {len(self._skills)} 个技能")
        except Exception as e:
            self.logger.error(f"保存技能失败: {e}")

    def register(self, skill: DetectiveSkill) -> bool:
        """注册一个新技能"""
        # 校验
        if not skill.name or not skill.knowledge:
            self.logger.warning(f"技能 {skill.skill_id} 缺少必要字段, 跳过")
            return False

        if not skill.category:
            skill.category = "reasoning_technique"

        self._skills[skill.skill_id] = skill
        self._save()
        self.logger.info(f"注册技能: {skill.skill_id} '{skill.name}' "
                         f"[{skill.expert_type}/{skill.category}]")
        return True

    def get(self, skill_id: str) -> Optional[DetectiveSkill]:
        return self._skills.get(skill_id)

    def find_relevant(self, expert_type: str,
                      case_type: str = None,
                      keywords: List[str] = None,
                      evidence_types: List[str] = None,
                      min_effectiveness: float = 0.3,
                      limit: int = 5) -> List[DetectiveSkill]:
        """
        查找与当前案件相关的技能
        
        匹配逻辑:
          1. 专家类型精确匹配
          2. 触发条件匹配(案件类型/关键词/证据类型)
          3. 按效果排序(可靠技能优先, 其次实验性)
        """
        candidates = []
        
        for skill in self._skills.values():
            # 专家类型过滤
            if skill.expert_type != expert_type:
                continue

            # 效果过滤
            if skill.times_used >= 3 and skill.effectiveness < min_effectiveness:
                continue  # 淘汰低效的成熟技能

            # 触发条件匹配评分
            score = 0.0
            tc = skill.trigger_conditions

            # 案件类型匹配
            if case_type and "case_type" in tc:
                if case_type in tc["case_type"]:
                    score += 3.0
                elif any(ct in case_type for ct in tc["case_type"]):
                    score += 1.5

            # 关键词匹配
            if keywords and "keywords" in tc:
                for kw in keywords:
                    if kw in tc["keywords"]:
                        score += 1.0

            # 证据类型匹配
            if evidence_types and "evidence_types" in tc:
                for et in evidence_types:
                    if et in tc["evidence_types"]:
                        score += 1.0

            # 无触发条件 = 通用技能, 给基础分
            if not tc:
                score += 0.5

            # 效果加权
            if skill.is_reliable:
                score *= (1.0 + skill.effectiveness)
            elif skill.is_experimental:
                score *= 0.8

            if score > 0:
                candidates.append((score, skill))

        # 排序: 分数降序
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in candidates[:limit]]

    def update_usage(self, skill_id: str, correct: bool):
        """更新技能使用效果"""
        skill = self._skills.get(skill_id)
        if skill:
            skill.update_effectiveness(correct)
            self._save()

    def get_all_skills(self, expert_type: str = None) -> List[DetectiveSkill]:
        """获取所有技能"""
        skills = list(self._skills.values())
        if expert_type:
            skills = [s for s in skills if s.expert_type == expert_type]
        return sorted(skills, key=lambda s: s.effectiveness, reverse=True)

    def prune_ineffective(self, threshold: float = 0.2, min_uses: int = 5):
        """淘汰低效技能"""
        pruned = []
        for sid, skill in list(self._skills.items()):
            if skill.times_used >= min_uses and skill.effectiveness < threshold:
                pruned.append(sid)
                del self._skills[sid]
        
        if pruned:
            self._save()
            self.logger.info(f"淘汰 {len(pruned)} 个低效技能: {pruned}")
        return pruned

    def get_stats(self) -> Dict[str, Any]:
        """技能统计"""
        by_expert = defaultdict(int)
        by_category = defaultdict(int)
        reliable = 0
        experimental = 0
        
        for s in self._skills.values():
            by_expert[s.expert_type] += 1
            by_category[s.category] += 1
            if s.is_reliable:
                reliable += 1
            if s.is_experimental:
                experimental += 1

        return {
            "total": len(self._skills),
            "reliable": reliable,
            "experimental": experimental,
            "by_expert": dict(by_expert),
            "by_category": dict(by_category),
        }
