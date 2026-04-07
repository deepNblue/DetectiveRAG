"""
犯罪模式库 (Pattern Library)
从已分析案件中提炼的犯罪模式和证据关联模式

模式类型:
  - 犯罪模式: 某类犯罪的典型特征
  - 证据关联模式: 哪些证据通常相关联
  - 嫌疑人行为模式: 不同角色在犯罪中的典型行为
  - 嫁祸/伪装模式: 常见的误导性证据模式
"""

import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger
from collections import defaultdict, Counter


@dataclass
class CrimePattern:
    """犯罪模式"""
    pattern_id: str
    name: str                       # 模式名称
    pattern_type: str               # crime/evidence/suspect/deception
    description: str                # 模式描述
    characteristics: List[str]      # 典型特征列表
    red_flags: List[str]            # 红旗信号(出现时需特别注意)
    common_mistakes: List[str]      # 侦查中常见的误判
    source_cases: List[str] = field(default_factory=list)
    frequency: int = 0             # 遇到次数
    confirmed_count: int = 0       # 确认次数
    case_types: List[str] = field(default_factory=list)  # 关联案件类型
    confidence: float = 0.5
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.updated_at == 0.0:
            self.updated_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'CrimePattern':
        valid_fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


# ============================================================
# 预设的初始模式 (从经典案件和侦查学常识中提炼)
# ============================================================
BUILTIN_PATTERNS = [
    CrimePattern(
        pattern_id="pat_crime_001",
        name="密室杀人模式",
        pattern_type="crime",
        description="现场呈现密室状态，需从门锁机制、窗户、通风管道、心理密室等角度分析",
        characteristics=[
            "门窗从内部锁闭",
            "无外力破坏痕迹",
            "受害者独自在封闭空间",
            "钥匙在室内或只有死者持有",
        ],
        red_flags=[
            "管家/物业有万能钥匙",
            "窗户有被擦过的痕迹",
            "门锁可从外部操作(如绳线诡计)",
            "死亡时间与发现时间有间隔(伪造密室时间)",
        ],
        common_mistakes=[
            "过早排除自杀可能",
            "忽略机械诡计",
            "忽视心理层面的'密室'(证人证词的矛盾)",
        ],
        case_types=["密室杀人", "谋杀案", "故意杀人"],
        confidence=0.9,
    ),
    CrimePattern(
        pattern_id="pat_crime_002",
        name="投毒案模式",
        pattern_type="crime",
        description="受害者被毒物致死，需追踪毒物来源、投毒时机、动机链",
        characteristics=[
            "死者无明显外伤",
            "有毒物残留(食物/饮品/药物)",
            "死者生前有呕吐/腹痛等症状",
            "毒物获取需要特定渠道",
        ],
        red_flags=[
            "最近购买保险/修改遗嘱",
            "与最近接触的人有利益冲突",
            "毒物与某嫌疑人的职业/知识相关",
            "食物被调换或有他人单独接触的机会",
        ],
        common_mistakes=[
            "只关注有明确动机者，忽视'不起眼'的人",
            "忽略慢性投毒(多次少量)",
            "未追踪毒物来源",
        ],
        case_types=["投毒案", "投放危险物质案", "谋杀案"],
        confidence=0.9,
    ),
    CrimePattern(
        pattern_id="pat_crime_003",
        name="嫁祸模式",
        pattern_type="deception",
        description="真凶故意伪造证据指向他人，需要从证据链的'过于完美'处识破",
        characteristics=[
            "某些证据过于明显/容易被发现",
            "被嫁祸者有明确动机但缺乏细节关联",
            "关键物证只有被嫁祸者的痕迹，过于'干净'",
        ],
        red_flags=[
            "被嫁祸者的不在场证明看似薄弱但实际可靠",
            "某些证据的出现位置不合理(随身携带不利物证)",
            "证人证词存在'恰好'指认的巧合",
            "真凶主动提供对被嫁祸者不利的证据",
        ],
        common_mistakes=[
            "被'完美证据链'说服，忽视疑点",
            "过早锁定被嫁祸者而停止调查",
        ],
        case_types=["密室杀人", "投毒案", "谋杀案", "商业间谍", "敲诈勒索案"],
        confidence=0.85,
    ),
    CrimePattern(
        pattern_id="pat_crime_004",
        name="合谋犯罪模式",
        pattern_type="crime",
        description="两人以上配合作案，互相提供不在场证明或分工行动",
        characteristics=[
            "多人受益",
            "不在场证明互为支撑",
            "证据分散指向不同人",
            "各嫌疑人之间存在隐秘关系",
        ],
        red_flags=[
            "两个以上嫌疑人的不在场证明相互印证但无第三方佐证",
            "案件涉及多个步骤/地点，单人难以完成",
            "某嫌疑人只做了一部分但'恰好'另一人完成了剩余部分",
        ],
        common_mistakes=[
            "假设凶手只有一人",
            "忽略嫌疑人之间的非公开关系",
        ],
        case_types=["连环盗窃", "绑架案", "诈骗案", "商业间谍", "抢劫案"],
        confidence=0.85,
    ),
    CrimePattern(
        pattern_id="pat_crime_005",
        name="连环案件模式",
        pattern_type="crime",
        description="同一罪犯/团伙连续作案，案件之间有模式可循",
        characteristics=[
            "作案手法相似",
            "目标选择有规律",
            "时间间隔有规律",
            "逐渐升级(犯罪升级理论)",
        ],
        red_flags=[
            "近期有类似案件",
            "手法高度一致",
            "受害者有共同特征",
        ],
        common_mistakes=[
            "将相关案件视为独立案件",
            "忽略案件之间的微妙差异(可能是不同人模仿)",
        ],
        case_types=["连环盗窃", "网络犯罪", "诈骗案", "敲诈勒索案"],
        confidence=0.8,
    ),
    CrimePattern(
        pattern_id="pat_evidence_001",
        name="不在场证明验证模式",
        pattern_type="evidence",
        description="验证不在场证明的关键技巧",
        characteristics=[
            "时间戳的精确度(分钟级)",
            "证人可信度评估",
            "数字证据(监控/通话/网络记录)",
        ],
        red_flags=[
            "不在场证明只依赖一人证词",
            "时间精度不够(只说'大约')",
            "数字证据有篡改可能",
            "证人本身是嫌疑人",
        ],
        common_mistakes=[
            "接受模糊的不在场证明",
            "不验证证人本身的可靠性",
        ],
        case_types=[],  # 通用
        confidence=0.9,
    ),
    CrimePattern(
        pattern_id="pat_suspect_001",
        name="最不可能的人模式",
        pattern_type="suspect",
        description="凶手往往是最不起眼/看似无害的人",
        characteristics=[
            "身份低微(佣人/管家/清洁工)",
            "不起眼的社交角色",
            "与受害者有隐秘的利益关系",
        ],
        red_flags=[
            "某嫌疑人被所有其他人忽视",
            "某人的证词'过于诚实'",
            "某人主动配合调查但回避关键问题",
        ],
        common_mistakes=[
            "只关注有强烈动机的人",
            "忽视'服务型'角色(管家/秘书/司机)",
        ],
        case_types=["密室杀人", "投毒案", "谋杀案"],
        confidence=0.8,
    ),
]


class PatternLibrary:
    """
    犯罪模式库
    管理预设模式和从案件中学到的新模式
    """

    def __init__(self, path: str = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'memory', 'patterns.json'
            )
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.logger = logger.bind(module="PatternLibrary")
        self._patterns: Dict[str, CrimePattern] = {}
        self._load()

    def _load(self):
        """加载模式(预设+自定义)"""
        # 加载预设
        for p in BUILTIN_PATTERNS:
            self._patterns[p.pattern_id] = p

        # 加载自定义(覆盖同名预设)
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for d in data:
                    p = CrimePattern.from_dict(d)
                    self._patterns[p.pattern_id] = p
                self.logger.info(f"加载自定义模式: {len(data)}个")
            except Exception as e:
                self.logger.error(f"加载模式失败: {e}")

    def _save_custom(self):
        """保存自定义模式(不含内置)"""
        builtin_ids = {p.pattern_id for p in BUILTIN_PATTERNS}
        custom = [p.to_dict() for pid, p in self._patterns.items() if pid not in builtin_ids]
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(custom, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存模式失败: {e}")

    def add_pattern(self, pattern: CrimePattern):
        self._patterns[pattern.pattern_id] = pattern
        self._save_custom()
        self.logger.info(f"添加模式: {pattern.pattern_id} '{pattern.name}'")

    def find_matching(self, case_type: str = None,
                      keywords: List[str] = None,
                      pattern_type: str = None,
                      limit: int = 5) -> List[CrimePattern]:
        """
        查找匹配当前案件的模式
        
        返回最相关的模式及其红旗信号和常见误判
        """
        candidates = []

        for p in self._patterns.values():
            score = 0.0

            # 类型过滤
            if pattern_type and p.pattern_type != pattern_type:
                continue

            # 案件类型匹配
            if case_type:
                if case_type in p.case_types:
                    score += 3.0
                elif any(ct in case_type for ct in p.case_types):
                    score += 1.5

            # 关键词匹配
            if keywords:
                text = f"{p.name} {p.description} {' '.join(p.characteristics)}"
                for kw in keywords:
                    if kw in text:
                        score += 1.0

            # 通用模式(空case_types)给基础分
            if not p.case_types:
                score += 0.5

            # 置信度加权
            score *= p.confidence

            if score > 0:
                candidates.append((score, p))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in candidates[:limit]]

    def get_all_patterns(self, pattern_type: str = None) -> List[CrimePattern]:
        patterns = list(self._patterns.values())
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        return patterns

    def get_red_flags_for(self, case_type: str) -> List[str]:
        """获取某案件类型的所有红旗信号"""
        flags = []
        for p in self.find_matching(case_type=case_type, limit=10):
            flags.extend(p.red_flags)
        return list(dict.fromkeys(flags))  # 去重保序
