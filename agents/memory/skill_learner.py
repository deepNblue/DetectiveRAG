"""
技能学习器 (Skill Learner)
从案件分析结果中自动提炼技能和洞察

工作流程:
  1. 接收案件分析结果(各expert结论 + 实际答案)
  2. 分析成功/失败原因
  3. 提炼可复用的推理技巧 → 生成DetectiveSkill
  4. 提炼模式观察 → 生成CrimePattern
  5. 注册到SkillRegistry和PatternLibrary
"""

import json
import os
import time
import hashlib
from typing import Dict, List, Any, Optional
from loguru import logger

from .base_memory import MemoryStore, CaseMemory
from .skill_registry import SkillRegistry, DetectiveSkill
from .pattern_library import PatternLibrary, CrimePattern


class SkillLearner:
    """
    技能学习器
    在案件完成后，分析结果并提炼技能
    """

    def __init__(self, memory_store: MemoryStore = None,
                 skill_registry: SkillRegistry = None,
                 pattern_library: PatternLibrary = None,
                 llm_client=None):
        self.memory = memory_store or MemoryStore()
        self.skills = skill_registry or SkillRegistry()
        self.patterns = pattern_library or PatternLibrary()
        self.llm = llm_client
        self.logger = logger.bind(module="SkillLearner")

    def learn_from_case(self, case_id: str, case_data: Dict,
                        expert_results: Dict[str, Dict],
                        voting_result: Dict,
                        actual_culprit: str,
                        adjudicator_result: Dict = None) -> Dict[str, Any]:
        """
        从单案中学习
        
        Args:
            case_id: 案件ID
            case_data: 案件原始数据(含suspects/evidence/timeline等)
            expert_results: 各expert的分析结果 {expert_type: {culprit, confidence, reasoning}}
            voting_result: 投票结果
            actual_culprit: 实际凶手
            adjudicator_result: 裁判结果(如有)
        
        Returns:
            学习结果(新生成的技能/模式/记忆)
        """
        self.logger.info(f"开始学习: {case_id}, 实际凶手={actual_culprit}")

        case_type = case_data.get("case_type", case_data.get("title", ""))
        difficulty = case_data.get("difficulty", "中等")

        # 1. 为每个专家创建案件记忆
        memories_created = self._create_memories(
            case_id, case_type, difficulty,
            expert_results, voting_result, actual_culprit
        )

        # 2. 使用LLM提炼技能(如果有LLM)
        new_skills = []
        if self.llm:
            new_skills = self._learn_skills_with_llm(
                case_id, case_type, case_data,
                expert_results, voting_result, actual_culprit
            )
        else:
            # 简单规则提炼
            new_skills = self._learn_skills_by_rules(
                case_id, case_type,
                expert_results, actual_culprit
            )

        # 3. 提炼模式
        new_patterns = self._learn_patterns(
            case_id, case_type,
            expert_results, actual_culprit, case_data
        )

        result = {
            "case_id": case_id,
            "memories_created": len(memories_created),
            "new_skills": len(new_skills),
            "new_patterns": len(new_patterns),
            "skill_ids": [s.skill_id for s in new_skills],
            "pattern_ids": [p.pattern_id for p in new_patterns],
        }

        self.logger.info(f"学习完成: {case_id} → "
                         f"{result['memories_created']}记忆, "
                         f"{result['new_skills']}技能, "
                         f"{result['new_patterns']}模式")

        return result

    def _create_memories(self, case_id: str, case_type: str, difficulty: str,
                         expert_results: Dict, voting_result: Dict,
                         actual_culprit: str) -> List[CaseMemory]:
        """为每个专家创建案件记忆"""
        memories = []

        for expert_type, result in expert_results.items():
            data = result.get("data", result)
            culprit = data.get("culprit", "未知")
            confidence = data.get("confidence", 0.3)
            reasoning = data.get("reasoning", "")
            perspective = data.get("perspective", expert_type)

            # 判断对错
            correct = None
            if actual_culprit:
                correct = (actual_culprit in culprit or culprit in actual_culprit)

            # 提取洞察(从推理中提取关键短语)
            insights = self._extract_insights(reasoning, correct)
            missed = self._extract_missed_evidence(reasoning, correct, culprit, actual_culprit)

            mem = CaseMemory(
                case_id=case_id,
                expert_type=perspective,
                conclusion={"culprit": culprit, "confidence": confidence, "reasoning": reasoning},
                actual_culprit=actual_culprit,
                correct=correct,
                key_insights=insights,
                missed_evidence=missed,
                reasoning_patterns=self._classify_reasoning_pattern(reasoning),
                case_type=case_type,
                difficulty=difficulty,
            )

            memories.append(mem)

        # 裁判记忆
        if voting_result:
            winner = voting_result.get("winner", "未知")
            correct = (actual_culprit in winner or winner in actual_culprit) if actual_culprit else None

            memories.append(CaseMemory(
                case_id=case_id,
                expert_type="adjudicator",
                conclusion={
                    "winner": winner,
                    "confidence": voting_result.get("confidence", 0),
                    "consensus": voting_result.get("consensus_level", ""),
                },
                actual_culprit=actual_culprit,
                correct=correct,
                key_insights=self._extract_insights(
                    str(voting_result.get("vote_details", "")), correct
                ),
                case_type=case_type,
                difficulty=difficulty,
            ))

        self.memory.add_batch(memories)
        return memories

    def _learn_skills_with_llm(self, case_id: str, case_type: str,
                                case_data: Dict, expert_results: Dict,
                                voting_result: Dict, actual_culprit: str) -> List[DetectiveSkill]:
        """使用LLM从案件结果中提炼技能"""
        new_skills = []

        for expert_type, result in expert_results.items():
            data = result.get("data", result)
            culprit = data.get("culprit", "未知")
            reasoning = data.get("reasoning", "")
            perspective = data.get("perspective", expert_type)
            correct = (actual_culprit in culprit or culprit in actual_culprit) if actual_culprit else None

            if correct is None:
                continue

            prompt = f"""你是一个侦探技能分析专家。请从以下案件分析中提炼可复用的推理技能。

案件: {case_id} ({case_type})
专家类型: {perspective}
该专家结论: 凶手={culprit}, {'✅正确' if correct else '❌错误'}
实际凶手: {actual_culprit}
该专家推理过程: {reasoning[:800]}

请以JSON格式返回1-2个可复用的推理技巧:
[
  {{
    "name": "技能名称(简短)",
    "category": "reasoning_technique|evidence_analysis|pattern_recognition|anti_deception|crime_reconstruction|timeline_analysis|suspect_profiling|cross_validation",
    "knowledge": "具体知识内容(2-3句话，描述这个技巧如何使用)",
    "trigger_keywords": ["触发关键词列表"],
    "applicable_case_types": ["适用案件类型"]
  }}
]

要求:
- 只返回JSON数组
- 技能必须是可复用的(不是案件特定的)
- 如果正确, 提炼成功的推理技巧
- 如果错误, 提炼"应该避免的陷阱"或"应该注意的信号"
"""

            try:
                import urllib.request
                api_key = os.environ.get("ZHIPUAI_API_KEY", "")
                if not api_key:
                    continue

                req_body = json.dumps({
                    "model": "glm-4-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                }).encode()

                req = urllib.request.Request(
                    "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions",
                    data=req_body,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    }
                )

                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp_data = json.loads(resp.read())
                
                content = resp_data["choices"][0]["message"]["content"]

                # 解析JSON
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    skills_data = json.loads(json_match.group(0))
                    for sd in skills_data:
                        skill_id = f"skl_{perspective}_{hashlib.md5(sd['name'].encode()).hexdigest()[:8]}"
                        
                        skill = DetectiveSkill(
                            skill_id=skill_id,
                            name=sd["name"],
                            expert_type=perspective,
                            category=sd.get("category", "reasoning_technique"),
                            description=sd["knowledge"][:200],
                            knowledge=sd["knowledge"],
                            trigger_conditions={
                                "keywords": sd.get("trigger_keywords", []),
                                "case_type": sd.get("applicable_case_types", []),
                            },
                            source_cases=[case_id],
                            confidence=0.7 if correct else 0.4,
                        )
                        
                        # 检查是否已有相似技能
                        existing = self.skills.get(skill_id)
                        if existing:
                            # 合并: 追加来源案件
                            existing.source_cases = list(set(existing.source_cases + [case_id]))
                            existing.confidence = (existing.confidence + skill.confidence) / 2
                            existing.last_updated = time.time()
                        else:
                            self.skills.register(skill)
                            new_skills.append(skill)

            except Exception as e:
                self.logger.warning(f"LLM提炼技能失败 {perspective}: {e}")
                continue

        return new_skills

    def _learn_skills_by_rules(self, case_id: str, case_type: str,
                                expert_results: Dict, actual_culprit: str) -> List[DetectiveSkill]:
        """基于规则提炼技能(无LLM时的fallback)"""
        new_skills = []

        for expert_type, result in expert_results.items():
            data = result.get("data", result)
            culprit = data.get("culprit", "未知")
            perspective = data.get("perspective", expert_type)
            correct = (actual_culprit in culprit or culprit in actual_culprit) if actual_culprit else None

            if correct is None:
                continue

            # 基于对错创建简单技能
            if correct:
                name = f"{case_type}正确推理经验"
                knowledge = f"在{case_type}中，{perspective}专家正确识别了凶手{actual_culprit}。"
            else:
                name = f"{case_type}推理陷阱警示"
                knowledge = f"在{case_type}中，{perspective}专家错误地认为是{culprit}而非{actual_culprit}。需注意避免类似误判。"

            skill_id = f"skl_{perspective}_{hashlib.md5(f'{case_type}_{perspective}'.encode()).hexdigest()[:8]}"

            skill = DetectiveSkill(
                skill_id=skill_id,
                name=name,
                expert_type=perspective,
                category="reasoning_technique",
                description=knowledge[:200],
                knowledge=knowledge,
                trigger_conditions={"case_type": [case_type]} if case_type else {},
                source_cases=[case_id],
                confidence=0.7 if correct else 0.4,
            )

            existing = self.skills.get(skill_id)
            if existing:
                existing.source_cases = list(set(existing.source_cases + [case_id]))
                existing.last_updated = time.time()
            else:
                self.skills.register(skill)
                new_skills.append(skill)

        return new_skills

    def _learn_patterns(self, case_id: str, case_type: str,
                        expert_results: Dict, actual_culprit: str,
                        case_data: Dict) -> List[CrimePattern]:
        """从案件中提炼模式观察"""
        new_patterns = []

        # 检查是否大多数专家都错了(集体盲点)
        wrong_count = sum(
            1 for r in expert_results.values()
            if actual_culprit not in (r.get("data", r).get("culprit", ""))
        )
        total = len(expert_results)

        if total > 0 and wrong_count / total > 0.5:
            # 大多数专家错了 → 有价值的模式
            pattern = CrimePattern(
                pattern_id=f"pat_learned_{hashlib.md5(f'{case_id}_blindspot'.encode()).hexdigest()[:8]}",
                name=f"{case_type}的侦查盲点",
                pattern_type="crime",
                description=f"在{case_type}中，{wrong_count}/{total}个专家判断错误。实际凶手是{actual_culprit}。",
                characteristics=["专家共识可能被误导", "需要关注被忽视的嫌疑人"],
                red_flags=["多个专家一致指向同一嫌疑人时需警惕嫁祸"],
                common_mistakes=["跟随多数意见而非独立分析"],
                source_cases=[case_id],
                frequency=1,
                confirmed_count=1,
                case_types=[case_type],
                confidence=0.6,
            )
            self.patterns.add_pattern(pattern)
            new_patterns.append(pattern)

        return new_patterns

    # ============================================================
    # 辅助方法
    # ============================================================

    def _extract_insights(self, reasoning: str, correct: bool) -> List[str]:
        """从推理过程中提取关键洞察"""
        if not reasoning:
            return []

        insights = []
        # 简单的关键句提取
        sentences = reasoning.replace('。', '。\n').replace('；', '；\n').split('\n')
        keywords = ["关键", "重要", "突破", "矛盾", "异常", "发现", "表明", "证明",
                    "注意", "可疑", "忽略", "线索", "推断", "揭示"]

        for s in sentences:
            s = s.strip()
            if any(kw in s for kw in keywords) and len(s) > 10:
                insights.append(s[:100])

        return insights[:3]

    def _extract_missed_evidence(self, reasoning: str, correct: bool,
                                  culprit: str, actual_culprit: str) -> List[str]:
        """提取遗漏的证据(如果判断错误)"""
        if correct or not actual_culprit:
            return []

        return [f"将{actual_culprit}误认为{culprit}"]

    def _classify_reasoning_pattern(self, reasoning: str) -> List[str]:
        """分类推理模式"""
        patterns = []
        if not reasoning:
            return patterns

        if "排除" in reasoning or "不可能" in reasoning:
            patterns.append("排除法")
        if "时间" in reasoning and ("不在场" in reasoning or "时间线" in reasoning):
            patterns.append("时间线验证")
        if "矛盾" in reasoning:
            patterns.append("矛盾发现")
        if "动机" in reasoning:
            patterns.append("动机分析")
        if "物证" in reasoning or "证据" in reasoning:
            patterns.append("物证关联")
        if "心理" in reasoning or "性格" in reasoning:
            patterns.append("心理画像")

        return patterns or ["综合推理"]
