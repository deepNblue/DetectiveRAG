#!/usr/bin/env python3
"""
时序侦破推理引擎 (Stage Engine)
模拟真实侦破过程，将证据按时序分为多个阶段，逐步推理。

核心思路:
  1. plan_stages() — 根据案件自动规划侦破阶段，分配证据
  2. run_stage()   — 每阶段只使用当前已获得的证据进行推理
  3. run_investigation() — 运行完整时序侦破流程

每个阶段输出:
  - reasoning: 当前推理结论
  - suspect_ranking: 嫌疑人嫌疑度排名
  - investigation_advice: 下一阶段调查方向建议
  - hypotheses: 假设预测（如果查到X → 说明Y）
  - confidence: 当前整体置信度
  - key_turning_point: 本阶段是否有关键转折
"""

import json
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from loguru import logger
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ═══════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════

@dataclass
class InvestigationStage:
    """一个侦破阶段"""
    stage_id: int = 0                          # 1, 2, 3, 4...
    stage_name: str = ""                       # "报案与现场勘查" / "嫌疑人询问" 等
    stage_description: str = ""                # 本阶段的描述
    available_evidence: List[str] = field(default_factory=list)   # 本阶段可用的证据文本
    new_evidence: List[str] = field(default_factory=list)         # 本阶段新增的证据
    reasoning: str = ""                        # 本阶段推理结论
    suspect_ranking: List[Dict] = field(default_factory=list)     # 嫌疑人排名
    investigation_advice: List[Dict] = field(default_factory=list)  # 调查建议
    hypotheses: List[Dict] = field(default_factory=list)          # 假设预测
    confidence: float = 0.0                    # 当前整体置信度
    key_turning_point: Optional[str] = None    # 关键转折

    def to_dict(self) -> Dict:
        return asdict(self)

    def has_reasoning(self) -> bool:
        return bool(self.reasoning)


# ═══════════════════════════════════════════════════════════════
# 时序侦破推理引擎
# ═══════════════════════════════════════════════════════════════

class StageEngine:
    """时序侦破推理引擎"""

    def __init__(self, llm_client=None):
        """
        初始化引擎

        Args:
            llm_client: LLM客户端（LLMClient实例），需要支持 chat_with_system() 方法
        """
        self.llm = llm_client
        self.stages_completed: List[InvestigationStage] = []
        self.logger = logger.bind(module="StageEngine")

    # ──────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────

    def run_investigation(
        self,
        case_text: str,
        suspects: List[Dict],
        all_evidence: List[Any],
        timeline: List[Dict],
        evidence_stages: Optional[Dict] = None,
    ) -> List[InvestigationStage]:
        """
        运行完整时序侦破流程

        Args:
            case_text: 案件描述
            suspects: 嫌疑人列表 [{name, motive, opportunity, capability, alibi}]
            all_evidence: 所有证据（字符串列表 或 带 stage 属性的字典列表）
            timeline: 时间线 [{time, event}]
            evidence_stages: 可选的预设阶段划分 {stage_id: [evidence_indices]}

        Returns:
            完成的阶段列表
        """
        self.stages_completed = []

        # 解析证据格式
        evidence_texts, evidence_stage_map = self._parse_evidence(all_evidence, evidence_stages)

        # 步骤1: 规划阶段
        self.logger.info("📋 步骤1: 规划侦破阶段...")
        stages = self.plan_stages(
            case_text=case_text,
            suspects=suspects,
            all_evidence=evidence_texts,
            timeline=timeline,
            evidence_stage_map=evidence_stage_map,
        )
        self.logger.info(f"📋 规划完成: {len(stages)} 个阶段")

        # 步骤2: 逐阶段执行推理
        for i, stage in enumerate(stages):
            self.logger.info(f"🔍 步骤2.{i+1}: 执行阶段 {stage.stage_id} - {stage.stage_name}")
            print(f"\n{'='*60}", flush=True)
            print(f"🔍 阶段 {stage.stage_id}/{len(stages)}: {stage.stage_name}", flush=True)
            print(f"   可用证据: {len(stage.available_evidence)} 条, "
                  f"新增: {len(stage.new_evidence)} 条", flush=True)
            print(f"{'='*60}", flush=True)

            completed = self.run_stage(
                stage=stage,
                case_text=case_text,
                suspects=suspects,
                timeline=timeline,
                previous_stages=list(self.stages_completed),  # 传副本
            )

            self.stages_completed.append(completed)

            # 打印本阶段结论
            self._print_stage_summary(completed)

            # 每个LLM调用之间sleep 2秒
            if i < len(stages) - 1:
                print(f"\n⏳ 等待2秒后进入下一阶段...", flush=True)
                time.sleep(2)

        return self.stages_completed

    def plan_stages(
        self,
        case_text: str,
        suspects: List[Dict],
        all_evidence: List[str],
        timeline: List[Dict],
        evidence_stage_map: Optional[Dict] = None,
    ) -> List[InvestigationStage]:
        """
        规划侦破阶段 — 根据案件自动决定分几个阶段，每阶段包含哪些证据

        Args:
            case_text: 案件描述
            suspects: 嫌疑人列表
            all_evidence: 所有证据文本
            timeline: 时间线
            evidence_stage_map: 预设的阶段映射 {stage_id: [evidence_indices]}

        Returns:
            阶段列表（只有evidence分配，还没有reasoning）
        """
        # 如果有预设阶段划分，直接使用
        if evidence_stage_map:
            return self._build_preset_stages(all_evidence, evidence_stage_map)

        # 否则调用LLM来规划
        suspect_names = [s.get("name", f"嫌疑人{i+1}") for i, s in enumerate(suspects)]
        evidence_list_str = ""
        for i, ev in enumerate(all_evidence):
            evidence_list_str += f"  E{i+1:03d}: {ev}\n"

        timeline_str = ""
        for t in timeline:
            timeline_str += f"  {t.get('time', '?')}: {t.get('event', '?')}\n"

        prompt = f"""你是资深刑侦专家。请将以下案件的证据按时序分为3-5个侦破阶段。

## 案件概况
{case_text[:800]}

## 嫌疑人
{', '.join(suspect_names)}

## 所有证据（带编号）
{evidence_list_str}

## 时间线
{timeline_str}

## 规则
- 阶段1: 报案 + 现场勘查（最基础的证据，如现场状况、尸体检验、基本物证）
- 阶段2: 初步调查（证人询问、外围调查、基本信息收集）
- 阶段3: 深入调查（技术鉴定、专项排查、关键证据发现）
- 阶段4+: 根据案件复杂度可能需要更多阶段（如综合分析、终局推理）
- 最后一个阶段应包含所有证据（完整推理）
- 每个阶段的证据应该是上一阶段的超集（后一阶段包含前面所有阶段的证据）

请输出JSON数组（不要markdown代码块）:
[
  {{
    "stage_id": 1,
    "stage_name": "阶段名称",
    "evidence_ids": ["E001", "E003"],
    "stage_description": "本阶段可获取的证据说明"
  }}
]"""

        self.logger.info("  调用LLM规划阶段...")
        response = self._call_llm(prompt)
        stage_plans = self._extract_json_list(response)

        if not stage_plans:
            # LLM规划失败，使用默认3阶段
            self.logger.warning("  LLM阶段规划失败，使用默认3阶段划分")
            return self._build_default_stages(all_evidence)

        # 构建 InvestigationStage 列表
        stages = []
        for sp in stage_plans:
            sid = sp.get("stage_id", len(stages) + 1)
            ev_ids = sp.get("evidence_ids", [])
            # 解析 E001 -> index 0
            evidence_indices = []
            for eid in ev_ids:
                m = re.match(r'E(\d+)', eid)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(all_evidence):
                        evidence_indices.append(idx)

            available = [all_evidence[i] for i in evidence_indices if i < len(all_evidence)]

            stage = InvestigationStage(
                stage_id=sid,
                stage_name=sp.get("stage_name", f"阶段{sid}"),
                stage_description=sp.get("stage_description", ""),
                available_evidence=available,
                new_evidence=list(available),  # 初始时全部视为新增
            )
            stages.append(stage)

        # 确保阶段是递增超集
        stages = self._ensure_cumulative(stages, all_evidence)

        return stages

    def run_stage(
        self,
        stage: InvestigationStage,
        case_text: str,
        suspects: List[Dict],
        timeline: List[Dict],
        previous_stages: List[InvestigationStage],
    ) -> InvestigationStage:
        """
        执行一个阶段的推理

        Args:
            stage: 当前阶段（有evidence但没有reasoning）
            case_text: 案件描述
            suspects: 嫌疑人列表
            timeline: 时间线
            previous_stages: 之前已完成的阶段列表

        Returns:
            完成推理的阶段
        """
        suspect_names = [s.get("name", f"嫌疑人{i+1}") for i, s in enumerate(suspects)]

        # 构建上一阶段结论
        prev_reasoning = ""
        if previous_stages:
            for ps in previous_stages:
                prev_reasoning += f"### 阶段{ps.stage_id}: {ps.stage_name}\n"
                prev_reasoning += f"置信度: {ps.confidence:.0%}\n"
                prev_reasoning += f"推理结论: {ps.reasoning[:300]}\n"
                if ps.suspect_ranking:
                    prev_reasoning += "嫌疑人排名: "
                    prev_reasoning += " > ".join(
                        f"{sr.get('name','?')}({sr.get('suspicion_score',0):.0%})"
                        for sr in ps.suspect_ranking[:3]
                    )
                    prev_reasoning += "\n"
                prev_reasoning += "\n"
        else:
            prev_reasoning = "（本阶段是初始阶段，无之前结论）"

        # 区分新增证据和已有证据
        prev_evidence_set = set()
        for ps in previous_stages:
            prev_evidence_set.update(ps.available_evidence)

        current_evidence_str = ""
        new_evidence = []
        for ev in stage.available_evidence:
            if ev in prev_evidence_set:
                current_evidence_str += f"  [已有] {ev}\n"
            else:
                current_evidence_str += f"  🆕 {ev}\n"
                new_evidence.append(ev)

        stage.new_evidence = new_evidence

        # 时间线
        timeline_str = ""
        for t in timeline:
            timeline_str += f"  {t.get('time', '?')}: {t.get('event', '?')}\n"

        # 嫌疑人信息
        suspects_str = ""
        for s in suspects:
            suspects_str += f"- **{s.get('name','?')}**: 动机={s.get('motive','?')}, "
            suspects_str += f"机会={s.get('opportunity','?')}, 能力={s.get('capability','?')}, "
            suspects_str += f"不在场证明={s.get('alibi','?')}\n"

        prompt = f"""你是资深刑侦专家。请基于当前已获得的证据进行阶段性推理。

## 案件概况
{case_text[:800]}

## 嫌疑人
{suspects_str}

## 上一阶段结论
{prev_reasoning}

## 当前阶段可用证据（新增证据用🆕标注）
阶段名称: {stage.stage_name}
{current_evidence_str}

## 时间线
{timeline_str}

请严格按照以下JSON格式输出（不要markdown代码块），reasoning放最后:
{{
  "suspect_ranking": [
    {{"name": "嫌疑人名", "suspicion_score": 0.8, "reason": "嫌疑升高/降低的理由"}}
  ],
  "investigation_advice": [
    {{
      "direction": "调查方向",
      "reason": "为什么要查这个",
      "priority": "高/中/低",
      "expected_finding": "预期能发现什么"
    }}
  ],
  "hypotheses": [
    {{
      "condition": "如果查到XXX",
      "then": "说明YYY",
      "impact": "嫌疑人XX的嫌疑度将升高/降低"
    }}
  ],
  "confidence": 0.6,
  "key_turning_point": "本阶段是否有关键发现改变了推理方向（null或具体描述）",
  "reasoning": "200-400字精简推理过程，分析当前证据如何指向或排除各嫌疑人"
}}

注意（非常重要，必须严格遵守）:
- JSON字段顺序: suspect_ranking → investigation_advice → hypotheses → confidence → key_turning_point → reasoning
  即reasoning必须放最后！因为推理文字最长，如果被截断不影响前面的结构化字段
- reasoning精简到200-400字，分析当前证据如何指向或排除各嫌疑人即可
- suspect_ranking中所有嫌疑人都必须列出，suspicion_score范围0-1
- investigation_advice至少给出2条建议
- hypotheses至少给出2个假设预测
- confidence反映当前阶段对结论的确信程度
- key_turning_point如果不是转折点则填null"""

        self.logger.info(f"  调用LLM执行阶段 {stage.stage_id} 推理...")
        response = self._call_llm(prompt)
        result = self._extract_json_dict(response)

        if result:
            stage.reasoning = result.get("reasoning", "推理失败")
            stage.suspect_ranking = result.get("suspect_ranking", [])
            stage.investigation_advice = result.get("investigation_advice", [])
            stage.hypotheses = result.get("hypotheses", [])
            stage.confidence = float(result.get("confidence", 0.5))
            ktp = result.get("key_turning_point")
            stage.key_turning_point = ktp if ktp and ktp != "null" and ktp != "None" else None
        else:
            # LLM返回解析失败
            stage.reasoning = response[:500] if response else "推理失败"
            stage.confidence = 0.3
            self.logger.warning(f"  阶段 {stage.stage_id} LLM响应解析失败，使用原始文本")

        return stage

    # ──────────────────────────────────────
    # 内部辅助方法
    # ──────────────────────────────────────

    def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """调用LLM，支持多种客户端格式"""
        if self.llm is None:
            return '{"error": "无LLM客户端"}'

        try:
            # 尝试 chat_with_system 方法（LLMClient）
            if hasattr(self.llm, 'chat_with_system'):
                return self.llm.chat_with_system(
                    system_prompt="你是资深刑侦分析专家，擅长逻辑推理和犯罪分析。请用中文回答。",
                    user_prompt=prompt,
                    temperature=temperature,
                )
            # 尝试 chat_completion 方法
            elif hasattr(self.llm, 'chat_completion'):
                messages = [
                    {"role": "system", "content": "你是资深刑侦分析专家，擅长逻辑推理和犯罪分析。请用中文回答。"},
                    {"role": "user", "content": prompt},
                ]
                return self.llm.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            # 尝试 simple_chat 方法
            elif hasattr(self.llm, 'simple_chat'):
                return self.llm.simple_chat(prompt, temperature=temperature)
            else:
                return '{"error": "LLM客户端不支持已知的调用方法"}'
        except Exception as e:
            self.logger.error(f"LLM调用异常: {e}")
            return f'{{"error": "LLM调用异常: {str(e)[:100]}"}}'

    def _extract_json_list(self, text: str) -> List[Dict]:
        """从LLM响应中提取JSON数组"""
        if not text:
            return []

        # 去掉 [系统提示] 开头的内容
        if text.startswith("[系统提示]"):
            return []

        # 尝试直接解析
        try:
            result = json.loads(text.strip())
            if isinstance(result, list):
                return result
        except:
            pass

        # 尝试提取 ```json ... ``` 块
        m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(1))
                if isinstance(result, list):
                    return result
            except:
                pass

        # 尝试提取 [ ... ]
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    return result
            except:
                pass

        self.logger.warning(f"无法从LLM响应中提取JSON数组: {text[:200]}")
        return []

    def _extract_json_dict(self, text: str) -> Optional[Dict]:
        """从LLM响应中提取JSON对象，支持截断JSON修复"""
        if not text:
            return None

        if text.startswith("[系统提示]"):
            return None

        # 尝试1: 直接解析完整JSON
        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                return result
        except:
            pass

        # 尝试2: 提取 ```json ... ``` 块
        m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except:
                pass

        # 尝试3: 提取 { ... } (完整匹配)
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                # 尝试修复截断的JSON
                result = self._repair_truncated_json(m.group(0))
                if result:
                    return result

        # 尝试4: 提取从 { 开始到文本末尾（截断的JSON）
        m = re.search(r'\{', text)
        if m:
            partial = text[m.start():]
            result = self._repair_truncated_json(partial)
            if result:
                return result

        self.logger.warning(f"无法从LLM响应中提取JSON字典: {text[:200]}")
        return None

    def _repair_truncated_json(self, partial: str) -> Optional[Dict]:
        """修复截断的JSON — 尝试补全缺失的引号、括号等"""
        if not partial or not partial.strip().startswith('{'):
            return None

        s = partial.strip()

        # 策略: 从右往左找到最后一个完整的 key:value 对，然后补全括号
        # 尝试多种补全方式
        attempts = [
            s + '"}',           # 截断在字符串值中间
            s + '"}',           # 同上
            s + '}',            # 截断在非字符串位置
            s + ']}}',          # 截断在数组元素中间
            s + '"}]}}',        # 截断在数组中对象字符串值中间
        ]

        # 更智能的修复: 找最后一个完整的逗号分隔处
        last_comma = s.rfind('",')
        if last_comma > 0:
            # 截取到最后一个完整键值对的逗号处，去掉尾部不完整的值
            truncated_at = s.rfind('",')
            for cut_point in [truncated_at + 1, s.rfind('",') + 2]:
                if cut_point > 0:
                    cut = s[:cut_point]
                    # 计算需要补多少个 ]
                    open_brackets = cut.count('[') - cut.count(']')
                    open_braces = cut.count('{') - cut.count('}')
                    completion = ']' * max(0, open_brackets) + '}' * max(0, open_braces)
                    attempts.append(cut + completion)

        for attempt in attempts:
            try:
                result = json.loads(attempt)
                if isinstance(result, dict):
                    return result
            except:
                continue

        return None

    def _parse_evidence(
        self,
        all_evidence: List[Any],
        evidence_stages: Optional[Dict] = None,
    ) -> tuple:
        """
        解析证据格式，支持字符串列表和带stage属性的字典列表

        Returns:
            (evidence_texts: List[str], evidence_stage_map: Dict[int, List[int]])
        """
        evidence_texts = []
        evidence_stage_map = {}  # {stage_id: [index]}

        if evidence_stages:
            # 使用外部传入的阶段映射
            evidence_texts = [str(ev) for ev in all_evidence]
            evidence_stage_map = evidence_stages
        else:
            for i, ev in enumerate(all_evidence):
                if isinstance(ev, dict):
                    text = ev.get("text", str(ev))
                    evidence_texts.append(text)
                    stage = ev.get("stage", None)
                    if stage is not None:
                        stage = int(stage)
                        evidence_stage_map.setdefault(stage, []).append(i)
                else:
                    evidence_texts.append(str(ev))

        return evidence_texts, evidence_stage_map if evidence_stage_map else None

    def _build_preset_stages(
        self,
        all_evidence: List[str],
        evidence_stage_map: Dict[int, List[int]],
    ) -> List[InvestigationStage]:
        """根据预设的阶段映射构建阶段列表"""
        stage_names = {
            1: "报案与现场勘查",
            2: "初步调查与询问",
            3: "深入调查与鉴定",
            4: "综合分析与推理",
            5: "终局推理",
        }

        stages = []
        sorted_stage_ids = sorted(evidence_stage_map.keys())
        cumulative_evidence = []

        for sid in sorted_stage_ids:
            indices = evidence_stage_map[sid]
            stage_evidence = [all_evidence[i] for i in indices if 0 <= i < len(all_evidence)]

            # 累积：当前阶段 = 之前所有 + 本阶段新证据
            new_evidence = [ev for ev in stage_evidence if ev not in cumulative_evidence]
            cumulative_evidence.extend(new_evidence)

            stage = InvestigationStage(
                stage_id=sid,
                stage_name=stage_names.get(sid, f"阶段{sid}"),
                stage_description=f"本阶段包含 {len(new_evidence)} 条新增证据，共 {len(cumulative_evidence)} 条可用证据",
                available_evidence=list(cumulative_evidence),
                new_evidence=list(new_evidence),
            )
            stages.append(stage)

        return stages

    def _build_default_stages(self, all_evidence: List[str]) -> List[InvestigationStage]:
        """默认3阶段划分（当LLM规划失败时的后备方案）"""
        n = len(all_evidence)
        if n == 0:
            return [InvestigationStage(
                stage_id=1, stage_name="无证据",
                available_evidence=[], new_evidence=[],
            )]

        # 按 30% / 60% / 100% 分
        s1_end = max(1, n * 3 // 10)
        s2_end = max(s1_end + 1, n * 6 // 10)

        stages = [
            InvestigationStage(
                stage_id=1,
                stage_name="报案与现场勘查",
                stage_description="基础物证和现场状况",
                available_evidence=all_evidence[:s1_end],
                new_evidence=all_evidence[:s1_end],
            ),
            InvestigationStage(
                stage_id=2,
                stage_name="初步调查与询问",
                stage_description=f"新增证人证词和外调信息",
                available_evidence=all_evidence[:s2_end],
                new_evidence=all_evidence[s1_end:s2_end],
            ),
            InvestigationStage(
                stage_id=3,
                stage_name="综合分析与推理",
                stage_description="全部证据到手，最终推理",
                available_evidence=list(all_evidence),
                new_evidence=all_evidence[s2_end:],
            ),
        ]
        return stages

    def _ensure_cumulative(
        self,
        stages: List[InvestigationStage],
        all_evidence: List[str],
    ) -> List[InvestigationStage]:
        """确保每个阶段的证据是上一阶段的超集"""
        if not stages:
            return stages

        cumulative = set()
        for stage in stages:
            # 确保之前阶段的证据也在当前阶段
            stage_set = set(stage.available_evidence)
            stage_set.update(cumulative)
            stage.available_evidence = list(stage_set)
            stage.new_evidence = [ev for ev in stage.available_evidence if ev not in cumulative]
            cumulative.update(stage.available_evidence)

        # 最后一个阶段确保包含所有证据
        if stages:
            last = stages[-1]
            missing = [ev for ev in all_evidence if ev not in set(last.available_evidence)]
            if missing:
                last.available_evidence.extend(missing)
                last.new_evidence.extend(missing)

        return stages

    def _print_stage_summary(self, stage: InvestigationStage):
        """打印阶段摘要"""
        print(f"\n📊 阶段 {stage.stage_id} 结论:", flush=True)
        print(f"   阶段名: {stage.stage_name}", flush=True)
        print(f"   置信度: {stage.confidence:.0%}", flush=True)
        print(f"   推理: {stage.reasoning[:200]}...", flush=True)

        if stage.suspect_ranking:
            print(f"\n   📋 嫌疑人排名:", flush=True)
            for sr in stage.suspect_ranking[:5]:
                print(f"      {sr.get('name','?')}: {sr.get('suspicion_score',0):.0%} — "
                      f"{sr.get('reason','')[:60]}", flush=True)

        if stage.investigation_advice:
            print(f"\n   🎯 调查建议:", flush=True)
            for adv in stage.investigation_advice[:3]:
                print(f"      [{adv.get('priority','?')}] {adv.get('direction','?')} — "
                      f"{adv.get('reason','')[:60]}", flush=True)

        if stage.hypotheses:
            print(f"\n   🔮 假设预测:", flush=True)
            for h in stage.hypotheses[:3]:
                print(f"      {h.get('condition','?')} → {h.get('then','?')[:60]}", flush=True)

        if stage.key_turning_point:
            print(f"\n   ⚡ 关键转折: {stage.key_turning_point[:100]}", flush=True)


# ═══════════════════════════════════════════════════════════════
# 独立运行入口
# ═══════════════════════════════════════════════════════════════

def create_stage_llm_client():
    """创建专用于StageEngine的LLM客户端"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    try:
        from api.llm_client import LLMClient
        client = LLMClient()
        print(f"✅ 使用项目LLMClient, model={client.model}", flush=True)
        return client
    except Exception as e:
        print(f"⚠️ 项目LLMClient加载失败: {e}", flush=True)

    # 后备：直接用requests调用
    print("📦 使用直接HTTP调用...", flush=True)
    return DirectLLMClient()


class DirectLLMClient:
    """直接HTTP调用智谱API的后备客户端"""

    def __init__(self):
        self.api_key = os.environ.get("ZHIPUAI_API_KEY", "")
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.model = "glm-4-flash"

    def chat_with_system(self, system_prompt: str, user_prompt: str,
                         temperature: float = 0.7, max_tokens: int = 4096) -> str:
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(self.base_url, headers=headers, json=data, timeout=120)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API错误: {resp.status_code} {resp.text[:200]}")

    def simple_chat(self, prompt: str, temperature: float = 0.7) -> str:
        return self.chat_with_system("你是助手", prompt, temperature)


def test_with_case_001():
    """用 CASE-001 测试 stage_engine 独立运行"""
    print("=" * 70, flush=True)
    print("🧪 时序侦破推理引擎 — CASE-001 测试", flush=True)
    print("=" * 70, flush=True)

    from tests.test_detective_cases import get_all_test_cases

    cases = get_all_test_cases()
    case = cases[0]  # CASE-001

    print(f"\n📂 案件: {case.case_id} — {case.case_type}", flush=True)
    print(f"   嫌疑人: {len(case.suspects)} 人", flush=True)
    print(f"   证据: {len(case.evidence)} 条", flush=True)
    print(f"   时间线: {len(case.timeline)} 个事件", flush=True)
    print(f"   真凶: {case.expected_result['culprit']}", flush=True)

    # 创建引擎
    llm = create_stage_llm_client()
    engine = StageEngine(llm_client=llm)

    # 运行时序侦破
    stages = engine.run_investigation(
        case_text=case.case_text,
        suspects=case.suspects,
        all_evidence=case.evidence,
        timeline=case.timeline,
    )

    # 汇总报告
    print("\n" + "=" * 70, flush=True)
    print("📊 时序侦破汇总报告", flush=True)
    print("=" * 70, flush=True)

    for stage in stages:
        print(f"\n{'─'*50}", flush=True)
        print(f"阶段 {stage.stage_id}: {stage.stage_name}", flush=True)
        print(f"  证据: {len(stage.available_evidence)} 条可用, "
              f"{len(stage.new_evidence)} 条新增", flush=True)
        print(f"  置信度: {stage.confidence:.0%}", flush=True)

        if stage.suspect_ranking:
            top = stage.suspect_ranking[0] if stage.suspect_ranking else {}
            print(f"  最大嫌疑人: {top.get('name', '?')} "
                  f"({top.get('suspicion_score', 0):.0%})", flush=True)

        if stage.key_turning_point:
            print(f"  ⚡ 转折: {stage.key_turning_point[:80]}", flush=True)

    # 最终结论
    if stages:
        last = stages[-1]
        print(f"\n{'='*50}", flush=True)
        print(f"🏁 最终结论:", flush=True)
        if last.suspect_ranking:
            for sr in last.suspect_ranking[:3]:
                print(f"   {sr.get('name','?')}: {sr.get('suspicion_score',0):.0%}", flush=True)
        print(f"   置信度: {last.confidence:.0%}", flush=True)
        print(f"   真凶(参考): {case.expected_result['culprit']}", flush=True)

    return stages


if __name__ == "__main__":
    test_with_case_001()
