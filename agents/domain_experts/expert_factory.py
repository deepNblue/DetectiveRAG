"""
动态领域专家工厂 (Domain Expert Factory)
根据案件特征动态创建业务专家Agent

核心能力:
  1. 案件分析 → 识别涉及的领域
  2. 匹配专家模板 → 选择合适的专家
  3. 动态创建Agent → 注入专业知识
  4. 参与并行推理 → 贡献领域视角

这个工厂不使用硬编码的Agent类，而是根据模板动态构建Agent，
让prompt、知识库、分析角度都随案件类型灵活变化。
"""

import json
import time
from typing import Dict, List, Any, Optional
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from agents.domain_experts.expert_registry import ExpertRegistry, ExpertTemplate, get_default_registry
from agents.domain_experts.domain_knowledge_base import DomainKnowledgeBase
from agents.base_agent import BaseAgent


class DynamicDomainExpert(BaseAgent):
    """
    动态领域专家 — 由工厂根据模板创建
    
    不继承MemoryEnhancedMixin(因为专家是临时的),
    但会从知识库中获取专业知识注入prompt
    """
    
    def __init__(self, template: ExpertTemplate, knowledge_base: DomainKnowledgeBase,
                 config: Dict[str, Any] = None, llm_client=None):
        self.template = template
        self.knowledge_base = knowledge_base
        
        # 动态设置Agent名称
        agent_name = f"Domain-{template.name_en.replace(' ', '')}"
        
        super().__init__(name=agent_name, config=config, llm_client=llm_client)
        
        self.logger.info(f"动态领域专家初始化: {template.name} ({template.title})")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """领域专家分析案件"""
        self.log_processing(input_data)
        
        sk = input_data.get("structured_knowledge", {})
        search = input_data.get("search_results", {})
        suspects = input_data.get("suspects", [])
        case_text = input_data.get("case_text", "")
        
        suspect_names = [s.get("name", "?") if isinstance(s, dict) else s for s in suspects]
        
        timeline = sk.get("timeline", {}).get("data", {})
        person_rels = sk.get("person_relation", {}).get("data", {})
        evidence = sk.get("evidence", {}).get("data", {})
        
        motive_data = search.get("motive", {}).get("data", {}).get("motive_analysis", [])
        temporal_data = search.get("temporal", {}).get("data", {}).get("temporal_contradictions", [])
        
        # 从知识库检索相关知识
        knowledge_text = self.knowledge_base.get_knowledge_prompt(
            domain=self.template.knowledge_domain,
            case_context=case_text[:500],
            max_entries=5,
        )
        
        # 构建专家分析prompt
        prompt = self._build_prompt(
            suspect_names=suspect_names,
            timeline=timeline,
            person_rels=person_rels,
            evidence=evidence,
            motive_data=motive_data,
            temporal_data=temporal_data,
            knowledge_text=knowledge_text,
            case_text=case_text[:800],
        )
        
        response = self.call_llm(prompt, temperature=0.4)
        parsed = self.extract_json_from_response(response)
        
        if parsed and isinstance(parsed, dict):
            culprit = parsed.get("culprit", "未知")
            confidence = parsed.get("confidence", 0.3)
            reasoning = parsed.get("reasoning", "")
        else:
            culprit, confidence, reasoning = "未知", 0.2, f"{self.template.name}分析失败"
        
        self.logger.info(f"{self.template.name}分析完成: 真凶={culprit}, 置信度={confidence}")
        
        return self.format_output({
            "perspective": f"domain_{self.template.expert_id}",
            "culprit": culprit,
            "confidence": confidence,
            "reasoning": reasoning,
            "detail": parsed or {},
            "expert_meta": {
                "name": self.template.name,
                "title": self.template.title,
                "domain": self.template.knowledge_domain,
                "voting_weight": self.template.voting_weight,
            },
        })
    
    def _build_prompt(self, suspect_names, timeline, person_rels, evidence,
                      motive_data, temporal_data, knowledge_text, case_text) -> str:
        """构建领域专家的分析prompt"""
        
        # 根据模板的prompt_template选择不同的分析重点
        focus_map = {
            "financial_analysis": "金融资金流向、保险利益、经济动机",
            "accounting_analysis": "账目异常、财务造假、资金挪用",
            "medical_analysis": "死因、药物反应、症状分析、医疗痕迹",
            "toxicology_analysis": "毒物种类、中毒途径、致死浓度、代谢特征",
            "legal_analysis": "罪名定性、证据标准、法律适用",
            "insurance_analysis": "保险欺诈、受益人利益、投保异常",
            "real_estate_analysis": "房产价值、产权纠纷、继承份额",
            "social_analysis": "家庭关系、暴力模式、社会支持网络",
            "addiction_analysis": "成瘾行为、戒断反应、行为控制力",
            "cybersecurity_analysis": "网络攻击、电子证据、数字痕迹",
            "environmental_analysis": "环境污染、化学物质、致害因果",
        }
        
        analysis_focus = focus_map.get(self.template.prompt_template, self.template.analysis_focus)
        
        return f"""你是一位{self.template.name}（{self.template.title}），请从{analysis_focus}的角度分析此案。

你的专业背景: {self.template.description}
你的分析专长: {self.template.analysis_focus}
{knowledge_text}

嫌疑人: {', '.join(suspect_names)}

案件内容:
{case_text}

人物关系:
{json.dumps(person_rels.get('relations', []), ensure_ascii=False, indent=2)[:600]}

时间线:
{json.dumps(timeline.get('events', []), ensure_ascii=False, indent=2)[:600]}

时间矛盾:
{json.dumps(temporal_data, ensure_ascii=False, indent=2)[:400]}

物证关联:
{json.dumps(evidence.get('connections', []), ensure_ascii=False, indent=2)[:500]}

证言:
{json.dumps(evidence.get('testimony', []), ensure_ascii=False, indent=2)[:500]}

动机分析:
{json.dumps(motive_data, ensure_ascii=False, indent=2)[:400]}

请以JSON格式返回你的专业分析:
{{
    "culprit": "你认为的真凶",
    "confidence": 0.0-1.0,
    "domain_findings": {{
        "key_discovery": "你从专业角度发现的关键线索",
        "professional_opinion": "你的专业判断(如: 资金流向异常/死因与供述不符/证据链存在法律缺陷)",
        "evidence_gaps": ["当前证据在专业角度的不足"],
        "recommended_investigation": ["建议进一步调查的方向"]
    }},
    "suspect_domain_analysis": [
        {{
            "suspect": "嫌疑人",
            "domain_relevance": "该嫌疑人在你专业领域的可疑之处",
            "risk_level": "high/medium/low"
        }}
    ],
    "reasoning": "你的专业推理过程",
    "certainty_factors": [
        {{
            "factor": "判断依据",
            "supports": "支持的方向(某嫌疑人)",
            "strength": 0.0-1.0
        }}
    ]
}}

要求:
1. 严格从{self.template.name}的专业角度分析
2. 运用你的领域知识发现其他专家可能忽略的线索
3. 如果你的专业知识不足以确定凶手,可以降低confidence
4. 只返回JSON

⚠️ 姓名规范: culprit只写一个人名, 去掉头衔, 同一人不同称呼视为同一人"""


class DomainExpertFactory:
    """
    动态领域专家工厂
    
    使用流程:
        factory = DomainExpertFactory(llm_client=llm_client)
        
        # 1. 分析案件,识别需要的领域专家
        matched = factory.analyze_case(case_text, case_type)
        
        # 2. 创建动态专家Agent
        experts = factory.create_experts(matched)
        
        # 3. 参与并行推理
        for expert in experts:
            result = expert.process(expert_input)
    """
    
    def __init__(self, llm_client=None, registry: ExpertRegistry = None):
        self.llm_client = llm_client
        self.registry = registry or get_default_registry()
        self.knowledge_base = DomainKnowledgeBase()
        self.logger = logger.bind(component="DomainExpertFactory")
    
    def analyze_case(self, case_text: str, case_type: str = "",
                     structured_knowledge: Dict = None) -> List[Dict[str, Any]]:
        """
        分析案件，识别需要哪些领域专家
        
        Returns:
            [{"template": ExpertTemplate, "relevance_score": float, "reason": str}, ...]
        """
        self.logger.info(f"🔍 分析案件领域需求 (文本长度={len(case_text)}, 类型={case_type})")
        
        # 方法1: 关键词匹配
        keyword_matches = self.registry.match_by_keywords(case_text, case_type)
        
        # 方法2: LLM判断（如果关键词匹配不够，用LLM做补充判断）
        llm_matches = self._llm_analyze_domains(case_text, case_type) if self.llm_client else []
        
        # 合并去重
        matched = {}
        for template in keyword_matches:
            matched[template.expert_id] = {
                "template": template,
                "relevance_score": 0.8,
                "reason": "关键词匹配",
            }
        
        for m in llm_matches:
            eid = m["expert_id"]
            if eid in matched:
                # LLM也认为需要，提高分数
                matched[eid]["relevance_score"] = min(1.0, matched[eid]["relevance_score"] + 0.2)
                matched[eid]["reason"] += " + LLM确认"
            else:
                template = self.registry.get(eid)
                if template:
                    matched[eid] = {
                        "template": template,
                        "relevance_score": m.get("score", 0.6),
                        "reason": "LLM判断",
                    }
        
        # 按相关性排序
        result = sorted(matched.values(), key=lambda x: -x["relevance_score"])
        
        self.logger.info(f"🔍 领域分析完成: 匹配到 {len(result)} 个领域专家")
        for r in result:
            t = r["template"]
            self.logger.info(f"   - {t.name}({t.title}): 相关性={r['relevance_score']:.1f} ({r['reason']})")
        
        return result
    
    def create_experts(self, matched_list: List[Dict[str, Any]]) -> List[DynamicDomainExpert]:
        """
        根据匹配结果创建动态专家Agent
        """
        experts = []
        for m in matched_list:
            template = m["template"]
            try:
                expert = DynamicDomainExpert(
                    template=template,
                    knowledge_base=self.knowledge_base,
                    llm_client=self.llm_client,
                )
                experts.append(expert)
                self.logger.info(f"✅ 创建领域专家: {template.name}")
            except Exception as e:
                self.logger.warning(f"❌ 创建领域专家失败 {template.name}: {e}")
        
        return experts
    
    def _llm_analyze_domains(self, case_text: str, case_type: str) -> List[Dict]:
        """使用LLM判断案件需要哪些领域专家"""
        if not self.llm_client:
            return []
        
        all_experts = self.registry.get_all()
        expert_list = "\n".join([
            f"- {eid}: {t.name}({t.title}) — {t.description[:60]}"
            for eid, t in all_experts.items()
        ])
        
        prompt = f"""分析以下案件需要哪些专业领域专家参与调查。

可用专家:
{expert_list}

案件类型: {case_type}
案件内容(摘要): {case_text[:600]}

请返回JSON数组，列出需要参与此案件调查的领域专家:
[
    {{
        "expert_id": "专家ID",
        "score": 0.0-1.0,
        "reason": "为什么需要这位专家"
    }}
]

只返回与案件有实际关联的专家,不要返回无关专家。只返回JSON数组。"""
        
        try:
            response = self.llm_client.simple_chat(prompt, temperature=0.3)
            # 提取JSON
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            self.logger.warning(f"LLM领域分析失败: {e}")
        
        return []
    
    def get_voting_weight(self, expert_id: str) -> float:
        """获取动态专家的投票权重"""
        template = self.registry.get(expert_id)
        return template.voting_weight if template else 0.7
    
    def get_stats(self) -> Dict[str, Any]:
        """获取工厂状态"""
        kb_stats = self.knowledge_base.get_stats()
        registry_count = len(self.registry.get_all())
        
        return {
            "registered_experts": registry_count,
            "knowledge_base": kb_stats,
        }
