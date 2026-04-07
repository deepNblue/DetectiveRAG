"""
领域专家注册表 (Expert Registry)
预定义各行业领域的专家模板和对应知识库

每个专家模板定义:
  - 专家身份(name, title, description)
  - 分析角度(analysis_focus)
  - 关键词触发条件(keywords)
  - 适用案件类型(applicable_case_types)
  - 对应知识库路径(knowledge_domain)
  - 投票权重(voting_weight)
"""

import json
import os
from typing import Dict, List, Any, Optional
from loguru import logger
from dataclasses import dataclass, field


@dataclass
class ExpertTemplate:
    """领域专家模板"""
    expert_id: str              # 唯一ID, 如 "financial_analyst"
    name: str                   # 专家名称, 如 "金融分析师"
    name_en: str                # 英文名
    title: str                  # 头衔, 如 "注册金融分析师(CFA)"
    description: str            # 专家简介
    analysis_focus: str         # 分析角度描述
    trigger_keywords: List[str]  # 触发关键词
    applicable_case_types: List[str]  # 适用案件类型
    knowledge_domain: str       # 知识库领域
    voting_weight: float        # 投票权重
    prompt_template: str        # 分析prompt模板
    sample_questions: List[str] = field(default_factory=list)  # 该专家会问的问题


class ExpertRegistry:
    """领域专家注册表"""
    
    def __init__(self):
        self._templates: Dict[str, ExpertTemplate] = {}
        self._load_defaults()
    
    def register(self, template: ExpertTemplate):
        """注册一个专家模板"""
        self._templates[template.expert_id] = template
        logger.debug(f"注册领域专家: {template.expert_id} ({template.name})")
    
    def get(self, expert_id: str) -> Optional[ExpertTemplate]:
        """获取专家模板"""
        return self._templates.get(expert_id)
    
    def get_all(self) -> Dict[str, ExpertTemplate]:
        """获取所有已注册模板"""
        return self._templates
    
    def match_by_keywords(self, case_text: str, case_type: str = "") -> List[ExpertTemplate]:
        """根据案件文本和类型匹配相关专家"""
        matched = []
        text_lower = case_text.lower()
        
        for tid, template in self._templates.items():
            # 检查案件类型
            type_match = any(ct in case_type for ct in template.applicable_case_types)
            
            # 检查关键词命中
            keyword_hits = sum(1 for kw in template.trigger_keywords if kw in text_lower)
            
            # 至少命中2个关键词 或 案件类型匹配
            if keyword_hits >= 2 or (type_match and keyword_hits >= 1):
                matched.append(template)
        
        return matched
    
    def _load_defaults(self):
        """加载默认专家模板"""
        defaults = self._get_default_templates()
        for t in defaults:
            self.register(t)
    
    def _get_default_templates(self) -> List[ExpertTemplate]:
        """默认领域专家模板库"""
        return [
            # ====== 金融经济类 ======
            ExpertTemplate(
                expert_id="financial_analyst",
                name="金融分析师",
                name_en="Financial Analyst",
                title="注册金融分析师",
                description="擅长资金链追踪、投资收益分析、金融产品风险识别。分析案件中的经济利益动机。",
                analysis_focus="资金流向异常、保险欺诈、投资纠纷、经济利益链",
                trigger_keywords=["银行", "转账", "账户", "存款", "取款", "贷款", "债务", "借款",
                                  "欠款", "利息", "投资", "股票", "基金", "资产", "负债", "破产",
                                  "金融", "信贷", "抵押", "担保", "信用卡", "理财产品"],
                applicable_case_types=["economic", "financial", "fraud", "insurance"],
                knowledge_domain="finance",
                voting_weight=0.9,
                prompt_template="financial_analysis",
                sample_questions=[
                    "资金流向是否存在异常？",
                    "是否存在保险欺诈的可能？",
                    "经济利益链条中谁是最大受益者？",
                    "是否存在洗钱或资金转移行为？",
                ],
            ),
            ExpertTemplate(
                expert_id="forensic_accountant",
                name="司法会计专家",
                name_en="Forensic Accountant",
                title="注册会计师(CPA) / 司法会计鉴定人",
                description="擅长财务审计、账目核查、资金异常检测。识别虚假报表、挪用公款、贪污受贿的财务痕迹。",
                analysis_focus="账目异常、财务造假、资金挪用、税务问题",
                trigger_keywords=["账目", "报表", "审计", "发票", "报销", "公款", "贪污", "受贿",
                                  "挪用", "侵占", "税务", "逃税", "假账", "做账", "签字", "审批",
                                  "财务", "会计", "出纳", "预算", "拨款"],
                applicable_case_types=["economic", "corruption", "embezzlement"],
                knowledge_domain="accounting",
                voting_weight=0.9,
                prompt_template="accounting_analysis",
                sample_questions=[
                    "财务报表是否存在异常？",
                    "资金收支是否合理合法？",
                    "是否存在虚假交易或虚构支出？",
                    "涉案金额的来源和去向？",
                ],
            ),
            
            # ====== 医疗健康类 ======
            ExpertTemplate(
                expert_id="medical_expert",
                name="临床医学专家",
                name_en="Medical Expert",
                title="主任医师",
                description="擅长临床诊断、药物作用分析、病理分析。判断死亡原因、中毒症状、药物反应是否与案情吻合。",
                analysis_focus="死因分析、毒物反应、药物相互作用、医疗事故",
                trigger_keywords=["药物", "中毒", "过敏", "症状", "处方", "药片", "注射", "剂量",
                                  "治疗", "手术", "诊断", "病历", "住院", "急诊", "护理", "病房",
                                  "医生", "护士", "医院", "诊所", "药房", "抗生素", "安眠药", "毒药",
                                  "胰岛素", "氯化钾", "氰化物", "砒霜"],
                applicable_case_types=["medical", "poisoning", "medical_malpractice"],
                knowledge_domain="medicine",
                voting_weight=1.0,
                prompt_template="medical_analysis",
                sample_questions=[
                    "症状是否符合自然疾病？",
                    "药物剂量是否达到致死量？",
                    "是否存在药物相互作用？",
                    "死亡时间是否符合病理特征？",
                ],
            ),
            ExpertTemplate(
                expert_id="toxicology_expert",
                name="毒物分析专家",
                name_en="Toxicology Expert",
                title="毒理学博士",
                description="专精于毒物鉴定、毒物代谢动力学、中毒途径判断。区分投毒与自然中毒。",
                analysis_focus="毒物种类、中毒途径、代谢过程、致死浓度",
                trigger_keywords=["毒", "中毒", "投毒", "下毒", "砷", "氰化物", "农药", "杀虫剂",
                                  "老鼠药", "安眠药过量", "一氧化碳", "甲醛", "甲醇", "有机磷",
                                  "毒蘑菇", "河豚", "蛇毒", "铊", "汞", "铅中毒"],
                applicable_case_types=["poisoning", "toxicology"],
                knowledge_domain="toxicology",
                voting_weight=1.0,
                prompt_template="toxicology_analysis",
                sample_questions=[
                    "毒物可能的摄入途径？",
                    "毒物发作时间是否与时间线吻合？",
                    "死者体内毒物浓度是否达到致死量？",
                    "毒物来源可能是哪里？",
                ],
            ),
            
            # ====== 法律法规类 ======
            ExpertTemplate(
                expert_id="criminal_lawyer",
                name="刑法律师",
                name_en="Criminal Law Expert",
                title="高级律师",
                description="精通刑法、刑诉法，擅长罪名定性、法律适用、证据标准审查。确保系统推理符合法律规定。",
                analysis_focus="罪名定性、证据标准、法律适用、量刑情节",
                trigger_keywords=["法律", "罪名", "犯罪", "自首", "坦白", "故意", "过失", "正当防卫",
                                  "紧急避险", "未遂", "既遂", "共犯", "主犯", "从犯", "教唆",
                                  "刑责", "刑事责任", "证据", "证言", "物证", "书证"],
                applicable_case_types=["criminal", "homicide", "assault"],
                knowledge_domain="criminal_law",
                voting_weight=0.8,
                prompt_template="legal_analysis",
                sample_questions=[
                    "行为是否构成犯罪？构成何罪？",
                    "证据是否达到「排除合理怀疑」标准？",
                    "是否存在正当防卫或紧急避险？",
                    "是否有从轻/减轻处罚的情节？",
                ],
            ),
            ExpertTemplate(
                expert_id="insurance_expert",
                name="保险调查专家",
                name_en="Insurance Investigator",
                title="资深保险理赔调查员",
                description="擅长保险欺诈识别、理赔异常分析、保险利益判断。分析骗保杀人和自残骗保。",
                analysis_focus="保险欺诈、理赔异常、保险利益、受益人分析",
                trigger_keywords=["保险", "保单", "受益人", "理赔", "投保", "保费", "险种",
                                  "人寿保险", "意外险", "财产险", "赔偿", "赔付", "骗保",
                                  "承保", "被保险人", "投保人"],
                applicable_case_types=["insurance", "financial", "fraud"],
                knowledge_domain="insurance",
                voting_weight=0.85,
                prompt_template="insurance_analysis",
                sample_questions=[
                    "是否存在保险欺诈？",
                    "谁是保险受益人？收益金额多少？",
                    "投保时间与案发时间是否异常接近？",
                    "是否存在超额投保或多重投保？",
                ],
            ),
            
            # ====== 房地产与建筑工程类 ======
            ExpertTemplate(
                expert_id="real_estate_expert",
                name="房地产评估专家",
                name_en="Real Estate Expert",
                title="注册房地产估价师",
                description="擅长房产价值评估、产权分析、拆迁纠纷分析。涉及房产继承、房产争夺的案件。",
                analysis_focus="房产价值、产权纠纷、拆迁补偿、土地利益",
                trigger_keywords=["房产", "房屋", "产权", "拆迁", "补偿", "继承", "遗嘱", "赠与",
                                  "过户", "抵押", "房贷", "房租", "租赁", "物业", "开发商",
                                  "楼盘", "商铺", "宅基地", "土地", "建设用地"],
                applicable_case_types=["property", "inheritance", "real_estate"],
                knowledge_domain="real_estate",
                voting_weight=0.8,
                prompt_template="real_estate_analysis",
                sample_questions=[
                    "涉案房产的市场价值？",
                    "产权是否存在争议？",
                    "继承顺序和份额如何确定？",
                    "是否存在房产抵押或查封？",
                ],
            ),
            
            # ====== 心理与社会工作类 ======
            ExpertTemplate(
                expert_id="social_worker",
                name="社会工作者",
                name_en="Social Worker",
                title="高级社工师",
                description="擅长家庭关系分析、家暴识别、社会支持网络评估。分析家庭矛盾和情感纠纷的深层原因。",
                analysis_focus="家庭关系、家暴模式、社会支持、情感纠纷",
                trigger_keywords=["家暴", "离婚", "出轨", "婚外情", "家事", "子女", "抚养权",
                                  "赡养", "家庭暴力", "冷暴力", "婚姻", "夫妻", "情人", "小三",
                                  "婆媳", "继父", "继母", "收养", "离家出走"],
                applicable_case_types=["domestic", "family", "emotional"],
                knowledge_domain="social_work",
                voting_weight=0.7,
                prompt_template="social_analysis",
                sample_questions=[
                    "家庭关系是否存在系统性暴力？",
                    "受害者是否有求助记录？",
                    "是否存在权力控制模式？",
                    "社会支持网络是否完整？",
                ],
            ),
            ExpertTemplate(
                expert_id="addiction_specialist",
                name="成瘾行为专家",
                name_en="Addiction Specialist",
                title="精神科主治医师",
                description="擅长酒精/药物成瘾分析、戒断症状识别、成瘾者行为模式。分析成瘾相关的犯罪动机。",
                analysis_focus="成瘾行为、戒断反应、药物依赖、行为失控",
                trigger_keywords=["酒精", "酗酒", "吸毒", "毒品", "冰毒", "海洛因", "大麻",
                                  "成瘾", "戒断", "瘾君子", "酒瘾", "药瘾", "赌博", "赌债",
                                  "戒酒", "戒毒", "复吸", "幻觉", "妄想"],
                applicable_case_types=["drug", "addiction", "alcohol"],
                knowledge_domain="addiction",
                voting_weight=0.8,
                prompt_template="addiction_analysis",
                sample_questions=[
                    "嫌疑人是否存在成瘾行为？",
                    "成瘾是否影响其行为控制能力？",
                  "戒断症状是否可能导致暴力行为？",
                  "是否存在因成瘾导致的经济压力？",
                ],
            ),
            
            # ====== 信息安全类 ======
            ExpertTemplate(
                expert_id="cybersecurity_expert",
                name="网络安全专家",
                name_en="Cybersecurity Expert",
                title="CISSP / CISP",
                description="擅长网络攻击分析、数据泄露追踪、电子取证。分析网络犯罪和黑客行为。",
                analysis_focus="网络攻击、数据泄露、电子取证、网络追踪",
                trigger_keywords=["黑客", "入侵", "数据泄露", "木马", "病毒", "勒索", "钓鱼",
                                  "网络攻击", "IP", "服务器", "数据库", "加密", "解密", "暗网",
                                  "虚拟货币", "比特币", "区块链", "电子签名", "人脸识别", "监控"],
                applicable_case_types=["cyber", "hacking", "data_breach"],
                knowledge_domain="cybersecurity",
                voting_weight=0.85,
                prompt_template="cybersecurity_analysis",
                sample_questions=[
                    "是否存在网络入侵痕迹？",
                    "数据是否被篡改或删除？",
                    "攻击来源是否可以追踪？",
                    "电子证据的完整性是否可验证？",
                ],
            ),
            
            # ====== 环境与化学类 ======
            ExpertTemplate(
                expert_id="environmental_expert",
                name="环境检测专家",
                name_en="Environmental Expert",
                title="环境工程高级工程师",
                description="擅长环境污染检测、化学物质分析、环境致害判断。分析环境污染相关的案件。",
                analysis_focus="环境污染、化学物质、环境致害、危险废物",
                trigger_keywords=["污染", "废气", "废水", "排放", "化学品", "辐射", "噪音",
                                  "粉尘", "重金属", "土壤污染", "水源", "大气", "臭氧",
                                  "二噁英", "苯", "甲醛超标", "雾霾"],
                applicable_case_types=["environmental", "pollution"],
                knowledge_domain="environment",
                voting_weight=0.8,
                prompt_template="environmental_analysis",
                sample_questions=[
                    "污染物是否超标？",
                    "污染与健康损害的因果关系？",
                    "排放是否符合标准？",
                    "污染治理责任如何认定？",
                ],
            ),
        ]


def get_default_registry() -> ExpertRegistry:
    """获取默认专家注册表"""
    return ExpertRegistry()
