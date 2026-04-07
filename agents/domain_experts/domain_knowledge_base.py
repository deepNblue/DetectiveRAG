"""
领域专业知识库 (Domain Knowledge Base)
为每个业务专家提供结构化的专业知识

知识条目结构:
  - 领域(domain)
  - 类别(category) 
  - 标题(title)
  - 内容(content)
  - 关键词(keywords)
  - 适用场景(applicable_scenarios)
  - 来源(source)

知识来源:
  1. 预设领域知识（代码内置）
  2. 从案件结果中自动学习积累
  3. 外部知识文件导入(data/knowledge/)
"""

import json
import os
from typing import Dict, List, Any, Optional
from loguru import logger
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class KnowledgeEntry:
    """知识条目"""
    entry_id: str
    domain: str          # 领域: finance, medicine, law, ...
    category: str        # 类别: concept, method, red_flag, case_pattern
    title: str
    content: str
    keywords: List[str] = field(default_factory=list)
    applicable_scenarios: List[str] = field(default_factory=list)
    source: str = "preset"  # preset / learned / imported
    created_at: str = ""
    confidence: float = 1.0
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()[:10]


class DomainKnowledgeBase:
    """领域专业知识库"""
    
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'knowledge')
    
    def __init__(self):
        self._entries: Dict[str, List[KnowledgeEntry]] = {}  # domain -> entries
        self._index: Dict[str, List[str]] = {}  # keyword -> [entry_ids]
        self._load_all()
        self._build_index()
    
    def add_entry(self, entry: KnowledgeEntry):
        """添加知识条目"""
        domain = entry.domain
        if domain not in self._entries:
            self._entries[domain] = []
        self._entries[domain].append(entry)
        
        # 更新索引
        for kw in entry.keywords:
            kw_lower = kw.lower()
            if kw_lower not in self._index:
                self._index[kw_lower] = []
            self._index[kw_lower].append(entry.entry_id)
    
    def search(self, query: str, domain: str = None, top_k: int = 10) -> List[KnowledgeEntry]:
        """搜索相关知识"""
        query_lower = query.lower()
        scored = []
        
        entries_to_search = (
            self._entries.get(domain, []) if domain
            else [e for entries in self._entries.values() for e in entries]
        )
        
        for entry in entries_to_search:
            score = 0.0
            # 关键词匹配
            for kw in entry.keywords:
                if kw.lower() in query_lower:
                    score += 2.0
            # 标题匹配
            if entry.title.lower() in query_lower:
                score += 3.0
            # 内容片段匹配
            content_lower = entry.content.lower()
            for word in query_lower.split():
                if word in content_lower:
                    score += 0.5
            # 场景匹配
            for scenario in entry.applicable_scenarios:
                if scenario.lower() in query_lower:
                    score += 1.5
            
            if score > 0:
                scored.append((score, entry))
        
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]
    
    def get_domain_knowledge(self, domain: str) -> List[KnowledgeEntry]:
        """获取某个领域的所有知识"""
        return self._entries.get(domain, [])
    
    def get_knowledge_prompt(self, domain: str, case_context: str = "", max_entries: int = 5) -> str:
        """
        获取领域知识注入prompt的文本
        
        根据案件上下文检索最相关的知识条目，生成可注入的文本
        """
        if case_context:
            entries = self.search(case_context, domain=domain, top_k=max_entries)
        else:
            entries = self.get_domain_knowledge(domain)[:max_entries]
        
        if not entries:
            return ""
        
        lines = [f"\n📚 【{domain}领域专业知识库】"]
        for i, entry in enumerate(entries, 1):
            lines.append(f"\n{i}. [{entry.category}] {entry.title}")
            lines.append(f"   {entry.content[:200]}")
            if entry.applicable_scenarios:
                lines.append(f"   适用: {', '.join(entry.applicable_scenarios[:3])}")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计"""
        stats = {}
        for domain, entries in self._entries.items():
            categories = {}
            for e in entries:
                categories[e.category] = categories.get(e.category, 0) + 1
            stats[domain] = {
                "total": len(entries),
                "categories": categories,
            }
        return stats
    
    def save(self):
        """保存知识库到文件"""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        for domain, entries in self._entries.items():
            filepath = os.path.join(self.DATA_DIR, f"{domain}.json")
            data = [asdict(e) for e in entries]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"知识库已保存: {len(self._entries)}个领域")
    
    def _load_all(self):
        """加载所有知识"""
        # 1. 加载文件中的知识
        if os.path.exists(self.DATA_DIR):
            for filename in os.listdir(self.DATA_DIR):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.DATA_DIR, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        domain = filename.replace('.json', '')
                        for item in data:
                            self.add_entry(KnowledgeEntry(**item))
                    except Exception as e:
                        logger.warning(f"加载知识文件失败 {filename}: {e}")
        
        # 2. 加载预设知识
        self._load_preset_knowledge()
    
    def _build_index(self):
        """构建关键词索引"""
        self._index = {}
        for domain, entries in self._entries.items():
            for entry in entries:
                for kw in entry.keywords:
                    kw_lower = kw.lower()
                    if kw_lower not in self._index:
                        self._index[kw_lower] = []
                    self._index[kw_lower].append(entry.entry_id)
    
    def _load_preset_knowledge(self):
        """加载预设领域知识"""
        presets = self._get_preset_knowledge()
        for entry_data in presets:
            entry = KnowledgeEntry(**entry_data)
            # 避免重复
            existing_ids = {e.entry_id for e in self._entries.get(entry.domain, [])}
            if entry.entry_id not in existing_ids:
                self.add_entry(entry)
    
    def _get_preset_knowledge(self) -> List[Dict]:
        """预设领域知识条目"""
        return [
            # ====== 金融领域 ======
            {
                "entry_id": "fin_red_flag_001",
                "domain": "finance",
                "category": "red_flag",
                "title": "大额异常转账红旗信号",
                "content": "案发前30天内的大额转账（超过月收入的3倍）是经济动机的重要红旗信号。需要关注: 1)转账时间与案发时间的关系 2)收款方身份 3)转账理由的合理性 4)是否分散多笔以规避监管(拆分交易/Smurfing)",
                "keywords": ["转账", "大额", "异常", "拆分交易", "红旗信号", "经济动机"],
                "applicable_scenarios": ["保险欺诈", "经济犯罪", "遗产纠纷", "离婚案件"],
            },
            {
                "entry_id": "fin_concept_001",
                "domain": "finance",
                "category": "concept",
                "title": "保险欺诈常见模式",
                "content": "保险欺诈的典型模式: 1)投保后短期内出险(投保<6个月) 2)超额投保(保额远超实际价值) 3)多重投保(在多家公司投保同一标的) 4)受益人变更后短期内出险 5)投保时隐瞒重要事实。在命案中，需要特别关注人寿保险受益人是否为嫌疑人。",
                "keywords": ["保险欺诈", "投保", "受益人", "超额投保", "多重投保", "隐瞒"],
                "applicable_scenarios": ["保险诈骗", "杀人骗保", "自残骗保"],
            },
            {
                "entry_id": "fin_method_001",
                "domain": "finance",
                "category": "method",
                "title": "资金链追踪方法论",
                "content": "资金链追踪的核心方法: 1)确定资金起点(谁的账户) 2)追踪资金流向(转入→转出) 3)识别中间账户(是否为过桥账户) 4)计算实际获利(扣除成本) 5)分析资金用途(消费/投资/转移)。关键指标: 资金周转率异常、大额现金提取、跨境转账。",
                "keywords": ["资金链", "追踪", "过桥账户", "洗钱", "资金流向"],
                "applicable_scenarios": ["贪污案件", "洗钱", "经济犯罪"],
            },
            
            # ====== 会计领域 ======
            {
                "entry_id": "acc_red_flag_001",
                "domain": "accounting",
                "category": "red_flag",
                "title": "财务造假红旗信号",
                "content": "财务造假常见迹象: 1)收入与现金流不匹配 2)应收账款异常增长 3)存货周转率骤降 4)关联交易频繁 5)审计意见非标 6)频繁更换会计政策 7)高管频繁变动。在案件中，需要关注涉案人员是否利用职务便利篡改财务记录。",
                "keywords": ["造假", "财务报表", "审计", "关联交易", "应收账款"],
                "applicable_scenarios": ["贪污挪用", "职务犯罪", "经济诈骗"],
            },
            {
                "entry_id": "acc_method_001",
                "domain": "accounting",
                "category": "method",
                "title": "挪用公款常见手法",
                "content": "挪用公款的典型手法: 1)虚构报销(伪造发票/虚构支出) 2)截留收入(收取现金不入账) 3)公款私用(以公司名义借款) 4)关联交易(利益输送) 5)阴阳合同(合同金额与实际不符)。关键审计方法: 银行对账单与账面核对、大额凭证抽查、函证。",
                "keywords": ["挪用公款", "虚构报销", "截留收入", "阴阳合同", "审计"],
                "applicable_scenarios": ["贪污", "挪用公款", "职务侵占"],
            },
            
            # ====== 医学领域 ======
            {
                "entry_id": "med_red_flag_001",
                "domain": "medicine",
                "category": "red_flag",
                "title": "非自然死亡医学红旗信号",
                "content": "提示非自然死亡的医学迹象: 1)尸斑位置与发现姿势不符(死后被移动) 2)颈部有压痕但报告为心脏病发作 3)胃内容物发现未知药物成分 4)伤口方向与自伤不符 5)存在两种以上致命因素。法医需要区分自然死亡、意外死亡和他杀。",
                "keywords": ["非自然死亡", "尸斑", "伤口", "药物", "他杀"],
                "applicable_scenarios": ["疑似他杀", "猝死鉴定", "医疗事故"],
            },
            {
                "entry_id": "med_concept_001",
                "domain": "medicine",
                "category": "concept",
                "title": "常见致死药物及特征",
                "content": "常见致死药物: 1)氰化物—迅速死亡,杏仁味,皮肤樱桃红色 2)胰岛素—低血糖昏迷,需大剂量注射 3)氯化钾—心脏骤停,注射用,难以检出 4)有机磷—瞳孔缩小,分泌增加,肌颤 5)安眠药(巴比妥类)—深度昏迷,呼吸抑制。注意: 部分药物代谢快,需及时取样。",
                "keywords": ["氰化物", "胰岛素", "氯化钾", "有机磷", "安眠药", "致死剂量"],
                "applicable_scenarios": ["投毒案件", "药物过量", "注射杀人"],
            },
            {
                "entry_id": "med_concept_002",
                "domain": "medicine",
                "category": "concept",
                "title": "死亡时间推断方法",
                "content": "死亡时间推断依据: 1)尸温—每小时下降约1°C(受环境温度影响) 2)尸僵—死后2-4小时开始,12-24小时最硬,48小时缓解 3)尸斑—死后30分钟出现,12小时固定 4)胃内容物消化程度—空腹/半消化/全消化 5)角膜混浊程度 6)腐败程度(蝇蛆生长周期)。综合多种方法可提高准确性。",
                "keywords": ["死亡时间", "尸温", "尸僵", "尸斑", "胃内容物", "法医"],
                "applicable_scenarios": ["死亡时间推断", "不在场证明验证"],
            },
            
            # ====== 毒理学领域 ======
            {
                "entry_id": "tox_concept_001",
                "domain": "toxicology",
                "category": "concept",
                "title": "毒物分类与中毒特征",
                "content": "常见毒物分类: 1)挥发性毒物(氰化物/甲醇)—迅速发作,有特殊气味 2)金属毒物(砷/汞/铊)—慢性中毒,毛发指甲可检出 3)农药(有机磷/百草枯)—农业场景常见 4)药物类(安眠药/胰岛素)—医疗场景常见 5)植物/动物毒素(河豚/蛇毒)—特殊食物/场景。关键: 毒物检出窗口期各不同。",
                "keywords": ["毒物分类", "氰化物", "有机磷", "砷", "中毒特征"],
                "applicable_scenarios": ["投毒", "食物中毒", "药物过量"],
            },
            {
                "entry_id": "tox_method_001",
                "domain": "toxicology",
                "category": "method",
                "title": "投毒 vs 自杀/意外中毒鉴别",
                "content": "鉴别要点: 1)毒物来源—嫌疑人能否获取? 2)摄入途径—口服(可能被下药)/注射(需要接触)/吸入(是否意外) 3)剂量—致死量vs治疗量,是否超大剂量 4)场景—食物/饮料中检出毒物强烈指向投毒 5)动机—是否有保险/遗产/仇恨 6)历史—是否有类似中毒史。投毒案常见于家庭内部。",
                "keywords": ["投毒", "自杀", "意外", "鉴别", "剂量", "摄入途径"],
                "applicable_scenarios": ["疑似投毒", "中毒鉴定"],
            },
            
            # ====== 刑法领域 ======
            {
                "entry_id": "law_concept_001",
                "domain": "criminal_law",
                "category": "concept",
                "title": "刑事案件证据标准",
                "content": "中国刑事案件定罪标准: 「案件事实清楚，证据确实、充分」。具体要求: 1)定罪量刑的事实都有证据证明 2)据以定案的证据均经法定程序查证属实 3)综合全案证据，对所认定事实已排除合理怀疑。间接证据定案需要形成完整证据链，不能有断裂。",
                "keywords": ["证据标准", "排除合理怀疑", "证据链", "确实充分", "间接证据"],
                "applicable_scenarios": ["证据审查", "定罪标准", "疑罪从无"],
            },
            {
                "entry_id": "law_concept_002",
                "domain": "criminal_law",
                "category": "concept",
                "title": "故意杀人罪构成要件",
                "content": "故意杀人罪(刑法第232条): 客体—他人的生命权; 客观方面—非法剥夺他人生命的行为; 主体—已满14周岁的自然人; 主观方面—故意(直接/间接)。需要区分: 故意杀人vs故意伤害致死(主观故意内容不同), 故意杀人vs过失致人死亡(主观心态不同), 正当防卫vs防卫过当。",
                "keywords": ["故意杀人", "构成要件", "故意", "过失", "正当防卫", "防卫过当"],
                "applicable_scenarios": ["命案定性", "罪名区分"],
            },
            {
                "entry_id": "law_red_flag_001",
                "domain": "criminal_law",
                "category": "red_flag",
                "title": "非法证据排除规则",
                "content": "以下证据应当排除: 1)采用刑讯逼供等非法方法收集的犯罪嫌疑人供述 2)采用暴力、威胁等非法方法收集的证人证言 3)收集物证、书证不符合法定程序,可能严重影响司法公正且不能补正。在推理中要注意: 只有合法取得的证据才有证明力。",
                "keywords": ["非法证据", "排除规则", "刑讯逼供", "合法性"],
                "applicable_scenarios": ["证据审查", "程序正义"],
            },
            
            # ====== 保险领域 ======
            {
                "entry_id": "ins_concept_001",
                "domain": "insurance",
                "category": "concept",
                "title": "人寿保险受益人规则",
                "content": "人寿保险受益人关键规则: 1)指定受益人优于法定受益人 2)受益人先于被保险人死亡且无其他受益人→保险金作为遗产 3)受益人故意造成被保险人死亡→丧失受益权 4)受益人变更需被保险人同意 5)投保人对被保险人有保险利益。在命案中: 如果嫌疑人是受益人,强烈指向骗保杀人。",
                "keywords": ["人寿保险", "受益人", "丧失受益权", "保险利益", "指定受益人"],
                "applicable_scenarios": ["杀人骗保", "保险欺诈", "受益人争议"],
            },
            
            # ====== 社会工作领域 ======
            {
                "entry_id": "soc_concept_001",
                "domain": "social_work",
                "category": "concept",
                "title": "家庭暴力循环理论",
                "content": "家暴三阶段循环: 1)蓄积期—紧张积累,小摩擦频发 2)爆发期—暴力行为发生,程度可能逐步升级 3)蜜月期—施暴者道歉/保证/示好,受害者原谅。循环往复,间隔逐渐缩短,暴力逐渐升级。关键洞察: 家暴受害者往往不是第一次被暴力对待,且有反复原谅的心理特征。",
                "keywords": ["家暴", "暴力循环", "蓄积期", "爆发期", "蜜月期", "升级"],
                "applicable_scenarios": ["家庭暴力", "杀亲案", "反杀案"],
            },
            {
                "entry_id": "soc_red_flag_001",
                "domain": "social_work",
                "category": "red_flag",
                "title": "亲密关系暴力危险信号",
                "content": "亲密关系暴力的高危信号: 1)极端控制(限制社交/经济控制/手机监控) 2)病态嫉妒(无端怀疑出轨) 3)隔离(切断与家人朋友的联系) 4)威胁(以自杀/伤害家人相威胁) 5)暴力升级(从言语→推搡→殴打) 6)武器获取。当受害者试图离开时是最危险的时刻。",
                "keywords": ["亲密关系暴力", "控制", "嫉妒", "隔离", "危险信号"],
                "applicable_scenarios": ["家庭命案", "感情纠纷"],
            },
            
            # ====== 网络安全领域 ======
            {
                "entry_id": "cyber_method_001",
                "domain": "cybersecurity",
                "category": "method",
                "title": "电子证据固定与保全",
                "content": "电子证据固定关键原则: 1)完整性—使用哈希值(MD5/SHA256)验证 2)原始性—对原始存储介质做镜像备份,分析在副本上进行 3)链式保管—记录证据从获取到呈堂的完整链条 4)时间戳—使用可信时间源标注操作时间 5)合法授权—搜查令/协助函。注意: 电子证据极易被篡改,需要专业人员操作。",
                "keywords": ["电子证据", "哈希值", "镜像备份", "链式保管", "时间戳"],
                "applicable_scenarios": ["电子取证", "网络犯罪", "证据固定"],
            },
            
            # ====== 环境领域 ======
            {
                "entry_id": "env_concept_001",
                "domain": "environment",
                "category": "concept",
                "title": "环境污染致害因果关系认定",
                "content": "环境侵权因果关系认定标准(低于一般侵权): 1)污染者排放了污染物 2)被侵权人受到了损害 3)污染物与损害之间存在关联性。举证责任倒置: 污染者应证明不存在因果关系。在刑事案件(污染环境罪)中,需要达到「排除合理怀疑」的更高标准。",
                "keywords": ["环境污染", "因果关系", "举证责任倒置", "污染环境罪"],
                "applicable_scenarios": ["环境污染案件", "群体性健康损害"],
            },
            
            # ====== 成瘾领域 ======
            {
                "entry_id": "add_concept_001",
                "domain": "addiction",
                "category": "concept",
                "title": "酒精与暴力行为关系",
                "content": "酒精与暴力的关系: 1)酒精抑制前额叶功能→冲动控制力下降→攻击性增强 2)血液酒精浓度0.08%-0.15%时暴力风险最高 3)慢性酗酒者在戒断期(停酒后12-72小时)可能出现震颤谵妄→极度兴奋→暴力行为 4)酒精会影响记忆形成→断片→嫌疑人可能确实不记得作案过程。不能以醉酒为免罪理由。",
                "keywords": ["酒精", "暴力", "冲动控制", "戒断", "断片", "醉酒"],
                "applicable_scenarios": ["醉酒杀人", "家暴", "酒后犯罪"],
            },
        ]
