#!/usr/bin/env python3
"""
图片证据图谱构建模块 v2 — 参照 RAG-Anything 设计
核心改动:
1. ContextExtractor: 提取图片周围的文本上下文增强分析
2. 视觉分析prompt对标RAG-Anything的vision_prompt (含context)
3. JSON实体提取对标RAG-Anything的_robust_json_parse
4. belongs_to关系 (weight=10.0)
5. 跨模态关联 (图片实体 ↔ 嫌疑人/物证)
6. 图谱感知检索: 向量+图混合检索增强ASMR
"""

import json
import re
import time
import hashlib
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger


# ══════════════════════════════════════════════════════════════
# 数据模型
# ══════════════════════════════════════════════════════════════

@dataclass
class EvidenceNode:
    """证据节点 — 对应 RAG-Anything 的 knowledge graph node"""
    node_id: str
    name: str
    node_type: str  # image_evidence, suspect, weapon, substance, location, time, etc.
    description: str
    source_image: Optional[str] = None
    confidence: float = 1.0
    summary: str = ""  # RAG-Anything entity_info.summary
    chunk_id: Optional[str] = None  # RAG-Anything chunk hash
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceEdge:
    """证据关系边 — 对应 RAG-Anything 的 knowledge graph edge"""
    edge_id: str
    source_id: str
    target_id: str
    relation_type: str  # belongs_to, proves, contradicts, mentions, etc.
    description: str
    keywords: str = ""
    weight: float = 10.0  # RAG-Anything 标准权重
    source_id_chunk: Optional[str] = None  # chunk来源
    metadata: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
# ContextExtractor — 参照 RAG-Anything ContextExtractor
# ══════════════════════════════════════════════════════════════

class ContextExtractor:
    """
    上下文提取器 — 从案件文本中提取图片周围的上下文
    
    RAG-Anything的做法: 图片所在的页面/段落的文本作为上下文
    我们的适配: 图片caption相关的案件段落作为上下文
    """
    
    def __init__(self, context_window: int = 500, max_context_tokens: int = 2000):
        self.context_window = context_window
        self.max_context_tokens = max_context_tokens
    
    def extract_context(
        self, 
        case_text: str, 
        caption: str = "",
        image_index: int = 0,
    ) -> str:
        """
        从案件文本中提取与图片相关的上下文
        
        策略:
        1. 如果caption有关键词，提取包含这些关键词的句子
        2. 否则提取案件文本的开头和结尾
        """
        if not case_text:
            return ""
        
        # 策略1: 用caption关键词定位上下文
        if caption:
            keywords = re.findall(r'[\u4e00-\u9fff]{2,}', caption)
            relevant_sentences = []
            
            for line in case_text.split('\n'):
                for kw in keywords:
                    if kw in line:
                        relevant_sentences.append(line.strip())
                        break
                if len(relevant_sentences) >= 3:
                    break
            
            if relevant_sentences:
                context = '\n'.join(relevant_sentences)
                return self._truncate(context)
        
        # 策略2: 按图片位置比例取文本段
        text_len = len(case_text)
        start = min(int(text_len * image_index / max(3, 1)), text_len - self.context_window)
        start = max(0, start)
        context = case_text[start:start + self.context_window]
        return self._truncate(context)
    
    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_context_tokens:
            return text
        # 在句子边界截断
        truncated = text[:self.max_context_tokens]
        last_period = truncated.rfind('。')
        if last_period > len(truncated) * 0.7:
            return truncated[:last_period + 1]
        return truncated + "..."


# ══════════════════════════════════════════════════════════════
# EvidenceGraphBuilder — 核心构建器
# ══════════════════════════════════════════════════════════════

class EvidenceGraphBuilder:
    """
    图片证据图谱构建器 v2
    
    完整参照 RAG-Anything 的 ImageModalProcessor:
    - ContextExtractor 提取上下文
    - vision_prompt_with_context 分析图片
    - _robust_json_parse 解析JSON
    - _create_entity_and_chunk 创建实体和关系
    - belongs_to 边 (weight=10.0)
    - extract_entities 提取子实体
    - merge_nodes_and_edges 合并到全局图谱
    """
    
    # 刑侦专用实体类型
    FORENSIC_ENTITY_TYPES = [
        "suspect", "victim", "weapon", "substance", "location",
        "time", "document", "digital_device", "physical_evidence",
        "vehicle", "communication", "financial", "biometric",
        "measurement", "surveillance", "other",
    ]
    
    # 关系类型
    RELATION_TYPES = [
        "belongs_to", "depicts", "contains", "indicates",
        "proves", "contradicts", "supports", "eliminates",
        "connects_to", "same_as", "mentions", "shows",
    ]
    
    def __init__(self, llm_client=None, context_extractor=None):
        self.llm_client = llm_client
        self.context_extractor = context_extractor or ContextExtractor()
        self.nodes: Dict[str, EvidenceNode] = {}
        self.edges: List[EvidenceEdge] = []
    
    def _compute_id(self, content: str, prefix: str = "chunk") -> str:
        """生成ID — 对标 RAG-Anything 的 compute_mdhash_id"""
        md5 = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"{prefix}-{md5}"
    
    def build_from_image_analyses(
        self,
        image_descriptions: List[Dict[str, Any]],
        suspects: List[Dict[str, Any]] = None,
        case_text: str = "",
    ) -> Dict[str, Any]:
        """
        完整的图片证据图谱构建流程
        
        流程(对标 RAG-Anything):
        1. 对每张图片: ContextExtractor提取上下文
        2. 对每张图片: 创建主实体节点(image_evidence)
        3. 对每张图片: LLM/正则提取子实体
        4. 创建 belongs_to 关系 (weight=10.0)
        5. 跨模态关联: 图片实体 ↔ 嫌疑人
        6. 生成图谱文本(注入到case_text)
        7. 返回完整图谱数据
        """
        suspects = suspects or []
        suspect_names = [s.get("name", "") if isinstance(s, dict) else str(s) for s in suspects]
        
        # Step 1+2: 创建主实体节点
        for desc in image_descriptions:
            self._create_image_evidence_node(desc, case_text)
        
        # Step 3+4: 提取子实体 + belongs_to关系
        for desc in image_descriptions:
            img_index = desc.get("index", 0)
            analysis = desc.get("analysis", "")
            caption = desc.get("caption", "")
            main_node_id = f"img_evidence_{img_index}"
            
            # LLM提取子实体
            entities = self._extract_forensic_entities(analysis, caption, img_index)
            
            # 创建子实体节点 + belongs_to 关系
            for entity in entities:
                entity_id = self._compute_id(f"{img_index}_{entity['name']}_{entity['type']}", "ent")
                
                if entity_id not in self.nodes:
                    node = EvidenceNode(
                        node_id=entity_id,
                        name=entity["name"],
                        node_type=entity["type"],
                        description=entity.get("description", ""),
                        source_image=desc.get("path", ""),
                        confidence=entity.get("confidence", 0.8),
                        summary=entity.get("description", "")[:100],
                        chunk_id=self._compute_id(analysis, "chunk"),
                    )
                    self.nodes[entity_id] = node
                
                # belongs_to 关系 (RAG-Anything 标准权重=10.0)
                edge_id = self._compute_id(f"{entity_id}_belongs_to_{main_node_id}", "rel")
                self.edges.append(EvidenceEdge(
                    edge_id=edge_id,
                    source_id=entity_id,
                    target_id=main_node_id,
                    relation_type="belongs_to",
                    description=f"{entity['name']} 出现在图片证据{img_index}({caption})中",
                    keywords="belongs_to,part_of,contained_in",
                    weight=10.0,
                    source_id_chunk=self._compute_id(analysis, "chunk"),
                ))
        
        # Step 5: 跨模态关联
        cross_modal_links = self._build_cross_modal_links(suspect_names)
        
        # Step 6: 跨图片实体关联 (同名实体合并)
        self._merge_duplicate_entities()
        
        # Step 7: 生成图谱文本
        graph_text = self._generate_graph_text()
        
        # Step 8: 生成检索向量描述(供向量DB使用)
        retrieval_chunks = self._generate_retrieval_chunks()
        
        return {
            "nodes": [{"id": n.node_id, "name": n.name, "type": n.node_type,
                        "desc": n.description[:100], "summary": n.summary,
                        "chunk_id": n.chunk_id} for n in self.nodes.values()],
            "edges": [{"id": e.edge_id, "src": e.source_id, "tgt": e.target_id,
                        "rel": e.relation_type, "desc": e.description,
                        "keywords": e.keywords, "weight": e.weight} for e in self.edges],
            "graph_text": graph_text,
            "cross_modal_links": cross_modal_links,
            "retrieval_chunks": retrieval_chunks,
            "stats": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "image_evidence_nodes": sum(1 for n in self.nodes.values() if n.node_type == "image_evidence"),
                "cross_modal_links": len(cross_modal_links),
                "retrieval_chunks": len(retrieval_chunks),
            }
        }
    
    def _create_image_evidence_node(self, desc: Dict, case_text: str = ""):
        """创建图片证据主实体节点 — 对标 RAG-Anything 的 _create_entity_and_chunk"""
        img_index = desc.get("index", 0)
        caption = desc.get("caption", f"图片{img_index}")
        analysis = desc.get("analysis", "")
        img_path = desc.get("path", "")
        
        # 提取上下文
        context = self.context_extractor.extract_context(case_text, caption, img_index)
        
        # 生成chunk_id (RAG-Anything: compute_mdhash_id)
        chunk_id = self._compute_id(analysis, "chunk")
        
        # 生成摘要 (对标 RAG-Anything entity_info.summary, max 100 words)
        summary = analysis[:200] if analysis else caption
        
        main_node_id = f"img_evidence_{img_index}"
        self.nodes[main_node_id] = EvidenceNode(
            node_id=main_node_id,
            name=f"图片证据{img_index}: {caption}",
            node_type="image_evidence",
            description=analysis,
            source_image=img_path,
            confidence=1.0,
            summary=summary,
            chunk_id=chunk_id,
            metadata={
                "caption": caption,
                "img_index": img_index,
                "context": context[:200] if context else "",
            }
        )
    
    def _extract_forensic_entities(
        self, 
        analysis_text: str, 
        caption: str, 
        img_index: int
    ) -> List[Dict[str, Any]]:
        """从图片分析文本中提取刑侦实体"""
        if self.llm_client:
            return self._llm_extract_entities(analysis_text, caption, img_index)
        else:
            return self._regex_extract_entities(analysis_text, caption, img_index)
    
    def _llm_extract_entities(
        self, analysis_text: str, caption: str, img_index: int
    ) -> List[Dict[str, Any]]:
        """
        LLM提取子实体 — 对标 RAG-Anything 的 extract_entities
        
        RAG-Anything的做法: 
        - 先用 vision_prompt 获取 detailed_description + entity_info
        - 再对 chunk 调用 extract_entities 提取子实体和关系
        
        我们的适配:
        - vision分析已在Stage 0完成，analysis_text就是结果
        - 这里额外调用LLM提取刑侦实体
        """
        prompt = f"""你是一名刑侦证据分析师。请从以下图片分析结果中提取所有具有刑侦价值的实体。

图片标题: {caption}
图片分析结果:
{analysis_text}

请提取所有关键实体，严格按以下JSON格式返回:
{{
  "entities": [
    {{
      "name": "实体名称(必须具体，如'氰化钾'而非'毒物')",
      "type": "实体类型(从以下选择: suspect/victim/weapon/substance/location/time/document/digital_device/physical_evidence/vehicle/communication/financial/biometric/measurement/surveillance/other)",
      "description": "该实体在案件中的意义(一句话)",
      "confidence": 0.9
    }}
  ]
}}

提取规则:
1. 提取所有具体人名(包括"某人"、"经理"等称呼)→ type=suspect
2. 提取所有化学物质、药物、毒物 → type=substance
3. 提取所有具体时间点 → type=time
4. 提取所有地点 → type=location
5. 提取所有数字测量数据 → type=measurement
6. 提取所有武器/工具 → type=weapon
7. 提取所有通讯记录 → type=communication
8. 提取所有监控/摄像头相关 → type=surveillance
9. 提取所有生物特征(DNA/指纹) → type=biometric
10. 不要提取抽象概念，只提取具体实体
"""
        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            if isinstance(response, str):
                entities = self._robust_json_parse(response)
                return entities
            return []
        except Exception as e:
            logger.warning(f"LLM实体提取失败(图片{img_index}): {e}, 使用正则回退")
            return self._regex_extract_entities(analysis_text, caption, img_index)
    
    def _robust_json_parse(self, response: str) -> List[Dict[str, Any]]:
        """
        健壮的JSON解析 — 对标 RAG-Anything 的 _robust_json_parse
        
        多策略解析:
        1. 直接JSON解析
        2. 代码块提取
        3. 平衡括号提取
        4. 正则字段提取
        """
        # 清理 thinking tags (某些模型)
        cleaned = re.sub(r'```(?:json)?\s*', '', response)
        cleaned = re.sub(r'```', '', cleaned)
        
        # Strategy 1: 直接解析
        try:
            data = json.loads(cleaned)
            return self._validate_entities(data)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Strategy 2: 提取JSON块
        json_blocks = re.findall(r'\{[\s\S]*\}', response)
        for block in json_blocks:
            try:
                data = json.loads(block)
                return self._validate_entities(data)
            except (json.JSONDecodeError, ValueError):
                continue
        
        # Strategy 3: 正则提取实体列表
        entities = []
        entity_pattern = re.compile(
            r'"name":\s*"([^"]+)"\s*,\s*"type":\s*"([^"]+)"\s*,\s*"description":\s*"([^"]*)"',
            re.DOTALL,
        )
        for match in entity_pattern.finditer(response):
            name, etype, desc = match.groups()
            if etype not in self.FORENSIC_ENTITY_TYPES:
                etype = "other"
            entities.append({
                "name": name,
                "type": etype,
                "description": desc,
                "confidence": 0.7,
            })
        
        return entities
    
    def _validate_entities(self, data: dict) -> List[Dict[str, Any]]:
        """验证并清理提取的实体"""
        entities = data.get("entities", [])
        validated = []
        for e in entities:
            if not isinstance(e, dict):
                continue
            name = e.get("name", "").strip()
            if not name:
                continue
            etype = e.get("type", "other")
            if etype not in self.FORENSIC_ENTITY_TYPES:
                etype = "other"
            validated.append({
                "name": name,
                "type": etype,
                "description": e.get("description", ""),
                "confidence": min(1.0, max(0.0, float(e.get("confidence", 0.8)))),
            })
        return validated
    
    def _regex_extract_entities(
        self, analysis_text: str, caption: str, img_index: int
    ) -> List[Dict[str, Any]]:
        """正则启发式实体提取(回退方案)"""
        entities = []
        seen = set()
        
        patterns = [
            # 人名 (称呼模式: 如"王经理", "李夫人") — 姓+称呼
            (r'[\u4e00-\u9fff](?:经理|律师|管家|商人|教授|医生|护士|厨师|保安|司机|助手|秘书|仆人|园丁|保镖|夫人|太太|小姐|先生|女士)',
             "suspect", 0.85),
            # 化学物质
            (r'氰化物|氰化钾|氰化钠|砒霜|砷|安眠药|巴比妥|有机磷|农药|敌敌畏|百草枯|甲醇|乙醇|一氧化碳|硫化氢|甲醛',
             "substance", 0.9),
            # 时间
            (r'\d{1,2}:\d{2}(?::\d{2})?|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日',
             "time", 0.8),
            # 测量数据
            (r'\d+\.?\d*\s*(?:mg|ml|克|kg|cm|m|km|楼|层|室|号|%|ppm|度)',
             "measurement", 0.7),
            # 地点
            (r'[\u4e00-\u9fff]{2,8}(?:室|楼|层|区|栋|房间|仓库|车库|花园|厨房|客厅|卧室|书房|地下室|阳台|走廊|楼梯|电梯|大厅)',
             "location", 0.8),
            # 监控
            (r'监控|摄像头|录像|CCTV|摄像头',
             "surveillance", 0.85),
            # 指纹/DNA
            (r'DNA|指纹|血迹|毛发|脚印|足迹|牙印|体液|精液|唾液',
             "biometric", 0.9),
            # 通讯
            (r'短信|微信|电话|邮件|QQ|WhatsApp|通话记录|通讯录',
             "communication", 0.85),
            # 武器
            (r'刀|枪|斧头|锤子|绳索|毒药|安眠药|匕首|棍棒|扳手|刀具|凶器',
             "weapon", 0.85),
            # 文件
            (r'(?:保险|遗嘱|合同|协议|报告|证书|日记|账本|收据|发票|遗嘱)',
             "document", 0.75),
        ]
        
        for pattern, etype, confidence in patterns:
            matches = re.findall(pattern, analysis_text)
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    entities.append({
                        "name": m,
                        "type": etype,
                        "description": f"图片中发现的{etype}: {m}",
                        "confidence": confidence,
                    })
        
        return entities
    
    def _build_cross_modal_links(self, suspect_names: List[str]) -> List[Dict[str, Any]]:
        """
        跨模态关联: 图片实体 ↔ 嫌疑人
        
        多策略匹配:
        1. 直接姓名匹配 (实体名/描述含嫌疑人名)
        2. 图片证据主节点描述匹配 (最常用)
        3. 角色称呼匹配 (如"管家" → 王福来(管家))
        4. 去括号/去头衔后的核心名匹配
        """
        links = []
        seen_links = set()
        
        # 预处理嫌疑人名: 提取核心名和角色
        suspect_parts = []  # [(full_name, core_name, role), ...]
        for sn in suspect_names:
            if not sn:
                continue
            # 去括号: "王福来（管家）" → "王福来", "管家"
            core = re.sub(r'[（(].*?[）)]', '', sn).strip()
            role_match = re.search(r'[（(](.*?)[）)]', sn)
            role = role_match.group(1) if role_match else ""
            suspect_parts.append((sn, core, role))
        
        for node_id, node in self.nodes.items():
            node_text = f"{node.name} {node.description} {node.summary}"
            
            for full_name, core_name, role in suspect_parts:
                # 策略1: 核心名直接匹配 (如"王福来" 在 "王福来（管家）"中)
                if core_name and core_name in node_text:
                    link_key = f"{node_id}_{full_name}_core"
                    if link_key not in seen_links:
                        seen_links.add(link_key)
                        links.append({
                            "image_entity": node.name[:50],
                            "image_entity_type": node.node_type,
                            "suspect_name": full_name,
                            "link_type": "name_in_evidence",
                            "source_node_id": node_id,
                            "confidence": 0.9 if core_name in node.name else 0.7,
                        })
                
                # 策略2: 角色匹配 (如"管家" 在描述中)
                if role and len(role) >= 2 and role in node_text:
                    link_key = f"{node_id}_{full_name}_role"
                    if link_key not in seen_links:
                        seen_links.add(link_key)
                        links.append({
                            "image_entity": node.name[:50],
                            "image_entity_type": node.node_type,
                            "suspect_name": full_name,
                            "link_type": "role_in_evidence",
                            "source_node_id": node_id,
                            "confidence": 0.6,
                        })
                
                # 策略3: 全名匹配 (如"王福来（管家）" 完整出现在描述中)
                if full_name in node_text:
                    link_key = f"{node_id}_{full_name}_full"
                    if link_key not in seen_links:
                        seen_links.add(link_key)
                        links.append({
                            "image_entity": node.name[:50],
                            "image_entity_type": node.node_type,
                            "suspect_name": full_name,
                            "link_type": "full_name_match",
                            "source_node_id": node_id,
                            "confidence": 0.95,
                        })
        
        return links
    
    def _merge_duplicate_entities(self):
        """合并同名实体 — 跨图片的同一实体创建 same_as 关系"""
        name_to_nodes: Dict[str, List[str]] = {}
        for node_id, node in self.nodes.items():
            if node.node_type == "image_evidence":
                continue
            key = f"{node.name}_{node.node_type}"
            name_to_nodes.setdefault(key, []).append(node_id)
        
        for key, node_ids in name_to_nodes.items():
            if len(node_ids) > 1:
                # 创建 same_as 关系
                for i in range(len(node_ids)):
                    for j in range(i + 1, len(node_ids)):
                        self.edges.append(EvidenceEdge(
                            edge_id=self._compute_id(f"same_{node_ids[i]}_{node_ids[j]}", "rel"),
                            source_id=node_ids[i],
                            target_id=node_ids[j],
                            relation_type="same_as",
                            description=f"同一实体: {self.nodes[node_ids[i]].name}",
                            keywords="same_as,equivalent",
                            weight=5.0,
                        ))
    
    def _generate_graph_text(self) -> str:
        """生成图谱的文本描述 — 注入到case_text供ASMR使用"""
        lines = ["\n\n【🕸️ 证据知识图谱】"]

        # 主实体（图片证据）
        image_nodes = [n for n in self.nodes.values() if n.node_type == "image_evidence"]
        if image_nodes:
            lines.append(f"\n📷 图片证据节点 ({len(image_nodes)}张):")
            for node in image_nodes:
                lines.append(f"  • [{node.chunk_id}] {node.name}")
                if node.summary:
                    lines.append(f"    摘要: {node.summary[:100]}")

        # 子实体(按类型分组，包含文本+图片实体)
        sub_nodes = [n for n in self.nodes.values() if n.node_type != "image_evidence"]
        if sub_nodes:
            by_type = {}
            for n in sub_nodes:
                by_type.setdefault(n.node_type, []).append(n)

            type_labels = {
                "suspect": "👤 嫌疑人", "victim": "💀 受害者", "weapon": "🔪 凶器",
                "substance": "⚗️ 物质", "location": "📍 地点", "time": "🕐 时间",
                "document": "📋 文件", "digital_device": "📱 电子设备",
                "physical_evidence": "🔎 物证", "biometric": "🧬 生物特征",
                "measurement": "📏 测量数据", "communication": "📞 通讯记录",
                "financial": "💰 财务记录", "surveillance": "📹 监控",
                "motive": "🎯 动机", "other": "📌 其他",
            }

            lines.append(f"\n🔍 提取的刑侦实体 ({len(sub_nodes)}个):")
            for ntype, nodes in by_type.items():
                label = type_labels.get(ntype, ntype)
                names = [f"{n.name}(置信{n.confidence:.0%})" for n in nodes]
                lines.append(f"  {label}: {', '.join(names)}")

        # 按关系类型分组输出
        if self.edges:
            by_rel = {}
            for e in self.edges:
                by_rel.setdefault(e.relation_type, []).append(e)

            rel_labels = {
                "belongs_to": "📎 归属关系",
                "same_as": "🔄 同一实体",
                "has_motive": "🎯 动机关联",
                "has_opportunity": "🕐 作案机会",
                "has_means": "🔧 作案手段",
                "contradicts": "⚡ 矛盾冲突",
                "proves": "✅ 证据证明",
                "implies": "🔍 暗示推断",
                "alibis": "🛡️ 不在场证明",
                "happened_at": "📍 发生于",
                "located_at": "📍 位于/出现于",
                "contacted_with": "📞 接触/通讯",
                "owns": "💰 拥有",
                "knows": "🔑 知道/掌握",
                "disputes_with": "⚔️ 纠纷",
                "threatens": "⚠️ 威胁",
                "hides": "🔒 隐藏/掩盖",
                "tampered": "🔧 篡改/破坏",
                "witnessed": "👁️ 目击",
                "suspicious_of": "❓ 可疑行为",
            }

            lines.append(f"\n🔗 实体关系 ({len(self.edges)}条):")
            for rel_type, edges in by_rel.items():
                label = rel_labels.get(rel_type, rel_type)
                lines.append(f"  {label} ({len(edges)}条):")
                for edge in edges[:8]:
                    src = self.nodes.get(edge.source_id)
                    tgt = self.nodes.get(edge.target_id)
                    if src and tgt:
                        lines.append(f"    {src.name} → {tgt.name} [{edge.description[:60]}]")
                if len(edges) > 8:
                    lines.append(f"    ... 还有 {len(edges)-8} 条")

        lines.append(f"\n")
        return "\n".join(lines)
    
    def _generate_retrieval_chunks(self) -> List[Dict[str, Any]]:
        """
        生成检索chunks — 对标 RAG-Anything 的 chunks_vdb
        
        每个chunk包含:
        - 图片主实体描述
        - 子实体列表
        - 关键关系
        """
        chunks = []
        
        image_nodes = [n for n in self.nodes.values() if n.node_type == "image_evidence"]
        for img_node in image_nodes:
            # 找到该图片的所有子实体
            children = []
            for edge in self.edges:
                if edge.target_id == img_node.node_id and edge.relation_type == "belongs_to":
                    child = self.nodes.get(edge.source_id)
                    if child:
                        children.append(child)
            
            # 构建chunk文本
            chunk_text = f"图片证据: {img_node.name}\n"
            chunk_text += f"分析: {img_node.description[:500]}\n"
            if children:
                chunk_text += f"包含实体: {', '.join(c.name for c in children)}\n"
            
            chunks.append({
                "chunk_id": img_node.chunk_id,
                "content": chunk_text,
                "image_index": img_node.metadata.get("img_index", 0),
                "entity_count": len(children),
            })
        
        return chunks
    
    def query_graph(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        图谱检索 — 简单关键词匹配 (对标 RAG-Anything 的混合检索)
        
        生产环境应该用向量检索，这里用关键词作为MVP
        """
        results = []
        query_keywords = set(re.findall(r'[\u4e00-\u9fff]{2,}', query))
        
        # 在节点中搜索
        for node_id, node in self.nodes.items():
            score = 0
            node_text = f"{node.name} {node.description} {node.summary}"
            node_chars = set(re.findall(r'[\u4e00-\u9fff]{2,}', node_text))
            
            # 关键词重叠度
            overlap = query_keywords & node_chars
            if overlap:
                score = len(overlap) / max(len(query_keywords), 1)
            
            if score > 0:
                # 获取关联节点
                related = []
                for edge in self.edges:
                    if edge.source_id == node_id:
                        rel_node = self.nodes.get(edge.target_id)
                        if rel_node:
                            related.append({"name": rel_node.name, "rel": edge.relation_type})
                    elif edge.target_id == node_id:
                        rel_node = self.nodes.get(edge.source_id)
                        if rel_node:
                            related.append({"name": rel_node.name, "rel": edge.relation_type})
                
                results.append({
                    "node_id": node_id,
                    "name": node.name,
                    "type": node.node_type,
                    "score": score,
                    "related": related[:5],
                })
        
        # 按score排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_contradiction_hints(self) -> List[Dict[str, Any]]:
        """
        从图谱中检测潜在的图片-文本矛盾
        
        策略:
        1. 同一实体在不同图片中有不同描述
        2. 图片中的时间与文本时间线矛盾
        3. 图片中出现的嫌疑人vs文本中声称的不在场
        """
        contradictions = []
        
        # 找 same_as 实体对，检查描述是否一致
        same_edges = [e for e in self.edges if e.relation_type == "same_as"]
        for edge in same_edges:
            src = self.nodes.get(edge.source_id)
            tgt = self.nodes.get(edge.target_id)
            if src and tgt and src.description != tgt.description:
                contradictions.append({
                    "type": "entity_description_mismatch",
                    "entity": src.name,
                    "desc_1": src.description[:100],
                    "desc_2": tgt.description[:100],
                    "confidence": 0.7,
                })
        
        return contradictions
    
    # ══════════════════════════════════════════════════════════════
    # 文本图谱构建 — 从案件文本提取实体+语义关系
    # ══════════════════════════════════════════════════════════════

    def build_from_text(
        self,
        case_text: str,
        suspects: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        从案件文本构建知识图谱（不依赖图片）

        流程:
        1. LLM 从案件文本提取实体 (人/物/地点/时间/证据)
        2. LLM 从案件文本提取实体间语义关系 (动机/矛盾/证明/时间线...)
        3. 创建嫌疑人与实体间的关联
        4. 合并同名实体
        5. 生成图谱文本

        Returns: 同 build_from_image_analyses 的格式
        """
        suspects = suspects or []
        suspect_names = [s.get("name", "") if isinstance(s, dict) else str(s) for s in suspects]

        # Step 1: 从文本提取实体
        text_entities = self._extract_text_entities(case_text)
        logger.info(f"📝 文本实体提取: {len(text_entities)} 个实体")

        # 创建文本实体节点
        for entity in text_entities:
            entity_id = self._compute_id(f"text_{entity['name']}_{entity['type']}", "ent")
            if entity_id not in self.nodes:
                self.nodes[entity_id] = EvidenceNode(
                    node_id=entity_id,
                    name=entity["name"],
                    node_type=entity["type"],
                    description=entity.get("description", ""),
                    source_image="",
                    confidence=entity.get("confidence", 0.8),
                    summary=entity.get("description", "")[:100],
                    chunk_id=self._compute_id(case_text[:500], "chunk"),
                    metadata={"source": "text"},
                )

        # Step 2: 从文本提取语义关系
        text_relations = self._extract_text_relations(case_text, text_entities)
        logger.info(f"📝 文本关系提取: {len(text_relations)} 条关系")

        for rel in text_relations:
            src_id = self._find_entity_id(rel.get("source", ""))
            tgt_id = self._find_entity_id(rel.get("target", ""))
            if src_id and tgt_id and src_id != tgt_id:
                edge_id = self._compute_id(f"txtrel_{src_id}_{tgt_id}_{rel['relation']}", "rel")
                # 避免重复边
                existing = {(e.source_id, e.target_id, e.relation_type) for e in self.edges}
                if (src_id, tgt_id, rel["relation"]) not in existing:
                    self.edges.append(EvidenceEdge(
                        edge_id=edge_id,
                        source_id=src_id,
                        target_id=tgt_id,
                        relation_type=rel["relation"],
                        description=rel.get("description", ""),
                        keywords=rel.get("relation", ""),
                        weight=rel.get("weight", 5.0),
                        source_id_chunk=self._compute_id(case_text[:500], "chunk"),
                    ))

        # Step 3: 嫌疑人 → 实体关联
        self._link_suspects_to_entities(suspect_names)

        # Step 4: 合并同名实体
        self._merge_duplicate_entities()

        # Step 5: 生成图谱文本
        graph_text = self._generate_graph_text()

        # Step 6: 检索 chunks
        retrieval_chunks = self._generate_retrieval_chunks()

        return {
            "nodes": [{"id": n.node_id, "name": n.name, "type": n.node_type,
                        "desc": n.description[:100], "summary": n.summary,
                        "chunk_id": n.chunk_id} for n in self.nodes.values()],
            "edges": [{"id": e.edge_id, "src": e.source_id, "tgt": e.target_id,
                        "rel": e.relation_type, "desc": e.description,
                        "keywords": e.keywords, "weight": e.weight} for e in self.edges],
            "graph_text": graph_text,
            "cross_modal_links": [],
            "retrieval_chunks": retrieval_chunks,
            "stats": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "text_entities": len(text_entities),
                "text_relations": len(text_relations),
                "retrieval_chunks": len(retrieval_chunks),
            }
        }

    def _extract_text_entities(self, case_text: str) -> List[Dict[str, Any]]:
        """LLM 从案件文本提取刑侦实体"""
        if not self.llm_client:
            return self._regex_extract_entities(case_text, "案件文本", 0)

        prompt = f"""你是一名刑侦证据分析师。请从以下案件文本中提取所有具有刑侦价值的实体。

案件文本:
{case_text}

请提取所有关键实体，严格按以下JSON格式返回:
{{
  "entities": [
    {{
      "name": "实体名称(必须具体，如'氰化钾'而非'毒物')",
      "type": "实体类型(从以下选择: suspect/victim/weapon/substance/location/time/document/digital_device/physical_evidence/vehicle/communication/financial/biometric/measurement/surveillance/motive/other)",
      "description": "该实体在案件中的具体角色和意义(一句话)",
      "confidence": 0.9
    }}
  ]
}}

提取规则:
1. 提取所有具体人名 → type=suspect (如果明显是死者用 victim)
2. 提取所有化学物质、药物、毒物 → type=substance
3. 提取所有具体时间点 → type=time
4. 提取所有地点 → type=location
5. 提取所有数字/测量数据 → type=measurement
6. 提取所有武器/工具 → type=weapon
7. 提取所有通讯记录(短信/通话) → type=communication
8. 提取所有监控/摄像头相关 → type=surveillance
9. 提取所有生物特征(DNA/指纹) → type=biometric
10. 提取所有文件/遗嘱/合同 → type=document
11. 提取所有财务/金钱相关 → type=financial
12. 提取明显的动机描述 → type=motive
13. 不要提取抽象概念，只提取具体可操作的实体"""

        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            if isinstance(response, str):
                return self._robust_json_parse(response)
            return []
        except Exception as e:
            logger.warning(f"文本实体提取失败: {e}, 使用正则回退")
            return self._regex_extract_entities(case_text, "案件文本", 0)

    def _extract_text_relations(
        self, case_text: str, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """LLM 从案件文本提取实体间的语义关系"""
        if not self.llm_client or not entities:
            return []

        # 拼接实体名供 LLM 参考
        entity_names = [e["name"] for e in entities]
        entity_list_str = "\n".join(f"  - {name} ({e['type']})" for name, e in zip(entity_names, entities))

        prompt = f"""你是一名刑侦图谱分析师。请从以下案件文本中提取实体之间的关系。

案件文本:
{case_text}

已识别的实体:
{entity_list_str}

请提取所有实体间的刑侦关系，严格按以下JSON格式返回:
{{
  "relations": [
    {{
      "source": "源实体名称(必须是上面列出的实体)",
      "target": "目标实体名称(必须是上面列出的实体)",
      "relation": "关系类型(从以下选择: has_motive/has_opportunity/has_means/contradicts/proves/implies/alibis/happened_at/located_at/contacted_with/owns/knows/disputes_with/threatens/hides/tampered/witnessed/suspicious_of)",
      "description": "关系描述(一句话说明这条关系在案件中的意义)",
      "weight": 7.0
    }}
  ]
}}

关系类型说明:
- has_motive: A有杀害B的动机
- has_opportunity: A有机会作案(时间/地点上)
- has_means: A有作案手段(技能/工具/接触途径)
- contradicts: A与B的陈述矛盾
- proves: A证明B的某个行为或状态
- implies: A暗示B的某个可能
- alibis: A为B提供不在场证明
- happened_at: 事件A发生在地点/时间B
- located_at: A位于B处
- contacted_with: A与B有接触/通讯
- owns: A拥有B
- knows: A知道B(如密码/秘密)
- disputes_with: A与B有纠纷
- threatens: A威胁B
- hides: A隐藏/掩盖B
- tampered: A篡改/破坏了B
- witnessed: A目击了B
- suspicious_of: A对B有可疑行为

权重规则:
- 直接证据(motivation/opportunity/means): 8-10
- 间接证据(implies/knows): 5-7
- 弱关联(suspicious_of): 3-5"""

        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            if isinstance(response, str):
                return self._robust_json_parse_relations(response)
            return []
        except Exception as e:
            logger.warning(f"文本关系提取失败: {e}")
            return []

    def _robust_json_parse_relations(self, response: str) -> List[Dict[str, Any]]:
        """健壮的JSON关系解析"""
        cleaned = re.sub(r'```(?:json)?\s*', '', response)
        cleaned = re.sub(r'```', '', cleaned)

        # Strategy 1: 直接解析
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "relations" in data:
                return [r for r in data["relations"] if isinstance(r, dict)
                        and "source" in r and "target" in r and "relation" in r]
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: 提取 JSON 块
        json_blocks = re.findall(r'\{[\s\S]*\}', response)
        for block in json_blocks:
            try:
                data = json.loads(block)
                if isinstance(data, dict) and "relations" in data:
                    return [r for r in data["relations"] if isinstance(r, dict)
                            and "source" in r and "target" in r and "relation" in r]
            except (json.JSONDecodeError, ValueError):
                continue

        return []

    def _find_entity_id(self, entity_name: str) -> Optional[str]:
        """根据实体名查找节点ID（精确+模糊匹配）"""
        if not entity_name:
            return None

        # 精确匹配
        for nid, node in self.nodes.items():
            if node.name == entity_name:
                return nid

        # 包含匹配
        for nid, node in self.nodes.items():
            if entity_name in node.name or node.name in entity_name:
                return nid

        # 核心名匹配（去括号/头衔后）
        core = re.sub(r'[（(].*?[）)]', '', entity_name).strip()
        if core:
            for nid, node in self.nodes.items():
                node_core = re.sub(r'[（(].*?[）)]', '', node.name).strip()
                if core == node_core:
                    return nid

        return None

    def _link_suspects_to_entities(self, suspect_names: List[str]):
        """为嫌疑人创建与图谱实体的关联边"""
        for suspect in suspect_names:
            # 核心名
            core = re.sub(r'[（(].*?[）)]', '', suspect).strip()
            if not core:
                continue

            # 找图谱中匹配的嫌疑人节点
            suspect_id = self._find_entity_id(suspect)
            if not suspect_id:
                continue

            # 为嫌疑人关联动机/物证/地点等
            for nid, node in self.nodes.items():
                if nid == suspect_id:
                    continue

                # 嫌疑人 → 地点: located_at
                if node.node_type == "location":
                    edge_id = self._compute_id(f"loc_{suspect_id}_{nid}", "rel")
                    existing = {(e.source_id, e.target_id, e.relation_type) for e in self.edges}
                    if (suspect_id, nid, "located_at") not in existing:
                        self.edges.append(EvidenceEdge(
                            edge_id=edge_id,
                            source_id=suspect_id,
                            target_id=nid,
                            relation_type="located_at",
                            description=f"{suspect}与地点{node.name}有关联",
                            keywords="located_at,presence",
                            weight=4.0,
                        ))

                # 嫌疑人 → 动机: has_motive
                if node.node_type == "motive":
                    edge_id = self._compute_id(f"mot_{suspect_id}_{nid}", "rel")
                    existing = {(e.source_id, e.target_id, e.relation_type) for e in self.edges}
                    if (suspect_id, nid, "has_motive") not in existing:
                        self.edges.append(EvidenceEdge(
                            edge_id=edge_id,
                            source_id=suspect_id,
                            target_id=nid,
                            relation_type="has_motive",
                            description=f"{suspect}有{node.name}的动机",
                            keywords="has_motive,motive",
                            weight=8.0,
                        ))

    def reset(self):
        """重置图谱"""
        self.nodes.clear()
        self.edges.clear()
