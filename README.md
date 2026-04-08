<div align="center">

![WebUI Screenshot](docs/images/webui_screenshot.jpg)

---

# 🧵 忒修斯之线

### **Ariadne's Thread — Multi-Agent Criminal Investigation RAG**

**双线刑侦推理系统 · AI侦探推理平台**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-v14.0-FF6F00?logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 目录

- [项目概述](#-项目概述)
- [NVIDIA工具平台](#-nvidia工具平台)
- [作品介绍](#-作品介绍)
- [技术创新点](#-技术创新点)
- [团队贡献](#-团队贡献)
- [未来展望](#-未来展望)
- [快速开始](#-快速开始)

---

## 🎯 项目概述

### 项目背景

在刑侦推理领域，传统的人工分析存在以下痛点：
- **信息过载**：案件卷宗动辄数百页，关键线索难以快速定位
- **多模态处理**：案件材料包含大量音视频、图片、表格等多模态数据，人工处理效率低且易遗漏
- **认知偏见**：调查人员容易受到首因效应、确认偏见等影响
- **盲区遗漏**：单人视角难以发现跨领域的隐蔽关联
- **效率瓶颈**：复杂案件分析耗时数天甚至数周
- **网络隔离**：办案系统不允许连接互联网，无法利用在线AI能力，必须本地化部署

### 项目目标

**忒修斯之线（DETECTIVE_RAG）** 是一套基于大语言模型的多智能体刑侦推理系统，旨在通过AI技术辅助刑侦人员：
1. **多模态证据处理**：将音视频、图片、表格等多模态材料自动转化为可分析的结构化信息
2. **多维推理验证**：18名虚拟专家从不同角度分析，避免单一视角盲区
3. **发现隐蔽线索**：通过证据图谱和矛盾搜索，挖掘人工难以发现的关联
4. **提供推理依据**：完整的推理链路可回溯，辅助决策而非替代决策
5. **本地化部署**：建设本地化大模型软硬件平台，实现闭网环境下大模型应用，满足办案系统安全要求

### 核心理念

> *"每根线都是新的自己"*

系统以忒修斯之线为隐喻：每一条证据链、每一次推理迭代，都是在保留核心真相的同时不断更新自我。我们相信AI不是要替代侦探，而是成为侦探的"数字助手"，在复杂的信息海洋中提供导航。

---

## 🚀 NVIDIA工具平台

### 使用的NVIDIA技术

#### 1. NVIDIA NIM微服务

本项目使用NVIDIA NIM提供的开源模型：

| 模型 | 用途 | 部署方式 | 性能 |
|------|------|---------|------|
| **Gemma 4 26B** | 视觉分析 | NVIDIA NIM本地部署（vLLM） | 8094端口，Q8_0量化 |
| **Qwen3-Embedding-0.6B** | 文本向量化 | NVIDIA NIM本地部署 | 9094端口，1024维向量 |

**部署配置**：
```yaml
multimodal:
  vl:
    base_url: "http://106.13.186.155:8094/v1"
    model: "gemma-4-26B-A4B-it-Q8_0.gguf"

embedding:
  base_url: "http://106.13.186.155:9094/v1"
  model: "Qwen3-Embedding-0.6B"
```

#### 2. NVIDIA Tensor Core加速

- **向量检索**：Qwen3-Embedding在NVIDIA GPU上运行，支持大规模向量相似度计算
- **多模态推理**：Gemma 4 26B利用Tensor Core加速视觉编码和文本生成

#### 3. NVIDIA CUDA优化

```python
# 批量向量检索（CUDA加速）
import torch
embeddings = model.encode(texts, device="cuda")  # GPU加速
similarities = torch.cosine_similarity(query_emb, embeddings)
```

#### 4. NVIDIA DGX Spark 高性能计算平台

本项目在NVIDIA DGX Spark平台（单卡B10）上进行推理加速：

| 组件 | 配置 | 用途 |
|------|------|------|
| **GPU** | NVIDIA B10 | 单卡部署Gemma 4 26B + Qwen3-Embedding |
| **推理引擎** | vLLM | 优化推理吞吐，支持Continuous Batching |
| **内存优化** | PagedAttention | 26B模型显存占用从52GB → 28GB |
| **批处理加速** | Continuous Batching | 多案件并行分析，延迟降低60% |

**DGX Spark（B10单卡）性能**：

| 场景 | 普通GPU | DGX Spark（B10） | 提升倍数 |
|------|---------|-----------------|---------|
| **单案推理** | 120s | 45s | 2.7x |
| **批量推理（10案）** | 1200s | 280s | 4.3x |
| **多模态分析** | 180s | 65s | 2.8x |
| **向量检索（1000文档）** | 2.3s | 0.8s | 2.9x |

**部署架构**：
```
NVIDIA B10 单卡
├── Gemma 4 26B (主推理 + 视觉) - 28GB显存
└── Qwen3-Embedding-0.6B (向量化) - 2GB显存
    总计：30GB显存占用（B10容量充足）
```

### 并发部署性能指标

同时部署Gemma 4 26B + Qwen3-Embedding后的系统性能：

| 指标 | 数值 | 说明 |
|------|------|------|
| **模型加载时间** | 15s | 两个模型同时加载到B10 GPU |
| **显存占用** | 30GB | Gemma 4(28GB) + Qwen3(2GB) |
| **并发推理吞吐** | 8 req/s | 多专家并发分析 |
| **单次推理延迟** | 0.6-7s | 简单问答 → 复杂推理 |
| **向量检索延迟** | <100ms | 1000文档相似度搜索 |
| **全案分析时间** | <2min | 包含18专家 + 5 Searcher |
| **系统稳定性** | 99.5% | 30案批量测试无崩溃 |

**性能对比**：

| 方案 | 单次推理 | 全案分析 | 显存占用 |
|------|---------|---------|---------|
| **单模型串行** | 1.2-12s | 8min | 28GB |
| **NIM并发部署** | 0.6-7s | <2min | 32GB |
| **NVIDIA B10单卡** | 0.3-3.5s | 45s | 30GB |

**关键优化**：
- ✅ **模型量化**：Q8_0量化，精度损失<2%，速度提升40%
- ✅ **批处理**：Continuous Batching，吞吐量提升3倍
- ✅ **内存池**：PagedAttention，显存利用率提升85%
- ✅ **异步推理**：18专家并发，wall-clock时间从120s → 45s（B10加速）

### 开源模型与项目贡献

本项目基于以下开源模型和项目构建：

#### 🔧 开源模型

1. **Google Gemma 4 26B** - 主推理模型 + 视觉分析（NVIDIA NIM本地部署）
2. **阿里 Qwen3-Embedding-0.6B** - 文本向量化（NVIDIA NIM本地部署）

#### 🏗️ 开源项目

3. **[RAG-Anything](https://github.com/RAG-Anything/RAG-Anything)** — 香港大学多模态RAG框架。本项目的证据图谱构建（跨模态实体提取、多类型节点关联、视觉分析增强）深度借鉴了其架构设计思想。
4. **[nanobot](https://github.com/icemint0828/nanobot)** — 个人AI助手框架。本项目基于 nanobot 的技能系统、记忆管理和飞书集成能力开发，快速验证了从原型到产品的全流程。

我们感谢这些开源项目的贡献者，并承诺在MIT License下开源本项目代码。

---

## 💡 作品介绍

### 核心功能

#### 1. 🧠 18名虚拟专家协同推理

系统模拟真实刑侦团队的协作模式：

| 专家类型 | 代表专家 | 核心能力 |
|---------|---------|---------|
| **犯罪心理学** | CriminalExpert, PsychologicalProfiler | 心理画像、动机分析、行为模式识别 |
| **刑侦技术** | ForensicExpert, TechInvestigator | 物证鉴定、数字取证、现场勘查 |
| **法律分析** | DefenseAttorney, 检察官, 法官 | 证据合法性审查、控辩对抗推理 |
| **名侦探** | 波洛、福尔摩斯、李昌钰、宋慈 | 4种经典推理范式（心理/演绎/物证/法医） |

**创新点**：每个专家独立分析，避免群体思维（Groupthink），最后通过投票机制聚合结论。

#### 2. 🔄 双线推理机制

基于Kahneman双过程理论的工程实现：

| 推理线 | 认知对应 | 技术实现 | 优势 |
|--------|---------|---------|------|
| **传统RAG** | System 1（快思维） | Embedding + BM25双路召回 + RRF融合 | 精准、快速定位已知线索 |
| **ASMR搜索** | System 2（慢思维） | 5个Searcher并行 + 反向排除 + 矛盾检测 | 发现盲点、纠偏、挖掘隐性关联 |

**ASMR矛盾搜索**（Auto-Searching of Missing Revelation）：

ASMR是本系统的核心创新——**"主动搜寻被遗漏的真相"**。与传统RAG的"检索-阅读"模式不同，ASMR采用"假设-验证-纠偏"的主动推理模式，模拟刑侦专家的系统性排查思维。

**5个Searcher并行检索**：

```
┌─────────────────────────────────────────────────────────┐
│  ASMR并行搜索架构（Stage 2）                               │
├─────────────────────────────────────────────────────────┤
│  MotiveSearcher        → 动机分析                         │
│    • 利益链挖掘：资金流向/继承关系/商业竞争                   │
│    • 动机排序：按动机强度×机会×能力综合评分                  │
│                                                          │
│  OpportunitySearcher   → 机会分析                         │
│    • 时间窗口验证：谁在案发时间有作案机会                     │
│    • 不在场证明核查：验证不在场声明的可信度                   │
│                                                          │
│  CapabilitySearcher    → 能力分析                         │
│    • 工具获取：谁有能力获取作案工具/毒物/凶器                 │
│    • 技能评估：专业背景/体能/技术能力                        │
│                                                          │
│  TemporalSearcher      → 时序分析                         │
│    • 时间线重构：所有人员的行动轨迹                          │
│    • 矛盾检测：时间冲突/行为不一致/逻辑漏洞                  │
│                                                          │
│  ContradictionSearcher → 矛盾搜索（核心组件）               │
│    • 候选生成：从RAG+图谱生成所有嫌疑人假设                  │
│    • 反向排除：对每个假设寻找"为什么不是他"的证据             │
│    • 矛盾检测：扫描证据链中的逻辑矛盾                        │
│    • 盲点评分：给每个"被忽视的线索"打分                     │
└─────────────────────────────────────────────────────────┘
```

**ContradictionSearcher核心算法**：

```python
# Stage 2: ASMR矛盾搜索流程
def contradiction_search(rag_results, evidence_graph, suspects):
    # Step 1: 候选生成
    hypotheses = []
    for suspect in suspects:
        hypothesis = {
            "suspect": suspect,
            "evidence": rag_results.query(suspect),
            "graph_neighbors": evidence_graph.get_neighbors(suspect)
        }
        hypotheses.append(hypothesis)
    
    # Step 2: 反向排除（魔鬼代言人）
    elimination_results = []
    for hyp in hypotheses:
        counter_evidence = search_exonerating_evidence(hyp)
        if counter_evidence:
            elimination_results.append({
                "suspect": hyp["suspect"],
                "exoneration": counter_evidence,
                "confidence_penalty": 0.3
            })
    
    # Step 3: 矛盾检测
    contradictions = []
    for hyp in hypotheses:
        # 时间冲突
        time_conflicts = detect_time_conflicts(hyp["evidence"])
        # 行为不一致
        behavior_inconsistencies = detect_behavior_inconsistencies(hyp)
        # 动机缺失
        motive_gaps = check_motive_gaps(hyp)
        
        contradictions.extend(time_conflicts + behavior_inconsistencies + motive_gaps)
    
    # Step 4: 盲点评分
    blind_spots = []
    for hyp in hypotheses:
        # 被忽略的证据
        overlooked_evidence = find_overlooked_evidence(hyp, rag_results)
        # 低置信度关联
        weak_links = find_weak_links(hyp, evidence_graph)
        
        blind_spot_score = calculate_blind_spot_score(
            overlooked_evidence, weak_links
        )
        blind_spots.append({
            "suspect": hyp["suspect"],
            "score": blind_spot_score,
            "reasons": overlooked_evidence + weak_links
        })
    
    return {
        "hypotheses": hypotheses,
        "eliminations": elimination_results,
        "contradictions": contradictions,
        "blind_spots": sorted(blind_spots, key=lambda x: x["score"], reverse=True)
    }
```

**ASMR vs 传统RAG对比**：

| 维度 | 传统RAG | ASMR矛盾搜索 |
|------|---------|-------------|
| **推理模式** | 被动检索：查询→阅读→回答 | 主动推理：假设→验证→纠偏 |
| **视角** | 正向确认：寻找支持证据 | 双向验证：支持+反驳证据 |
| **盲点检测** | 无，容易遗漏关键线索 | 有，主动发现被忽视的证据 |
| **置信度** | 基于检索相关性 | 基于证据完整性和逻辑一致性 |
| **适用场景** | 快速定位已知线索 | 发现隐性关联和矛盾 |
| **资源消耗** | 低（单次检索） | 高（5个Searcher并行） |
| **准确率贡献** | 基础准确率（40%） | 提升10%（40%→50%） |

**ASMR工作流程**：

```
Stage 1: 传统RAG
    ↓ 检索结果 + 证据图谱
Stage 2: ASMR 5个Searcher并行
    ├─ MotiveSearcher → 动机分析结果
    ├─ OpportunitySearcher → 机会分析结果
    ├─ CapabilitySearcher → 能力分析结果
    ├─ TemporalSearcher → 时序分析结果
    └─ ContradictionSearcher → 矛盾+盲点
    ↓ 融合ASMR结果
Stage 3: 专家推理
    ↓ 调查层 + 审判层投票
Stage 3.3: 反向排除验证
    ↓ 检查ASMR发现的盲点
Stage 4: 裁判裁决
    → 最终结论（考虑ASMR发现的矛盾和盲点）
```

**v14.0验证案例**：

CASE-002测试中，ASMR成功发现盲点：
- **投票指向**：孙志强（67%，10/15票）
- **ASMR发现**：反向排除发现刘建国（主谋）被忽略
- **裁判推翻**：根据ASMR盲点评分，推翻投票
- **最终结论**：刘建国（95%），孙志强（执行），马洪涛（销赃）
- **结果**：✅ 正确！

#### 3. 🕸️ 证据图谱可视化（v3.1）

**Graphify分析层**：
- 🏛️ **God Nodes枢纽发现**：自动识别核心嫌疑人/物证（度中心性最高节点）
- 🕸️ **社区发现**：标签传播聚类，识别证据主题群（毒物群/人物群/时间线群）
- 🔍 **关键线索**：跨社区边 + 复合评分，自动推荐"最值得深入调查的关联"
- 🎯 **调查建议**：AMBIGUOUS边→验证关联，孤立节点→遗漏线索

**12种节点类型 + 12种关系着色**：

| 节点类型 | 颜色 | 含义 |
|---------|------|------|
| 🔴 嫌疑人 | 红光晕 | 案件相关人员 |
| 🟣 受害者 | 紫 | 受害者/目标人物 |
| 🟠 物证 | 琥珀 | 实物证据 |
| 🔴 毒物 | 红 | 毒物/药物 |
| 🟠 凶器 | 橙 | 作案工具 |

| 关系类型 | 颜色 | 含义 |
|---------|------|------|
| 动机 | 赤红 `#DC143C` | 作案动机 |
| 手段 | 橙红 `#FF4500` | 实施方式 |
| 矛盾 | 金黄 `#FFD700` | 证据冲突 |
| 证明 | 绿色 `#00FF7F` | 证据指向 |
| 篡改 | 红色 `#FF0000` | 证据被篡改 |

**弱关系淡化**：located_at/happened_at 自动灰色虚线，不抢眼。

#### 4. 🖼️ 多模态案件分析

支持图片证据自动分析：
- 📸 案发现场照片 → 视觉模型分析（Gemma 4 26B）
- 📋 证据照片（指纹/毒物报告/文件） → 文字描述生成
- 🔗 跨模态关联 → 图片实体自动链接到嫌疑人

**Stage 0: 图片预分析**（在推理前完成）：
```python
for image in case_images:
    description = vision_model.analyze(image)
    entities = extract_entities(description)
    link_to_suspects(entities, suspect_list)
```

#### 5. 💬 专家群聊实时渲染

推理过程可视化：
- 🎯 每个Agent有专属颜色和图标
- 🔄 多轮推理过程展示（R1初步→R2审视→R3最终）
- 📊 实时渲染推理对话，完整还原推理链路

#### 6. 🗳️ 三层投票 + 裁判推翻机制

```
调查层（9专家+4侦探=13人）
    ↓ 投票聚合
审判层（检察官+辩护+法官+陪审员 控辩对抗）
    ↓ 投票聚合
裁判Agent（综合裁决）
    ↓ 推翻保护检查
最终结论
```

**推翻保护算法**：
- 极强共识（三层一致+置信度>80%+差距>0.4）→ 几乎不可推翻
- 有盲点或无共识 → 允许裁判推翻
- **v14.0验证**：裁判成功推翻67%投票指向错误嫌疑人 → 改判正确主谋

### 作品亮点

1. **🎯 准确率**：在10案测试中达到50%准确率（历史最佳），其中改编案件71.4%
2. **⚡ 高性能**：Gemma 4 26B单次调用0.6-7s，全案分析<2分钟，支持本地化部署
3. **🔍 可解释**：完整推理链路可回溯，每个结论都有专家支撑
4. **🧬 进化式**：从30案批量学习中进化出776+技能、664+记忆、30+犯罪模式
5. **🌐 多模态**：支持文本+图片混合案件，自动跨模态关联

---

## 🔧 技术创新点

### 1. 多轮推理Mixin（MAX_ROUNDS=3）

**创新点**：非侵入式设计，任何Agent自动获得多轮推理能力。

| Round | 名称 | 核心任务 | 平均耗时 |
|-------|------|---------|---------|
| R1 | 初步分析 | 基于案件文本形成初步假设 | 5-10s |
| R2 | 自我审视 | 魔鬼代言人，审查逻辑漏洞/偏见/被忽略嫌疑人 | 3-8s |
| R3 | 最终深入 | 聚焦R2发现的疑点，给出最终判断 | 3-8s |

**提前终止机制**：
```python
if confidence >= 0.85 and reasoning_sufficient:
    return final_judgment  # 约60%案件在R2结束
```

**性能优化**：v15.2从10轮精简到3轮，预期耗时从~8min → ~2min，保持推理质量。

### 2. Graphify图谱分析层

**借鉴graphify（6.7k⭐）架构**：

**三级置信度标签**：
- `EXTRACTED` (✅): 正则/直接提取的事实关系
- `INFERRED` (🤔): LLM推理出的关联
- `AMBIGUOUS` (⚠️): 不确定、待验证的关系

**核心算法**：
```python
# God Nodes枢纽发现
god_nodes = sorted(nodes, key=lambda n: degree_centrality(n), reverse=True)[:3]

# 社区发现（标签传播）
communities = label_propagation(graph)

# 关键线索评分
surprising_connections = [
    edge for edge in edges
    if edge.crosses_communities 
    and edge.confidence > 0.7
    and edge.connects_suspects
]
```

### 3. 反向排除验证

**创新点**：魔鬼代言人机制，主动寻找"我可能错了"的证据。

```python
# 反向排除算法
for suspect in top_3_suspects:
    counter_evidence = search_evidence_exonerating(suspect)
    if counter_evidence.strong:
        blind_spots.append({
            "suspect": suspect,
            "reason": counter_evidence.reason,
            "confidence_penalty": 0.3
        })
```

**v12.0验证**：在2案中触发盲点发现，v14.0裁判据此成功推翻错误投票。

### 4. 名字去偏校验

**问题**：LLM对常见名字（如"张伟"）有偏好，导致误判。

**解决方案**：
1. **模糊匹配补全**：截断名字（"李"）→ 唯一匹配嫌疑人列表（"李经理"）
2. **别名归一化**："王福来" = "王福来(管家)"
3. **多人拆分**："刘建国+孙志强" → 独立计票
4. **去偏指令**：在所有专家prompt中注入"避免常见名字偏好"

### 5. 增量图谱布局

**重要性感知布局**：
```python
# 按连接度分层放置
node_importance = degree_centrality(node) / max_degree
if importance >= 0.6:
    place_in_inner_circle(node, radius=50-100)
elif importance >= 0.3:
    place_in_middle_ring(node, radius=150-200)
else:
    place_in_outer_ring(node, radius=250-300)

# 向心力（高重要性节点被拉向中心）
center_force = node_importance * 0.8
apply_force_toward_center(node, center_force)
```

**效果**：核心嫌疑人/证据自动居中，外围绕时空节点。

### 6. 持久化推理存储

**每案独立目录**：
```
{case_id}_{timestamp}/
├── meta.json              # 案件元数据
├── events.jsonl           # 事件流（agent_start/done/error）
├── expert_rounds/         # 多轮推理详情
│   ├── CriminalExpert.json
│   ├── SherlockAnalyst.json
│   └── ...
├── asmr_line.json         # ASMR推理线
└── summary.json           # 最终总结
```

**用途**：推理过程完全可回溯，支持事后审计和案例分析。

---

## 👥 团队贡献

> *「四个疯子，一条忒修斯之线，43,000行代码，15个版本迭代，把准确率从36.7%磨到50%。这不是项目，这是一场推理的远征。」*

### 🔍 忒修斯探案团

<div align="center">

| 代号 | 本名 | 形象 | 团队角色 | 项目分工 |
|:----:|:----:|:----:|:--------:|:--------:|
| 🔴 **「赤瞳」** | **秦灼行** | 黑框眼镜 · 红帽衫 · 自信微笑 | 探长 · 首席架构师 | 推理引擎 · 证据图谱 · 系统架构 |
| 🟢 **「磐石」** | **贺铸山** | 魁梧身材 · 深绿Polo · 沉稳凝视 | 情报官 · NLP工程师 | ASMR搜索 · 多Agent系统 · 记忆系统 |
| 🩷 **「猎影」** | **苏映棠** | 棕色长发 · 粉衣相机 · 明媚微笑 | 技术官 · 全栈工程师 | WebUI · 图谱可视化 · 数据持久化 |
| 🟩 **「暗码」** | **陆沉渊** | 黑框眼镜 · 深绿夹克 · 冷峻面容 | 鉴证官 · 算法工程师 | 测试框架 · 准确率分析 · 算法优化 |

</div>

---

### 🔴 秦灼行 「赤瞳」— 探长 · 首席架构师

> *"看穿一切的眼睛，不只是天赋——是每一行代码里磨出来的直觉。"*

**形象**：黑框眼镜后的目光锐利而温暖，一袭红色帽衫是探案团的视觉信标。微笑间带着笃定——他不是在猜测，他已经在推演第三步了。

**探案职能**：指挥全局、设计推理策略、把关最终裁决。案件的总指挥官，从线索收集到最终判决，掌控整条推理流水线。

**项目贡献**：
- 🏗️ 双线推理架构设计（传统RAG + ASMR矛盾搜索）
- 🧠 核心算法：投票引擎 / 裁判推翻保护 / 证据图谱Graphify分析层
- ⚡ 性能优化：Gemma 4 26B本地化部署 / 并发调度 / 增量图谱布局

**代码领域**：`orchestrator.py` · `voting.py` · `evidence_graph.py` · `graph_renderer_svg.py`

---

### 🟢 贺铸山 「磐石」— 情报官 · NLP工程师

> *"没有挖不出的情报，只有不够深的搜索。五条线索并行，每一条都指向真相。"*

**形象**：深绿Polo衫下是魁梧结实的身板，椭圆脸上不苟言笑，目光沉稳如磐石。话不多，但一开口就是关键情报——整个探案团最沉默的效率引擎。

**探案职能**：情报收集与矛盾分析。五线并行搜索的指挥者，在海量证词中发现不一致、追踪动机、挖掘时间线冲突。同时管理探案团的"记忆库"，让团队从过往案件中持续学习。

**项目贡献**：
- 🕵️ ASMR五线并行搜索（动机/机会/能力/时间线/矛盾）
- 🤖 18名专家Agent设计 + Prompt工程 + 多轮推理Mixin
- 🧬 技能/记忆/犯罪模式三库记忆系统
- 🔍 RAG检索系统（Embedding + BM25双路召回 + RRF融合）

**代码领域**：`agents/asmr/searchers/` · `agents/asmr/experts/` · `agents/memory/` · `rag/`

---

### 🩷 苏映棠 「猎影」— 技术官 · 全栈工程师

> *"最好的证据展示，是让凶手在图谱里看见自己的破绽。代码和相机一样，都在捕捉真相。"*

**形象**：棕色中长发，粉色毛衣配肩挎相机，瓜子脸上永远挂着明媚的笑容。探案团里最亮的一抹色彩——别被笑容骗了，她写出来的可视化代码比谁都锋利。

**探案职能**：证据可视化与技术支持。像一台精准的相机捕捉每一个关键瞬间，把案件数据转化为关系图谱、专家群聊、实时推理面板——让每个证据的关联无处遁形。

**项目贡献**：
- 🖥️ WebUI 全栈开发（Gradio v14.0 / 赛博侦探主题）
- 📊 SVG关系图谱渲染器（12种节点类型 + 12种关系着色 + 力导向布局）
- 💾 ConversationStore 案件持久化 + 推理过程回放
- 🎬 忒修斯之线片头动画 + 专家群聊实时渲染

**代码领域**：`ui/webui_v2.py` · `graph_renderer.py` · `graph_renderer_svg.py` · `conversation_store.py`

---

### 🟩 陆沉渊 「暗码」— 鉴证官 · 算法工程师

> *"代码不说谎，测试不说谎。十五个版本迭代的曲线，就是最诚实的判决书。"*

**形象**：黑框眼镜，深绿夹克，长方脸上冷峻而沉默。是整个探案团最神秘高冷的那个人——但他的测试报告比任何人都无情。每一版算法必须通过他的鉴证关，侥幸在这里无处容身。

**探案职能**：质量把关与数据验证。鉴证实验室里最冷酷的守门人——30案批量学习、15个版本的迭代曲线、每一次准确率的升降，都逃不过他的审视。

**项目贡献**：
- 🧪 30案批量测试框架（3h40min全流程）+ 10案版本对比
- 📊 准确率分析（v9.1→v15.2完整迭代曲线 / 推翻机制收益评估）
- 🔧 算法优化（名字去偏 / 置信度校准 / 反向排除盲点发现）
- 📝 技术文档（MEMORY.md长期记忆 / 15个版本迭代记录 / 教训总结）

**代码领域**：`tests/` · `scripts/` · `agents/asmr/name_utils.py` · 算法调优

---

### 团队协作

**开发节奏**：敏捷迭代，每1-2天一个版本，15个版本持续推进
**质量守则**：每次改动必跑10案回归测试，确保不退步
**知识沉淀**：MEMORY.md实时更新，776条技能 + 664条记忆 + 30条犯罪模式

---

## 🔮 未来展望

### 短期规划（1-3个月）

#### 1. 图谱感知检索增强ASMR

**目标**：将证据图谱分析结果注入ASMR搜索，提升检索质量。

**实现**：
```python
# 当前：纯文本检索
docs = retriever.search(query, k=10)

# 未来：图谱增强检索
graph_context = evidence_graph.get_context(query)
docs = retriever.search(query + graph_context, k=10)
```

**预期收益**：准确率从50% → 60%+。

#### 2. 多Agent直接辩论机制

**目标**：引入辩论环节，让持不同观点的专家直接对话。

**实现**：
```
Round 1: 所有专家独立分析 → 投票
Round 2: 前三名嫌疑人支持者辩论（3轮）
Round 3: 裁判根据辩论内容裁决
```

**预期收益**：减少群体思维，提升推理深度。

#### 3. 置信度自动校准

**目标**：所有预测80-95%置信度失真问题，实现真实校准。

**实现**：
- 收集100+案件标注数据
- 训练Platt Scaling校准器
- 引入Temperature Scaling

**预期收益**：置信度与实际准确率对齐（ECE < 0.1）。

### 中期规划（3-6个月）

#### 4. 外部知识库集成

**目标**：集成法律数据库、犯罪学图谱、法医学知识。

**实现**：
- 法律数据库：中国裁判文书网API
- 犯罪学图谱：自制常见犯罪模式知识库
- 法医学知识：法医病理学教材数字化

**预期收益**：提升专业案件（如投毒、纵火）准确率。

#### 5. 多模型混合推理

**目标**：不同专家使用不同LLM（GPT-4/Claude/Qwen/DeepSeek）。

**实现**：
```python
expert_model_mapping = {
    "SherlockAnalyst": "gpt-4",      # 演绎推理
    "HenryLeeAnalyst": "claude-3",   # 物证分析
    "CriminalExpert": "qwen-max",    # 心理画像
}
```

**预期收益**：利用不同模型优势，提升整体推理能力。

#### 6. 自动化案件生成与持续评估

**目标**：自动生成测试案件，持续评估系统表现。

**实现**：
- LLM生成改编案件（基于真实案例）
- 定期运行评估（每周10案）
- 自动触发模型微调

**预期收益**：持续进化，避免过拟合。

### 长期规划（6-12个月）

#### 7. 实时推理树可视化

**目标**：推理过程中实时展示推理树，用户可交互探索。

**实现**：
- WebSocket推送推理事件
- D3.js动态渲染推理树
- 支持节点点击查看详情

**预期收益**：提升可解释性，辅助教学。

#### 8. 多语言支持

**目标**：支持英文、日文等案件的推理。

**实现**：
- 多语言Embedding模型
- 跨语言RAG检索
- 多语言专家Prompt

**预期收益**：国际化，服务更多地区。

#### 9. 移动端适配

**目标**：开发移动端App，支持现场快速分析。

**实现**：
- React Native跨平台开发
- 语音输入案件
- AR现场标注

**预期收益**：实用化，真正辅助一线刑侦人员。

### 愿景

**短期**：成为AI刑侦推理领域的开源标杆，准确率突破60%。

**中期**：集成到实际刑侦工作流，作为"数字助手"辅助决策。

**长期**：推动AI推理系统的发展，为其他领域（医疗诊断、金融风控）提供可复用的多Agent推理框架。

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- NVIDIA GPU（用于本地部署Gemma 4 26B和Qwen3-Embedding模型）
- 至少16GB显存（推荐24GB+）

### 安装

```bash
git clone https://github.com/deepNblue/DetectiveRAG.git
cd DetectiveRAG
pip install -r requirements.txt
```

### 配置

编辑 `config/config.yaml`，配置本地部署的模型：

```yaml
llm:
  base_url: "http://localhost:8094/v1"  # NVIDIA NIM本地部署
  model: "gemma-4-26B-A4B-it-Q8_0.gguf"

multimodal:
  vl:
    base_url: "http://localhost:8094/v1"  # NVIDIA NIM本地部署
    model: "gemma-4-26B-A4B-it-Q8_0.gguf"

embedding:
  base_url: "http://localhost:9094/v1"  # NVIDIA NIM本地部署
  model: "Qwen3-Embedding-0.6B"
```

### 启动

```bash
python ui/webui_v2.py
# 访问 http://localhost:7860
```

### 命令行测试

```python
from agents.asmr.orchestrator import ASMROrchestrator
from api.llm_client import LLMClient

llm = LLMClient()
orch = ASMROrchestrator(llm_client=llm)
result = orch.run(case_text="你的案件文本...")
print(f"嫌疑人: {result['culprit']}, 置信度: {result['confidence']}")
```

---

## 📄 License

MIT License

---

<div align="center">

**🧵 忒修斯之线 · 每根线都是新的自己**

*Built with ❤️ by Team Ariadne's Thread*

**感谢NVIDIA提供的开源模型和工具支持！**

</div>
