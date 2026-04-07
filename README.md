<div align="center">

# 🧵 忒修斯之线

### **Ariadne's Thread — Multi-Agent Criminal Investigation RAG**

**双线刑侦推理系统 · 深度技术白皮书 v14.0**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-v14.0-FF6F00?logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](#overview) · [系统架构](#-系统架构) · [核心算法](#-核心算法) · [专家系统](#-专家系统) · [快速开始](#-快速开始)

</div>

---

## Overview

**忒修斯之线（DETECTIVE_RAG）** 是一套基于大语言模型的多智能体刑侦推理系统，通过"双线推理"机制——传统 RAG 精准检索 + ASMR 矛盾搜索——实现案件真相的自动挖掘与多维度验证。

> *"每根线都是新的自己"*
> 
> 系统以忒修斯之船哲学命题为隐喻：每一条证据链、每一次推理迭代，都是在保留核心真相的同时不断更新自我。

### ✨ 核心亮点

- 🧠 **18名虚拟专家Agent** — 犯罪心理/刑侦技术/法律/医学/名侦探（波洛/福尔摩斯/李昌钰/宋慈）
- 🔄 **双线推理** — 传统RAG + ASMR矛盾搜索（Auto-Searching of Missing Revelation）
- 🔁 **多轮推理Mixin** — 每个专家最多10轮深度推理，置信度≥85%自动终止
- 🗳️ **三层投票机制** — 调查层→审判层→裁判Agent，含推翻保护机制
- 🕸️ **证据图谱** — 21种关系类型分类着色，跨模态关联（图片↔嫌疑人）
- 🖼️ **多模态分析** — Gemma 4 26B视觉模型自动分析案件图片
- 💬 **专家群聊面板** — 实时渲染Agent推理过程
- 📊 **SVG关系图谱** — 21种边类型自动着色渲染
- 💾 **推理持久化** — 每案独立存储，推理过程完全可回溯

---

## 📊 测试结果

| 版本 | 日期 | 测试案数 | 准确率 | 关键改动 |
|------|------|---------|--------|---------|
| **v14.0** | 04-07 | CASE-002 | ✅ 正确 | 多轮推理 + 群聊 + 图谱着色 |
| **v13.0** | 04-06 | CASE-001 | ✅ 首次正确 | 多模态视觉 + 证据图谱v2 |
| v12.0 | 04-06 | 10案 | **50%** (持平最佳) | 反向排除 + 分歧放大 |
| v9.2 | 04-04 | 10案 | **50%** (历史最佳) | 推翻保护 + 保守裁判 |
| v9.1 | 04-04 | 10案 | 40% | 推理树 + 裁判推翻 |

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────┐
│               WebUI（Gradio v14.0）                    │
│   忒修斯片头 → 推理界面 → 群聊面板 → SVG图谱            │
├──────────────────────────────────────────────────────┤
│           Orchestrator（编排调度中心）                   │
│                                                       │
│   Stage 0:   多模态图片分析（Gemma 4 26B）              │
│   Stage 0.5: 证据图谱构建（NetworkX双向图谱）            │
│   Stage 1:   传统RAG检索（Embedding + BM25 + RRF）      │
│   Stage 2:   ASMR矛盾搜索（5个Searcher并行）             │
│   Stage 3.1: 调查层（9专家 + 4名侦探 = 13并行）         │
│   Stage 3.2: 审判层（检察官+辩护+法官+陪审员 控辩对抗）  │
│   Stage 3.3: 反向排除验证（魔鬼代言人）                   │
│   Stage 3.5: 推理树验证（3假设×5维度）                   │
│   Stage 4:   裁判Agent（综合裁决 + 推翻保护）            │
├──────────────────────────────────────────────────────┤
│  ASMR Agents (18名专家) │ Voting Engine │ Evidence Graph│
│  ConversationStore │ Memory System │ Skill Registry    │
├──────────────────────────────────────────────────────┤
│  GLM-5.1 (推理) │ Gemma 4 26B (视觉) │ Qwen3-0.6B (向量)│
└──────────────────────────────────────────────────────┘
```

---

## 🧠 核心算法

### 1. 双线推理（Dual-Track Reasoning）

基于Kahneman双过程理论的工程实现：

| 推理线 | 认知对应 | 算法 | 优势 |
|--------|---------|------|------|
| **传统RAG** | System 1（快思维） | Embedding + BM25双路召回 + RRF融合(k=60) | 精准、快速 |
| **ASMR搜索** | System 2（慢思维） | 5个Searcher并行 + 反向排除 + 矛盾检测 | 发现盲点、纠偏 |

### 2. ASMR矛盾搜索（5个Searcher并行）

```
MotiveSearcher        → 动机分析：利益分析 → 动机排序
OpportunitySearcher   → 机会分析：时间窗口 → 不在场验证
CapabilitySearcher    → 能力分析：工具/技能匹配 → 能力评估
TemporalSearcher      → 时序分析：时间线重建 → 矛盾检测
ContradictionSearcher → 矛盾搜索：证据冲突 → 盲点评分
```

### 3. 多轮推理Mixin（MAX_ROUNDS=10）

每个专家Agent通过非侵入式Mixin获得多轮推理能力：

| Round | 名称 | 核心任务 |
|-------|------|---------|
| R1 | 初步分析 | 基于案件文本形成初步假设 |
| R2 | 自我审视 | 魔鬼代言人，审查逻辑漏洞/偏见/被忽略嫌疑人 |
| R3 | 深入调查 | 聚焦R2发现的疑点重新审视 |
| R4 | 最终整合 | 综合前轮，给出最终判断（可改变嫌疑人） |
| R5-R10 | 动态深入 | 时间线→动机→证据链→行为→间接证据→心理学 |

**提前终止**：置信度 ≥ 0.85 且推理充分 → 自动终止，约60%案件在R2-R3结束。

### 4. 投票引擎算法

```python
# 多步加权聚合
1. 提取结论 + 名字模糊匹配补全（normalize_name → 前缀匹配）
2. LogicVerifier置信度调整（逻辑漏洞 × 0.85）
3. 加权投票 + 低置信度惩罚(<0.3 → ×0.1) + 少数派放大(≥0.7 → ×1.5)
4. 多人拆分 + 别名归一化（"王福来" = "王福来(管家)"）
5. 归一化 → 确定赢家
```

### 5. 裁判推翻机制

```python
# 推翻保护算法
if 三层一致 and confidence > 0.80 and gap > 0.40:
    return "strong_protect"    # 极强共识，几乎不可推翻
elif 三层一致 and not has_blind_spot:
    return "protect"           # 正常保护
else:
    return "allow_overturn"    # 允许推翻（有盲点或无共识）
```

**v14.0验证**：裁判成功推翻67%投票指向孙志强 → 改判刘建国（主谋），事实证明推翻正确。

### 6. 证据图谱（21种关系类型）

| 颜色分类 | 关系类型 | 色值 |
|---------|---------|------|
| 🔴 嫌疑 | 嫌疑人/指向/关联/涉及/动机 | `#FF003C` |
| 🟡 证据 | 证据/物证/证明/提供 | `#FFB800` |
| 🔵 时空 | 时间/地点/出现在/目击 | `#00F5FF` |
| 🟣 社交 | 认识/亲属/朋友/合作 | `#B37FEB` |
| 🟠 矛盾 | 矛盾/冲突/矛盾点 | `#FF6600` |
| ⚫ 默认 | 未知关系 | `#3A3A45` |

跨模态关联三策略：核心名匹配 → 角色过滤 → 全名匹配

---

## 🕵️ 专家系统

### 调查层（9名分析专家）

| 专家 | 专业领域 | 分析视角 |
|------|---------|---------|
| CriminalExpert | 犯罪心理学 | 心理画像、动机分析、行为模式 |
| ForensicExpert | 刑侦技术 | 物证鉴定、现场勘查、技术侦查 |
| PsychologicalProfiler | 心理画像 | 嫌疑人心理特征、犯罪倾向 |
| FinancialInvestigator | 金融调查 | 资金流向追踪、经济动机 |
| TechInvestigator | 数字取证 | 电子证据提取、通信记录分析 |
| IntelligenceAnalyst | 情报分析 | 多源情报整合、关联分析 |
| InterrogationAnalyst | 审讯分析 | 口供分析、矛盾检测 |
| LogicVerifier | 逻辑验证 | 验证其他专家逻辑一致性（串行） |
| DefenseAttorney | 辩护分析 | 质疑证据充分性、无罪推定 |

### 名侦探专家（4种推理范式）

| 专家 | 原型 | 推理范式 |
|------|------|---------|
| PoirotAnalyst | 赫尔克里·波洛 | 心理推理法——人性洞察和细微矛盾 |
| SherlockAnalyst | 夏洛克·福尔摩斯 | 演绎排除法——排除不可能后剩下的即真相 |
| HenryLeeAnalyst | 李昌钰 | 物证优先法——让证据说话 |
| SongCiAnalyst | 宋慈 | 法医检验法——死因分析、尸检线索 |

### 审判层（控辩对抗）

检察官（起诉强化） ↔ 辩护律师（质疑证据） → 法官（中立裁决） + 陪审员（常识判断）

---

## 🧬 进化式记忆系统

从30案批量学习中进化出的记忆资产：

| 资产 | 数量 | 用途 |
|------|------|------|
| 技能（Skill） | 776+ | 推理策略和模式识别规则 |
| 记忆（Memory） | 664+ | 案件关键特征编码和关联 |
| 犯罪模式（Pattern） | 30+ | 典型犯罪行为模式归纳 |

---

## 📁 项目结构

```
DETECTIVE_RAG_project/
├── agents/
│   ├── asmr/
│   │   ├── orchestrator.py          # 🧠 编排调度核心（Stage流水线）
│   │   ├── voting.py                # 🗳️ 投票引擎
│   │   ├── evidence_graph.py        # 🕸️ 证据图谱构建
│   │   ├── multi_round_mixin.py     # 🔄 多轮推理Mixin
│   │   ├── reasoning_tree.py        # 🌳 推理树验证
│   │   ├── stage_engine.py          # ⚙️ Stage引擎
│   │   ├── name_utils.py            # 📛 名字归一化/别名/拆分
│   │   ├── experts/                 # 👥 18个专家Agent
│   │   ├── searchers/               # 🔍 5个ASMR Searcher
│   │   └── readers/                 # 📖 3个证据Reader
│   ├── memory/                      # 💾 记忆系统（Skill/Memory/Pattern）
│   └── domain_experts/              # 🎯 动态领域专家
├── ui/
│   ├── webui_v2.py                  # 🖥️ Gradio Web界面
│   ├── boot_animation.py            # 🎬 忒修斯片头动画
│   ├── graph_renderer_svg.py        # 📊 SVG图谱渲染器
│   ├── expert_chat_renderer.py      # 💬 专家群聊渲染
│   └── conversation_store.py        # 💾 推理对话持久化
├── rag/                             # 📚 RAG检索模块
├── raganything/                     # 🔗 RAG-Anything集成
├── api/
│   └── llm_client.py                # 🔌 LLM API客户端
├── config/
│   └── config.yaml                  # ⚙️ 系统配置
├── scripts/                         # 📜 批量测试脚本
└── tests/                           # 🧪 测试用例
```

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- 智谱 GLM-5.1 API Key（或兼容 OpenAI API 的 LLM）
- 可选：本地部署 Gemma 4 26B（视觉模型）+ Qwen3-Embedding-0.6B

### 安装

```bash
git clone https://github.com/deepNblue/DetectiveRAG.git
cd DetectiveRAG
pip install -r requirements.txt
```

### 配置

编辑 `config/config.yaml`，填入 LLM API 信息：

```yaml
llm:
  base_url: "https://your-llm-api-endpoint/v1"
  model: "glm-5.1"
  api_key: "your-api-key"

multimodal:
  vl:
    base_url: "http://localhost:8094/v1"
    model: "gemma-4-26B"
```

### 启动

```bash
python -m ui.webui_v2
# 访问 http://localhost:7860
```

### 命令行测试

```python
from agents.asmr.orchestrator import Orchestrator

orch = Orchestrator()
result = await orch.analyze(case_text="你的案件文本...")
print(f"嫌疑人: {result['culprit']}, 置信度: {result['confidence']}")
```

---

## 🛠️ 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 主LLM | 智谱 GLM-5.1 | 200B参数级，推理/分析/投票/裁判 |
| 视觉模型 | Gemma 4 26B Q8_0 | 本地vLLM部署，案件图片分析 |
| Embedding | Qwen3-Embedding-0.6B | 1024维向量，RAG检索 |
| 前端 | Gradio v14.0 | Blocks + CSS/JS自定义 |
| 图谱 | NetworkX | 双向证据链 + SVG渲染 |
| 持久化 | JSONL + JSON | 事件日志 + 元数据 |

---

## 📈 准确率演进与核心教训

```
v14.0  ✅ CASE-002正确（裁判推翻→正确）
v13.0  ✅ CASE-001首次正确（多模态+图谱v2）
v12.0  ██████████ 50% (持平最佳)
v11.1  ██████     30% (强一致快速通路过于激进, 退步!)
v9.3   ████████   40% (去推理树+去裁判反而变差)
v9.2   ██████████ 50% (推翻保护+保守裁判, 历史最佳)
v9.1   ████████   40% (推翻净收益为负)
```

### 核心教训

1. **推翻是把双刃剑** — v9.1投票60%正确，推翻后降到40%。关键不是能否推翻，而是**何时推翻**
2. **方法数≠准确率** — v11.1引入6种新方法反而降到30%。同一LLM输出的多样化后处理 ≠ 多样化信号
3. **推理树+裁判有正收益** — v9.3去掉后降到40%，虽然不完美但确实有用
4. **反向排除是有效纠偏** — v12.0引入后在2案触发盲点发现，v14.0裁判据此成功推翻

---

## 🔮 路线图

- [ ] 图谱感知检索增强ASMR
- [ ] 多Agent直接辩论机制
- [ ] 置信度自动校准
- [ ] 外部知识库集成（法律数据库/犯罪学图谱）
- [ ] 多模型混合推理（不同专家用不同LLM）
- [ ] 自动化案件生成与持续评估
- [ ] 实时推理树可视化

---

## 📄 License

MIT License

---

<div align="center">

**🧵 忒修斯之线 · 每根线都是新的自己**

*Built with ❤️ by [deepNblue](https://github.com/deepNblue)*

</div>
