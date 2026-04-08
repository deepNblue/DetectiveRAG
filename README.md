<div align="center">

# 🧵 忒修斯之线

### **Ariadne's Thread — Multi-Agent Criminal Investigation RAG**

**双线刑侦推理系统 · 深度技术白皮书 v15.2**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-v14.0-FF6F00?logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](#overview) · [系统架构](#-系统架构) · [核心算法](#-核心算法) · [专家系统](#-专家系统) · [快速开始](#-快速开始)

</div>

---

## 📸 系统截图

<div align="center">

![WebUI Screenshot](docs/images/webui_screenshot.jpg)

*WebUI 推理界面 — 忒修斯之线启动画面*

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
- 🔁 **多轮推理Mixin** — 最多3轮深度推理（R1初步→R2审视→R3最终），置信度≥85%自动终止
- 🗳️ **三层投票机制** — 调查层→审判层→裁判Agent，含推翻保护机制
- 🕸️ **证据图谱v3.1** — Graphify分析层（God Nodes/社区发现/关键线索）+ 12种节点类型分类着色
- 🖼️ **多模态分析** — GLM-4-Flash 主模型 + Gemma 4 26B视觉模型自动分析案件图片
- 💬 **专家群聊面板** — 实时渲染Agent推理过程
- 📊 **SVG关系图谱v2** — 12种边类型自动着色渲染 + 弱关系淡化
- 💾 **推理持久化** — 每案独立存储，推理过程完全可回溯
- ⚡ **性能优化** — GLM-4-Flash 单次调用 0.6-7s（vs GLM-5.1 20-40s），并发度优化（max_workers=2）

---

## 📊 测试结果

| 版本 | 日期 | 测试案数 | 准确率 | 关键改动 |
|------|------|---------|--------|---------|
| **v15.2** | 04-08 | 测试中 | 🧪 | 多轮推理精简(10→3) + 图谱渲染v2 + 并发优化 |
| **v15.1** | 04-08 | - | ✅ | GLM-4-Flash切换（速度提升10倍） |
| **v14.0** | 04-07 | CASE-002 | ✅ 正确 | 多轮推理 + 群聊 + 图谱着色 |
| **v13.0** | 04-06 | CASE-001 | ✅ 首次正确 | 多模态视觉 + 证据图谱v2 |
| v12.0 | 04-06 | 10案 | **50%** (持平最佳) | 反向排除 + 分歧放大 |
| v9.2 | 04-04 | 10案 | **50%** (历史最佳) | 推翻保护 + 保守裁判 |

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
│   Stage 0.5: 证据图谱构建v3.1（Graphify分析层）         │
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
│  GLM-4-Flash (推理) │ Gemma 4 26B (视觉) │ Qwen3-0.6B (向量)│
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

### 3. 多轮推理Mixin（MAX_ROUNDS=3）

每个专家Agent通过非侵入式Mixin获得多轮推理能力：

| Round | 名称 | 核心任务 |
|-------|------|---------|
| R1 | 初步分析 | 基于案件文本形成初步假设 |
| R2 | 自我审视 | 魔鬼代言人，审查逻辑漏洞/偏见/被忽略嫌疑人 |
| R3 | 最终深入 | 聚焦R2发现的疑点，给出最终判断 |

**提前终止**：置信度 ≥ 0.85 且推理充分 → 自动终止，约60%案件在R2结束。

**v15.2优化**：从10轮精简到3轮，预期耗时从 ~8min → ~2min，保持推理质量。

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

### 6. 证据图谱v3.1 — Graphify分析层

**三级置信度标签**：
- `EXTRACTED` (✅): 正则/直接提取的事实关系
- `INFERRED` (🤔): LLM推理出的关联
- `AMBIGUOUS` (⚠️): 不确定、待验证的关系

**Graphify核心能力**：
- 🏛️ **God Nodes枢纽发现** — 度最高的节点 = 核心嫌疑人/物证
- 🕸️ **社区发现** — 标签传播聚类，识别证据主题群
- 🔍 **关键线索** — 跨社区边 + 复合评分，自动识别"最值得深入调查的证据关联"
- 🎯 **调查建议** — AMBIGUOUS边→验证关联，孤立节点→遗漏线索

**图谱渲染v2**（12种节点类型 + 12种关系着色）：

| 节点类型 | 颜色 | 关系类型 | 颜色 |
|---------|------|---------|------|
| 嫌疑人 | 🔴 红光晕 | 动机 | 赤红 `#DC143C` |
| 受害者 | 🟣 紫 | 手段 | 橙红 `#FF4500` |
| 物证 | 🟠 琥珀 | 矛盾 | 金黄 `#FFD700` |
| 毒物 | 🔴 红 | 证明 | 绿色 `#00FF7F` |
| 凶器 | 🟠 橙 | 篡改 | 红色 `#FF0000` |
| 时间/地点 | 🔵 青 | 认识 | 紫色 `#9370DB` |

**弱关系淡化**：located_at/happened_at 自动灰色虚线，不抢眼。

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
│   │   ├── evidence_graph.py        # 🕸️ 证据图谱构建v3.1
│   │   ├── multi_round_mixin.py     # 🔄 多轮推理Mixin（3轮）
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
│   ├── graph_renderer.py            # 📊 图谱渲染器v2（12类型着色）
│   ├── graph_renderer_svg.py        # 📊 SVG图谱渲染器
│   ├── expert_chat_renderer.py      # 💬 专家群聊渲染
│   └── conversation_store.py        # 💾 推理对话持久化
├── rag/                             # 📚 RAG检索模块
├── raganything/                     # 🔗 RAG-Anything集成
├── api/
│   └── llm_client.py                # 🔌 LLM API客户端（GLM-4-Flash）
├── config/
│   └── config.yaml                  # ⚙️ 系统配置（max_workers=2）
├── scripts/                         # 📜 批量测试脚本
└── tests/                           # 🧪 测试用例
```

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- 智谱 GLM-4-Flash API Key（或兼容 OpenAI API 的 LLM）
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
  model: "glm-4-flash-250414"
  api_key: "your-api-key"

multimodal:
  vl:
    base_url: "http://localhost:8094/v1"
    model: "gemma-4-26B"
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

## 🛠️ 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 主LLM | 智谱 GLM-4-Flash | 高速推理，单次调用 0.6-7s |
| 视觉模型 | Gemma 4 26B Q8_0 | 本地vLLM部署，案件图片分析 |
| Embedding | Qwen3-Embedding-0.6B | 1024维向量，RAG检索 |
| 前端 | Gradio v14.0 | Blocks + CSS/JS自定义 |
| 图谱 | NetworkX | 双向证据链 + Graphify分析 + SVG渲染 |
| 持久化 | JSONL + JSON | 事件日志 + 元数据 |

---

## 📈 准确率演进与核心教训

```
v15.2  🧪 测试中（多轮推理精简 10→3）
v15.1  ⚡ GLM-4-Flash切换（速度提升10倍）
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
5. **并发度需要平衡** — v15.2将max_workers从4降到2，避免API速率限制（429错误）

---

## 🔮 路线图

- [x] ~~MemPalace语义记忆宫殿~~ — 已移除，使用传统记忆检索
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
