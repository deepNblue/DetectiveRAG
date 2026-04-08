# Graphify 借鉴分析 → DETECTIVE_RAG 证据图谱升级

> 来源: [safishamsi/graphify](https://github.com/safishamsi/graphify) (6762⭐)
> 目标: `agents/asmr/evidence_graph.py` (753行)
> 日期: 2026-04-08

---

## 一、Graphify 核心架构

```
detect() → extract() → build_graph() → cluster() → analyze() → report() → export()
```

| 模块 | 职责 | 输入→输出 |
|------|------|-----------|
| `detect.py` | 文件发现 | 目录 → `[Path]` |
| `extract.py` | 实体/关系提取 | 文件 → `{nodes, edges}` |
| `build.py` | 图谱组装 | 提取结果 → `nx.Graph` |
| `cluster.py` | Leiden社区发现 | 图 → 社区标签 |
| `analyze.py` | 结构分析 | 图 → god nodes + surprises + questions |
| `report.py` | 报告生成 | 分析结果 → Markdown |
| `cache.py` | SHA256缓存 | 文件 → (cached, uncached) |

---

## 二、关键设计对比

### 2.1 边置信度标签 — ⭐⭐⭐ 高价值

| | Graphify | DETECTIVE_RAG |
|---|---------|---------------|
| 方式 | 三级标签: `EXTRACTED` / `INFERRED` / `AMBIGUOUS` | 浮点数 `confidence: 0.0~1.0` |
| 含义 | 明确告知"找到的 vs 猜的" | 只是一个模糊的数字 |
| 下游使用 | 裁判/分析器根据标签决策 | 几乎没被使用 |

**借鉴建议**: 在 `EvidenceEdge` 中新增 `confidence_label` 字段:
```python
class EvidenceEdge:
    ...
    confidence_label: str = "EXTRACTED"  # EXTRACTED | INFERRED | AMBIGUOUS
```
- 正则提取的实体 → `EXTRACTED`（在文本中明确出现的）
- LLM推断的关联 → `INFERRED`（模型推理出来的）
- 不确定的关联 → `AMBIGUOUS`（需要Agent进一步验证）

**对侦探系统的价值**: 裁判Agent可以根据 `AMBIGUOUS` 边生成"需要进一步调查的方向"，而不是盲目信任所有关系。

---

### 2.2 Leiden 社区发现 — ⭐⭐⭐ 高价值

| | Graphify | DETECTIVE_RAG |
|---|---------|---------------|
| 算法 | Leiden (graspologic) | 无社区发现 |
| 输出 | `{community_id: [node_ids]}` | 无 |
| 下游 | god nodes, surprising connections, cohesion | 无 |

**借鉴建议**: 对证据图谱运行 Leiden 社区检测:
```python
from graspologic.partition import leiden

def detect_evidence_communities(self) -> dict:
    """发现证据社区 — 如"毒物证据群"、"时间线证据群"、"人物关系群" """
    G = self._to_networkx()
    partition = leiden(G)
    # 结果: {node_id: community_id}
    # 同一社区的实体互相关联紧密
```

**对侦探系统的价值**:
1. **社区 = 证据主题群** — 如所有毒物相关证据会自然聚类
2. **跨社区边 = 关键线索** — 连接不同证据主题群的边，往往是破案关键
3. **孤立节点 = 遗漏证据** — 不属于任何社区的实体，可能是线索缺口

---

### 2.3 God Nodes（枢纽节点） — ⭐⭐⭐ 高价值

Graphify 的 `god_nodes()`: 找到连接度最高的核心节点。

**借鉴建议**:
```python
def find_god_nodes(self, top_n=5) -> list:
    """找到证据图谱中的枢纽实体 — 通常是案件核心人物/物证"""
    # 度最高的节点 = 与最多其他证据相关联的实体
    # 在刑侦语境中，这往往指向：
    # 1. 真凶（与最多证据关联）
    # 2. 核心物证（如毒物、凶器）
    # 3. 案件中心地点
```

**对侦探系统的价值**: 
- 度最高的嫌疑人节点 = 嫌疑最大的人（Occam's razor）
- 如果所有证据都指向某个"边缘人物"，可能需要重新审视
- 可作为 ASMR 矛盾搜索的线索优先级排序依据

---

### 2.4 Surprising Connections（意外关联） — ⭐⭐⭐ 高价值

Graphify 的 `surprising_connections()`: 找到跨文件/跨社区的意外边，用复合评分:
```
score = confidence_weight + cross_filetype_bonus + cross_community_bonus + peripheral_hub_bonus
```

**借鉴建议** — 改造为"关键线索发现":
```python
def find_surprising_connections(self, communities, top_n=5) -> list:
    """发现证据图谱中的意外关联 — 刑侦版"""
    # 评分因素:
    # 1. AMBIGUOUS/INFERRED边比EXTRACTED更有调查价值
    # 2. 跨社区的边（如毒物证据群 ↔ 人物关系群）= 关键线索
    # 3. 低度节点连接到高度节点 = 小线索连到大人物
    # 4. 矛盾的边（同一实体有冲突描述）= 证人撒谎信号
```

**对侦探系统的价值**: 这是**最核心的借鉴**——直接告诉Agent "哪些证据关联最值得深入调查"，而不是让Agent在所有证据中大海捞针。

---

### 2.5 图谱差异对比 (graph_diff) — ⭐⭐ 中等价值

Graphify 的 `graph_diff()`: 比较两次图谱快照的差异。

**借鉴建议**: 多轮推理时，每轮新增的证据可以增量分析:
```python
def evidence_diff(self, old_graph, new_graph) -> dict:
    """对比两轮推理的证据变化"""
    # new_nodes: 新发现的证据实体
    # new_edges: 新发现的关系
    # 这些增量可以注入下一轮推理的 prompt
```

**对侦探系统的价值**: 多轮推理Mixin可以利用增量图谱，告诉Agent "这轮新发现了什么"，避免重复分析。

---

### 2.6 SHA256 缓存 — ⭐⭐ 中等价值

Graphify 的 `cache.py`: 用 SHA256 哈希文件内容，避免重复提取。

**借鉴建议**: 缓存LLM实体提取结果:
```python
def _get_cached_entities(self, analysis_text: str) -> list | None:
    """缓存LLM提取结果，相同图片不重复调用"""
    content_hash = hashlib.sha256(analysis_text.encode()).hexdigest()
    cache_file = f"~/.mempalace/graph_cache/{content_hash}.json"
    if os.path.exists(cache_file):
        return json.load(cache_file)
    return None
```

**价值**: Stage 0.5 的 LLM 实体提取是性能瓶颈（~71s），缓存可以避免重复调用。

---

### 2.7 Schema 验证 — ⭐ 低价值（已有）

Graphify 有 `validate.py` 严格校验提取结果的 schema。
DETECTIVE_RAG 已有 `_validate_entities()` + `_robust_json_parse()`，覆盖了此需求。

---

### 2.8 内聚度评分 — ⭐⭐ 中等价值

Graphify 的 `cohesion_score()`: 实际内部边数 / 最大可能边数。

**借鉴建议**: 评估证据社区的"紧密程度":
```python
def cohesion_score(self, community_nodes) -> float:
    """内聚度 — 证据链的完整度指标"""
    # 高内聚 = 这条证据链自洽完整（如毒物链: 购买→投毒→检测→死亡）
    # 低内聚 = 证据链有缺口，需要补充
```

---

### 2.9 Suggested Questions — ⭐⭐ 中等价值

Graphify 的 `suggest_questions()`: 基于 AMBIGUOUS 边、bridge nodes、孤立节点生成"图谱想回答的问题"。

**借鉴建议**: 为Agent生成"待调查方向":
```python
def suggest_investigation_directions(self) -> list:
    """基于图谱结构生成调查建议"""
    # 1. AMBIGUOUS边 → "需要验证的关系"
    # 2. 孤立节点 → "与其他证据无关联的实体，可能是遗漏线索"
    # 3. 跨社区桥 → "连接不同证据主题的关键线索"
```

---

## 三、优先级排序与实施建议

### 🥇 第一优先级 — 直接影响准确率

| 改进 | 预期效果 | 工作量 |
|------|---------|--------|
| **置信度三级标签** | 裁判Agent能区分"事实vs推测" | 小 (1h) |
| **God Nodes 发现** | 自动识别核心嫌疑人/物证 | 小 (1h) |
| **Surprising Connections** | 关键线索自动排序 | 中 (3h) |

### 🥈 第二优先级 — 增强推理质量

| 改进 | 预期效果 | 工作量 |
|------|---------|--------|
| **Leiden 社区发现** | 证据主题聚类 | 中 (2h, 需装 graspologic) |
| **调查方向建议** | Agent 不再盲目搜索 | 中 (2h) |
| **图谱差异对比** | 多轮推理增量分析 | 小 (1h) |

### 🥉 第三优先级 — 性能优化

| 改进 | 预期效果 | 工作量 |
|------|---------|--------|
| **SHA256 缓存** | Stage 0.5 从71s降到<1s(缓存命中时) | 小 (1h) |
| **内聚度评分** | 证据链完整度量化 | 小 (0.5h) |

---

## 四、Graphify 不适用的设计

| 设计 | 原因 |
|------|------|
| tree-sitter AST 解析 | 面向代码，不适用于案件文本 |
| vis.js 交互式可视化 | 已有 SVG 渲染器 |
| MCP Server | 案件推理是一次性的，不需要持久查询服务 |
| Obsidian Vault 导出 | 非知识管理场景 |
| `--watch` 文件监控 | 非增量编辑场景 |

---

## 五、总结

Graphify 最值得借鉴的**不是某个具体算法**，而是它的**图谱分析哲学**：

> **不是构建图谱就完了，而是要让图谱"开口说话"——告诉你哪里有疑点、哪里有遗漏、哪里有关键线索。**

当前 DETECTIVE_RAG 的图谱是"静默的"——构建完只生成文本注入 case_text，没有结构化的分析输出。借鉴 Graphify 的 analyze 层，可以让图谱成为**主动的推理参与者**，而不仅仅是被动的数据容器。
