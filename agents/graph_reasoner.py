"""
知识图谱推理Agent v2 — 性能优化版
将 local + global + hybrid 三阶段合并为单次LLM调用 + 纯Python图算法

优化策略:
  - v1: 4次LLM调用(深度抽取 + local×4嫌疑人 + global + hybrid) → 253s
  - v2: 2次LLM调用(深度抽取 + 统一推理) + Python图算法 → 目标<150s

图谱分析用纯Python(邻接表BFS/DFS)，LLM只用于:
  1. 深度实体/关系抽取(仅在图谱简陋时)
  2. 统一推理(一次LLM完成local+global+hybrid的分析)
"""

import json
from typing import Dict, List, Any, Optional, Set
from loguru import logger
from collections import defaultdict

from .base_agent import BaseAgent


class GraphReasonerAgent(BaseAgent):
    """知识图谱推理Agent v2 — 性能优化版"""
    
    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__(name="图谱推理Agent", config=config, llm_client=llm_client)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_processing(input_data)
        
        graph = input_data.get("graph", {})
        case_clues = input_data.get("case_clues", {})
        suspect_analyses = input_data.get("suspect_analyses", [])
        evidence_connections = input_data.get("evidence_connections", {})
        
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        communities = graph.get("communities", [])
        
        # === 阶段0: 深度抽取(仅在图谱简陋时) ===
        if len(nodes) < 5 or len(edges) < 5:
            self.logger.info("图谱过于简陋，执行深度实体/关系抽取...")
            enriched = self._deep_extract(case_clues, suspect_analyses, evidence_connections)
            if enriched:
                nodes = enriched.get("nodes", nodes)
                edges = enriched.get("edges", edges)
                communities = enriched.get("communities", communities)
        
        if not nodes:
            return self.format_output({
                "mode": "bypass",
                "local_findings": [], "global_findings": [],
                "hybrid_synthesis": [], "reasoning_paths": [],
                "contradictions": [], "suspect_graph_scores": {},
                "reasoning_summary": "图谱数据不足，跳过图谱推理"
            })
        
        # === 阶段1: 纯Python图分析（不调LLM） ===
        node_map = {n["id"]: n for n in nodes}
        adj = defaultdict(list)
        for e in edges:
            adj[e["source"]].append((e["target"], e))
            adj[e["target"]].append((e["source"], e))
        
        # 1a. 对每个嫌疑人提取局部子图 + 计算图指标
        local_analyses = []
        suspect_nodes = [n for n in nodes if n.get("type") == "person"]
        
        for s_node in suspect_nodes[:6]:
            analysis = self._python_local_analysis(s_node, nodes, edges, adj, node_map)
            local_analyses.append(analysis)
        
        # 1b. 全局图指标
        global_metrics = self._python_global_analysis(nodes, edges, adj, node_map, communities)
        
        # === 阶段2: 单次LLM统一推理 ===
        # 将local分析结果 + 全局指标 + 图谱关系 一起给LLM
        unified_result = self._unified_llm_reasoning(
            local_analyses, global_metrics, nodes, edges,
            adj, node_map, communities, case_clues, suspect_analyses
        )
        
        # === 阶段3: 纯Python综合打分 ===
        suspect_scores = self._compute_suspect_scores(local_analyses, unified_result)
        
        # === 阶段4: 生成摘要 ===
        reasoning_summary = self._build_reasoning_summary(
            local_analyses, global_metrics, unified_result, suspect_scores
        )
        
        result = {
            "mode": "hybrid_v2",
            "local_findings": local_analyses,
            "global_findings": global_metrics.get("patterns", []),
            "hybrid_synthesis": unified_result.get("synthesis", []),
            "reasoning_paths": unified_result.get("paths", []),
            "contradictions": unified_result.get("contradictions", []),
            "suspect_graph_scores": suspect_scores,
            "reasoning_summary": reasoning_summary,
            "graph_stats": {"nodes": len(nodes), "edges": len(edges), "communities": len(communities)}
        }
        
        self.logger.info(f"图谱推理v2完成: local={len(local_analyses)}, "
                        f"paths={len(result['reasoning_paths'])}, "
                        f"contradictions={len(result['contradictions'])}")
        return self.format_output(result)
    
    # ================================================================
    #  阶段0: 深度抽取(保留，仅在需要时调用)
    # ================================================================
    
    def _deep_extract(self, case_clues, suspect_analyses, evidence_connections):
        clues_text = json.dumps(case_clues, ensure_ascii=False)[:1500]
        suspects_text = json.dumps(
            [s if isinstance(s, str) else s.get("name", s.get("suspect_name", ""))
             for s in suspect_analyses], ensure_ascii=False
        )
        prompt = (
            "你是知识图谱构建专家。请从以下案件信息中进行深度实体和关系抽取。\n\n"
            f"=== 案件信息 ===\n{clues_text}\n\n嫌疑人: {suspects_text}\n\n"
            "抽取要求:\n"
            "1. 实体类型: 人物(含身份)、地点、时间、物品/证据、事件\n"
            "2. 关系有方向、类型、权重(0-1)、描述\n"
            "3. 注意隐含关系和时间关系\n\n"
            '返回JSON: {"nodes":[{"id":"ID","label":"名","type":"person|location|time|evidence|event",'
            '"description":"描述","properties":{}}],'
            '"edges":[{"source":"ID","target":"ID","label":"关系","weight":0.8,'
            '"description":"描述","evidence":[]}],'
            '"communities":[{"name":"名","members":["ID"],"significance":"说明"}]}\n'
            "只返回JSON。"
        )
        response = self.call_llm(prompt, temperature=0.2)
        parsed = self.extract_json_from_response(response)
        if parsed and isinstance(parsed, dict) and "nodes" in parsed:
            return parsed
        return None
    
    # ================================================================
    #  阶段1: 纯Python图分析（零LLM调用）
    # ================================================================
    
    def _python_local_analysis(self, center_node, nodes, edges, adj, node_map):
        """
        纯Python局部子图分析 — 替代v1的LLM local_query
        计算: 度中心性、关系权重、证据距离、异常检测
        """
        cid = center_node["id"]
        cname = center_node.get("label", cid)
        
        # 度中心性
        degree = len(adj.get(cid, []))
        total_nodes = len(nodes)
        degree_centrality = degree / max(total_nodes - 1, 1)
        
        # 1跳和2跳邻居
        one_hop = set()
        one_hop_edges = []
        for neighbor, edge in adj.get(cid, []):
            one_hop.add(neighbor)
            one_hop_edges.append(edge)
        
        two_hop = set(one_hop)
        for n in one_hop:
            for nn, _ in adj.get(n, []):
                two_hop.add(nn)
        
        # 关系强度分析
        edge_weights = [e.get("weight", 0.5) for _, e in adj.get(cid, [])]
        avg_weight = sum(edge_weights) / max(len(edge_weights), 1)
        max_weight = max(edge_weights) if edge_weights else 0
        
        # 到证据节点的距离
        evidence_nodes = {n["id"] for n in nodes if n.get("type") == "evidence"}
        victim_nodes = {n["id"] for n in nodes if n.get("type") == "person" 
                       and "受害" in n.get("description", "")}
        
        evidence_dist = self._bfs_distance(cid, evidence_nodes, adj)
        victim_dist = self._bfs_distance(cid, victim_nodes, adj)
        
        # 异常检测: 高权重关系数量
        high_weight_count = sum(1 for w in edge_weights if w > 0.7)
        
        # 位置判断
        if degree_centrality > 0.4:
            position = "核心"
        elif degree_centrality > 0.2:
            position = "桥接"
        else:
            position = "边缘"
        
        # 构建关键连接描述（供LLM统一推理使用）
        key_connections = []
        for neighbor, edge in adj.get(cid, []):
            n = node_map.get(neighbor, {})
            key_connections.append({
                "target": n.get("label", neighbor),
                "type": n.get("type", "?"),
                "relation": edge.get("label", "?"),
                "weight": edge.get("weight", 0.5),
                "description": edge.get("description", "")
            })
        
        return {
            "suspect": cname,
            "node_id": cid,
            "position": position,
            "degree": degree,
            "degree_centrality": round(degree_centrality, 3),
            "avg_edge_weight": round(avg_weight, 3),
            "max_edge_weight": round(max_weight, 3),
            "high_weight_count": high_weight_count,
            "evidence_distance": evidence_dist,
            "victim_distance": victim_dist,
            "one_hop_count": len(one_hop),
            "two_hop_count": len(two_hop),
            "key_connections": sorted(key_connections, key=lambda x: -x["weight"])[:5],
            "anomaly_score": self._compute_anomaly(
                degree_centrality, avg_weight, evidence_dist, victim_dist, high_weight_count
            )
        }
    
    def _bfs_distance(self, start, targets, adj):
        """BFS计算start到targets中任一节点的最短距离"""
        if not targets:
            return -1
        if start in targets:
            return 0
        visited = {start}
        frontier = {start}
        for depth in range(1, 4):
            next_frontier = set()
            for node in frontier:
                for neighbor, _ in adj.get(node, []):
                    if neighbor in targets:
                        return depth
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        return -1  # 未找到
    
    def _compute_anomaly(self, degree_c, avg_w, ev_dist, vic_dist, high_w):
        """纯Python异常度计算"""
        score = 0.0
        # 高度中心性 → 更可疑（接触面广）
        score += degree_c * 0.3
        # 高平均关系权重 → 更可疑
        score += avg_w * 0.25
        # 到证据距离近 → 更可疑
        if ev_dist == 0:
            score += 0.25
        elif ev_dist == 1:
            score += 0.15
        elif ev_dist == 2:
            score += 0.05
        # 到受害者距离近 → 更可疑
        if vic_dist == 0:
            score += 0.1
        elif vic_dist == 1:
            score += 0.05
        # 多个高权重关系 → 更可疑
        score += min(high_w * 0.05, 0.1)
        
        return round(min(score, 1.0), 3)
    
    def _python_global_analysis(self, nodes, edges, adj, node_map, communities):
        """纯Python全局图分析"""
        # 度分布
        degree_dist = {n["id"]: len(adj.get(n["id"], [])) for n in nodes}
        
        # 桥接节点（连接不同社区的节点）
        community_map = defaultdict(set)
        for c in communities:
            for m in c.get("members", []):
                # Handle nested lists (LLM may return ["id1","id2"] as a sub-list)
                if isinstance(m, list):
                    for sub_m in m:
                        if isinstance(sub_m, (str, int)):
                            community_map[sub_m].add(c.get("name", ""))
                elif isinstance(m, (str, int)):
                    community_map[m].add(c.get("name", ""))
        
        bridge_nodes = []
        for nid, comms in community_map.items():
            if len(comms) > 1:
                bridge_nodes.append({
                    "node": node_map.get(nid, {}).get("label", nid),
                    "communities": list(comms)
                })
        
        # 高权重边（可疑关系）
        suspicious_edges = sorted(edges, key=lambda e: e.get("weight", 0), reverse=True)[:5]
        
        # 全局密度
        max_edges = len(nodes) * (len(nodes) - 1) / 2
        density = len(edges) / max(max_edges, 1)
        
        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "density": round(density, 3),
            "bridge_nodes": bridge_nodes,
            "top_weighted_edges": [
                {
                    "source": node_map.get(e["source"], {}).get("label", e["source"]),
                    "target": node_map.get(e["target"], {}).get("label", e["target"]),
                    "label": e.get("label", "?"),
                    "weight": e.get("weight", 0)
                }
                for e in suspicious_edges
            ],
            "community_count": len(communities),
            "patterns": []  # 将由LLM推理补充
        }
    
    # ================================================================
    #  阶段2: 单次LLM统一推理（替代v1的3次LLM调用）
    # ================================================================
    
    def _unified_llm_reasoning(self, local_analyses, global_metrics,
                                nodes, edges, adj, node_map, communities,
                                case_clues, suspect_analyses):
        """
        单次LLM调用完成 local分析解读 + global模式发现 + hybrid融合推理
        
        关键优化: 把Python计算的图指标作为结构化输入给LLM
        LLM只需要做"理解+推理"，不需要做"计算"
        """
        # 构建结构化输入
        suspects_summary = []
        for la in local_analyses:
            conn_strs = [f"    → {c['target']}({c['type']}) [{c['relation']}] w={c['weight']}"
                        for c in la["key_connections"][:4]]
            suspects_summary.append(
                f"【{la['suspect']}】\n"
                f"  图位置: {la['position']} | 度={la['degree']} | 中心性={la['degree_centrality']}\n"
                f"  关系强度: 平均={la['avg_edge_weight']} 最高={la['max_edge_weight']} 高权重关系数={la['high_weight_count']}\n"
                f"  距离: 到证据={la['evidence_distance']}跳 到受害者={la['victim_distance']}跳\n"
                f"  异常度: {la['anomaly_score']}\n"
                f"  关键连接:\n" + "\n".join(conn_strs)
            )
        suspects_text = "\n\n".join(suspects_summary)
        
        global_text = (
            f"节点={global_metrics['total_nodes']}, 边={global_metrics['total_edges']}, "
            f"密度={global_metrics['density']}\n"
            f"社区={global_metrics['community_count']}个\n"
            f"桥接节点: {json.dumps(global_metrics['bridge_nodes'], ensure_ascii=False)}\n"
            f"高权重关系: {json.dumps(global_metrics['top_weighted_edges'], ensure_ascii=False)}"
        )
        
        clues_text = json.dumps(case_clues, ensure_ascii=False)[:500]
        
        prompt = (
            "你是刑侦图谱推理专家。以下是通过图算法分析案件知识图谱得到的结构化数据。\n"
            "请基于这些数据进行统一推理。\n\n"
            f"=== 各嫌疑人图分析 ===\n{suspects_text}\n\n"
            f"=== 全局图谱指标 ===\n{global_text}\n\n"
            f"=== 案件线索 ===\n{clues_text}\n\n"
            "请完成:\n"
            "1. 对每个嫌疑人，从图结构角度评估可疑程度\n"
            "2. 发现推理路径（嫌疑人→关键证据→受害者的链条）\n"
            "3. 发现矛盾（时间矛盾、关系异常、证据冲突）\n"
            "4. 综合排名\n\n"
            '返回JSON:\n'
            '{\n'
            '  "suspect_evaluations": [\n'
            '    {"suspect":"名","graph_suspicion":0.8,"key_evidence":["证据"],'
            '"reasoning":"基于图分析的推理"}\n'
            '  ],\n'
            '  "paths": [\n'
            '    {"suspect":"名","to":"目标","steps":["步骤1","步骤2"],'
            '"strength":0.8,"key_evidence":["证据"],"reasoning":"说明"}\n'
            '  ],\n'
            '  "contradictions": [\n'
            '    {"type":"时间矛盾|关系异常|证据冲突|逻辑漏洞",'
            '"description":"描述","severity":"high|medium|low",'
            '"implication":"暗示","related_suspect":"嫌疑人"}\n'
            '  ],\n'
            '  "synthesis": [\n'
            '    {"suspect":"名","rank":1,"score":0.9,'
            '"path_strength":"强|中|弱","contradictions_count":0,'
            '"conclusion":"综合评价"}\n'
            '  ]\n'
            '}\n'
            "只返回JSON。"
        )
        
        response = self.call_llm(prompt, temperature=0.3)
        parsed = self.extract_json_from_response(response)
        
        if parsed and isinstance(parsed, dict):
            return parsed
        return {"suspect_evaluations": [], "paths": [], "contradictions": [], "synthesis": []}
    
    # ================================================================
    #  阶段3: 纯Python综合打分
    # ================================================================
    
    def _compute_suspect_scores(self, local_analyses, unified_result):
        """融合Python图指标 + LLM推理结果"""
        scores = {}
        
        # Python图分析贡献(权重0.4)
        for la in local_analyses:
            name = la["suspect"]
            scores[name] = la["anomaly_score"] * 0.4
        
        # LLM评估贡献(权重0.4)
        for ev in unified_result.get("suspect_evaluations", []):
            name = ev.get("suspect", "")
            suspicion = ev.get("graph_suspicion", 0)
            if name:
                scores[name] = scores.get(name, 0) + suspicion * 0.4
        
        # LLM路径贡献(权重0.15)
        for path in unified_result.get("paths", []):
            name = path.get("suspect", "")
            strength = path.get("strength", 0)
            if name:
                scores[name] = scores.get(name, 0) + strength * 0.15
        
        # LLM矛盾贡献(权重0.05)
        for c in unified_result.get("contradictions", []):
            name = c.get("related_suspect", "")
            sev_map = {"high": 0.05, "medium": 0.02, "low": 0.005}
            if name:
                scores[name] = scores.get(name, 0) + sev_map.get(c.get("severity", "low"), 0.005)
        
        # 归一化
        if scores:
            max_s = max(scores.values())
            if max_s > 1:
                scores = {k: min(v / max_s, 1.0) for k, v in scores.items()}
        
        return scores
    
    # ================================================================
    #  阶段4: 摘要生成（纯Python，不调LLM）
    # ================================================================
    
    def _build_reasoning_summary(self, local_analyses, global_metrics,
                                  unified_result, scores):
        parts = []
        
        # Python图分析结果
        parts.append("=== 图算法分析 ===")
        for la in local_analyses:
            parts.append(
                f"  [{la['suspect']}] {la['position']} | 异常度={la['anomaly_score']:.2f} | "
                f"度={la['degree']} | 到证据={la['evidence_distance']}跳 | "
                f"到受害者={la['victim_distance']}跳"
            )
        
        parts.append(f"\n全局: {global_metrics['total_nodes']}节点, "
                    f"{global_metrics['total_edges']}边, "
                    f"密度={global_metrics['density']:.3f}")
        
        if global_metrics["bridge_nodes"]:
            parts.append(f"桥接节点: {[b['node'] for b in global_metrics['bridge_nodes']]}")
        
        # LLM推理结果
        evals = unified_result.get("suspect_evaluations", [])
        if evals:
            parts.append("\n=== LLM图推理评估 ===")
            for ev in evals:
                parts.append(f"  [{ev.get('suspect','?')}] "
                            f"图可疑度={ev.get('graph_suspicion',0):.2f}: "
                            f"{ev.get('reasoning','')[:80]}")
        
        paths = unified_result.get("paths", [])
        if paths:
            parts.append("\n=== 推理路径 ===")
            for p in paths:
                parts.append(f"  {p.get('suspect','?')} → {p.get('to','?')}: "
                            f"强度={p.get('strength',0):.2f}")
                parts.append(f"    {' → '.join(p.get('steps', []))}")
        
        contradictions = unified_result.get("contradictions", [])
        if contradictions:
            parts.append("\n=== 矛盾检测 ===")
            for c in contradictions:
                parts.append(f"  [{c.get('severity','?')}] {c.get('type','?')}: {c.get('description','')}")
                if c.get("implication"):
                    parts.append(f"    → {c['implication']}")
        
        if scores:
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            parts.append("\n=== 图谱推理排名 ===")
            for i, (name, score) in enumerate(ranked, 1):
                parts.append(f"  #{i} {name}: {score:.3f}")
        
        return "\n".join(parts)
