"""
推理生成Agent - v2 简化版
修复f-string中JSON花括号冲突，简化async为同步调用
"""

from typing import Dict, List, Any
from .base_agent import BaseAgent
import json
from loguru import logger


class ReasoningGeneratorAgent(BaseAgent):
    """推理生成Agent - 生成推理链，提出假设，排除法推理"""
    
    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        super().__init__("ReasoningGeneratorAgent", config, llm_client)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成推理"""
        self.log_processing(input_data)
        
        if not input_data.get("case_clues"):
            return self.format_output({"error": "缺少案件线索"})
        
        try:
            case_clues = input_data["case_clues"]
            suspect_analyses = input_data.get("suspect_analyses", [])
            evidence_connections = input_data.get("evidence_connections", {})
            rag_context = input_data.get("rag_context", "")
            graph_reasoning = input_data.get("graph_reasoning", {})
            
            # 构建图谱增强上下文
            graph_context = self._build_graph_context(graph_reasoning)
            if graph_context:
                rag_context = (rag_context + "\n\n" + graph_context).strip() if rag_context else graph_context
            
            # Step 1: 生成假设
            hypotheses = self._generate_hypotheses(case_clues, suspect_analyses, rag_context)
            
            # Step 2: 构建推理链
            reasoning_chain = self._build_reasoning_chain(evidence_connections, case_clues, graph_context)
            
            # Step 3: 排除法推理
            elimination_results = self._elimination_reasoning(hypotheses, case_clues, graph_context)
            
            # Step 4: 最终结论（传入hypotheses以获取嫌疑人名）
            final_conclusion = self._draw_final_conclusion(elimination_results, reasoning_chain, hypotheses)
            
            # Step 5: 置信度
            confidence_score = self._calculate_confidence(final_conclusion, elimination_results, reasoning_chain)
            
            # Step 6: 如果有图谱打分，融合到最终置信度
            if graph_reasoning and isinstance(graph_reasoning, dict):
                graph_scores = graph_reasoning.get("suspect_graph_scores", {})
                top_suspect = final_conclusion.get("top_suspect", "")
                if top_suspect in graph_scores:
                    graph_boost = graph_scores[top_suspect] * 0.15  # 图谱最多贡献15%
                    confidence_score = min(confidence_score + graph_boost, 1.0)
                    self.logger.info(f"图谱增强置信度: +{graph_boost:.3f} → {confidence_score:.3f}")
            
            result = {
                "hypotheses": hypotheses,
                "reasoning_chain": reasoning_chain,
                "elimination_results": elimination_results,
                "final_conclusion": final_conclusion,
                "confidence_score": confidence_score,
                "graph_enhanced": bool(graph_context),
            }
            
            self.logger.info(f"推理完成: {len(hypotheses)}假设, {len(reasoning_chain)}步, 置信度={confidence_score:.2f}"
                           f"{'[图谱增强]' if graph_context else ''}")
            return self.format_output(result)
            
        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            return self.format_output({
                "error": str(e),
                "hypotheses": [],
                "reasoning_chain": [],
                "final_conclusion": {"conclusion": "推理失败", "confidence": 0.0}
            })
    
    def _build_graph_context(self, graph_reasoning: dict) -> str:
        """构建图谱推理上下文文本"""
        if not graph_reasoning or not isinstance(graph_reasoning, dict):
            return ""
        
        parts = []
        
        # 推理摘要
        summary = graph_reasoning.get("reasoning_summary", "")
        if summary:
            parts.append("=== 知识图谱推理发现 ===")
            parts.append(summary)
        
        # 路径证据
        paths = graph_reasoning.get("path_evidence", [])
        if paths:
            parts.append("\n=== 图谱路径分析 ===")
            for p in paths:
                suspect = p.get("suspect", "?")
                strength = p.get("overall_path_strength", 0)
                parts.append(f"- {suspect}: 路径强度={strength:.2f}")
                for path in p.get("paths", [])[:2]:
                    parts.append(f"  → {path.get('description', '')}")
                    if path.get("key_evidence"):
                        parts.append(f"  关键证据: {', '.join(path['key_evidence'][:3])}")
        
        # 矛盾检测
        contradictions = graph_reasoning.get("contradictions", [])
        if contradictions:
            parts.append("\n=== 图谱矛盾检测 ===")
            for c in contradictions:
                sev = c.get("severity", "low")
                parts.append(f"- [{sev}] {c.get('description', '')}")
                if c.get("implication"):
                    parts.append(f"  暗示: {c['implication']}")
        
        return "\n".join(parts)
    
    def _generate_hypotheses(self, case_clues, suspect_analyses, rag_context=""):
        """生成假设"""
        clues_str = json.dumps(case_clues, ensure_ascii=False)[:800]
        suspects_str = json.dumps(suspect_analyses, ensure_ascii=False)[:800]
        
        prompt = (
            "请基于以下信息生成破案假设：\n\n"
            f"案件线索: {clues_str}\n\n"
            f"嫌疑人分析: {suspects_str}\n\n"
        )
        if rag_context:
            prompt += f"参考信息: {rag_context[:500]}\n\n"
        
        prompt += (
            "请生成3-5个假设，以JSON数组格式返回:\n"
            '[{"hypothesis_id":"H1","description":"假设描述","suspect":"嫌疑人名",'
            '"supporting_evidence":["证据1"],"contradicting_evidence":["矛盾1"],'
            '"initial_confidence":0.6,"motive":"动机","opportunity":"机会","capability":"能力"}]\n\n'
            "⚠️ 重要: 除了单人作案假设外，请务必包含至少一个**多罪犯合谋**假设。"
            "如果案件涉及商业间谍、信息泄密、团伙犯罪等类型，"
            "检查是否有两个嫌疑人各掌握不同的关键犯罪要素(互补的动机+互补的渠道/能力)，"
            "合谋假设的suspect字段格式为'人A+人B'(用加号连接两人)。\n"
            "只返回JSON，不要其他文字。"
        )
        
        response = self.call_llm(prompt, temperature=0.7)
        hypotheses = self.extract_json_from_response(response)
        
        if hypotheses and isinstance(hypotheses, list):
            return hypotheses
        
        # 降级：默认假设
        return [
            {"hypothesis_id": "H1", "description": "主要嫌疑人作案", "suspect": "嫌疑人A",
             "supporting_evidence": [], "contradicting_evidence": [], "initial_confidence": 0.6},
            {"hypothesis_id": "H2", "description": "次要嫌疑人作案", "suspect": "嫌疑人B",
             "supporting_evidence": [], "contradicting_evidence": [], "initial_confidence": 0.3},
        ]
    
    def _build_reasoning_chain(self, evidence_connections, case_clues, graph_context=""):
        """构建推理链"""
        ev_str = json.dumps(evidence_connections, ensure_ascii=False)[:800]
        clues_str = json.dumps(case_clues, ensure_ascii=False)[:500]
        
        prompt = (
            "请基于以下信息构建推理链：\n\n"
            f"证据关联: {ev_str}\n\n"
            f"案件线索: {clues_str}\n\n"
        )
        if graph_context:
            prompt += f"图谱推理发现:\n{graph_context[:600]}\n\n"
        
        prompt += (
            "请构建5-8步推理链，以JSON数组返回:\n"
            '[{"step":1,"description":"步骤描述","evidence_used":["证据1"],'
            '"logic":"推理逻辑","conclusion":"中间结论","confidence":0.8}]\n'
            "只返回JSON。"
        )
        
        response = self.call_llm(prompt, temperature=0.6)
        chain = self.extract_json_from_response(response)
        
        if chain and isinstance(chain, list):
            return chain
        
        return [
            {"step": 1, "description": "分析证据", "evidence_used": [],
             "logic": "从证据出发", "conclusion": "初步判断", "confidence": 0.7}
        ]
    
    def _elimination_reasoning(self, hypotheses, case_clues, graph_context=""):
        """排除法推理 - 对每个假设逐一验证"""
        results = []
        
        for hyp in hypotheses:
            hyp_str = json.dumps(hyp, ensure_ascii=False)[:500]
            clues_str = json.dumps(case_clues, ensure_ascii=False)[:500]
            
            prompt = (
                "请用排除法分析以下假设：\n\n"
                f"假设: {hyp_str}\n\n"
                f"案件线索: {clues_str}\n\n"
            )
            if graph_context:
                prompt += f"图谱推理参考:\n{graph_context[:400]}\n\n"
            
            prompt += (
                "分析该假设是否可排除，以JSON返回:\n"
                '{"hypothesis_id":"H1","can_eliminate":false,'
                '"elimination_reason":"原因","supporting_points":["支持1"],'
                '"contradicting_points":["矛盾1"],"updated_confidence":0.7,'
                '"key_evidence":["证据1"]}\n'
                "只返回JSON。"
            )
            
            response = self.call_llm(prompt, temperature=0.5)
            result = self.extract_json_from_response(response)
            
            if result and isinstance(result, dict):
                result.setdefault("hypothesis_id", hyp.get("hypothesis_id", "H1"))
                result.setdefault("suspect", hyp.get("suspect", ""))
                result.setdefault("can_eliminate", False)
                result.setdefault("updated_confidence", hyp.get("initial_confidence", 0.5))
                results.append(result)
            else:
                results.append({
                    "hypothesis_id": hyp.get("hypothesis_id", "H1"),
                    "suspect": hyp.get("suspect", ""),
                    "can_eliminate": False,
                    "elimination_reason": "无法确定",
                    "supporting_points": [],
                    "contradicting_points": [],
                    "updated_confidence": hyp.get("initial_confidence", 0.5)
                })
        
        return results
    
    def _draw_final_conclusion(self, elimination_results, reasoning_chain, hypotheses=None):
        """得出最终结论"""
        hypotheses = hypotheses or []
        # 构建 hypothesis_id → suspect 的映射
        hyp_suspect_map = {}
        for h in hypotheses:
            hid = h.get("hypothesis_id", "")
            sname = h.get("suspect", "")
            if hid:
                hyp_suspect_map[hid] = sname

        valid = [r for r in elimination_results if not r.get("can_eliminate", True)]
        
        if not valid:
            return {
                "conclusion": "无法得出结论",
                "top_suspect": "未知",
                "confidence": 0.0,
                "reason": "所有假设被排除",
                "suspect_ranking": [],
            }
        
        best = max(valid, key=lambda x: x.get("updated_confidence", 0))
        key_conclusions = [s.get("conclusion", "") for s in reasoning_chain[-3:] if s.get("conclusion")]
        
        # 获取真凶名（优先suspect字段，否则hypothesis_id映射）
        best_hid = best.get("hypothesis_id", "")
        top_suspect = hyp_suspect_map.get(best_hid, best.get("suspect", best_hid))
        
        # 按置信度排序所有未排除的假设
        suspect_ranking = sorted(valid, key=lambda x: -x.get("updated_confidence", 0))
        
        return {
            "conclusion": best.get("elimination_reason", ""),
            "top_suspect": top_suspect,
            "confidence": best.get("updated_confidence", 0),
            "supporting_evidence": best.get("supporting_points", []),
            "key_conclusions": key_conclusions,
            "suspect_ranking": [
                {
                    "name": hyp_suspect_map.get(r.get("hypothesis_id", ""), r.get("hypothesis_id", "?")),
                    "score": r.get("updated_confidence", 0),
                    "can_eliminate": r.get("can_eliminate", False),
                }
                for r in suspect_ranking
            ],
        }
    
    def _calculate_confidence(self, final_conclusion, elimination_results, reasoning_chain):
        """计算综合置信度"""
        base = final_conclusion.get("confidence", 0)
        chain_avg = sum(s.get("confidence", 0.7) for s in reasoning_chain) / max(len(reasoning_chain), 1)
        elim_rate = sum(1 for r in elimination_results if r.get("can_eliminate", False)) / max(len(elimination_results), 1)
        
        return min(max(base * 0.5 + chain_avg * 0.3 + elim_rate * 0.2, 0), 1)
