"""
双路径融合算法
融合RAG-Anything和Agentic RAG的结果
"""

from typing import Dict, List, Any, Optional
from loguru import logger
import json


class FusionEngine:
    """
    双路径融合引擎
    融合RAG-Anything和Agentic RAG的推理结果
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化融合引擎
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logger.bind(module="fusion")
        
        # 融合配置
        fusion_config = self.config.get("rag", {}).get("fusion", {})
        self.method = fusion_config.get("method", "weighted")
        self.rag_anything_weight = fusion_config.get("rag_anything_weight", 0.6)
        self.agentic_rag_weight = fusion_config.get("agentic_rag_weight", 0.4)
        
        self.logger.info(
            f"融合引擎初始化完成: 方法={self.method}, "
            f"权重={self.rag_anything_weight}/{self.agentic_rag_weight}"
        )
    
    def fuse(
        self,
        rag_anything_result: Dict[str, Any],
        agentic_rag_result: Dict[str, Any],
        query: str = ""
    ) -> Dict[str, Any]:
        """
        融合双路径结果
        
        Args:
            rag_anything_result: RAG-Anything路径结果
            agentic_rag_result: Agentic RAG路径结果
            query: 原始查询
            
        Returns:
            融合后的结果
        """
        self.logger.info(f"开始融合双路径结果: {query[:50] if query else '无查询'}...")
        
        if self.method == "weighted":
            return self._weighted_fusion(rag_anything_result, agentic_rag_result)
        elif self.method == "voting":
            return self._voting_fusion(rag_anything_result, agentic_rag_result)
        elif self.method == "ensemble":
            return self._ensemble_fusion(rag_anything_result, agentic_rag_result)
        else:
            self.logger.warning(f"未知融合方法: {self.method}, 使用加权融合")
            return self._weighted_fusion(rag_anything_result, agentic_rag_result)
    
    def _weighted_fusion(
        self,
        rag_anything_result: Dict[str, Any],
        agentic_rag_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        加权融合
        
        Args:
            rag_anything_result: RAG-Anything结果
            agentic_rag_result: Agentic RAG结果
            
        Returns:
            融合结果
        """
        # 提取置信度
        rag_confidence = rag_anything_result.get("confidence", 0.5)
        agentic_confidence = agentic_rag_result.get("confidence", 0.5)
        
        # 计算加权置信度
        weighted_confidence = (
            rag_confidence * self.rag_anything_weight +
            agentic_confidence * self.agentic_rag_weight
        )
        
        # 融合答案
        rag_answer = rag_anything_result.get("answer", "")
        agentic_answer = agentic_rag_result.get("answer", "")
        
        # 如果两个答案都存在，组合它们
        if rag_answer and agentic_answer:
            fused_answer = self._combine_answers(rag_answer, agentic_answer)
        else:
            fused_answer = rag_answer or agentic_answer
        
        # 融合推理链
        rag_chain = rag_anything_result.get("reasoning_chain", [])
        agentic_chain = agentic_rag_result.get("reasoning_chain", [])
        
        fused_chain = self._merge_reasoning_chains(rag_chain, agentic_chain)
        
        # 融合证据
        rag_evidence = rag_anything_result.get("evidence", [])
        agentic_evidence = agentic_rag_result.get("evidence", [])
        
        fused_evidence = self._merge_evidence(rag_evidence, agentic_evidence)
        
        # 构建融合结果
        result = {
            "fusion_method": "weighted",
            "confidence": weighted_confidence,
            "answer": fused_answer,
            "reasoning_chain": fused_chain,
            "evidence": fused_evidence,
            "path_contributions": {
                "rag_anything": {
                    "weight": self.rag_anything_weight,
                    "confidence": rag_confidence,
                    "contribution": rag_confidence * self.rag_anything_weight
                },
                "agentic_rag": {
                    "weight": self.agentic_rag_weight,
                    "confidence": agentic_confidence,
                    "contribution": agentic_confidence * self.agentic_rag_weight
                }
            },
            "fusion_metadata": {
                "rag_answer_length": len(rag_answer),
                "agentic_answer_length": len(agentic_answer),
                "chain_steps": len(fused_chain),
                "evidence_count": len(fused_evidence)
            }
        }
        
        self.logger.info(f"加权融合完成: 置信度={weighted_confidence:.2f}")
        
        return result
    
    def _voting_fusion(
        self,
        rag_anything_result: Dict[str, Any],
        agentic_rag_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        投票融合
        
        Args:
            rag_anything_result: RAG-Anything结果
            agentic_rag_result: Agentic RAG结果
            
        Returns:
            融合结果
        """
        # 提取答案
        rag_answer = rag_anything_result.get("answer", "")
        agentic_answer = agentic_rag_result.get("answer", "")
        
        # 简单的投票机制（这里简化为选择置信度更高的）
        rag_confidence = rag_anything_result.get("confidence", 0.5)
        agentic_confidence = agentic_rag_result.get("confidence", 0.5)
        
        if rag_confidence >= agentic_confidence:
            winner = "rag_anything"
            winner_result = rag_anything_result
        else:
            winner = "agentic_rag"
            winner_result = agentic_rag_result
        
        result = {
            "fusion_method": "voting",
            "winner": winner,
            "confidence": winner_result.get("confidence", 0.5),
            "answer": winner_result.get("answer", ""),
            "reasoning_chain": winner_result.get("reasoning_chain", []),
            "evidence": winner_result.get("evidence", []),
            "voting_details": {
                "rag_confidence": rag_confidence,
                "agentic_confidence": agentic_confidence,
                "margin": abs(rag_confidence - agentic_confidence)
            }
        }
        
        self.logger.info(f"投票融合完成: 获胜者={winner}")
        
        return result
    
    def _ensemble_fusion(
        self,
        rag_anything_result: Dict[str, Any],
        agentic_rag_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        集成融合（组合所有信息）
        
        Args:
            rag_anything_result: RAG-Anything结果
            agentic_rag_result: Agentic RAG结果
            
        Returns:
            融合结果
        """
        # 集成所有信息
        all_evidence = []
        all_reasoning = []
        
        # 收集证据
        all_evidence.extend(rag_anything_result.get("evidence", []))
        all_evidence.extend(agentic_rag_result.get("evidence", []))
        
        # 收集推理步骤
        all_reasoning.extend(rag_anything_result.get("reasoning_chain", []))
        all_reasoning.extend(agentic_rag_result.get("reasoning_chain", []))
        
        # 去重
        unique_evidence = self._deduplicate_list(all_evidence, key="id")
        unique_reasoning = self._deduplicate_list(all_reasoning, key="step")
        
        # 计算平均置信度
        rag_confidence = rag_anything_result.get("confidence", 0.5)
        agentic_confidence = agentic_rag_result.get("confidence", 0.5)
        avg_confidence = (rag_confidence + agentic_confidence) / 2
        
        # 组合答案
        rag_answer = rag_anything_result.get("answer", "")
        agentic_answer = agentic_rag_result.get("answer", "")
        fused_answer = self._combine_answers(rag_answer, agentic_answer)
        
        result = {
            "fusion_method": "ensemble",
            "confidence": avg_confidence,
            "answer": fused_answer,
            "reasoning_chain": unique_reasoning,
            "evidence": unique_evidence,
            "ensemble_metadata": {
                "total_evidence": len(unique_evidence),
                "total_reasoning_steps": len(unique_reasoning),
                "rag_confidence": rag_confidence,
                "agentic_confidence": agentic_confidence
            }
        }
        
        self.logger.info(f"集成融合完成: {len(unique_evidence)}条证据, {len(unique_reasoning)}步推理")
        
        return result
    
    def _combine_answers(self, answer1: str, answer2: str) -> str:
        """
        组合两个答案
        
        Args:
            answer1: 答案1
            answer2: 答案2
            
        Returns:
            组合后的答案
        """
        if not answer1:
            return answer2
        if not answer2:
            return answer1
        
        # 简单的组合（实际应该更智能）
        if answer1 == answer2:
            return answer1
        
        # 如果答案不同，组合它们
        return f"{answer1}\n\n【补充分析】\n{answer2}"
    
    def _merge_reasoning_chains(
        self,
        chain1: List[Dict[str, Any]],
        chain2: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        合并推理链
        
        Args:
            chain1: 推理链1
            chain2: 推理链2
            
        Returns:
            合并后的推理链
        """
        merged = []
        
        # 添加路径1的推理步骤
        for i, step in enumerate(chain1):
            merged.append({
                **step,
                "source": "rag_anything",
                "step": len(merged) + 1
            })
        
        # 添加路径2的推理步骤
        for i, step in enumerate(chain2):
            merged.append({
                **step,
                "source": "agentic_rag",
                "step": len(merged) + 1
            })
        
        return merged
    
    def _merge_evidence(
        self,
        evidence1: List[Dict[str, Any]],
        evidence2: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        合并证据
        
        Args:
            evidence1: 证据列表1
            evidence2: 证据列表2
            
        Returns:
            合并后的证据列表
        """
        merged = []
        
        # 添加路径1的证据
        for evidence in evidence1:
            merged.append({
                **evidence,
                "source": "rag_anything"
            })
        
        # 添加路径2的证据
        for evidence in evidence2:
            merged.append({
                **evidence,
                "source": "agentic_rag"
            })
        
        return merged
    
    def _deduplicate_list(
        self,
        items: List[Dict[str, Any]],
        key: str = "id"
    ) -> List[Dict[str, Any]]:
        """
        去重列表
        
        Args:
            items: 项目列表
            key: 去重键
            
        Returns:
            去重后的列表
        """
        seen = set()
        unique = []
        
        for item in items:
            item_key = item.get(key, str(item))
            
            if item_key not in seen:
                seen.add(item_key)
                unique.append(item)
        
        return unique
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计数据
        """
        return {
            "fusion_method": self.method,
            "rag_anything_weight": self.rag_anything_weight,
            "agentic_rag_weight": self.agentic_rag_weight,
            "status": "initialized"
        }
