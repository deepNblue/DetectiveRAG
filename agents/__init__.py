"""
Agent系统初始化 - ASMR双线架构版
集成传统RAG线路 + ASMR多智能体线路
"""

from .base_agent import BaseAgent
from .clue_extractor import ClueExtractorAgent
from .suspect_analyzer import SuspectAnalyzerAgent
from .evidence_connector import EvidenceConnectorAgent
from .reasoning_generator import ReasoningGeneratorAgent
from .graph_builder import GraphBuilderAgent

# ASMR线路
from .asmr import (
    ASMROrchestrator,
    ExpertVotingEngine,
)
from .asmr.dual_track_fusion import DualTrackFusionEngine

from loguru import logger

__all__ = [
    "BaseAgent",
    "ClueExtractorAgent",
    "SuspectAnalyzerAgent",
    "EvidenceConnectorAgent",
    "ReasoningGeneratorAgent",
    "GraphBuilderAgent",
    "ASMROrchestrator",
    "ExpertVotingEngine",
    "DualTrackFusionEngine",
]


class AgentOrchestrator:
    """
    Agent协调器（ASMR双线架构版）
    协调传统RAG线路 + ASMR多智能体线路，融合输出最终结论
    """

    def __init__(self, config: dict = None, llm_client=None):
        self.config = config or {}
        self.llm_client = llm_client
        self.logger = logger.bind(module="AgentOrchestrator")

        # 传统线路: 5个专业化Agent
        self.clue_extractor = ClueExtractorAgent(config, llm_client)
        self.suspect_analyzer = SuspectAnalyzerAgent(config, llm_client)
        self.evidence_connector = EvidenceConnectorAgent(config, llm_client)
        self.reasoning_generator = ReasoningGeneratorAgent(config, llm_client)
        self.graph_builder = GraphBuilderAgent(config, llm_client)

        # ASMR线路
        self.asmr_orchestrator = ASMROrchestrator(llm_client=llm_client, config=config)

        # 双线融合
        self.fusion_engine = DualTrackFusionEngine()

        # 传统RAG系统（可选注入）
        self.rag_anything = None
        self.agentic_rag = None

        self.logger.info("Agent协调器(ASMR双线版)初始化完成")

    def set_rag_systems(self, rag_anything, agentic_rag):
        """注入传统RAG系统"""
        self.rag_anything = rag_anything
        self.agentic_rag = agentic_rag
        self.logger.info("传统RAG系统已注入")

    def run_full_investigation(self, case_data: dict) -> dict:
        """
        运行完整双线调查流程

        Args:
            case_data: {
                "case_text": str,
                "images": list,
                "suspects": list,
                "evidence": list,
                "case_type": str
            }

        Returns:
            双线融合调查结果
        """
        import time
        start_time = time.time()
        results = {}

        case_text = case_data.get("case_text", "")
        suspects = case_data.get("suspects", [])
        evidence_list = case_data.get("evidence", [])
        case_type = case_data.get("case_type", "modern")

        # ==========================================
        # 线路A: 传统RAG (5步流水线，保持不变)
        # ==========================================
        self.logger.info("🔵 线路A: 传统RAG启动")

        # Step 1: 提取线索
        self.logger.info("  Step 1: 提取线索...")
        clues_result = self.clue_extractor.process({
            "case_text": case_text,
            "images": case_data.get("images", []),
            "case_type": case_type
        })
        results["clues"] = clues_result

        # Step 2: 分析嫌疑人
        self.logger.info("  Step 2: 分析嫌疑人...")
        suspect_analyses = []
        for suspect in suspects:
            analysis = self.suspect_analyzer.process({
                "suspect_info": suspect,
                "case_clues": clues_result["data"],
                "evidence": evidence_list
            })
            suspect_analyses.append(analysis)
        results["suspect_analyses"] = suspect_analyses

        # Step 3: 关联证据
        self.logger.info("  Step 3: 关联证据...")
        evidence_result = self.evidence_connector.process({
            "evidence_list": evidence_list,
            "suspects": suspects,
            "case_clues": clues_result["data"]
        })
        results["evidence_connections"] = evidence_result

        # Step 4: 生成推理
        self.logger.info("  Step 4: 生成推理...")
        reasoning_result = self.reasoning_generator.process({
            "case_clues": clues_result["data"],
            "suspect_analyses": suspect_analyses,
            "evidence_connections": evidence_result["data"]
        })
        results["reasoning"] = reasoning_result

        # Step 5: 构建图谱
        self.logger.info("  Step 5: 构建知识图谱...")
        graph_result = self.graph_builder.process({
            "case_clues": clues_result["data"],
            "suspects": suspects,
            "evidence": evidence_list,
            "reasoning_chain": reasoning_result["data"].get("reasoning_chain", [])
        })
        results["graph"] = graph_result

        self.logger.info("🔵 线路A完成")

        # ==========================================
        # 线路B: ASMR多智能体
        # ==========================================
        self.logger.info("🟢 线路B: ASMR多智能体启动")
        asmr_result = self.asmr_orchestrator.run(
            case_text=case_text,
            suspects=suspects,
            case_type=case_type
        )
        results["asmr"] = asmr_result
        self.logger.info("🟢 线路B完成")

        # ==========================================
        # 双线融合
        # ==========================================
        self.logger.info("🔀 双线融合开始")
        fusion_result = self.fusion_engine.fuse(
            traditional_result=results,
            asmr_result=asmr_result,
            case_data=case_data
        )
        results["fusion"] = fusion_result

        # 生成报告
        results["report"] = self.fusion_engine.generate_report(fusion_result, asmr_result)

        total_time = time.time() - start_time
        results["timing"] = {
            "total": round(total_time, 1),
            "traditional": "included",
            "asmr": asmr_result.get("timing", {}),
            "fusion": fusion_result.get("timing", {}),
        }

        self.logger.info(f"🕵️ 双线调查完成! 总耗时: {total_time:.1f}s")
        self.logger.info(f"结论: {fusion_result['conclusion']['culprit']}, "
                         f"置信度: {fusion_result['conclusion']['confidence']:.2%}")

        return results
