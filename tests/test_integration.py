"""
集成测试脚本 - Day 3
测试完整的双路径RAG + Agent系统
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents import AgentOrchestrator
from api.llm_client import LLMClient
from rag.rag_anything import RAGAnything
from rag.agentic_rag import AgenticRAG
from rag.fusion import FusionEngine
from loguru import logger
import yaml


class IntegrationTest:
    """集成测试类"""
    
    def __init__(self):
        """初始化测试"""
        self.logger = logger.bind(module="test")
        
        # 加载配置
        with open("./config/config.yaml", 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化组件
        self.llm_client = None
        self.orchestrator = None
        self.rag_anything = None
        self.agentic_rag = None
        self.fusion_engine = None
    
    def setup(self):
        """设置测试环境"""
        self.logger.info("=" * 60)
        self.logger.info("设置测试环境...")
        
        # 1. 初始化LLM客户端
        self.logger.info("1. 初始化LLM客户端...")
        self.llm_client = LLMClient("./config/api_keys.yaml")
        
        # 测试LLM连接
        test_result = self.llm_client.test_connection()
        self.logger.info(f"LLM连接测试: {test_result}")
        
        # 2. 初始化Agent系统
        self.logger.info("2. 初始化Agent系统...")
        self.orchestrator = AgentOrchestrator(self.config, self.llm_client)
        
        # 3. 初始化RAG系统（可选）
        self.logger.info("3. 初始化RAG系统...")
        try:
            self.rag_anything = RAGAnything(self.config)
            self.agentic_rag = AgenticRAG(self.config)
            self.agentic_rag.set_llm_client(self.llm_client)
            self.fusion_engine = FusionEngine(self.config)
            
            # 注入到协调器
            self.orchestrator.set_rag_systems(
                self.rag_anything,
                self.agentic_rag,
                self.fusion_engine
            )
            
            self.logger.info("✅ RAG系统初始化成功")
            
        except Exception as e:
            self.logger.warning(f"⚠️ RAG系统初始化失败: {e}")
            self.logger.info("继续测试，跳过RAG功能")
        
        self.logger.info("=" * 60)
    
    def test_llm_client(self):
        """测试LLM客户端"""
        self.logger.info("\n测试1: LLM客户端")
        self.logger.info("-" * 60)
        
        # 测试简单对话
        response = self.llm_client.simple_chat("你好，请用一句话介绍自己。")
        self.logger.info(f"LLM响应: {response[:100]}...")
        
        # 测试JSON提取
        json_response = self.llm_client.simple_chat(
            '请返回一个简单的JSON: {"name": "测试", "value": 123}'
        )
        json_obj = self.llm_client.extract_json(json_response)
        self.logger.info(f"JSON提取: {json_obj}")
        
        # 查看统计
        stats = self.llm_client.get_stats()
        self.logger.info(f"LLM统计: {stats}")
    
    def test_agents(self):
        """测试Agent系统"""
        self.logger.info("\n测试2: Agent系统")
        self.logger.info("-" * 60)
        
        # 准备测试案例
        test_case = {
            "case_text": """
            2024年3月15日晚上，富豪张三被发现死在自己的别墅中。
            尸体是在第二天早上由管家李四发现的。
            
            嫌疑人：
            1. 管家李四 - 有别墅钥匙，当晚在值班
            2. 商业竞争对手王五 - 近期与张三有商业纠纷
            3. 张三的儿子张小三 - 近期因遗产问题与父亲争吵
            
            证据：
            1. 现场发现一把带血的刀
            2. 监控显示王五在案发前离开别墅
            3. 张小三的手机定位显示他在案发时在附近
            """,
            "images": [],
            "suspects": [
                {"name": "李四", "role": "管家"},
                {"name": "王五", "role": "竞争对手"},
                {"name": "张小三", "role": "儿子"}
            ],
            "evidence": [
                "带血的刀",
                "监控录像",
                "手机定位记录"
            ],
            "case_type": "modern"
        }
        
        # 运行调查
        self.logger.info("开始运行完整调查...")
        results = self.orchestrator.run_full_investigation(test_case)
        
        # 显示结果
        self.logger.info("\n调查结果:")
        self.logger.info(f"✅ 线索数量: {results['summary']['clue_count']}")
        self.logger.info(f"✅ 嫌疑人数量: {results['summary']['suspect_count']}")
        self.logger.info(f"✅ 证据数量: {results['summary']['evidence_count']}")
        self.logger.info(f"✅ RAG增强: {'是' if results['summary']['has_rag'] else '否'}")
        
        if results.get("reasoning", {}).get("data", {}).get("conclusion"):
            self.logger.info(f"\n推理结论:")
            self.logger.info(results["reasoning"]["data"]["conclusion"][:200])
    
    def test_rag_systems(self):
        """测试RAG系统"""
        if not self.rag_anything or not self.agentic_rag:
            self.logger.warning("\n跳过测试3: RAG系统未初始化")
            return
        
        self.logger.info("\n测试3: RAG系统")
        self.logger.info("-" * 60)
        
        # 测试检索
        query = "谋杀案件的线索提取"
        
        # RAG-Anything检索
        self.logger.info(f"RAG-Anything检索: {query}")
        rag_docs = self.rag_anything.retrieve(query, top_k=3)
        self.logger.info(f"检索到 {len(rag_docs)} 个文档")
        
        # Agentic RAG检索
        self.logger.info(f"Agentic RAG检索: {query}")
        agentic_result = self.agentic_rag.retrieve_with_reasoning(
            query,
            context={"test": True}
        )
        self.logger.info(f"完成 {len(agentic_result['iterations'])} 次迭代")
        
        # 融合
        if rag_docs and agentic_result:
            self.logger.info("融合双路径结果...")
            fused = self.fusion_engine.fuse(
                {"documents": rag_docs, "confidence": 0.7},
                agentic_result,
                query
            )
            self.logger.info(f"融合置信度: {fused.get('confidence', 0):.2f}")
    
    def run_all_tests(self):
        """运行所有测试"""
        try:
            self.setup()
            
            self.test_llm_client()
            self.test_agents()
            self.test_rag_systems()
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("✅ 所有测试完成！")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    
    # 运行测试
    test = IntegrationTest()
    test.run_all_tests()


if __name__ == "__main__":
    main()
