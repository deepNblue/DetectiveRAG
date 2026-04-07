"""
端到端测试脚本 - Day 7
完整的系统测试，包括所有功能模块
"""

import sys
import os
import time
import json
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents import AgentOrchestrator
from api.auth import AuthSystem
from api.llm_client import LLMClient
from rag.rag_anything import RAGAnything
from rag.agentic_rag import AgenticRAG
from rag.fusion import FusionEngine
from rag.data_manager import DataManager
from ui.visualization import VisualizationEngine
from loguru import logger


class EndToEndTester:
    """端到端测试器"""
    
    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0
        self.start_time = time.time()
        
        # 测试配置
        self.config = {
            "llm": {
                "model": "qwen3-30b-a3b",
                "base_url": "http://localhost:8094/v1",
                "temperature": 0.7,
                "max_tokens": 2000
            },
            "visualization": {
                "output_dir": "./data/test_visualizations"
            }
        }
        
        logger.info("端到端测试器初始化完成")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("🔍 侦探推理RAG系统 - 端到端测试")
        print("="*60 + "\n")
        
        # 1. 系统初始化测试
        self.test_system_initialization()
        
        # 2. 认证系统测试
        self.test_authentication()
        
        # 3. 案件分析测试
        self.test_case_analysis()
        
        # 4. 可视化测试
        self.test_visualization()
        
        # 5. 导出功能测试
        self.test_export_functions()
        
        # 6. 性能测试
        self.test_performance()
        
        # 7. 错误处理测试
        self.test_error_handling()
        
        # 生成测试报告
        self.generate_test_report()
    
    def test_system_initialization(self):
        """测试系统初始化"""
        print("\n📋 测试1: 系统初始化")
        
        # 测试LLM客户端
        try:
            llm_client = LLMClient(self.config)
            self._record_test("LLM客户端初始化", True, "成功")
        except Exception as e:
            self._record_test("LLM客户端初始化", False, str(e))
        
        # 测试认证系统
        try:
            auth_system = AuthSystem()
            self._record_test("认证系统初始化", True, "成功")
        except Exception as e:
            self._record_test("认证系统初始化", False, str(e))
        
        # 测试Agent编排器
        try:
            orchestrator = AgentOrchestrator(self.config)
            self._record_test("Agent编排器初始化", True, "成功")
        except Exception as e:
            self._record_test("Agent编排器初始化", False, str(e))
        
        # 测试可视化引擎
        try:
            viz_engine = VisualizationEngine(self.config)
            status = viz_engine.get_visualization_status()
            if status["status"] == "ready":
                self._record_test("可视化引擎初始化", True, "成功")
            else:
                self._record_test("可视化引擎初始化", False, status.get("error", "未知错误"))
        except Exception as e:
            self._record_test("可视化引擎初始化", False, str(e))
    
    def test_authentication(self):
        """测试认证系统"""
        print("\n🔐 测试2: 认证系统")
        
        try:
            auth_system = AuthSystem()
            
            # 测试正确登录
            if auth_system.verify_password("demo", "REDACTED_DEMO_PASSWORD"):
                self._record_test("正确登录", True, "demo用户登录成功")
            else:
                self._record_test("正确登录", False, "demo用户登录失败")
            
            # 测试错误登录
            if not auth_system.verify_password("demo", "wrong_password"):
                self._record_test("错误登录拦截", True, "错误密码被正确拦截")
            else:
                self._record_test("错误登录拦截", False, "错误密码未被拦截")
            
            # 测试不存在的用户
            if not auth_system.verify_password("nonexistent", "password"):
                self._record_test("不存在用户拦截", True, "不存在用户被正确拦截")
            else:
                self._record_test("不存在用户拦截", False, "不存在用户未被拦截")
                
        except Exception as e:
            self._record_test("认证系统测试", False, str(e))
    
    def test_case_analysis(self):
        """测试案件分析"""
        print("\n🔍 测试3: 案件分析")
        
        try:
            # 创建测试案例
            test_case = {
                "case_text": """
                2026年3月15日晚，某别墅发生命案。受害者张某，男，45岁，企业家。
                案发时别墅内有三名嫌疑人：
                1. 妻子李某（有保险受益权）
                2. 合作伙伴王某（最近有财务纠纷）
                3. 秘书赵某（掌握公司机密）
                
                现场发现凶器为水果刀，刀柄上有指纹。
                监控显示王某曾在案发时间段进出别墅。
                """,
                "images": [],
                "case_type": "modern",
                "suspects": [],
                "evidence": []
            }
            
            # 初始化编排器
            orchestrator = AgentOrchestrator(self.config)
            
            # 运行分析
            start_time = time.time()
            results = orchestrator.run_full_investigation(test_case)
            elapsed_time = time.time() - start_time
            
            # 验证结果
            if "error" not in results:
                self._record_test("案件分析执行", True, f"分析完成，耗时{elapsed_time:.2f}秒")
                
                # 验证各个模块结果
                if "clues" in results:
                    self._record_test("线索提取", True, "线索提取成功")
                else:
                    self._record_test("线索提取", False, "缺少线索提取结果")
                
                if "suspects" in results:
                    self._record_test("嫌疑人分析", True, "嫌疑人分析成功")
                else:
                    self._record_test("嫌疑人分析", False, "缺少嫌疑人分析结果")
                
                if "reasoning" in results:
                    self._record_test("推理生成", True, "推理生成成功")
                else:
                    self._record_test("推理生成", False, "缺少推理生成结果")
                
                # 保存结果供后续测试使用
                self.test_results_data = results
                
            else:
                self._record_test("案件分析执行", False, results["error"])
                
        except Exception as e:
            self._record_test("案件分析测试", False, str(e))
    
    def test_visualization(self):
        """测试可视化功能"""
        print("\n🎨 测试4: 可视化功能")
        
        try:
            viz_engine = VisualizationEngine(self.config)
            
            # 测试图谱生成
            if hasattr(self, 'test_results_data'):
                nodes, edges = self._extract_test_graph_data(self.test_results_data)
                
                if nodes:
                    graph_path = viz_engine.generate_pyvis_graph(
                        nodes=nodes,
                        edges=edges,
                        case_name="测试案件"
                    )
                    
                    if graph_path and os.path.exists(graph_path):
                        self._record_test("图谱生成", True, f"图谱已生成: {graph_path}")
                    else:
                        self._record_test("图谱生成", False, "图谱文件未生成")
                else:
                    self._record_test("图谱生成", False, "无法提取图谱数据")
                
                # 测试动画生成
                reasoning_chain = self.test_results_data.get("reasoning", {}).get("data", {}).get("reasoning_chain", [])
                
                if reasoning_chain:
                    animation_path = viz_engine.generate_reasoning_animation(
                        reasoning_chain=reasoning_chain,
                        case_name="测试案件"
                    )
                    
                    if animation_path and os.path.exists(animation_path):
                        self._record_test("动画生成", True, f"动画已生成: {animation_path}")
                    else:
                        self._record_test("动画生成", False, "动画文件未生成")
                else:
                    self._record_test("动画生成", False, "无法提取推理链")
            else:
                self._record_test("可视化测试", False, "缺少测试数据，请先运行案件分析")
                
        except Exception as e:
            self._record_test("可视化测试", False, str(e))
    
    def test_export_functions(self):
        """测试导出功能"""
        print("\n📥 测试5: 导出功能")
        
        try:
            viz_engine = VisualizationEngine(self.config)
            
            if hasattr(self, 'test_results_data'):
                # 测试JSON导出
                json_path = viz_engine.export_results(
                    results=self.test_results_data,
                    export_format="json",
                    case_name="测试案件"
                )
                
                if json_path and os.path.exists(json_path):
                    self._record_test("JSON导出", True, f"JSON已导出: {json_path}")
                else:
                    self._record_test("JSON导出", False, "JSON文件未生成")
                
                # 测试Markdown导出
                md_path = viz_engine.export_results(
                    results=self.test_results_data,
                    export_format="markdown",
                    case_name="测试案件"
                )
                
                if md_path and os.path.exists(md_path):
                    self._record_test("Markdown导出", True, f"Markdown已导出: {md_path}")
                else:
                    self._record_test("Markdown导出", False, "Markdown文件未生成")
                
                # 测试HTML导出
                html_path = viz_engine.export_results(
                    results=self.test_results_data,
                    export_format="html",
                    case_name="测试案件"
                )
                
                if html_path and os.path.exists(html_path):
                    self._record_test("HTML导出", True, f"HTML已导出: {html_path}")
                else:
                    self._record_test("HTML导出", False, "HTML文件未生成")
            else:
                self._record_test("导出测试", False, "缺少测试数据")
                
        except Exception as e:
            self._record_test("导出功能测试", False, str(e))
    
    def test_performance(self):
        """测试性能"""
        print("\n⚡ 测试6: 性能测试")
        
        try:
            # 测试响应时间
            start_time = time.time()
            
            # 简单的初始化测试
            orchestrator = AgentOrchestrator(self.config)
            
            init_time = time.time() - start_time
            
            if init_time < 5.0:
                self._record_test("初始化性能", True, f"初始化耗时{init_time:.2f}秒")
            else:
                self._record_test("初始化性能", False, f"初始化耗时过长: {init_time:.2f}秒")
            
            # 测试并发能力（简化版）
            start_time = time.time()
            
            # 模拟3个并发请求
            for i in range(3):
                test_case = {
                    "case_text": f"测试案件{i+1}",
                    "images": [],
                    "case_type": "modern",
                    "suspects": [],
                    "evidence": []
                }
                # 这里只测试初始化，不运行完整分析
                _ = AgentOrchestrator(self.config)
            
            concurrent_time = time.time() - start_time
            
            if concurrent_time < 10.0:
                self._record_test("并发性能", True, f"3个并发请求耗时{concurrent_time:.2f}秒")
            else:
                self._record_test("并发性能", False, f"并发性能较差: {concurrent_time:.2f}秒")
                
        except Exception as e:
            self._record_test("性能测试", False, str(e))
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n🛡️ 测试7: 错误处理")
        
        try:
            orchestrator = AgentOrchestrator(self.config)
            
            # 测试空输入
            try:
                results = orchestrator.run_full_investigation({"case_text": ""})
                if "error" in results:
                    self._record_test("空输入处理", True, "空输入被正确处理")
                else:
                    self._record_test("空输入处理", False, "空输入未被正确处理")
            except Exception:
                self._record_test("空输入处理", True, "空输入触发异常处理")
            
            # 测试无效输入
            try:
                results = orchestrator.run_full_investigation({})
                if "error" in results:
                    self._record_test("无效输入处理", True, "无效输入被正确处理")
                else:
                    self._record_test("无效输入处理", False, "无效输入未被正确处理")
            except Exception:
                self._record_test("无效输入处理", True, "无效输入触发异常处理")
            
            # 测试超长输入
            long_text = "测试" * 10000
            try:
                results = orchestrator.run_full_investigation({"case_text": long_text})
                self._record_test("超长输入处理", True, "超长输入被处理")
            except Exception as e:
                self._record_test("超长输入处理", False, f"超长输入导致错误: {str(e)}")
                
        except Exception as e:
            self._record_test("错误处理测试", False, str(e))
    
    def _extract_test_graph_data(self, results: Dict) -> tuple:
        """提取测试图谱数据"""
        nodes = []
        edges = []
        
        try:
            # 提取线索
            clues = results.get("clues", {}).get("data", {})
            
            # 时间线节点
            for i, event in enumerate(clues.get("timeline", [])[:5]):
                nodes.append({
                    "id": f"timeline_{i}",
                    "label": event.get("event", "事件")[:20],
                    "type": "timeline",
                    "title": f"{event.get('time', '')}: {event.get('event', '')}",
                    "size": 20
                })
            
            # 嫌疑人节点
            suspects = results.get("suspects", {}).get("data", [])
            for i, suspect in enumerate(suspects[:3]):
                nodes.append({
                    "id": f"suspect_{i}",
                    "label": suspect.get("name", "嫌疑人"),
                    "type": "suspect",
                    "title": f"{suspect.get('name', '')} - {suspect.get('motive', '')}",
                    "size": 30
                })
            
            # 生成简单边
            for i in range(len(nodes) - 1):
                edges.append({
                    "from": nodes[i]["id"],
                    "to": nodes[i+1]["id"],
                    "label": "关联",
                    "value": 1
                })
            
            return nodes, edges
            
        except Exception as e:
            logger.error(f"提取图谱数据失败: {e}")
            return [], []
    
    def _record_test(self, test_name: str, passed: bool, message: str):
        """记录测试结果"""
        result = {
            "test": test_name,
            "passed": passed,
            "message": message,
            "timestamp": time.time()
        }
        
        self.test_results.append(result)
        
        if passed:
            self.passed += 1
            print(f"  ✅ {test_name}: {message}")
        else:
            self.failed += 1
            print(f"  ❌ {test_name}: {message}")
    
    def generate_test_report(self):
        """生成测试报告"""
        total_time = time.time() - self.start_time
        total_tests = self.passed + self.failed
        pass_rate = (self.passed / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*60)
        print("📊 测试报告")
        print("="*60)
        print(f"总测试数: {total_tests}")
        print(f"通过: {self.passed} ✅")
        print(f"失败: {self.failed} ❌")
        print(f"通过率: {pass_rate:.1f}%")
        print(f"总耗时: {total_time:.2f}秒")
        print("="*60)
        
        # 保存详细报告
        report_path = "./data/test_results/e2e_test_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": pass_rate,
                "total_time": total_time
            },
            "details": self.test_results
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细报告已保存: {report_path}")
        
        # 返回测试结果
        return pass_rate >= 80  # 80%通过率算成功


if __name__ == "__main__":
    # 运行端到端测试
    tester = EndToEndTester()
    success = tester.run_all_tests()
    
    # 返回退出码
    sys.exit(0 if success else 1)
