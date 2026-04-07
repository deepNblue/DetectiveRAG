"""
WebUI主程序
黑客帝国风格的侦探推理RAG系统界面
"""

import gradio as gr
from typing import Dict, List, Any
import yaml
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents import AgentOrchestrator
from api.auth import AuthSystem
from api.llm_client import LLMClient
from rag.rag_anything import RAGAnything
from rag.agentic_rag import AgenticRAG
from rag.fusion import FusionEngine
from rag.data_manager import DataManager
from loguru import logger


class DetectiveWebUI:
    """
    侦探推理RAG系统WebUI
    黑客帝国风格界面
    """
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        初始化WebUI
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = logger.bind(module="webui")
        self.config = self._load_config(config_path)
        
        # 初始化Agent系统
        self.orchestrator = AgentOrchestrator(self.config)
        
        # 初始化认证系统
        self.auth_system = AuthSystem()
        
        # 初始化LLM客户端
        self.llm_client = LLMClient()
        
        # 初始化RAG系统（延迟初始化）
        self.rag_anything = None
        self.agentic_rag = None
        self.fusion_engine = None
        self.data_manager = None
        
        # 当前用户
        self.current_user = None
        
        self.logger.info("WebUI初始化完成")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def login(self, username: str, password: str) -> tuple:
        """
        用户登录
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            (登录消息, 是否成功)
        """
        if self.auth_system.verify_password(username, password):
            self.current_user = username
            self.logger.info(f"用户登录成功: {username}")
            return f"✅ 登录成功！欢迎, {username}!", True
        else:
            self.logger.warning(f"登录失败: {username}")
            return "❌ 登录失败！用户名或密码错误", False
    
    def init_rag_systems(self):
        """
        初始化RAG系统（运行时调用）
        """
        try:
            # 初始化RAG-Anything
            self.rag_anything = RAGAnything(self.config)
            self.rag_anything.init_vector_db()
            
            # 初始化Agentic RAG
            self.agentic_rag = AgenticRAG(self.config)
            self.agentic_rag.set_llm_client(self.llm_client)
            
            # 初始化融合引擎
            self.fusion_engine = FusionEngine(self.config)
            
            # 初始化数据管理器
            self.data_manager = DataManager(self.config)
            self.data_manager.set_rag_anything(self.rag_anything)
            
            self.logger.info("RAG系统初始化完成")
            
            return True
            
        except Exception as e:
            self.logger.error(f"RAG系统初始化失败: {e}")
            return False
    
    def analyze_case(
        self,
        case_text: str,
        case_images: List[str],
        progress=gr.Progress()
    ) -> Dict[str, Any]:
        """
        分析案件
        
        Args:
            case_text: 案件文本
            case_images: 案件图片列表
            progress: 进度条
            
        Returns:
            分析结果
        """
        if not self.current_user:
            return {"error": "请先登录"}
        
        if not case_text:
            return {"error": "请输入案件描述"}
        
        progress(0.1, desc="🔍 提取线索...")
        
        # 准备案例数据
        case_data = {
            "case_text": case_text,
            "images": case_images or [],
            "case_type": "modern",
            "suspects": [],  # TODO: 从文本中提取
            "evidence": []   # TODO: 从文本中提取
        }
        
        progress(0.3, desc="🕵️ 分析嫌疑人...")
        
        # 运行完整调查
        results = self.orchestrator.run_full_investigation(case_data)
        
        progress(0.8, desc="🔗 关联证据...")
        
        progress(0.9, desc="🧠 生成推理...")
        
        progress(1.0, desc="✅ 分析完成！")
        
        return results
    
    def init_rag_system(self) -> bool:
        """
        初始化RAG系统
        
        Returns:
            是否初始化成功
        """
        try:
            # 导入必要的模块
            from rag.rag_anything import RAGAnything
            from rag.agentic_rag import AgenticRAG
            from rag.fusion import FusionEngine
            from rag.data_manager import DataManager
            
            # 初始化RAG-Anything
            self.rag_anything = RAGAnything(self.config)
            
            # 初始化Agentic RAG
            self.agentic_rag = AgenticRAG(self.config)
            
            # 设置LLM客户端
            llm_client = LLMClient()
            self.agentic_rag.set_llm_client(llm_client)
            
            # 初始化融合引擎
            self.fusion_engine = FusionEngine(self.config)
            
            # 初始化数据管理器
            self.data_manager = DataManager(self.config)
            self.data_manager.set_rag_anything(self.rag_anything)
            
            self.logger.info("RAG系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"RAG系统初始化失败: {e}")
            return False
    
    def scan_data(self) -> str:
        """
        扫描可用数据
        
        Returns:
            扫描结果（Markdown格式）
        """
        if not self.data_manager:
            return "❌ 请先初始化RAG系统"
        
        data_info = self.data_manager.scan_available_data()
        
        if "error" in data_info:
            return f"❌ 扫描失败: {data_info['error']}"
        
        output = f"""
# 📊 数据扫描结果

## 📁 文件统计

- **总文件数**: {data_info['total_files']}
- **Sherlock案例**: {data_info['sherlock_cases']}个
- **现代案例**: {data_info['modern_cases']}个
- **逻辑题**: {data_info['logic_puzzles']}个
- **图片证据**: {data_info['images']}张

## 📋 文件列表

"""
        
        for file_info in data_info['files']:
            output += f"- **{file_info['filename']}** ({file_info['type']}) - {file_info['size']} bytes\n"
        
        return output
    
    def import_data(self, data_types: List[str], progress=gr.Progress()) -> str:
        """
        导入数据到向量数据库
        
        Args:
            data_types: 数据类型列表
            progress: 进度条
            
        Returns:
            导入结果
        """
        if not self.data_manager:
            return "❌ 请先初始化RAG系统"
        
        progress(0.1, desc="🔍 扫描数据...")
        
        # 导入数据
        result = self.data_manager.import_data(data_types if data_types else None)
        
        progress(1.0, desc="✅ 导入完成")
        
        if result["success"]:
            output = f"""
# ✅ 数据导入成功

- **成功导入**: {result['imported_count']}个文档
- **导入失败**: {result['failed_count']}个文档
"""
            if result['errors']:
                output += "\n## ⚠️ 错误信息\n\n"
                for error in result['errors']:
                    output += f"- {error}\n"
        else:
            output = f"""
# ❌ 数据导入失败

"""
            for error in result['errors']:
                output += f"- {error}\n"
        
        return output
    
    def get_index_status(self) -> str:
        """
        获取索引状态
        
        Returns:
            索引状态（Markdown格式）
        """
        if not self.data_manager:
            return "❌ 请先初始化RAG系统"
        
        status = self.data_manager.get_indexing_status()
        
        if status["is_indexing"]:
            output = f"""
# 🔄 索引进行中

- **进度**: {status['progress']}/{status['total']}
- **当前文件**: {status['current_file']}
"""
        else:
            output = """
# ✅ 索引空闲

当前没有正在进行的索引任务。
"""
        
        if status['errors']:
            output += "\n## ⚠️ 错误信息\n\n"
            for error in status['errors']:
                output += f"- {error}\n"
        
        return output
    
    def clear_index(self) -> str:
        """
        清空索引
        
        Returns:
            清空结果
        """
        if not self.data_manager:
            return "❌ 请先初始化RAG系统"
        
        result = self.data_manager.clear_index()
        
        if result["success"]:
            return f"✅ {result['message']}"
        else:
            return f"❌ 清空失败: {result['error']}"
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """格式化结果为Markdown"""
        if "error" in results:
            return f"❌ 错误: {results['error']}"
        
        # 提取关键信息
        clues = results.get("clues", {}).get("data", {})
        reasoning = results.get("reasoning", {}).get("data", {})
        
        # 构建Markdown输出
        output = f"""
# 🔍 侦探推理报告

## 📋 案件线索

### ⏰ 时间线
"""
        
        # 时间线
        for timeline in clues.get("timeline", [])[:5]:
            output += f"- **{timeline.get('time', '')}**: {timeline.get('event', '')}\n"
        
        output += "\n### 📍 关键地点\n"
        
        # 地点
        for location in clues.get("locations", [])[:5]:
            output += f"- **{location.get('name', '')}** ({location.get('type', '')}) - 重要度: {location.get('importance', 0)}/5\n"
        
        output += "\n## 🧠 推理结论\n"
        
        # 推理结论
        conclusion = reasoning.get("final_conclusion", {})
        output += f"""
**结论**: {conclusion.get('description', '')}

**置信度**: {reasoning.get('confidence_score', 0):.2%}

**支持证据**:
"""
        
        for evidence in conclusion.get("supporting_evidence", [])[:5]:
            output += f"- {evidence}\n"
        
        output += "\n## 🔗 推理链\n"
        
        # 推理链
        for i, step in enumerate(reasoning.get("reasoning_chain", [])[:5], 1):
            output += f"\n**步骤 {i}**: {step.get('description', '')}\n"
            output += f"- 结论: {step.get('conclusion', '')}\n"
        
        return output
    
    def create_interface(self):
        """创建Gradio界面"""
        
        # 自定义CSS（黑客帝国风格）
        matrix_css = """
        /* 黑客帝国主题 */
        .gradio-container {
            background: #0D0208 !important;
            color: #00FF00 !important;
            font-family: 'Courier New', monospace !important;
        }
        
        .gr-button {
            background: #003B00 !important;
            color: #00FF00 !important;
            border: 2px solid #00FF00 !important;
            font-family: 'Courier New', monospace !important;
        }
        
        .gr-button:hover {
            background: #00FF00 !important;
            color: #0D0208 !important;
        }
        
        .gr-textbox, .gr-textarea {
            background: #0D0208 !important;
            color: #00FF00 !important;
            border: 1px solid #003B00 !important;
            font-family: 'Courier New', monospace !important;
        }
        
        .gr-input {
            background: #0D0208 !important;
            color: #00FF00 !important;
            border: 1px solid #003B00 !important;
        }
        
        .gr-panel {
            background: #0D0208 !important;
            border: 2px solid #003B00 !important;
        }
        
        .gr-box {
            background: #0D0208 !important;
            border: 1px solid #003B00 !important;
        }
        
        h1, h2, h3 {
            color: #00FF00 !important;
            font-family: 'Courier New', monospace !important;
        }
        
        /* 标题动画 */
        @keyframes glow {
            from {text-shadow: 0 0 5px #00FF00, 0 0 10px #00FF00;}
            to {text-shadow: 0 0 10px #00FF00, 0 0 20px #00FF00, 0 0 30px #00FF00;}
        }
        
        h1 {
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        /* 输入框聚焦效果 */
        .gr-textbox:focus, .gr-textarea:focus {
            border-color: #00FF00 !important;
            box-shadow: 0 0 10px #00FF00 !important;
        }
        """
        
        # 创建界面
        with gr.Blocks(
            title="🔍 侦探推理RAG系统",
            css=matrix_css,
            theme=gr.themes.Base()
        ) as demo:
            
            # 标题
            gr.Markdown(
                """
                # 🔍 侦探推理RAG系统
                
                **双路径智能推理** | **多模态支持** | **黑客帝国风格**
                
                ---
                """
            )
            
            # 登录区域
            with gr.Row():
                with gr.Column(scale=1):
                    login_username = gr.Textbox(
                        label="用户名",
                        placeholder="demo",
                        elem_id="login_username"
                    )
                with gr.Column(scale=1):
                    login_password = gr.Textbox(
                        label="密码",
                        type="password",
                        placeholder="REDACTED_DEMO_PASSWORD",
                        elem_id="login_password"
                    )
                with gr.Column(scale=1):
                    login_btn = gr.Button("登录", variant="primary")
                    login_status = gr.Textbox(label="登录状态", interactive=False)
            
            # 主界面
            with gr.Tab("🕵️ 案件分析"):
                with gr.Row():
                    # 左侧输入
                    with gr.Column(scale=1):
                        case_input = gr.Textbox(
                            label="📝 案件描述",
                            placeholder="请输入案件详细信息...",
                            lines=10,
                            elem_id="case_input"
                        )
                        
                        image_input = gr.File(
                            label="📷 证据图片（可选）",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        
                        analyze_btn = gr.Button(
                            "🔍 开始分析",
                            variant="primary",
                            size="lg"
                        )
                    
                    # 右侧输出
                    with gr.Column(scale=1):
                        output_text = gr.Markdown(
                            label="📊 分析结果",
                            elem_id="output_text"
                        )
            
            with gr.Tab("📊 知识图谱"):
                gr.Markdown("## 🔗 案件关系图谱")
                graph_output = gr.HTML(
                    value="<p style='color: #00FF00;'>图谱将在这里显示...</p>",
                    elem_id="graph_output"
                )
            
            with gr.Tab("⚙️ 系统设置"):
                gr.Markdown("## ⚙️ 系统配置")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎨 显示设置")
                        theme_select = gr.Dropdown(
                            label="主题",
                            choices=["matrix", "dark", "light"],
                            value="matrix"
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🤖 Agent设置")
                        enable_clue = gr.Checkbox(label="启用线索提取", value=True)
                        enable_suspect = gr.Checkbox(label="启用嫌疑人分析", value=True)
                        enable_evidence = gr.Checkbox(label="启用证据关联", value=True)
            
            with gr.Tab("💾 数据管理"):
                gr.Markdown("## 💾 运行时数据管理")
                gr.Markdown("**注意**: 数据导入和向量化操作在系统运行时执行")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        init_rag_btn = gr.Button("🔧 初始化RAG系统", variant="primary", size="lg")
                        init_status = gr.Textbox(label="初始化状态", interactive=False)
                    
                    with gr.Column(scale=1):
                        scan_data_btn = gr.Button("🔍 扫描可用数据", variant="secondary")
                        scan_output = gr.Markdown(elem_id="scan_output")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 数据导入")
                        import_types = gr.CheckboxGroup(
                            label="选择数据类型",
                            choices=["sherlock_cases", "modern_cases", "logic_puzzles"],
                            value=["sherlock_cases", "modern_cases"]
                        )
                        import_btn = gr.Button("📥 开始导入", variant="primary")
                        import_output = gr.Markdown(elem_id="import_output")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 索引状态")
                        check_status_btn = gr.Button("📊 检查索引状态", variant="secondary")
                        status_output = gr.Markdown(elem_id="status_output")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        clear_btn = gr.Button("🗑️ 清空索引", variant="stop")
                        clear_output = gr.Textbox(label="清空状态", interactive=False)
            
            # 示例
            with gr.Accordion("📚 示例案件", open=False):
                gr.Examples(
                    examples=[
                        ["2026年3月15日晚，某别墅发生命案。受害者张某，男，45岁，企业家。案发时别墅内有三名嫌疑人：妻子李某、合作伙伴王某、秘书赵某。现场发现凶器为水果刀，刀柄上有指纹。监控显示王某曾在案发时间段进出别墅。"],
                        ["某公司财务室被盗，损失现金50万元。案发时间为凌晨2点。监控录像显示，一名戴口罩男子从后门进入。嫌疑人包括：保安刘某（有后门钥匙）、会计陈某（熟悉财务室）、前员工孙某（最近被解雇）。"],
                    ],
                    inputs=case_input,
                    label="选择示例案件"
                )
            
            # 事件绑定
            login_btn.click(
                fn=self.login,
                inputs=[login_username, login_password],
                outputs=[login_status, gr.State()]
            )
            
            analyze_btn.click(
                fn=self.analyze_case,
                inputs=[case_input, image_input],
                outputs=gr.State()
            ).then(
                fn=self.format_results,
                inputs=gr.State(),
                outputs=output_text
            )
            
            # 数据管理事件绑定
            init_rag_btn.click(
                fn=self.init_rag_system,
                inputs=[],
                outputs=[init_status]
            )
            
            scan_data_btn.click(
                fn=self.scan_available_data,
                inputs=[],
                outputs=[scan_output]
            )
            
            import_btn.click(
                fn=self.import_data,
                inputs=[import_types],
                outputs=[import_output]
            )
            
            check_status_btn.click(
                fn=self.check_index_status,
                inputs=[],
                outputs=[status_output]
            )
            
            clear_btn.click(
                fn=self.clear_index,
                inputs=[],
                outputs=[clear_output]
            )
        
        return demo
    
    def launch(self):
        """启动WebUI"""
        demo = self.create_interface()
        
        # 启动参数
        host = self.config.get("webui", {}).get("host", "0.0.0.0")
        port = self.config.get("webui", {}).get("port", 7860)
        
        self.logger.info(f"WebUI启动: http://{host}:{port}")
        self.logger.info("演示账号: demo / REDACTED_DEMO_PASSWORD")
        
        demo.launch(
            server_name=host,
            server_port=port,
            share=False,
            show_error=True
        )


if __name__ == "__main__":
    # 启动WebUI
    webui = DetectiveWebUI()
    webui.launch()
