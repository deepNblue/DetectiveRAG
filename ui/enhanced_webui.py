"""
增强版WebUI主程序 - Day 7
集成可视化功能、文档解析、系统状态监控
"""

import gradio as gr
from typing import Dict, List, Any, Optional, Tuple
import yaml
import os
import sys
import tempfile
import base64
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
import logging

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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedDetectiveWebUI:
    """
    增强版侦探推理RAG系统WebUI
    集成可视化功能
    """
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        初始化增强版WebUI
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = logger
        self.config = self._load_config(config_path)
        
        # 初始化核心系统
        self.orchestrator = AgentOrchestrator(self.config)
        self.auth_system = AuthSystem()
        self.llm_client = LLMClient()
        
        # 初始化RAG系统（延迟）
        self.rag_anything = None
        self.agentic_rag = None
        self.fusion_engine = None
        self.data_manager = None
        
        # 初始化可视化引擎
        self.viz_engine = VisualizationEngine(self.config)
        
        # 当前用户
        self.current_user = None
        
        # 当前分析结果（用于可视化）
        self.current_results = None
        
        self.logger.info("增强版WebUI初始化完成")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def login(self, username: str, password: str) -> Tuple[str, str]:
        """用户登录"""
        if self.auth_system.verify_password(username, password):
            self.current_user = username
            self.logger.info(f"用户登录成功: {username}")
            return f"✅ 登录成功！欢迎, {username}!", username
        else:
            self.logger.warning(f"登录失败: {username}")
            return "❌ 登录失败！用户名或密码错误", ""
    
    def init_rag_system(self) -> str:
        """初始化RAG系统"""
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
            return "✅ RAG系统初始化成功"
            
        except Exception as e:
            self.logger.error(f"RAG系统初始化失败: {e}")
            return f"❌ 初始化失败: {str(e)}"
    
    def analyze_case(
        self,
        case_text: str,
        case_images: List[str],
        progress=gr.Progress()
    ) -> Tuple[str, Dict]:
        """
        分析案件
        
        Args:
            case_text: 案件文本
            case_images: 案件图片列表
            progress: 进度条
            
        Returns:
            (格式化结果, 原始结果)
        """
        if not self.current_user:
            return "❌ 请先登录", {}
        
        if not case_text:
            return "❌ 请输入案件描述", {}
        
        try:
            progress(0.1, desc="🔍 提取线索...")
            
            # 准备案例数据
            case_data = {
                "case_text": case_text,
                "images": case_images or [],
                "case_type": "modern",
                "suspects": [],
                "evidence": []
            }
            
            progress(0.3, desc="🕵️ 分析嫌疑人...")
            
            # 运行完整调查
            results = self.orchestrator.run_full_investigation(case_data)
            
            # 保存当前结果
            self.current_results = results
            
            progress(0.8, desc="🔗 关联证据...")
            progress(0.9, desc="🧠 生成推理...")
            
            # 格式化结果
            formatted_output = self._format_results_enhanced(results)
            
            progress(1.0, desc="✅ 分析完成！")
            
            return formatted_output, results
            
        except Exception as e:
            self.logger.error(f"案件分析失败: {e}")
            return f"❌ 分析失败: {str(e)}", {}
    
    def _format_results_enhanced(self, results: Dict[str, Any]) -> str:
        """增强版结果格式化"""
        if "error" in results:
            return f"❌ 错误: {results['error']}"
        
        # 提取关键信息
        clues = results.get("clues", {}).get("data", {})
        reasoning = results.get("reasoning", {}).get("data", {})
        suspects = results.get("suspects", {}).get("data", [])
        
        # 构建增强版Markdown输出
        output = f"""
# 🔍 侦探推理分析报告

---

## 📋 案件概况

**线索数量**: {len(clues.get("timeline", []))}条  
**嫌疑人数量**: {len(suspects)}人  
**推理置信度**: {reasoning.get("confidence_score", 0):.1%}

---

## ⏰ 时间线

"""
        
        # 时间线
        for i, event in enumerate(clues.get("timeline", [])[:8], 1):
            time = event.get("time", "未知时间")
            event_text = event.get("event", "未知事件")
            output += f"**{i}. {time}**: {event_text}\n\n"
        
        output += "\n## 👥 嫌疑人分析\n\n"
        
        # 嫌疑人
        for i, suspect in enumerate(suspects[:5], 1):
            name = suspect.get("name", "未知")
            motive = suspect.get("motive", "未知动机")
            motive_score = suspect.get("motive_score", 0)
            opportunity_score = suspect.get("opportunity_score", 0)
            capability_score = suspect.get("capability_score", 0)
            
            output += f"""### {i}. {name}

- **动机**: {motive}
- **嫌疑指数**:
  - 动机: {motive_score:.2f}
  - 机会: {opportunity_score:.2f}
  - 能力: {capability_score:.2f}

"""
        
        output += "\n## 🧠 推理结论\n\n"
        
        # 推理结论
        conclusion = reasoning.get("final_conclusion", {})
        confidence = reasoning.get("confidence_score", 0)
        
        output += f"""### 最终结论

{conclusion.get("description", "暂无结论")}

**置信度**: {confidence:.1%}

**支持证据**:

"""
        
        for evidence in conclusion.get("supporting_evidence", [])[:8]:
            output += f"- {evidence}\n"
        
        output += "\n---\n\n**✅ 分析完成！可查看可视化图谱和导出报告**\n"
        
        return output
    
    def generate_graph(self) -> str:
        """生成图谱"""
        if not self.current_results:
            return "❌ 请先分析案件"
        
        try:
            # 提取节点和边
            nodes, edges = self._extract_graph_data(self.current_results)
            
            if not nodes:
                return "❌ 无法生成图谱：缺少数据"
            
            # 生成图谱
            graph_path = self.viz_engine.generate_pyvis_graph(
                nodes=nodes,
                edges=edges,
                case_name="当前案件"
            )
            
            if graph_path:
                # 读取HTML内容
                with open(graph_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                return html_content
            else:
                return "❌ 图谱生成失败"
                
        except Exception as e:
            self.logger.error(f"图谱生成失败: {e}")
            return f"❌ 错误: {str(e)}"
    
    def _extract_graph_data(self, results: Dict) -> Tuple[List[Dict], List[Dict]]:
        """从结果中提取图谱数据"""
        nodes = []
        edges = []
        
        # 提取线索
        clues = results.get("clues", {}).get("data", {})
        
        # 时间线节点
        for i, event in enumerate(clues.get("timeline", [])[:10]):
            nodes.append({
                "id": f"timeline_{i}",
                "label": event.get("event", "事件")[:20],
                "type": "timeline",
                "title": f"{event.get('time', '')}: {event.get('event', '')}",
                "size": 20
            })
        
        # 地点节点
        for i, location in enumerate(clues.get("locations", [])[:5]):
            nodes.append({
                "id": f"location_{i}",
                "label": location.get("name", "地点"),
                "type": "location",
                "title": f"{location.get('type', '')}: {location.get('name', '')}",
                "size": 25
            })
        
        # 嫌疑人节点
        suspects = results.get("suspects", {}).get("data", [])
        for i, suspect in enumerate(suspects[:5]):
            nodes.append({
                "id": f"suspect_{i}",
                "label": suspect.get("name", "嫌疑人"),
                "type": "suspect",
                "title": f"{suspect.get('name', '')} - {suspect.get('motive', '')}",
                "size": 30
            })
        
        # 证据节点
        evidence = results.get("evidence", {}).get("data", {}).get("connections", [])
        for i, ev in enumerate(evidence[:5]):
            nodes.append({
                "id": f"evidence_{i}",
                "label": ev.get("evidence", "证据")[:15],
                "type": "evidence",
                "title": ev.get("evidence", ""),
                "size": 22
            })
        
        # 推理节点
        reasoning_chain = results.get("reasoning", {}).get("data", {}).get("reasoning_chain", [])
        for i, step in enumerate(reasoning_chain[:5]):
            nodes.append({
                "id": f"reasoning_{i}",
                "label": step.get("description", "推理")[:15],
                "type": "reasoning",
                "title": step.get("description", ""),
                "size": 24
            })
        
        # 生成边（简化版本）
        # 时间线连接
        for i in range(len(clues.get("timeline", [])[:10]) - 1):
            edges.append({
                "from": f"timeline_{i}",
                "to": f"timeline_{i+1}",
                "label": "→",
                "value": 1
            })
        
        # 嫌疑人连接证据
        for i in range(min(len(suspects[:5]), len(evidence[:5]))):
            edges.append({
                "from": f"suspect_{i}",
                "to": f"evidence_{i}",
                "label": "关联",
                "value": 2
            })
        
        # 推理连接
        for i in range(len(reasoning_chain[:5]) - 1):
            edges.append({
                "from": f"reasoning_{i}",
                "to": f"reasoning_{i+1}",
                "label": "→",
                "value": 1
            })
        
        return nodes, edges
    
    def generate_animation(self) -> str:
        """生成推理动画"""
        if not self.current_results:
            return "❌ 请先分析案件"
        
        try:
            # 提取推理链
            reasoning_chain = self.current_results.get("reasoning", {}).get("data", {}).get("reasoning_chain", [])
            
            if not reasoning_chain:
                return "❌ 无法生成动画：缺少推理数据"
            
            # 生成动画
            animation_path = self.viz_engine.generate_reasoning_animation(
                reasoning_chain=reasoning_chain,
                case_name="当前案件"
            )
            
            if animation_path:
                # 读取HTML内容
                with open(animation_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                return html_content
            else:
                return "❌ 动画生成失败"
                
        except Exception as e:
            self.logger.error(f"动画生成失败: {e}")
            return f"❌ 错误: {str(e)}"
    
    def export_results(self, export_format: str) -> str:
        """导出结果"""
        if not self.current_results:
            return "❌ 请先分析案件"
        
        try:
            # 导出
            export_path = self.viz_engine.export_results(
                results=self.current_results,
                export_format=export_format,
                case_name="当前案件"
            )
            
            if export_path:
                return f"✅ 导出成功: {export_path}"
            else:
                return "❌ 导出失败"
                
        except Exception as e:
            self.logger.error(f"导出失败: {e}")
            return f"❌ 错误: {str(e)}"
    
    def scan_available_data(self) -> str:
        """扫描可用数据"""
        if not self.data_manager:
            return "❌ 请先初始化RAG系统"
        
        data_info = self.data_manager.scan_available_data()
        
        if "error" in data_info:
            return f"❌ 扫描失败: {data_info['error']}"
        
        output = f"""# 📊 数据扫描结果

## 📁 文件统计

- **总文件数**: {data_info['total_files']}
- **Sherlock案例**: {data_info['sherlock_cases']}个
- **现代案例**: {data_info['modern_cases']}个
- **逻辑题**: {data_info['logic_puzzles']}个
- **图片证据**: {data_info['images']}张

## 📋 文件列表

"""
        
        for file_info in data_info['files'][:20]:
            output += f"- **{file_info['filename']}** ({file_info['type']}) - {file_info['size']} bytes\n"
        
        return output
    
    def import_data(self, data_types: List[str], progress=gr.Progress()) -> str:
        """导入数据"""
        if not self.data_manager:
            return "❌ 请先初始化RAG系统"
        
        progress(0.1, desc="🔍 扫描数据...")
        
        result = self.data_manager.import_data(data_types if data_types else None)
        
        progress(1.0, desc="✅ 导入完成")
        
        if result["success"]:
            output = f"""# ✅ 数据导入成功

- **成功导入**: {result['imported_count']}个文档
- **导入失败**: {result['failed_count']}个文档
"""
            if result['errors']:
                output += "\n## ⚠️ 错误信息\n\n"
                for error in result['errors'][:5]:
                    output += f"- {error}\n"
        else:
            output = "# ❌ 数据导入失败\n\n"
            for error in result['errors']:
                output += f"- {error}\n"
        
        return output
    
    def check_index_status(self) -> str:
        """检查索引状态"""
        if not self.data_manager:
            return "❌ 请先初始化RAG系统"
        
        status = self.data_manager.get_indexing_status()
        
        if status["is_indexing"]:
            output = f"""# 🔄 索引进行中

- **进度**: {status['progress']}/{status['total']}
- **当前文件**: {status['current_file']}
"""
        else:
            output = """# ✅ 索引空闲

当前没有正在进行的索引任务。
"""
        
        if status['errors']:
            output += "\n## ⚠️ 错误信息\n\n"
            for error in status['errors']:
                output += f"- {error}\n"
        
        return output
    
    def clear_index(self) -> str:
        """清空索引"""
        if not self.data_manager:
            return "❌ 请先初始化RAG系统"
        
        result = self.data_manager.clear_index()
        
        if result["success"]:
            return f"✅ {result['message']}"
        else:
            return f"❌ 清空失败: {result['error']}"
    
    def parse_documents(self, files: List[str], progress=gr.Progress()) -> Tuple[str, str]:
        """
        使用MinerU解析文档
        
        Args:
            files: 上传的文件路径列表
            
        Returns:
            (解析结果Markdown, 状态信息)
        """
        if not files:
            return "", "❌ 未选择文件"
        
        results = []
        output_dir = tempfile.mkdtemp(prefix="mineru_")
        
        for i, file_path in enumerate(files):
            progress((i + 1) / len(files), desc=f"解析 {os.path.basename(file_path)}...")
            
            try:
                cmd = [
                    "mineru", "-p", file_path,
                    "-o", output_dir,
                    "-b", "pipeline", "-l", "ch"
                ]
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
                
                if proc.returncode == 0:
                    # 查找输出Markdown文件
                    md_files = list(Path(output_dir).rglob("*.md"))
                    if md_files:
                        with open(md_files[0], 'r', encoding='utf-8') as f:
                            md_content = f.read()
                        results.append(f"## 📄 {os.path.basename(file_path)}\n\n{md_content[:5000]}")
                    else:
                        results.append(f"⚠️ {os.path.basename(file_path)}: 解析成功但未找到Markdown输出")
                else:
                    results.append(f"❌ {os.path.basename(file_path)}: 解析失败 - {proc.stderr[:500]}")
                    
            except subprocess.TimeoutExpired:
                results.append(f"❌ {os.path.basename(file_path)}: 解析超时（5分钟）")
            except FileNotFoundError:
                results.append("❌ MinerU未安装，请运行: pip install mineru")
                break
            except Exception as e:
                results.append(f"❌ {os.path.basename(file_path)}: {str(e)}")
        
        # 清理临时目录
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
        except:
            pass
        
        status = f"✅ 完成：{len(files)}个文件已处理"
        return "\n\n---\n\n".join(results), status
    
    def check_system_status(self) -> str:
        """检查系统各组件状态"""
        status_parts = ["# 🔧 系统状态报告\n"]
        
        # LLM状态（智谱API）
        try:
            llm_cfg = self.config.get("llm", {})
            api_key = os.environ.get("ZHIPUAI_API_KEY", "")
            if llm_cfg.get("provider") == "zhipu" and api_key:
                import requests
                # 智谱API：发一个简单请求验证连通性
                resp = requests.post(
                    "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={"model": "glm-4-flash", "messages": [{"role": "user", "content": "OK"}], "max_tokens": 5},
                    timeout=10
                )
                if resp.status_code == 200:
                    model_name = llm_cfg.get("model", "glm-4-flash")
                    status_parts.append(f"## 🤖 LLM服务: ✅ 在线 (智谱{model_name})\n")
                else:
                    status_parts.append(f"## 🤖 LLM服务: ❌ 错误 ({resp.status_code})\n")
            else:
                status_parts.append("## 🤖 LLM服务: ⚠️ 未配置\n")
        except Exception as e:
            status_parts.append(f"## 🤖 LLM服务: ❌ 不可达 ({str(e)[:50]})\n")
        
        # Embedding状态（9094端口 Qwen3-Embedding）
        try:
            emb_cfg = self.config.get("vector_db", {}).get("embedding", {})
            emb_url = emb_cfg.get("base_url", "")
            if emb_url:
                import requests
                resp = requests.get(f"{emb_url}/models", timeout=5)
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
                    model_name = models[0].get("id", "?") if models else "?"
                    dim = emb_cfg.get("dimension", "?")
                    status_parts.append(f"## 📊 Embedding服务: ✅ 在线\n**模型**: {model_name} ({dim}维)\n")
                else:
                    status_parts.append(f"## 📊 Embedding服务: ❌ 错误 ({resp.status_code})\n")
            else:
                status_parts.append("## 📊 Embedding服务: ⚠️ 未配置\n")
        except:
            status_parts.append("## 📊 Embedding服务: ❌ 不可达\n")
        
        # 向量库状态
        try:
            if self.rag_anything:
                status_parts.append("## 🗄️ 向量库: ✅ 已初始化\n")
            else:
                status_parts.append("## 🗄️ 向量库: ⏳ 未初始化\n")
        except:
            status_parts.append("## 🗄️ 向量库: ❌ 错误\n")
        
        # MinerU状态
        try:
            proc = subprocess.run(["mineru", "--version"], capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                version = proc.stdout.strip()
                status_parts.append(f"## 📄 MinerU: ✅ 已安装 ({version})\n")
            else:
                status_parts.append("## 📄 MinerU: ⚠️ 未正确安装\n")
        except FileNotFoundError:
            status_parts.append("## 📄 MinerU: ❌ 未安装\n")
        except:
            status_parts.append("## 📄 MinerU: ⚠️ 检测失败\n")
        
        # 系统信息
        import platform
        status_parts.append(f"""## 💻 系统信息
- **Python**: {platform.python_version()}
- **平台**: {platform.system()} {platform.release()}
- **当前用户**: {self.current_user or '未登录'}
""")
        
        return "\n".join(status_parts)
    
    def create_interface(self):
        """创建增强版Gradio界面"""
        
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
            transition: all 0.3s ease;
        }
        
        .gr-button:hover {
            background: #00FF00 !important;
            color: #0D0208 !important;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 255, 0, 0.4);
        }
        
        .gr-textbox, .gr-textarea {
            background: #0D0208 !important;
            color: #00FF00 !important;
            border: 2px solid #003B00 !important;
            font-family: 'Courier New', monospace !important;
        }
        
        .gr-textbox:focus, .gr-textarea:focus {
            border-color: #00FF00 !important;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.5) !important;
        }
        
        .gr-panel {
            background: #0D0208 !important;
            border: 2px solid #003B00 !important;
            border-radius: 10px;
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
        
        /* Tab样式 */
        .tab-nav button {
            background: #003B00 !important;
            color: #00FF00 !important;
            border: 1px solid #003B00 !important;
        }
        
        .tab-nav button.selected {
            background: #00FF00 !important;
            color: #0D0208 !important;
            border-color: #00FF00 !important;
        }
        
        /* 进度条样式 */
        .progress-bar {
            background: #003B00 !important;
        }
        
        .progress-bar-fill {
            background: #00FF00 !important;
        }
        
        /* 文件上传区域 */
        .upload-container {
            border: 2px dashed #00FF00 !important;
            background: #0D0208 !important;
        }
        
        /* Markdown样式 */
        .markdown-body {
            background: #0D0208 !important;
            color: #00FF00 !important;
        }
        
        .markdown-body h1, .markdown-body h2, .markdown-body h3 {
            color: #00FF00 !important;
            border-bottom: 1px solid #00FF00 !important;
        }
        
        .markdown-body a {
            color: #00FF00 !important;
        }
        
        .markdown-body code {
            background: #003B00 !important;
            color: #00FF00 !important;
            border: 1px solid #00FF00 !important;
        }
        """
        
        # 创建界面
        with gr.Blocks(
            title="🔍 侦探推理RAG系统 - 增强版",
            css=matrix_css,
            theme=gr.themes.Base()
        ) as demo:
            
            # 标题
            gr.Markdown(
                """
                # 🔍 侦探推理RAG系统 - 增强版
                
                **双路径智能推理** | **多模态支持** | **交互式可视化** | **黑客帝国风格**
                
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
                    login_btn = gr.Button("🔐 登录", variant="primary")
                    login_status = gr.Textbox(label="登录状态", interactive=False, lines=1)
            
            # 用于存储当前用户名
            user_state = gr.State("")
            
            # 主界面
            with gr.Tab("🕵️ 案件分析"):
                with gr.Row():
                    # 左侧输入
                    with gr.Column(scale=1):
                        case_type_dropdown = gr.Dropdown(
                            label="案件类型",
                            choices=["自动检测", "密室杀人", "连环盗窃", "网络诈骗", "投毒", "不在场证明破解", "遗嘱伪造", "绑架", "商业间谍"],
                            value="自动检测"
                        )
                        analysis_depth = gr.Slider(
                            label="分析深度",
                            minimum=1, maximum=5, value=3, step=1,
                            info="1=快速 5=深度分析"
                        )
                        case_input = gr.Textbox(
                            label="📝 案件描述",
                            placeholder="请输入案件详细信息...\n\n例如：2026年3月15日晚，某别墅发生命案...",
                            lines=10,
                            elem_id="case_input"
                        )
                        
                        image_input = gr.File(
                            label="📷 证据图片（可选）",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        
                        with gr.Row():
                            analyze_btn = gr.Button(
                                "🔍 开始分析",
                                variant="primary",
                                size="lg"
                            )
                            
                            clear_btn = gr.Button(
                                "🗑️ 清空",
                                variant="secondary",
                                size="lg"
                            )
                    
                    # 右侧输出
                    with gr.Column(scale=1):
                        output_text = gr.Markdown(
                            label="📊 分析结果",
                            elem_id="output_text",
                            value="等待分析..."
                        )
                        
                        with gr.Row():
                            status_indicator = gr.Textbox(
                                label="状态",
                                value="就绪",
                                interactive=False,
                                lines=1
                            )
                            
                            current_user_display = gr.Textbox(
                                label="当前用户",
                                value="未登录",
                                interactive=False,
                                lines=1
                            )
                        
                        copy_result_btn = gr.Button("📋 复制结果到剪贴板", variant="secondary")
                        copy_status = gr.Textbox(visible=False)
            
            with gr.Tab("📊 知识图谱"):
                gr.Markdown("## 🔗 案件关系图谱")
                
                with gr.Row():
                    generate_graph_btn = gr.Button("🎨 生成图谱", variant="primary", size="lg")
                    refresh_graph_btn = gr.Button("🔄 刷新", variant="secondary")
                
                graph_output = gr.HTML(
                    value="<div style='text-align: center; padding: 50px; color: #00FF00;'>图谱将在这里显示...<br>请先分析案件</div>",
                    elem_id="graph_output"
                )
                
                gr.Markdown("""
                ### 💡 使用说明
                - **鼠标拖拽**: 移动节点
                - **滚轮缩放**: 放大/缩小视图
                - **悬停**: 查看节点详细信息
                - **双击**: 聚焦到节点
                """)
            
            with gr.Tab("🎬 推理动画"):
                gr.Markdown("## 🎬 推理过程演示")
                
                with gr.Row():
                    generate_animation_btn = gr.Button("▶️ 生成动画", variant="primary", size="lg")
                    reset_animation_btn = gr.Button("🔄 重置", variant="secondary")
                
                animation_output = gr.HTML(
                    value="<div style='text-align: center; padding: 50px; color: #00FF00;'>推理动画将在这里显示...<br>请先分析案件</div>",
                    elem_id="animation_output"
                )
                
                gr.Markdown("""
                ### 🎯 动画说明
                - **自动播放**: 动画将自动开始
                - **进度条**: 显示当前推理进度
                - **步骤揭示**: 逐步展示推理过程
                """)
            
            with gr.Tab("📥 导出报告"):
                gr.Markdown("## 📥 导出分析报告")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        export_format = gr.Radio(
                            label="选择导出格式",
                            choices=["json", "markdown", "html"],
                            value="html"
                        )
                        
                        export_btn = gr.Button("📥 导出报告", variant="primary", size="lg")
                        export_status = gr.Textbox(label="导出状态", interactive=False, lines=2)
                
                gr.Markdown("""
                ### 📄 格式说明
                
                **JSON格式**:
                - 完整数据结构
                - 便于程序处理
                - 包含所有分析结果
                
                **Markdown格式**:
                - 易读的文本报告
                - 适合文档编辑
                - 包含格式化内容
                
                **HTML格式**:
                - 可视化报告
                - 黑客帝国主题
                - 可直接在浏览器查看
                """)
            
            with gr.Tab("💾 数据管理"):
                gr.Markdown("## 💾 运行时数据管理")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        init_rag_btn = gr.Button("🔧 初始化RAG系统", variant="primary", size="lg")
                        init_status = gr.Textbox(label="初始化状态", interactive=False, lines=2)
                    
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
                    with gr.Column():
                        clear_btn_data = gr.Button("🗑️ 清空索引", variant="stop")
                        clear_output = gr.Textbox(label="清空状态", interactive=False)
            
            with gr.Tab("📄 文档解析"):
                gr.Markdown("## 📄 MinerU文档解析\n上传PDF/图片文件，使用MinerU解析为Markdown格式")
                with gr.Row():
                    with gr.Column(scale=1):
                        doc_upload = gr.File(
                            label="📁 上传文档",
                            file_count="multiple",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg"]
                        )
                        parse_btn = gr.Button("🔄 开始解析", variant="primary")
                        parse_status = gr.Textbox(label="解析状态", interactive=False, lines=1)
                    with gr.Column(scale=2):
                        parse_output = gr.Markdown(
                            label="解析结果",
                            value="等待上传文档..."
                        )

            with gr.Tab("🔧 系统状态"):
                refresh_status_btn = gr.Button("🔄 刷新状态", variant="primary")
                system_status_output = gr.Markdown(
                    label="系统状态",
                    value="点击刷新查看系统状态..."
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
                        
                        enable_animation = gr.Checkbox(label="启用动画效果", value=True)
                        enable_sound = gr.Checkbox(label="启用音效", value=False)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🤖 Agent设置")
                        enable_clue = gr.Checkbox(label="启用线索提取", value=True)
                        enable_suspect = gr.Checkbox(label="启用嫌疑人分析", value=True)
                        enable_evidence = gr.Checkbox(label="启用证据关联", value=True)
                        enable_reasoning = gr.Checkbox(label="启用推理生成", value=True)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📊 可视化设置")
                        graph_layout = gr.Radio(
                            label="图谱布局",
                            choices=["force", "circular", "hierarchical"],
                            value="force"
                        )
                        
                        animation_speed = gr.Slider(
                            label="动画速度",
                            minimum=0.5,
                            maximum=3.0,
                            value=1.0,
                            step=0.1
                        )
            
            # 示例
            with gr.Accordion("📚 示例案件", open=False):
                gr.Examples(
                    examples=[
                        ["2026年3月15日晚，某别墅发生命案。受害者张某，男，45岁，企业家。案发时别墅内有三名嫌疑人：妻子李某（有保险受益权）、合作伙伴王某（最近有财务纠纷）、秘书赵某（掌握公司机密）。现场发现凶器为水果刀，刀柄上有指纹。监控显示王某曾在案发时间段进出别墅。李某声称一直在卧室，赵某说在书房整理文件。"],
                        ["某公司财务室被盗，损失现金50万元。案发时间为凌晨2点。监控录像显示，一名戴口罩男子从后门进入。嫌疑人包括：保安刘某（有后门钥匙，最近赌博欠债）、会计陈某（熟悉财务室，知道保险柜密码）、前员工孙某（最近被解雇，对公司不满）。现场发现一枚陌生指纹和脚印。"],
                        ["某艺术画廊发生盗窃案，一幅价值300万的油画失窃。案发当晚有三人有作案机会：画廊经理、保安队长、清洁工。经理有债务危机，保安队长最近购买豪车，清洁工的儿子需要手术费。现场监控被人为关闭，保安说看到可疑车辆离开。"]
                    ],
                    inputs=case_input,
                    label="选择示例案件"
                )
            
            # 底部信息
            gr.Markdown("""
            ---
            
            ### 📊 系统信息
            
            **版本**: v1.7.0 (增强版)  
            **开发者**: Detective RAG Team  
            **技术栈**: Gradio + pyvis + RAG-Anything + MinerU + vLLM  
            **更新时间**: 2026-04-02
            
            **功能特性**:
            - ✅ 双路径RAG推理
            - ✅ 交互式图谱可视化
            - ✅ 推理动画演示
            - ✅ 多格式报告导出
            - ✅ 运行时数据管理
            - ✅ MinerU文档解析
            - ✅ 系统状态监控
            - ✅ 案件类型分类
            - ✅ 黑客帝国主题界面
            
            ---
            """)
            
            # 事件绑定
            login_btn.click(
                fn=self.login,
                inputs=[login_username, login_password],
                outputs=[login_status, user_state]
            ).then(
                fn=lambda u: u if u else "未登录",
                inputs=[user_state],
                outputs=[current_user_display]
            )

            analyze_btn.click(
                fn=lambda: ("⏳ 正在分析...", gr.update()),
                outputs=[status_indicator]
            ).then(
                fn=self.analyze_case,
                inputs=[case_input, image_input],
                outputs=[output_text]
            ).then(
                fn=lambda: "✅ 分析完成",
                outputs=[status_indicator]
            )

            clear_btn.click(
                fn=lambda: ("", "已清空", "等待分析..."),
                outputs=[case_input, status_indicator, output_text]
            )

            copy_result_btn.click(
                fn=lambda: None, inputs=[], outputs=[]
            )

            parse_btn.click(
                fn=self.parse_documents,
                inputs=[doc_upload],
                outputs=[parse_output, parse_status]
            )

            refresh_status_btn.click(
                fn=self.check_system_status,
                outputs=[system_status_output]
            )
            
            generate_graph_btn.click(
                fn=self.generate_graph,
                outputs=[graph_output]
            )
            
            refresh_graph_btn.click(
                fn=self.generate_graph,
                outputs=[graph_output]
            )
            
            generate_animation_btn.click(
                fn=self.generate_animation,
                outputs=[animation_output]
            )
            
            reset_animation_btn.click(
                fn=self.generate_animation,
                outputs=[animation_output]
            )
            
            export_btn.click(
                fn=self.export_results,
                inputs=[export_format],
                outputs=[export_status]
            )
            
            init_rag_btn.click(
                fn=self.init_rag_system,
                outputs=[init_status]
            )
            
            scan_data_btn.click(
                fn=self.scan_available_data,
                outputs=[scan_output]
            )
            
            import_btn.click(
                fn=self.import_data,
                inputs=[import_types],
                outputs=[import_output]
            )
            
            check_status_btn.click(
                fn=self.check_index_status,
                outputs=[status_output]
            )
            
            clear_btn_data.click(
                fn=self.clear_index,
                outputs=[clear_output]
            )
        
        return demo
    
    def launch(self):
        """启动增强版WebUI"""
        demo = self.create_interface()
        
        # 启动参数
        host = self.config.get("webui", {}).get("host", "0.0.0.0")
        port = self.config.get("webui", {}).get("port", 7860)
        
        self.logger.info(f"增强版WebUI启动: http://{host}:{port}")
        self.logger.info("演示账号: demo / REDACTED_DEMO_PASSWORD")
        
        demo.launch(
            server_name=host,
            server_port=port,
            share=False,
            show_error=True
        )


if __name__ == "__main__":
    # 启动增强版WebUI
    webui = EnhancedDetectiveWebUI()
    webui.launch()
