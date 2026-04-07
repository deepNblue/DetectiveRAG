"""
可视化增强模块 - Day 5
提供交互式图谱、推理动画和结果导出功能
"""

import json
import os
import tempfile
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    可视化引擎
    提供图谱可视化、推理动画、结果导出等功能
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化可视化引擎"""
        self.config = config or {}
        self.output_dir = self.config.get("visualization", {}).get("output_dir", "./data/visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 黑客帝国配色
        self.matrix_colors = {
            "primary": "#00FF00",
            "secondary": "#003B00",
            "background": "#0D0208",
            "text": "#00FF00",
            "clue": "#00FF00",
            "suspect": "#FF6B6B",
            "evidence": "#4ECDC4",
            "location": "#FFE66D",
            "timeline": "#95E1D3",
            "reasoning": "#DDA0DD",
            "edge": "#00FF00"
        }
        
        logger.info("可视化引擎初始化完成")
    
    def generate_pyvis_graph(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        case_name: str = "案件分析"
    ) -> str:
        """
        生成pyvis交互式图谱
        
        Args:
            nodes: 节点列表
            edges: 边列表
            case_name: 案件名称
            
        Returns:
            HTML文件路径
        """
        try:
            from pyvis.network import Network
            import networkx as nx
            
            # 创建pyvis网络
            net = Network(
                height="600px",
                width="100%",
                bgcolor=self.matrix_colors["background"],
                font_color=self.matrix_colors["text"],
                directed=True
            )
            
            # 设置物理引擎
            net.barnes_hut(
                gravity=2000,
                central_gravity=0.3,
                spring_length=100,
                damping=0.09
            )
            
            # 添加节点
            for node in nodes:
                node_id = node.get("id", str(hash(node.get("label", ""))))
                label = node.get("label", "")
                node_type = node.get("type", "clue")
                title = node.get("title", "")
                size = node.get("size", 20)
                color = self.matrix_colors.get(node_type, self.matrix_colors["clue"])
                
                # 添加节点
                net.add_node(
                    node_id,
                    label=label,
                    title=title,
                    size=size,
                    color=color,
                    group=node_type,
                    borderWidth=2,
                    borderColor=self.matrix_colors["primary"],
                    shadow=True
                )
            
            # 添加边
            for edge in edges:
                source = edge.get("from", edge.get("source", ""))
                target = edge.get("to", edge.get("target", ""))
                label = edge.get("label", "")
                value = edge.get("value", 1)
                color = edge.get("color", self.matrix_colors["edge"])
                
                if source and target:
                    net.add_edge(
                        source,
                        target,
                        label=label,
                        value=value,
                        color=color,
                        width=2,
                        arrows="to",
                        arrowStrikethrough=False
                    )
            
            # 设置选项
            net.set_options("""
            {
                "physics": {
                    "barnesHut": {
                        "gravitationalConstant": -2000,
                        "centralGravity": 0.3,
                        "springLength": 100,
                        "damping": 0.09
                    },
                    "minVelocity": 0.75,
                    "maxVelocity": 50
                },
                "interaction": {
                    "hover": true,
                    "tooltipDelay": 200,
                    "zoomView": true,
                    "dragView": true
                },
                "nodes": {
                    "font": {
                        "size": 16,
                        "face": "Courier New",
                        "color": "#00FF00"
                    },
                    "borderWidth": 2,
                    "borderWidthSelected": 4
                },
                "edges": {
                    "font": {
                        "size": 12,
                        "face": "Courier New",
                        "color": "#00FF00"
                    },
                    "smooth": {
                        "type": "continuous"
                    }
                }
            }
            """)
            
            # 保存HTML
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graph_{case_name}_{timestamp}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            net.save_graph(filepath)
            
            # 添加自定义样式
            self._add_custom_styles(filepath)
            
            logger.info(f"图谱生成成功: {filepath}")
            return filepath
            
        except ImportError:
            logger.warning("pyvis未安装，使用备用方案")
            return self._generate_static_graph(nodes, edges, case_name)
        except Exception as e:
            logger.error(f"图谱生成失败: {e}")
            return ""
    
    def _generate_static_graph(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        case_name: str
    ) -> str:
        """
        生成静态图谱（备用方案）
        
        Args:
            nodes: 节点列表
            edges: 边列表
            case_name: 案件名称
            
        Returns:
            HTML文件路径
        """
        try:
            import networkx as nx
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # 创建有向图
            G = nx.DiGraph()
            
            # 添加节点
            for node in nodes:
                node_id = node.get("id", str(hash(node.get("label", ""))))
                G.add_node(node_id, **node)
            
            # 添加边
            for edge in edges:
                source = edge.get("from", edge.get("source", ""))
                target = edge.get("to", edge.get("target", ""))
                if source and target:
                    G.add_edge(source, target, **edge)
            
            # 设置画布
            plt.figure(figsize=(14, 10), facecolor=self.matrix_colors["background"])
            ax = plt.gca()
            ax.set_facecolor(self.matrix_colors["background"])
            
            # 设置布局
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # 绘制节点
            node_colors = [
                self.matrix_colors.get(
                    G.nodes[n].get("type", "clue"),
                    self.matrix_colors["clue"]
                ) for n in G.nodes()
            ]
            
            nx.draw_networkx_nodes(
                G, pos,
                node_color=node_colors,
                node_size=1000,
                alpha=0.9,
                edgecolors=self.matrix_colors["primary"],
                linewidths=2
            )
            
            # 绘制边
            nx.draw_networkx_edges(
                G, pos,
                edge_color=self.matrix_colors["edge"],
                arrows=True,
                arrowsize=20,
                width=2,
                alpha=0.7
            )
            
            # 绘制标签
            nx.draw_networkx_labels(
                G, pos,
                font_color=self.matrix_colors["text"],
                font_size=10,
                font_family="Courier New",
                font_weight="bold"
            )
            
            # 设置标题
            plt.title(
                f"🔍 {case_name} - 关系图谱",
                color=self.matrix_colors["primary"],
                fontsize=16,
                fontweight="bold",
                pad=20
            )
            
            # 保存图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graph_{case_name}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            plt.savefig(
                filepath,
                facecolor=self.matrix_colors["background"],
                dpi=150,
                bbox_inches='tight'
            )
            plt.close()
            
            logger.info(f"静态图谱生成成功: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"静态图谱生成失败: {e}")
            return ""
    
    def _add_custom_styles(self, filepath: str):
        """添加自定义样式到HTML"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # 添加自定义CSS
            custom_css = """
            <style>
                body {
                    background-color: #0D0208;
                    color: #00FF00;
                    font-family: 'Courier New', monospace;
                }
                
                .title {
                    color: #00FF00;
                    text-align: center;
                    font-size: 24px;
                    margin: 20px 0;
                    text-shadow: 0 0 10px #00FF00;
                }
                
                #mynetwork {
                    border: 2px solid #00FF00;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
                }
            </style>
            """
            
            # 插入到head中
            html_content = html_content.replace('</head>', f'{custom_css}</head>')
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            logger.warning(f"添加自定义样式失败: {e}")
    
    def generate_reasoning_animation(
        self,
        reasoning_chain: List[Dict],
        case_name: str = "案件分析"
    ) -> str:
        """
        生成推理动画
        
        Args:
            reasoning_chain: 推理链
            case_name: 案件名称
            
        Returns:
            动画HTML路径
        """
        try:
            # 生成动画HTML
            html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 推理动画 - {case_name}</title>
    <style>
        body {{
            background-color: #0D0208;
            color: #00FF00;
            font-family: 'Courier New', monospace;
            margin: 0;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            color: #00FF00;
            text-shadow: 0 0 10px #00FF00;
            animation: glow 2s ease-in-out infinite alternate;
        }}
        
        @keyframes glow {{
            from {{ text-shadow: 0 0 5px #00FF00, 0 0 10px #00FF00; }}
            to {{ text-shadow: 0 0 10px #00FF00, 0 0 20px #00FF00, 0 0 30px #00FF00; }}
        }}
        
        .timeline {{
            position: relative;
            padding: 20px 0;
        }}
        
        .timeline::before {{
            content: '';
            position: absolute;
            left: 50%;
            top: 0;
            bottom: 0;
            width: 2px;
            background: linear-gradient(to bottom, #00FF00, #003B00);
        }}
        
        .step {{
            position: relative;
            margin: 40px 0;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.5s ease;
        }}
        
        .step.visible {{
            opacity: 1;
            transform: translateY(0);
        }}
        
        .step:nth-child(odd) {{
            margin-left: 55%;
        }}
        
        .step:nth-child(even) {{
            margin-right: 55%;
        }}
        
        .step-content {{
            background: #0D0208;
            border: 2px solid #00FF00;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
        }}
        
        .step-number {{
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            width: 40px;
            height: 40px;
            background: #00FF00;
            color: #0D0208;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 18px;
        }}
        
        .step-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #00FF00;
        }}
        
        .step-description {{
            font-size: 14px;
            line-height: 1.6;
            color: #00FF00;
        }}
        
        .step-conclusion {{
            margin-top: 15px;
            padding: 10px;
            background: #003B00;
            border-radius: 5px;
            border-left: 4px solid #00FF00;
        }}
        
        .controls {{
            text-align: center;
            margin: 30px 0;
        }}
        
        .btn {{
            background: #003B00;
            color: #00FF00;
            border: 2px solid #00FF00;
            padding: 10px 30px;
            font-size: 16px;
            font-family: 'Courier New', monospace;
            cursor: pointer;
            margin: 0 10px;
            transition: all 0.3s ease;
        }}
        
        .btn:hover {{
            background: #00FF00;
            color: #0D0208;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 4px;
            background: #003B00;
            margin: 20px 0;
            border-radius: 2px;
        }}
        
        .progress {{
            height: 100%;
            background: #00FF00;
            width: 0%;
            transition: width 0.5s ease;
            box-shadow: 0 0 10px #00FF00;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 推理动画演示</h1>
        
        <div class="progress-bar">
            <div class="progress" id="progress"></div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="startAnimation()">▶️ 开始演示</button>
            <button class="btn" onclick="resetAnimation()">🔄 重置</button>
        </div>
        
        <div class="timeline" id="timeline">
            {self._generate_timeline_html(reasoning_chain)}
        </div>
    </div>
    
    <script>
        let currentStep = 0;
        const steps = document.querySelectorAll('.step');
        const progressBar = document.getElementById('progress');
        
        function startAnimation() {{
            if (currentStep < steps.length) {{
                showStep(currentStep);
                currentStep++;
                setTimeout(startAnimation, 2000);
            }}
        }}
        
        function showStep(index) {{
            if (index < steps.length) {{
                steps[index].classList.add('visible');
                const progress = ((index + 1) / steps.length) * 100;
                progressBar.style.width = progress + '%';
            }}
        }}
        
        function resetAnimation() {{
            currentStep = 0;
            steps.forEach(step => step.classList.remove('visible'));
            progressBar.style.width = '0%';
        }}
        
        // 自动开始
        setTimeout(startAnimation, 1000);
    </script>
</body>
</html>
            """
            
            # 保存HTML
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reasoning_animation_{case_name}_{timestamp}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_template)
            
            logger.info(f"推理动画生成成功: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"推理动画生成失败: {e}")
            return ""
    
    def _generate_timeline_html(self, reasoning_chain: List[Dict]) -> str:
        """生成时间线HTML"""
        if not reasoning_chain:
            return "<p>暂无推理数据</p>"
        
        html = ""
        for i, step in enumerate(reasoning_chain, 1):
            title = step.get("description", step.get("title", f"步骤 {i}"))
            conclusion = step.get("conclusion", "")
            
            html += f"""
            <div class="step">
                <div class="step-number">{i}</div>
                <div class="step-content">
                    <div class="step-title">🔍 {title}</div>
                    <div class="step-description">
                        分析证据，提取关键信息...
                    </div>
                    <div class="step-conclusion">
                        <strong>结论:</strong> {conclusion}
                    </div>
                </div>
            </div>
            """
        
        return html
    
    def export_results(
        self,
        results: Dict[str, Any],
        export_format: str = "json",
        case_name: str = "案件分析"
    ) -> str:
        """
        导出结果
        
        Args:
            results: 分析结果
            export_format: 导出格式 (json/markdown/html/pdf)
            case_name: 案件名称
            
        Returns:
            文件路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if export_format == "json":
                filename = f"report_{case_name}_{timestamp}.json"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
            elif export_format == "markdown":
                filename = f"report_{case_name}_{timestamp}.md"
                filepath = os.path.join(self.output_dir, filename)
                
                markdown_content = self._generate_markdown_report(results, case_name)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
            elif export_format == "html":
                filename = f"report_{case_name}_{timestamp}.html"
                filepath = os.path.join(self.output_dir, filename)
                
                html_content = self._generate_html_report(results, case_name)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
            else:
                raise ValueError(f"不支持的导出格式: {export_format}")
            
            logger.info(f"结果导出成功: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"结果导出失败: {e}")
            return ""
    
    def _generate_markdown_report(self, results: Dict, case_name: str) -> str:
        """生成Markdown报告"""
        timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        
        # 提取关键信息
        clues = results.get("clues", {}).get("data", {})
        reasoning = results.get("reasoning", {}).get("data", {})
        suspects = results.get("suspects", {}).get("data", [])
        
        markdown = f"""# 🔍 侦探推理分析报告

**案件名称**: {case_name}

**生成时间**: {timestamp}

**系统版本**: Detective RAG v10.0

---

## 📋 案件概况

{results.get("case_text", "无案件描述")[:500]}...

---

## 🔍 线索提取

### ⏰ 时间线

"""
        
        # 时间线
        for event in clues.get("timeline", [])[:10]:
            time = event.get("time", "未知时间")
            event_text = event.get("event", "未知事件")
            markdown += f"- **{time}**: {event_text}\n"
        
        markdown += "\n### 📍 关键地点\n\n"
        
        # 地点
        for location in clues.get("locations", [])[:10]:
            name = location.get("name", "未知地点")
            loc_type = location.get("type", "未知类型")
            importance = location.get("importance", 0)
            markdown += f"- **{name}** ({loc_type}) - 重要度: {importance}/5\n"
        
        markdown += "\n## 👥 嫌疑人分析\n\n"
        
        # 嫌疑人
        for suspect in suspects[:5]:
            name = suspect.get("name", "未知")
            motive = suspect.get("motive", "未知动机")
            motive_score = suspect.get("motive_score", 0)
            opportunity_score = suspect.get("opportunity_score", 0)
            
            markdown += f"""### {name}

- **动机**: {motive}
- **动机分数**: {motive_score:.2f}
- **机会分数**: {opportunity_score:.2f}

"""
        
        markdown += "\n## 🧠 推理结论\n\n"
        
        # 推理结论
        conclusion = reasoning.get("final_conclusion", {})
        markdown += f"""### 最终结论

{conclusion.get("description", "暂无结论")}

**置信度**: {reasoning.get("confidence_score", 0):.2%}

**支持证据**:

"""
        
        for evidence in conclusion.get("supporting_evidence", [])[:10]:
            markdown += f"- {evidence}\n"
        
        markdown += "\n---\n\n**报告生成完毕** ✅\n"
        
        return markdown
    
    def _generate_html_report(self, results: Dict, case_name: str) -> str:
        """生成HTML报告"""
        timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        
        # 提取关键信息
        clues = results.get("clues", {}).get("data", {})
        reasoning = results.get("reasoning", {}).get("data", {})
        suspects = results.get("suspects", {}).get("data", [])
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔍 侦探推理报告 - {case_name}</title>
    <style>
        body {{
            background-color: #0D0208;
            color: #00FF00;
            font-family: 'Courier New', monospace;
            margin: 0;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: #0D0208;
            border: 2px solid #00FF00;
            padding: 30px;
            box-shadow: 0 0 30px rgba(0, 255, 0, 0.3);
        }}
        
        h1 {{
            text-align: center;
            color: #00FF00;
            text-shadow: 0 0 10px #00FF00;
            border-bottom: 2px solid #00FF00;
            padding-bottom: 20px;
        }}
        
        h2 {{
            color: #00FF00;
            border-left: 4px solid #00FF00;
            padding-left: 10px;
            margin-top: 30px;
        }}
        
        .section {{
            margin: 20px 0;
            padding: 20px;
            background: #003B00;
            border-radius: 10px;
        }}
        
        .meta {{
            text-align: center;
            margin-bottom: 30px;
            color: #00FF00;
        }}
        
        .timeline-item, .location-item, .suspect-item, .evidence-item {{
            margin: 10px 0;
            padding: 15px;
            background: #0D0208;
            border: 1px solid #00FF00;
            border-radius: 5px;
        }}
        
        .score {{
            display: inline-block;
            padding: 2px 10px;
            background: #00FF00;
            color: #0D0208;
            border-radius: 3px;
            font-weight: bold;
        }}
        
        .confidence {{
            font-size: 24px;
            text-align: center;
            color: #00FF00;
            text-shadow: 0 0 10px #00FF00;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #00FF00;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 侦探推理分析报告</h1>
        
        <div class="meta">
            <p><strong>案件名称:</strong> {case_name}</p>
            <p><strong>生成时间:</strong> {timestamp}</p>
            <p><strong>系统版本:</strong> Detective RAG v10.0</p>
        </div>
        
        <h2>📋 案件概况</h2>
        <div class="section">
            <p>{results.get("case_text", "无案件描述")[:500]}...</p>
        </div>
        
        <h2>🔍 线索提取</h2>
        
        <h3>⏰ 时间线</h3>
        <div class="section">
"""
        
        # 时间线
        for event in clues.get("timeline", [])[:10]:
            time = event.get("time", "未知时间")
            event_text = event.get("event", "未知事件")
            html += f"""
            <div class="timeline-item">
                <strong>{time}</strong>: {event_text}
            </div>
"""
        
        html += """
        </div>
        
        <h3>📍 关键地点</h3>
        <div class="section">
"""
        
        # 地点
        for location in clues.get("locations", [])[:10]:
            name = location.get("name", "未知地点")
            loc_type = location.get("type", "未知类型")
            importance = location.get("importance", 0)
            html += f"""
            <div class="location-item">
                <strong>{name}</strong> ({loc_type}) - 重要度: <span class="score">{importance}/5</span>
            </div>
"""
        
        html += """
        </div>
        
        <h2>👥 嫌疑人分析</h2>
        <div class="section">
"""
        
        # 嫌疑人
        for suspect in suspects[:5]:
            name = suspect.get("name", "未知")
            motive = suspect.get("motive", "未知动机")
            motive_score = suspect.get("motive_score", 0)
            opportunity_score = suspect.get("opportunity_score", 0)
            
            html += f"""
            <div class="suspect-item">
                <h3>{name}</h3>
                <p><strong>动机:</strong> {motive}</p>
                <p><strong>动机分数:</strong> <span class="score">{motive_score:.2f}</span></p>
                <p><strong>机会分数:</strong> <span class="score">{opportunity_score:.2f}</span></p>
            </div>
"""
        
        html += """
        </div>
        
        <h2>🧠 推理结论</h2>
        <div class="section">
"""
        
        # 推理结论
        conclusion = reasoning.get("final_conclusion", {})
        html += f"""
            <h3>最终结论</h3>
            <p>{conclusion.get("description", "暂无结论")}</p>
            
            <p class="confidence">置信度: {reasoning.get("confidence_score", 0):.2%}</p>
            
            <h3>支持证据</h3>
            <ul>
"""
        
        for evidence in conclusion.get("supporting_evidence", [])[:10]:
            html += f"                <li>{evidence}</li>\n"
        
        html += """
            </ul>
        </div>
        
        <div class="footer">
            <p>**报告生成完毕** ✅</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def create_summary_dashboard(
        self,
        analysis_results: Dict[str, Any]
    ) -> str:
        """
        创建摘要仪表板
        
        Args:
            analysis_results: 分析结果
            
        Returns:
            仪表板HTML路径
        """
        try:
            # 统计数据
            total_clues = len(analysis_results.get("clues", {}).get("data", {}).get("timeline", []))
            total_suspects = len(analysis_results.get("suspects", {}).get("data", []))
            confidence = analysis_results.get("reasoning", {}).get("data", {}).get("confidence_score", 0)
            
            html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 分析仪表板</title>
    <style>
        body {{
            background-color: #0D0208;
            color: #00FF00;
            font-family: 'Courier New', monospace;
            margin: 0;
            padding: 20px;
        }}
        
        .dashboard {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            color: #00FF00;
            text-shadow: 0 0 10px #00FF00;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .stat-card {{
            background: #0D0208;
            border: 2px solid #00FF00;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-value {{
            font-size: 48px;
            font-weight: bold;
            color: #00FF00;
            text-shadow: 0 0 10px #00FF00;
        }}
        
        .stat-label {{
            font-size: 16px;
            margin-top: 10px;
            color: #00FF00;
        }}
        
        .progress-container {{
            margin: 20px 0;
        }}
        
        .progress-label {{
            margin-bottom: 10px;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #003B00;
            border-radius: 15px;
            overflow: hidden;
            border: 2px solid #00FF00;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #00FF00, #003B00);
            width: {confidence * 100}%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: #0D0208;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>📊 分析仪表板</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_clues}</div>
                <div class="stat-label">🔍 线索数量</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-value">{total_suspects}</div>
                <div class="stat-label">👥 嫌疑人数量</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-value">{confidence:.0%}</div>
                <div class="stat-label">🎯 置信度</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-value">✅</div>
                <div class="stat-label">分析完成</div>
            </div>
        </div>
        
        <div class="progress-container">
            <div class="progress-label">推理置信度</div>
            <div class="progress-bar">
                <div class="progress-fill">{confidence:.0%}</div>
            </div>
        </div>
    </div>
</body>
</html>
            """
            
            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_{timestamp}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            
            logger.info(f"仪表板生成成功: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"仪表板生成失败: {e}")
            return ""
    
    def get_visualization_status(self) -> Dict[str, Any]:
        """获取可视化状态"""
        try:
            # 检查输出目录
            files = []
            if os.path.exists(self.output_dir):
                for filename in os.listdir(self.output_dir):
                    filepath = os.path.join(self.output_dir, filename)
                    if os.path.isfile(filepath):
                        stat = os.stat(filepath)
                        files.append({
                            "filename": filename,
                            "size": stat.st_size,
                            "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                            "type": filename.split('.')[-1]
                        })
            
            return {
                "status": "ready",
                "output_dir": self.output_dir,
                "total_files": len(files),
                "files": files,
                "supported_formats": ["json", "markdown", "html"],
                "features": {
                    "pyvis_graph": True,
                    "reasoning_animation": True,
                    "result_export": True,
                    "dashboard": True
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
