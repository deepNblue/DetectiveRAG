"""
多模态处理器
支持文本、图片、表格、公式等多种模态的处理
"""

from typing import Dict, List, Any, Optional, Union
from PIL import Image
import base64
import io
import os
from loguru import logger
import json


class MultimodalProcessor:
    """
    多模态处理器
    处理文本、图片、表格、公式等多种数据类型
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化多模态处理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logger.bind(module="multimodal_processor")
        
        # 图片处理配置
        self.image_config = self.config.get("multimodal", {}).get("image_processing", {})
        self.max_image_size = self.image_config.get("max_size", 512)
        self.image_format = self.image_config.get("format", "JPEG")
        self.image_quality = self.image_config.get("quality", 85)
        
        self.logger.info("多模态处理器初始化完成")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理多模态数据
        
        Args:
            input_data: {
                "text": "文本内容",
                "images": ["图片路径1", "图片路径2"],
                "tables": ["表格数据"],
                "formulas": ["公式数据"]
            }
            
        Returns:
            {
                "text_chunks": [...],  # 文本块
                "image_descriptions": [...],  # 图片描述
                "table_data": [...],  # 表格数据
                "formula_data": [...],  # 公式数据
                "combined_text": "组合文本"
            }
        """
        self.logger.info("开始处理多模态数据")
        
        results = {}
        
        # 处理文本
        results["text_chunks"] = self._process_text(input_data.get("text", ""))
        
        # 处理图片
        results["image_descriptions"] = self._process_images(
            input_data.get("images", [])
        )
        
        # 处理表格
        results["table_data"] = self._process_tables(
            input_data.get("tables", [])
        )
        
        # 处理公式
        results["formula_data"] = self._process_formulas(
            input_data.get("formulas", [])
        )
        
        # 组合所有文本
        results["combined_text"] = self._combine_all_text(results)
        
        self.logger.info(
            f"多模态处理完成: {len(results['text_chunks'])}个文本块, "
            f"{len(results['image_descriptions'])}张图片"
        )
        
        return results
    
    def _process_text(self, text: str) -> List[Dict[str, Any]]:
        """
        处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            文本块列表
        """
        if not text:
            return []
        
        # 分块配置
        chunk_size = self.config.get("rag", {}).get("rag_anything", {}).get("chunk_size", 512)
        chunk_overlap = self.config.get("rag", {}).get("rag_anything", {}).get("chunk_overlap", 50)
        
        # 简单分块（按段落和句子）
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "content": current_chunk.strip(),
                        "type": "text",
                        "length": len(current_chunk)
                    })
                    chunk_id += 1
                current_chunk = para + "\n\n"
        
        # 添加最后一个块
        if current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "content": current_chunk.strip(),
                "type": "text",
                "length": len(current_chunk)
            })
        
        return chunks
    
    def _process_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        处理图片
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            图片描述列表
        """
        if not image_paths:
            return []
        
        descriptions = []
        
        for idx, image_path in enumerate(image_paths):
            try:
                # 检查文件是否存在
                if not os.path.exists(image_path):
                    self.logger.warning(f"图片不存在: {image_path}")
                    continue
                
                # 加载图片
                with Image.open(image_path) as img:
                    # 压缩图片
                    compressed_img = self._compress_image(img)
                    
                    # 生成描述（这里先返回基础信息，实际需要多模态模型）
                    description = {
                        "image_id": idx,
                        "path": image_path,
                        "type": "image",
                        "size": img.size,
                        "format": img.format,
                        "compressed_size": compressed_img.size,
                        "description": f"图片证据 {idx + 1}",
                        "base64": self._image_to_base64(compressed_img),
                        "metadata": {
                            "original_size": img.size,
                            "mode": img.mode
                        }
                    }
                    
                    descriptions.append(description)
                    
            except Exception as e:
                self.logger.error(f"图片处理失败 {image_path}: {e}")
        
        return descriptions
    
    def _compress_image(self, img: Image.Image) -> Image.Image:
        """
        压缩图片
        
        Args:
            img: PIL图片对象
            
        Returns:
            压缩后的图片
        """
        # 调整大小
        if max(img.size) > self.max_image_size:
            ratio = self.max_image_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # 转换格式
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        return img
    
    def _image_to_base64(self, img: Image.Image) -> str:
        """
        将图片转换为base64编码
        
        Args:
            img: PIL图片对象
            
        Returns:
            base64编码字符串
        """
        buffer = io.BytesIO()
        img.save(buffer, format=self.image_format, quality=self.image_quality)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_base64
    
    def _process_tables(self, tables: List[Any]) -> List[Dict[str, Any]]:
        """
        处理表格数据
        
        Args:
            tables: 表格数据列表
            
        Returns:
            处理后的表格数据
        """
        if not tables:
            return []
        
        processed_tables = []
        
        for idx, table in enumerate(tables):
            # TODO: 实现表格解析（CSV/Excel/HTML等）
            processed_table = {
                "table_id": idx,
                "type": "table",
                "data": table,
                "description": f"表格数据 {idx + 1}",
                "text_representation": str(table)
            }
            
            processed_tables.append(processed_table)
        
        return processed_tables
    
    def _process_formulas(self, formulas: List[str]) -> List[Dict[str, Any]]:
        """
        处理公式
        
        Args:
            formulas: 公式列表
            
        Returns:
            处理后的公式数据
        """
        if not formulas:
            return []
        
        processed_formulas = []
        
        for idx, formula in enumerate(formulas):
            # TODO: 实现公式解析（LaTeX/MathML等）
            processed_formula = {
                "formula_id": idx,
                "type": "formula",
                "content": formula,
                "description": f"公式 {idx + 1}",
                "text_representation": formula
            }
            
            processed_formulas.append(processed_formula)
        
        return processed_formulas
    
    def _combine_all_text(self, results: Dict[str, Any]) -> str:
        """
        组合所有文本内容
        
        Args:
            results: 处理结果
            
        Returns:
            组合后的文本
        """
        combined_parts = []
        
        # 添加文本块
        for chunk in results.get("text_chunks", []):
            combined_parts.append(chunk["content"])
        
        # 添加图片描述
        for img_desc in results.get("image_descriptions", []):
            combined_parts.append(f"[图片] {img_desc['description']}")
        
        # 添加表格描述
        for table in results.get("table_data", []):
            combined_parts.append(f"[表格] {table['text_representation']}")
        
        # 添加公式描述
        for formula in results.get("formula_data", []):
            combined_parts.append(f"[公式] {formula['text_representation']}")
        
        return "\n\n".join(combined_parts)
    
    def extract_case_info(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从案例数据中提取关键信息
        
        Args:
            case_data: 案例数据
            
        Returns:
            提取的关键信息
        """
        # 处理多模态数据
        processed = self.process(case_data)
        
        # 提取关键信息
        case_info = {
            "case_id": case_data.get("id", ""),
            "title": case_data.get("title", ""),
            "type": case_data.get("type", "modern"),
            "text_chunks": processed["text_chunks"],
            "image_count": len(processed["image_descriptions"]),
            "table_count": len(processed["table_data"]),
            "formula_count": len(processed["formula_data"]),
            "combined_text": processed["combined_text"],
            "metadata": {
                "processed_at": self._get_timestamp(),
                "total_length": len(processed["combined_text"])
            }
        }
        
        return case_info
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
