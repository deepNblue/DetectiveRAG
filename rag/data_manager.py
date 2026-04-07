"""
数据管理器
运行时控制数据导入和向量化
"""

from typing import Dict, List, Any, Optional
from loguru import logger
import os
import json
import shutil


class DataManager:
    """
    数据管理器
    管理数据导入、向量化、索引构建等运行时操作
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据管理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logger.bind(module="data_manager")
        
        # 数据路径配置
        self.raw_data_path = self.config.get("data", {}).get(
            "raw_data_path",
            "/home/dudu/detective_system/data"
        )
        self.processed_data_path = self.config.get("data", {}).get(
            "processed_data_path",
            "./data/processed"
        )
        
        # 向量数据库管理器（稍后注入）
        self.rag_anything = None
        
        # 状态追踪
        self.indexing_status = {
            "is_indexing": False,
            "progress": 0,
            "total": 0,
            "current_file": "",
            "errors": []
        }
        
        self.logger.info("数据管理器初始化完成")
    
    def set_rag_anything(self, rag_anything):
        """
        设置RAG-Anything实例
        
        Args:
            rag_anything: RAG-Anything实例
        """
        self.rag_anything = rag_anything
        self.logger.info("RAG-Anything已设置")
    
    def scan_available_data(self) -> Dict[str, Any]:
        """
        扫描可用数据
        
        Returns:
            可用数据信息
        """
        self.logger.info(f"扫描数据目录: {self.raw_data_path}")
        
        data_info = {
            "sherlock_cases": 0,
            "modern_cases": 0,
            "logic_puzzles": 0,
            "images": 0,
            "total_files": 0,
            "files": []
        }
        
        try:
            # 扫描processed目录
            processed_path = os.path.join(self.raw_data_path, "processed")
            
            if os.path.exists(processed_path):
                for filename in os.listdir(processed_path):
                    if filename.endswith(".json"):
                        file_path = os.path.join(processed_path, filename)
                        
                        # 读取文件统计
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        file_info = {
                            "filename": filename,
                            "path": file_path,
                            "size": os.path.getsize(file_path),
                            "type": self._detect_data_type(filename)
                        }
                        
                        # 统计数据类型
                        if "sherlock" in filename.lower():
                            if isinstance(data, list):
                                data_info["sherlock_cases"] += len(data)
                            elif isinstance(data, dict):
                                data_info["sherlock_cases"] += 1
                        elif "modern" in filename.lower():
                            if isinstance(data, list):
                                data_info["modern_cases"] += len(data)
                            elif isinstance(data, dict):
                                data_info["modern_cases"] += 1
                        elif "logic" in filename.lower():
                            if isinstance(data, list):
                                data_info["logic_puzzles"] += len(data)
                            elif isinstance(data, dict):
                                data_info["logic_puzzles"] += 1
                        
                        data_info["files"].append(file_info)
                        data_info["total_files"] += 1
            
            # 扫描图片目录
            images_path = os.path.join(self.raw_data_path, "images_customized")
            
            if os.path.exists(images_path):
                image_count = len([
                    f for f in os.listdir(images_path)
                    if f.endswith(('.jpg', '.jpeg', '.png'))
                ])
                data_info["images"] = image_count
            
            self.logger.info(
                f"扫描完成: {data_info['total_files']}个文件, "
                f"{data_info['images']}张图片"
            )
            
            return data_info
            
        except Exception as e:
            self.logger.error(f"扫描失败: {e}")
            return {"error": str(e)}
    
    def _detect_data_type(self, filename: str) -> str:
        """
        检测数据类型
        
        Args:
            filename: 文件名
            
        Returns:
            数据类型
        """
        filename_lower = filename.lower()
        
        if "sherlock" in filename_lower:
            return "sherlock_cases"
        elif "modern" in filename_lower:
            return "modern_cases"
        elif "logic" in filename_lower:
            return "logic_puzzles"
        elif "integrated" in filename_lower:
            return "integrated"
        else:
            return "unknown"
    
    def import_data(
        self,
        data_types: List[str] = None,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        导入数据到向量数据库（运行时调用）
        
        Args:
            data_types: 要导入的数据类型列表（None表示全部）
            batch_size: 批处理大小
            
        Returns:
            导入结果
        """
        if not self.rag_anything:
            return {"error": "RAG-Anything未初始化"}
        
        if self.indexing_status["is_indexing"]:
            return {"error": "已有索引任务在运行"}
        
        self.logger.info("开始导入数据...")
        
        # 更新状态
        self.indexing_status["is_indexing"] = True
        self.indexing_status["progress"] = 0
        self.indexing_status["errors"] = []
        
        result = {
            "success": False,
            "imported_count": 0,
            "failed_count": 0,
            "errors": []
        }
        
        try:
            # 扫描可用数据
            data_info = self.scan_available_data()
            
            if "error" in data_info:
                result["errors"].append(data_info["error"])
                return result
            
            # 准备文档列表
            all_documents = []
            
            for file_info in data_info["files"]:
                file_type = file_info["type"]
                
                # 检查是否需要导入此类型
                if data_types and file_type not in data_types:
                    continue
                
                # 读取文件
                with open(file_info["path"], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 处理数据
                if isinstance(data, list):
                    for item in data:
                        doc = self._convert_to_document(item, file_type)
                        if doc:
                            all_documents.append(doc)
                elif isinstance(data, dict):
                    doc = self._convert_to_document(data, file_type)
                    if doc:
                        all_documents.append(doc)
            
            # 批量索引
            self.indexing_status["total"] = len(all_documents)
            
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i + batch_size]
                
                try:
                    self.rag_anything.index_documents(batch)
                    result["imported_count"] += len(batch)
                    
                    # 更新进度
                    self.indexing_status["progress"] = i + len(batch)
                    self.indexing_status["current_file"] = f"Batch {i//batch_size + 1}"
                    
                except Exception as e:
                    result["failed_count"] += len(batch)
                    result["errors"].append(f"Batch {i//batch_size + 1} 失败: {str(e)}")
                    self.indexing_status["errors"].append(str(e))
            
            result["success"] = result["imported_count"] > 0
            
            self.logger.info(
                f"数据导入完成: 成功{result['imported_count']}, "
                f"失败{result['failed_count']}"
            )
            
        except Exception as e:
            self.logger.error(f"数据导入失败: {e}")
            result["errors"].append(str(e))
            
        finally:
            self.indexing_status["is_indexing"] = False
        
        return result
    
    def _convert_to_document(
        self,
        item: Dict[str, Any],
        data_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        将数据项转换为文档格式
        
        Args:
            item: 数据项
            data_type: 数据类型
            
        Returns:
            文档字典
        """
        try:
            # 提取内容
            content = ""
            
            if data_type == "sherlock_cases":
                content = self._extract_sherlock_content(item)
            elif data_type == "modern_cases":
                content = self._extract_modern_content(item)
            elif data_type == "logic_puzzles":
                content = self._extract_logic_content(item)
            else:
                content = str(item)
            
            if not content:
                return None
            
            # 创建文档
            doc = {
                "id": item.get("id", f"{data_type}_{hash(content)}"),
                "content": content,
                "metadata": {
                    "type": data_type,
                    "title": item.get("title", ""),
                    "source": item.get("source", "")
                }
            }
            
            return doc
            
        except Exception as e:
            self.logger.error(f"文档转换失败: {e}")
            return None
    
    def _extract_sherlock_content(self, item: Dict[str, Any]) -> str:
        """提取Sherlock案例内容"""
        parts = []
        
        if "title" in item:
            parts.append(f"标题: {item['title']}")
        
        if "description" in item:
            parts.append(f"描述: {item['description']}")
        
        if "solution" in item:
            parts.append(f"解答: {item['solution']}")
        
        return "\n\n".join(parts)
    
    def _extract_modern_content(self, item: Dict[str, Any]) -> str:
        """提取现代案例内容"""
        parts = []
        
        if "title" in item:
            parts.append(f"标题: {item['title']}")
        
        if "description" in item:
            parts.append(f"描述: {item['description']}")
        
        if "evidence" in item:
            parts.append(f"证据: {', '.join(item['evidence'])}")
        
        return "\n\n".join(parts)
    
    def _extract_logic_content(self, item: Dict[str, Any]) -> str:
        """提取逻辑题内容"""
        parts = []
        
        if "question" in item:
            parts.append(f"问题: {item['question']}")
        
        if "options" in item:
            parts.append(f"选项: {', '.join(item['options'])}")
        
        if "answer" in item:
            parts.append(f"答案: {item['answer']}")
        
        return "\n\n".join(parts)
    
    def get_indexing_status(self) -> Dict[str, Any]:
        """
        获取索引状态
        
        Returns:
            索引状态
        """
        return self.indexing_status.copy()
    
    def clear_index(self) -> Dict[str, Any]:
        """
        清空索引（运行时调用）
        
        Returns:
            清空结果
        """
        if not self.rag_anything:
            return {"error": "RAG-Anything未初始化"}
        
        try:
            # 删除集合
            if self.rag_anything.collection:
                self.rag_anything.chroma_client.delete_collection(
                    self.rag_anything.collection_name
                )
                
                # 重新创建集合
                self.rag_anything.init_vector_db()
            
            self.logger.info("索引已清空")
            
            return {
                "success": True,
                "message": "索引已清空"
            }
            
        except Exception as e:
            self.logger.error(f"清空索引失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计数据
        """
        stats = {
            "data_manager": {
                "raw_data_path": self.raw_data_path,
                "processed_data_path": self.processed_data_path,
                "rag_anything_connected": self.rag_anything is not None
            },
            "indexing_status": self.indexing_status.copy()
        }
        
        # 添加可用数据信息
        data_info = self.scan_available_data()
        if "error" not in data_info:
            stats["available_data"] = data_info
        
        return stats
