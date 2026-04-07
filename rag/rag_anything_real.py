"""
RAG-Anything真实实现 - 使用本地Embedding模型
修复核心功能的伪实现问题
"""

from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings
from loguru import logger
import os
import numpy as np

# 真实的Embedding模型
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("sentence-transformers未安装，将使用简化版本")


class RAGAnythingReal:
    """
    RAG-Anything真实实现
    使用真实的Embedding模型和向量检索
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化RAG-Anything
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logger.bind(module="rag_anything_real")
        
        # RAG配置
        rag_config = self.config.get("rag", {}).get("rag_anything", {})
        self.chunk_size = rag_config.get("chunk_size", 512)
        self.chunk_overlap = rag_config.get("chunk_overlap", 50)
        self.top_k = rag_config.get("top_k", 5)
        
        # 向量数据库配置
        db_config = self.config.get("vector_db", {})
        self.persist_dir = db_config.get("persist_directory", "./data/vector_db")
        self.collection_name = db_config.get("collection_name", "detective_cases")
        
        # 初始化Embedding模型
        self.embedding_model = None
        self.embedding_dimension = 384  # 默认维度
        
        if EMBEDDING_AVAILABLE:
            try:
                self.logger.info("正在加载本地Embedding模型...")
                self.embedding_model = SentenceTransformer(
                    'paraphrase-multilingual-MiniLM-L12-v2'
                )
                self.embedding_dimension = 384
                self.logger.info("✅ 本地Embedding模型加载成功")
            except Exception as e:
                self.logger.error(f"❌ Embedding模型加载失败: {e}")
        
        # 初始化向量数据库（延迟初始化）
        self.chroma_client = None
        self.collection = None
        
        self.logger.info("RAG-Anything真实实现初始化完成")
    
    def init_vector_db(self):
        """
        初始化向量数据库
        """
        try:
            # 确保目录存在
            os.makedirs(self.persist_dir, exist_ok=True)
            
            # 初始化ChromaDB
            self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            
            # 创建或获取集合
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "侦探案例知识库",
                    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
                    "dimension": self.embedding_dimension
                }
            )
            
            self.logger.info(f"✅ 向量数据库初始化完成: {self.persist_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ 向量数据库初始化失败: {e}")
            raise
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        使用真实的Embedding模型编码文本
        
        Args:
            texts: 文本列表
            
        Returns:
            Embedding向量列表
        """
        if not texts:
            return []
        
        try:
            if self.embedding_model:
                # 使用真实的SentenceTransformer模型
                self.logger.info(f"正在编码 {len(texts)} 个文本...")
                embeddings = self.embedding_model.encode(
                    texts,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                return embeddings.tolist()
            else:
                # 降级方案：使用简单的TF-IDF
                self.logger.warning("⚠️ Embedding模型不可用，使用TF-IDF降级方案")
                return self._tfidf_encode(texts)
                
        except Exception as e:
            self.logger.error(f"❌ 文本编码失败: {e}")
            # 降级到简单编码
            return self._simple_encode(texts)
    
    def _tfidf_encode(self, texts: List[str]) -> List[List[float]]:
        """
        TF-IDF编码（降级方案）
        
        Args:
            texts: 文本列表
            
        Returns:
            TF-IDF向量列表
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vectorizer = TfidfVectorizer(max_features=self.embedding_dimension)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # 转换为dense并pad到目标维度
            dense_matrix = tfidf_matrix.toarray()
            
            # Pad或truncate到固定维度
            result = []
            for vec in dense_matrix:
                if len(vec) < self.embedding_dimension:
                    vec = np.pad(vec, (0, self.embedding_dimension - len(vec)))
                else:
                    vec = vec[:self.embedding_dimension]
                result.append(vec.tolist())
            
            return result
            
        except Exception as e:
            self.logger.warning(f"TF-IDF编码失败: {e}，使用简单编码")
            return self._simple_encode(texts)
    
    def _simple_encode(self, texts: List[str]) -> List[List[float]]:
        """
        简单编码（最终降级方案）
        
        Args:
            texts: 文本列表
            
        Returns:
            简单向量列表
        """
        import hashlib
        
        embeddings = []
        for text in texts:
            # 使用哈希生成伪向量
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            
            # 扩展到目标维度
            vector = []
            for i in range(self.embedding_dimension):
                byte_val = hash_bytes[i % len(hash_bytes)]
                vector.append(byte_val / 255.0)
            
            embeddings.append(vector)
        
        return embeddings
    
    def index_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]] = None):
        """
        索引文档到向量数据库
        
        Args:
            documents: 文档列表
            embeddings: 预计算的embedding向量（可选）
        """
        if not self.collection:
            self.init_vector_db()
        
        if not documents:
            self.logger.warning("没有文档需要索引")
            return
        
        try:
            # 准备数据
            ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
            texts = [doc.get("content", "") for doc in documents]
            metadatas = [doc.get("metadata", {}) for doc in documents]
            
            # 如果没有提供embedding，使用真实模型编码
            if embeddings is None:
                self.logger.info("正在生成文档Embedding...")
                embeddings = self.encode_texts(texts)
            
            # 添加到集合
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            self.logger.info(f"✅ 成功索引 {len(documents)} 个文档")
            
        except Exception as e:
            self.logger.error(f"❌ 文档索引失败: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        if not self.collection:
            self.logger.warning("向量数据库未初始化")
            return []
        
        top_k = top_k or self.top_k
        
        try:
            # 使用真实模型编码查询
            query_embedding = self.encode_texts([query])[0]
            
            # 查询
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # 格式化结果
            documents = []
            for i in range(len(results["ids"][0])):
                doc = {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                }
                documents.append(doc)
            
            self.logger.info(f"✅ 检索到 {len(documents)} 个相关文档")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"❌ 检索失败: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计数据
        """
        if not self.collection:
            return {"status": "not_initialized"}
        
        try:
            count = self.collection.count()
            
            return {
                "status": "initialized",
                "document_count": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_dir,
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2" if self.embedding_model else "fallback",
                "embedding_dimension": self.embedding_dimension
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# 使用示例
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 RAG-Anything真实实现测试")
    print("="*60 + "\n")
    
    # 初始化
    rag = RAGAnythingReal()
    
    # 测试编码
    print("测试文本编码...")
    texts = ["这是一个测试文本", "这是另一个测试"]
    embeddings = rag.encode_texts(texts)
    print(f"✅ 编码成功: {len(embeddings)} 个向量，维度: {len(embeddings[0])}")
    
    # 测试索引和检索
    print("\n测试文档索引和检索...")
    docs = [
        {"id": "1", "content": "嫌疑人张某在案发现场出现", "metadata": {"type": "evidence"}},
        {"id": "2", "content": "证人李某看到了案发过程", "metadata": {"type": "witness"}}
    ]
    
    rag.init_vector_db()
    rag.index_documents(docs)
    
    results = rag.retrieve("谁在现场")
    print(f"✅ 检索结果: {len(results)} 个文档")
    
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['content'][:50]}... (距离: {result['distance']:.4f})")
    
    print("\n✅ 测试完成！")
