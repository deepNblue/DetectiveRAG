"""
侦探推理RAG系统 - RAG-Anything集成适配器
将GitHub上的RAG-Anything源码与侦探推理系统对接
基于: https://github.com/HKUDS/RAG-Anything (v1.2.10)
依赖: LightRAG + 智谱AI API (全部走API，不使用本地模型)

模型配置:
  - LLM:      glm-5.1 (Coding Plan API) / glm-4-flash (免费备用)
  - Vision:   glm-4v-flash (免费)
  - Embedding: embedding-2/3 (智谱API)
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ============================================================
#  智谱AI API 统一配置
# ============================================================

ZHIPU_API_KEY = os.getenv("ZHIPUAI_API_KEY")
ZHIPU_CHAT_BASE = "https://open.bigmodel.cn/api/coding/paas/v4"       # Coding Plan API
ZHIPU_GENERAL_BASE = "https://open.bigmodel.cn/api/paas/v4"           # 普通API


# ============================================================
#  LLM 函数 (GLM-5.1 / GLM-4-Flash)
# ============================================================

def create_llm_func(
    api_key: str = None,
    base_url: str = None,
    model: str = "glm-5.1",
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> Callable:
    """
    创建LLM函数（全部走API）
    
    默认使用 Coding Plan API + glm-5.1
    备用使用 普通API + glm-4-flash（免费）
    """
    import httpx
    api_key = api_key or ZHIPU_API_KEY
    base_url = base_url or ZHIPU_CHAT_BASE
    
    async def llm_func(prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        use_model = kwargs.get("model", model)
        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", temperature),
            "max_tokens": kwargs.get("max_tokens", max_tokens),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{base_url}/chat/completions", headers=headers, json=payload
            )
            
            # 如果主模型失败，自动降级到免费模型
            if response.status_code != 200:
                error_msg = ""
                try:
                    error_msg = response.json().get("error", {}).get("message", "")
                except:
                    error_msg = response.text[:100]
                
                # 降级到glm-4-flash
                if use_model != "glm-4-flash":
                    logger.warning(f"⚠️ {use_model}失败({response.status_code}), 降级到glm-4-flash")
                    payload["model"] = "glm-4-flash"
                    response = await client.post(
                        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                        headers=headers, json=payload
                    )
            
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"].get("content", "")
            # 某些模型有reasoning_content字段
            if not content:
                content = result["choices"][0]["message"].get("reasoning_content", "")
            return content
    
    return llm_func


# ============================================================
#  Vision 函数 (GLM-4V-Flash 免费)
# ============================================================

def create_vision_func(
    api_key: str = None,
    base_url: str = None,
    model: str = "glm-4v-flash",
    temperature: float = 0.5,
    max_tokens: int = 1000,
) -> Callable:
    """
    创建视觉模型函数（全部走API）
    
    默认使用 glm-4v-flash（免费）
    """
    import httpx
    api_key = api_key or ZHIPU_API_KEY
    base_url = base_url or ZHIPU_GENERAL_BASE
    
    async def vision_func(
        prompt: str, image_data: str = None, messages: List[Dict] = None,
        system_prompt: str = None, **kwargs
    ) -> str:
        if messages:
            payload_messages = messages
        else:
            content_parts = []
            if image_data:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                })
            if prompt:
                content_parts.append({"type": "text", "text": prompt})
            
            payload_messages = []
            if system_prompt:
                payload_messages.append({"role": "system", "content": system_prompt})
            payload_messages.append({"role": "user", "content": content_parts})
        
        payload = {
            "model": kwargs.get("model", model),
            "messages": payload_messages,
            "temperature": kwargs.get("temperature", temperature),
            "max_tokens": kwargs.get("max_tokens", max_tokens),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{base_url}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    
    return vision_func


# ============================================================
#  Embedding 函数 (智谱API - embedding-2/3)
# ============================================================

def create_embedding_func(
    api_key: str = None,
    base_url: str = None,
    model: str = "embedding-2",
    embedding_dim: int = 1024,
) -> Callable:
    """
    创建Embedding函数（全部走API，不使用本地模型）
    
    使用智谱 embedding-2 API
    """
    import httpx
    api_key = api_key or ZHIPU_API_KEY
    base_url = base_url or ZHIPU_GENERAL_BASE
    
    async def _api_embed(texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 分批处理（API限制）
        batch_size = 16
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{base_url}/embeddings",
                    headers=headers,
                    json={"model": model, "input": batch}
                )
                
                if response.status_code != 200:
                    error_msg = f"Embedding API错误: {response.status_code}"
                    try:
                        error_msg = response.json().get("error", {}).get("message", error_msg)
                    except:
                        pass
                    raise RuntimeError(error_msg)
                
                result = response.json()
                batch_embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    # 包装为LightRAG需要的EmbeddingFunc格式
    try:
        from lightrag.utils import EmbeddingFunc
        return EmbeddingFunc(embedding_dim=embedding_dim, func=_api_embed)
    except ImportError:
        return _api_embed


# ============================================================
#  侦探推理RAG-Anything 集成器
# ============================================================

class DetectiveRAGAnything:
    """
    侦探推理RAG系统 - RAG-Anything集成器
    
    源码来源: https://github.com/HKUDS/RAG-Anything (v1.2.10)
    核心依赖: LightRAG + 智谱AI API（全部走API）
    
    模型配置:
    - LLM:      glm-5.1 (Coding Plan) → 自动降级 glm-4-flash (免费)
    - Vision:   glm-4v-flash (免费)
    - Embedding: embedding-2 (智谱API)
    """
    
    def __init__(
        self,
        working_dir: str = "./data/rag_storage",
        api_key: str = None,
        llm_model: str = "glm-5.1",
        vl_model: str = "glm-4v-flash",
        embedding_model: str = "embedding-2",
        base_url: str = None,
    ):
        self.working_dir = working_dir
        self.api_key = api_key or ZHIPU_API_KEY
        self.llm_model = llm_model
        self.vl_model = vl_model
        self.embedding_model = embedding_model
        self.base_url = base_url or ZHIPU_CHAT_BASE
        
        # 创建模型函数（全部走API）
        self.llm_func = create_llm_func(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.llm_model,
        )
        self.vision_func = create_vision_func(
            api_key=self.api_key,
            model=self.vl_model,
        )
        self.embedding_func = create_embedding_func(
            api_key=self.api_key,
            model=self.embedding_model,
        )
        
        self.rag_anything = None
        self._initialized = False
        
        logger.info(f"侦探推理RAG-Anything集成器: LLM={self.llm_model}(API), Vision={self.vl_model}(API), Embedding={self.embedding_model}(API)")
    
    async def initialize(self):
        """初始化RAG-Anything + LightRAG"""
        if self._initialized:
            return
        
        from raganything import RAGAnything, RAGAnythingConfig
        
        os.makedirs(self.working_dir, exist_ok=True)
        
        config = RAGAnythingConfig(
            working_dir=self.working_dir,
            parser="docling",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        
        self.rag_anything = RAGAnything(
            llm_model_func=self.llm_func,
            vision_model_func=self.vision_func,
            embedding_func=self.embedding_func,
            config=config,
            lightrag_kwargs={
                "working_dir": self.working_dir,
            }
        )
        
        # 跳过parser安装检查
        self.rag_anything._parser_installation_checked = True
        
        result = await self.rag_anything._ensure_lightrag_initialized()
        
        if result.get("success"):
            self._initialized = True
            logger.info("✅ RAG-Anything + LightRAG初始化成功")
        else:
            error = result.get("error", "Unknown error")
            logger.error(f"❌ 初始化失败: {error}")
            raise RuntimeError(f"RAG-Anything初始化失败: {error}")
    
    async def insert_text(self, text: str, description: str = "") -> Dict[str, Any]:
        """插入纯文本到RAG系统"""
        if not self._initialized:
            await self.initialize()
        try:
            await self.rag_anything.lightrag.ainsert(text)
            logger.info(f"✅ 文本插入成功: {description[:50] if description else text[:50]}...")
            return {"success": True, "description": description}
        except Exception as e:
            logger.error(f"❌ 文本插入失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def insert_document(self, file_path: str) -> Dict[str, Any]:
        """插入文档到RAG系统（支持PDF/图片/Office）"""
        if not self._initialized:
            await self.initialize()
        try:
            result = await self.rag_anything.aprocess_file(file_path)
            logger.info(f"✅ 文档处理完成: {file_path}")
            return result
        except Exception as e:
            logger.error(f"❌ 文档处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def query(self, query: str, mode: str = "mix") -> str:
        """查询RAG系统 (local/global/hybrid/naive/mix)"""
        if not self._initialized:
            await self.initialize()
        try:
            result = await self.rag_anything.aquery(query, mode=mode)
            logger.info(f"✅ 查询完成: {query[:50]}...")
            return result
        except Exception as e:
            logger.error(f"❌ 查询失败: {e}")
            return f"查询失败: {str(e)}"
    
    async def query_with_multimodal(
        self, query: str, multimodal_content: List[Dict[str, Any]] = None, mode: str = "mix"
    ) -> str:
        """多模态查询"""
        if not self._initialized:
            await self.initialize()
        try:
            return await self.rag_anything.aquery_with_multimodal(
                query, multimodal_content=multimodal_content, mode=mode
            )
        except Exception as e:
            logger.error(f"❌ 多模态查询失败: {e}")
            return f"查询失败: {str(e)}"
    
    async def detective_query(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """侦探推理专用查询"""
        case_text = case_data.get("case_text", "")
        question = case_data.get("question", "")
        mode = case_data.get("mode", "mix")
        multimodal = case_data.get("multimodal")
        
        detective_prompt = f"""你是侦探推理专家。请基于以下案件信息进行分析和推理。

## 案件信息
{case_text}

## 问题
{question}

请提供详细的推理分析，包括：
1. 关键线索识别
2. 嫌疑人分析
3. 证据关联
4. 推理结论
"""
        
        if multimodal:
            result = await self.query_with_multimodal(
                detective_prompt, multimodal_content=multimodal, mode=mode
            )
        else:
            result = await self.query(detective_prompt, mode=mode)
        
        return {"success": True, "question": question, "analysis": result, "mode": mode}
    
    def get_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "initialized": self._initialized,
            "models": {
                "llm": f"{self.llm_model} (API: Coding Plan)",
                "vision": f"{self.vl_model} (API: 免费)",
                "embedding": f"{self.embedding_model} (API: 智谱)",
            },
            "working_dir": self.working_dir,
            "source": "GitHub: HKUDS/RAG-Anything v1.2.10",
            "note": "全部模型调用走API，不使用本地模型",
        }
    
    async def close(self):
        """关闭并清理资源"""
        if self.rag_anything:
            try:
                await self.rag_anything.finalize_storages()
            except:
                pass
            logger.info("✅ RAG-Anything资源已清理")


def create_detective_rag(
    working_dir: str = "./data/rag_storage",
    api_key: str = None,
) -> DetectiveRAGAnything:
    """创建侦探推理RAG-Anything实例"""
    return DetectiveRAGAnything(working_dir=working_dir, api_key=api_key)


# ============================================================
#  测试
# ============================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    
    print("\n" + "="*70)
    print("🧪 RAG-Anything集成测试 (全部API，无本地模型)")
    print("="*70)
    
    async def test():
        rag = create_detective_rag()
        print(f"\n1️⃣ 创建实例: LLM={rag.llm_model}, Vision={rag.vl_model}, Embedding={rag.embedding_model}")
        
        # LLM测试
        print("\n2️⃣ 测试LLM (glm-5.1 API)...")
        try:
            result = await rag.llm_func("用一句话介绍你自己")
            print(f"   ✅ LLM响应: {result[:80]}")
        except Exception as e:
            print(f"   ❌ LLM失败: {e}")
        
        # Vision测试
        print("\n3️⃣ 测试Vision (glm-4v-flash API)...")
        try:
            result = await rag.vision_func("描述一个苹果的外观")
            print(f"   ✅ Vision响应: {result[:80]}")
        except Exception as e:
            print(f"   ❌ Vision失败: {e}")
        
        # Embedding测试
        print("\n4️⃣ 测试Embedding (embedding-2 API)...")
        try:
            embeddings = await rag.embedding_func(["测试文本"])
            dim = len(embeddings[0]) if isinstance(embeddings, list) else "N/A"
            print(f"   ✅ Embedding维度: {dim}")
        except Exception as e:
            print(f"   ❌ Embedding失败: {e}")
        
        # 初始化
        print("\n5️⃣ 初始化RAG-Anything...")
        try:
            await rag.initialize()
            print(f"   ✅ 初始化成功!")
        except Exception as e:
            print(f"   ❌ 初始化失败: {e}")
        
        await rag.close()
        print("\n✅ 测试完成!")
    
    asyncio.run(test())
