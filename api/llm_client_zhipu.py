"""
智谱AI客户端 - 支持GLM-4、GLM-4V、Embedding-2
使用环境变量配置API密钥
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


class ZhipuClient:
    """智谱AI客户端"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化智谱AI客户端
        
        Args:
            config: 配置字典，可包含api_key, base_url等
        """
        # 从环境变量或配置读取API密钥
        self.api_key = self._get_api_key(config)
        self.base_url = config.get("base_url", "https://open.bigmodel.cn/api/coding/paas/v4")  # Coding Plan API
        
        # 模型配置
        self.models = {
            "text": config.get("model", "glm-4"),
            "vl": config.get("vl_model", "glm-4v"),
            "embedding": config.get("embedding_model", "embedding-2")
        }
        
        # 请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"智谱AI客户端初始化成功 - 模型: {self.models}")
    
    def _get_api_key(self, config: Dict = None) -> str:
        """
        获取API密钥（优先从环境变量）
        
        Args:
            config: 配置字典
            
        Returns:
            API密钥
        """
        # 1. 优先从环境变量读取
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if api_key:
            logger.info("✅ 从环境变量读取ZHIPUAI_API_KEY")
            return api_key
        
        # 2. 从配置读取（支持环境变量占位符）
        if config and "api_key" in config:
            key = config["api_key"]
            # 处理环境变量占位符 ${VAR_NAME}
            if key.startswith("${") and key.endswith("}"):
                env_var = key[2:-1].split(":")[0]  # 提取变量名
                api_key = os.getenv(env_var)
                if api_key:
                    logger.info(f"✅ 从环境变量{env_var}读取API密钥")
                    return api_key
            elif not key.startswith("$"):
                # 直接配置的密钥
                logger.warning("⚠️ 配置文件中直接使用了API密钥，建议使用环境变量")
                return key
        
        raise ValueError(
            "❌ 未找到智谱AI API密钥！请设置环境变量ZHIPUAI_API_KEY或在配置文件中配置"
        )
    
    def chat(
        self,
        messages: List[Dict],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        聊天补全
        
        Args:
            messages: 消息列表
            model: 模型名称（默认glm-4）
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            响应结果
        """
        model = model or self.models["text"]
        url = f"{self.base_url}chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ 聊天请求成功 - 模型: {model}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 聊天请求失败: {e}")
            raise
    
    def vision(
        self,
        messages: List[Dict],
        image_url: str = None,
        model: str = None,
        temperature: float = 0.5,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        视觉语言理解
        
        Args:
            messages: 消息列表
            image_url: 图片URL或base64
            model: 模型名称（默认glm-4v）
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            响应结果
        """
        model = model or self.models["vl"]
        url = f"{self.base_url}chat/completions"
        
        # 构建消息
        if image_url:
            # 在消息中添加图片
            messages_with_image = []
            for msg in messages:
                if msg["role"] == "user":
                    msg_with_image = {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            },
                            {
                                "type": "text",
                                "text": msg["content"]
                            }
                        ]
                    }
                    messages_with_image.append(msg_with_image)
                else:
                    messages_with_image.append(msg)
            messages = messages_with_image
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ 视觉理解请求成功 - 模型: {model}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 视觉理解请求失败: {e}")
            raise
    
    def embedding(
        self,
        text: str,
        model: str = None,
        **kwargs
    ) -> List[float]:
        """
        文本嵌入
        
        Args:
            text: 输入文本
            model: 模型名称（默认embedding-2）
            
        Returns:
            嵌入向量
        """
        model = model or self.models["embedding"]
        url = f"{self.base_url}embeddings"
        
        payload = {
            "model": model,
            "input": text,
            **kwargs
        }
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result["data"][0]["embedding"]
            logger.info(f"✅ 嵌入请求成功 - 模型: {model}, 维度: {len(embedding)}")
            return embedding
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 嵌入请求失败: {e}")
            raise
    
    def batch_embedding(
        self,
        texts: List[str],
        model: str = None,
        **kwargs
    ) -> List[List[float]]:
        """
        批量文本嵌入
        
        Args:
            texts: 文本列表
            model: 模型名称
            
        Returns:
            嵌入向量列表
        """
        model = model or self.models["embedding"]
        url = f"{self.base_url}embeddings"
        
        payload = {
            "model": model,
            "input": texts,
            **kwargs
        }
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            logger.info(f"✅ 批量嵌入请求成功 - 模型: {model}, 数量: {len(embeddings)}")
            return embeddings
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 批量嵌入请求失败: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        测试API连接
        
        Returns:
            是否连接成功
        """
        try:
            # 发送简单测试请求
            result = self.chat(
                messages=[{"role": "user", "content": "测试"}],
                max_tokens=10
            )
            
            if "choices" in result:
                logger.info("✅ 智谱AI API连接测试成功")
                return True
            else:
                logger.error("❌ 智谱AI API连接测试失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ 智谱AI API连接测试失败: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    # 测试连接
    client = ZhipuClient()
    
    # 测试聊天
    print("测试聊天...")
    result = client.chat([
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ])
    print(f"回复: {result['choices'][0]['message']['content']}")
    
    # 测试嵌入
    print("\n测试嵌入...")
    embedding = client.embedding("这是一个测试文本")
    print(f"嵌入维度: {len(embedding)}")
    print(f"前5个值: {embedding[:5]}")
    
    print("\n✅ 所有测试通过！")
