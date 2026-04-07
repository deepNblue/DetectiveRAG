"""
LLM API客户端 - Day 3 增强版
支持多种LLM API调用，带缓存、重试、错误处理
"""

from typing import Dict, List, Optional, Any
import yaml
import os
from loguru import logger
import requests
import time
import hashlib
import json


class LLMClient:
    """
    LLM API客户端（增强版）
    支持OpenAI、DeepSeek、GLM等多种API
    """
    
    def __init__(self, config_path: str = "./config/api_keys.yaml"):
        """
        初始化LLM客户端
        
        Args:
            config_path: API配置文件路径
        """
        self.logger = logger.bind(module="llm_client")
        self.config = self._load_config(config_path)
        
        # API配置
        self.api_key = None
        self.base_url = None
        self.model = None
        
        # 缓存
        self.cache = {}
        self.cache_enabled = True
        self.max_cache_size = 100
        
        # 重试配置
        self.max_retries = 3
        self.retry_delay = 2
        
        # 统计
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "cache_hits": 0,
            "total_tokens": 0
        }
        
        # 初始化客户端
        self._init_client()
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"API配置加载成功")
                return config
        else:
            self.logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "default_provider": "custom",
            "custom": {
                "api_key": "EMPTY",
                "base_url": "http://REDACTED_INTERNAL_IP:8094/v1",
                "model": "qwen3-30b-a3b"
            }
        }
    
    def _init_client(self):
        """初始化API客户端"""
        provider = self.config.get("default_provider", "custom")
        provider_config = self.config.get(provider, {})
        
        self.api_key = provider_config.get("api_key", "EMPTY")
        self.base_url = provider_config.get("base_url", "")
        self.model = provider_config.get("model", "qwen3-30b-a3b")
        
        self.logger.info(f"使用API提供商: {provider}")
        self.logger.info(f"Base URL: {self.base_url}")
        self.logger.info(f"Model: {self.model}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        use_cache: bool = True,
        timeout: int = None
    ) -> str:
        """
        调用聊天API（带缓存和重试）
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            use_cache: 是否使用缓存
            
        Returns:
            API响应文本
        """
        model = model or self.model
        self.stats["total_calls"] += 1
        
        # 检查缓存
        if use_cache and self.cache_enabled:
            cache_key = self._generate_cache_key(messages, model, temperature)
            
            if cache_key in self.cache:
                self.logger.debug("命中缓存")
                self.stats["cache_hits"] += 1
                return self.cache[cache_key]
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                result = self._call_api(
                    messages, model, temperature, max_tokens, timeout
                )
                
                # 缓存结果
                if use_cache and self.cache_enabled:
                    self._cache_result(cache_key, result)
                
                self.stats["successful_calls"] += 1
                return result
                
            except Exception as e:
                self.logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    self.stats["failed_calls"] += 1
                    self.logger.error(f"API调用最终失败: {e}")
                    return self._handle_error(e, messages)
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int = None
    ) -> str:
        """
        实际调用API
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时秒数
            
        Returns:
            API响应
        """
        _timeout = timeout or 180
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=_timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # 统计token使用
            if "usage" in result:
                self.stats["total_tokens"] += result["usage"].get("total_tokens", 0)
            
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API返回错误: {response.status_code} - {response.text}")
    
    def _generate_cache_key(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float
    ) -> str:
        """生成缓存键"""
        content = json.dumps(messages, sort_keys=True) + model + str(temperature)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cache_result(self, cache_key: str, result: str):
        """缓存结果"""
        # 检查缓存大小
        if len(self.cache) >= self.max_cache_size:
            # 删除最旧的缓存项
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
    
    def _handle_error(self, error: Exception, messages: List[Dict[str, str]]) -> str:
        """
        错误处理
        
        Args:
            error: 异常
            messages: 原始消息
            
        Returns:
            错误响应
        """
        error_msg = str(error)
        
        # 根据错误类型返回不同的响应
        if "Empty reply from server" in error_msg:
            return "[系统提示] API服务暂时不可用，请稍后再试"
        elif "timeout" in error_msg.lower():
            return "[系统提示] API调用超时，请简化输入或稍后再试"
        elif "401" in error_msg or "403" in error_msg:
            return "[系统提示] API认证失败，请检查API密钥"
        else:
            return f"[系统提示] API调用失败: {error_msg[:100]}"
    
    def simple_chat(self, prompt: str, temperature: float = 0.7) -> str:
        """
        简单聊天接口
        
        Args:
            prompt: 提示词
            temperature: 温度参数
            
        Returns:
            回复文本
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, temperature=temperature)
    
    def chat_with_images(
        self,
        prompt: str,
        image_paths: List[str],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 180,
    ) -> str:
        """
        多模态调用：文本+图片
        
        Args:
            prompt: 文本提示词
            image_paths: 本地图片文件路径列表
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时秒数（图片传输较慢）
            
        Returns:
            API响应文本
        """
        import base64
        
        # 构建 content 列表（文本 + 图片）
        content = [{"type": "text", "text": prompt}]
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                self.logger.warning(f"图片不存在: {img_path}")
                continue
            
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            
            ext = os.path.splitext(img_path)[1].lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
            mime = mime_map.get(ext, "image/png")
            
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            })
        
        messages = [{"role": "user", "content": content}]
        
        # 查找视觉模型配置（支持多种配置格式）
        vision_provider = (
            self.config.get("zhipu_vision") or 
            self.config.get("vl") or 
            self.config.get("vision")
        )
        if vision_provider:
            model = vision_provider.get("model", "gemma-4-26B-A4B-it-Q8_0.gguf")
            saved_api_key = self.api_key
            saved_base_url = self.base_url
            self.api_key = vision_provider.get("api_key", self.api_key)
            self.base_url = vision_provider.get("base_url", self.base_url)
            
            try:
                self.logger.info(f"[Vision] 调用视觉模型: {model} @ {self.base_url}")
                result = self.chat_completion(
                    messages, model=model, temperature=temperature,
                    max_tokens=max_tokens, use_cache=False, timeout=timeout
                )
                return result
            finally:
                self.api_key = saved_api_key
                self.base_url = saved_base_url
        else:
            # fallback: 用默认模型发（可能失败）
            return self.chat_completion(messages, temperature=temperature, timeout=timeout)
    
    def chat_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7
    ) -> str:
        """
        带系统提示的聊天
        
        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            temperature: 温度参数
            
        Returns:
            回复文本
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.chat_completion(messages, temperature=temperature)
    
    def extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """
        从响应中提取JSON
        
        Args:
            response: API响应
            
        Returns:
            解析后的JSON对象
        """
        try:
            # 直接解析
            return json.loads(response)
        except:
            pass
        
        # 尝试提取JSON块
        import re
        
        # 查找 ```json ... ``` 块
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # 查找 { ... } 或 [ ... ]
        json_match = re.search(r'[\{\[].*[\}\]]', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        self.logger.warning(f"无法从响应中提取JSON: {response[:100]}...")
        return None
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.logger.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计数据
        """
        success_rate = (
            self.stats["successful_calls"] / self.stats["total_calls"] * 100
            if self.stats["total_calls"] > 0 else 0
        )
        
        cache_hit_rate = (
            self.stats["cache_hits"] / self.stats["total_calls"] * 100
            if self.stats["total_calls"] > 0 else 0
        )
        
        return {
            **self.stats,
            "success_rate": f"{success_rate:.1f}%",
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "cache_size": len(self.cache)
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        测试API连接
        
        Returns:
            测试结果
        """
        try:
            response = self.simple_chat("测试连接", temperature=0.1)
            
            return {
                "success": True,
                "message": "API连接正常",
                "response_length": len(response),
                "model": self.model
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.model
            }


# 兼容OpenAI SDK的接口（如果安装了openai包）
try:
    from openai import OpenAI
    
    class OpenAICompatibleClient(LLMClient):
        """OpenAI兼容客户端"""
        
        def _init_client(self):
            """初始化OpenAI客户端"""
            provider = self.config.get("default_provider", "custom")
            provider_config = self.config.get(provider, {})
            
            api_key = provider_config.get("api_key", "EMPTY")
            base_url = provider_config.get("base_url", "")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            self.model = provider_config.get("model", "qwen3-30b-a3b")
            
            self.logger.info(f"OpenAI兼容客户端初始化完成")
        
        def _call_api(
            self,
            messages: List[Dict[str, str]],
            model: str,
            temperature: float,
            max_tokens: int
        ) -> str:
            """使用OpenAI SDK调用"""
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # 统计token
                if hasattr(response, 'usage'):
                    self.stats["total_tokens"] += response.usage.total_tokens
                
                return response.choices[0].message.content
                
            except Exception as e:
                raise Exception(f"OpenAI SDK调用失败: {e}")

except ImportError:
    # 如果没有安装openai包，使用基础客户端
    OpenAICompatibleClient = LLMClient
    logger.info("未安装openai包，使用基础HTTP客户端")
