"""
Agent基类
提供所有Agent的通用功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from loguru import logger
import yaml
import os
import json


class BaseAgent(ABC):
    """
    Agent基类
    所有专业化Agent都继承此类
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None, llm_client=None):
        """
        初始化Agent
        
        Args:
            name: Agent名称
            config: 配置字典
            llm_client: LLM客户端实例
        """
        self.name = name
        self.config = config or self._load_default_config()
        self.logger = logger.bind(agent=name)
        self.llm_client = llm_client  # LLM客户端
        
        self.logger.info(f"初始化Agent: {name}")
        
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "../config/config.yaml"
        )
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return {}
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入数据（抽象方法，子类必须实现）
        
        Args:
            input_data: 输入数据字典
            
        Returns:
            处理结果字典
        """
        pass
    
    # 🆕 v9.3: 名字去偏指令 — 所有专家共享
    NAME_DEBIAS_PROMPT = """

【⚠️ 重要反偏见指令】
- 不要因为某个名字"听起来像坏人"就认为他/她是凶手
- 不要偏好常见名字（如张伟、王刚、李强等），这些名字出现频率高不代表犯罪概率高
- 必须基于具体的证据链（动机、机会、能力、异常行为）来判断，而非名字的"感觉"
- 如果两个嫌疑人证据相当，应仔细审查各自的作案条件和排除条件，而非凭直觉选一个
"""

    def call_llm(self, prompt: str, temperature: float = None) -> str:
        """
        调用LLM API
        
        Args:
            prompt: 提示词
            temperature: 温度参数（可选）
            
        Returns:
            LLM回复文本
        """
        if self.llm_client is None:
            self.logger.warning("LLM客户端未初始化，返回模拟响应")
            return self._generate_mock_response(prompt)
        
        # 🆕 v9.3: 自动注入名字去偏指令 (仅对包含"嫌疑人"的推理类prompt)
        if "嫌疑人" in prompt and "culprit" not in self.name.lower():
            prompt = prompt + self.NAME_DEBIAS_PROMPT

        try:
            # 使用配置的温度或默认温度
            temp = temperature or self.config.get('temperature', 0.7)
            
            # 使用增强后的LLM客户端
            response = self.llm_client.simple_chat(prompt, temperature=temp)
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return self._generate_mock_response(prompt, error=str(e))
    
    def call_llm_with_images(self, prompt: str, image_paths: List[str], temperature: float = None) -> str:
        """
        多模态LLM调用：文本+图片
        
        Args:
            prompt: 文本提示词
            image_paths: 本地图片文件路径列表
            temperature: 温度参数
            
        Returns:
            LLM回复文本（包含图片理解结果）
        """
        if self.llm_client is None:
            self.logger.warning("LLM客户端未初始化，无法处理图片")
            return "[系统提示] LLM客户端未初始化，无法分析图片"
        
        if not image_paths:
            return self.call_llm(prompt, temperature)
        
        try:
            temp = temperature or self.config.get('temperature', 0.7)
            # 注入名字去偏指令
            if "嫌疑人" in prompt and "culprit" not in self.name.lower():
                prompt = prompt + self.NAME_DEBIAS_PROMPT
            
            self.logger.info(f"🖼️ 多模态调用: {len(image_paths)}张图片")
            response = self.llm_client.chat_with_images(prompt, image_paths, temperature=temp)
            return response
        except Exception as e:
            self.logger.error(f"多模态LLM调用失败: {e}")
            # fallback: 跳过图片用纯文本
            self.logger.warning("回退到纯文本模式")
            return self.call_llm(prompt, temperature)
    
    def _generate_mock_response(self, prompt: str, error: str = None) -> str:
        """
        生成模拟响应（当LLM不可用时）
        
        Args:
            prompt: 原始提示词
            error: 错误信息
            
        Returns:
            模拟响应
        """
        if error:
            return f"[系统提示] LLM暂时不可用 ({error[:50]}...)"
        
        # 根据Agent类型生成不同的模拟响应
        if "线索" in self.name or "Clue" in self.name:
            return json.dumps([
                {"time": "案发当晚", "event": "案件发生", "order": 1},
                {"time": "第二天上午", "event": "发现现场", "order": 2}
            ], ensure_ascii=False)
        elif "嫌疑人" in self.name or "Suspect" in self.name:
            return json.dumps([
                {"name": "嫌疑人A", "motive": 0.8, "opportunity": 0.7, "capability": 0.6}
            ], ensure_ascii=False)
        elif "证据" in self.name or "Evidence" in self.name:
            return json.dumps([
                {"from": "证据1", "to": "证据2", "relation": "关联", "strength": 0.8}
            ], ensure_ascii=False)
        elif "推理" in self.name or "Reasoning" in self.name:
            return "基于现有证据，初步推断嫌疑人A有重大作案嫌疑。"
        else:
            return f"[{self.name}] 模拟响应"
    
    def format_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化输出（统一输出格式）
        
        Args:
            data: 原始数据
            
        Returns:
            格式化后的数据
        """
        return {
            "agent": self.name,
            "status": "success",
            "data": data,
            "metadata": {
                "timestamp": self._get_timestamp(),
                "version": "1.0"
            }
        }
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def log_processing(self, input_data: Dict[str, Any]):
        """记录处理日志"""
        self.logger.info(f"开始处理: {list(input_data.keys())}")
    
    def validate_input(self, input_data: Dict[str, Any], required_fields: List[str]) -> bool:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据
            required_fields: 必需字段列表
            
        Returns:
            是否验证通过
        """
        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"缺少必需字段: {field}")
                return False
        return True
    
    def extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        从LLM响应中提取JSON（增强鲁棒性）
        多策略解析: 直接解析 → markdown代码块 → 括号匹配 → 递归修复
        
        Args:
            response: LLM响应文本
            
        Returns:
            解析后的JSON对象
        """
        import json
        import re

        if not response or not isinstance(response, str):
            return None

        # Strategy 1: 直接解析
        try:
            return json.loads(response.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: 提取 ```json ... ``` 代码块
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)```', response, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 3: 找最外层 {...} 用括号平衡匹配（非贪婪，取最大的合法JSON）
        # 逐个尝试所有 '{' 开头的候选
        candidates = []
        for i, ch in enumerate(response):
            if ch == '{':
                # 从此位置尝试括号平衡
                depth = 0
                in_string = False
                escape_next = False
                for j in range(i, len(response)):
                    c = response[j]
                    if escape_next:
                        escape_next = False
                        continue
                    if c == '\\' and in_string:
                        escape_next = True
                        continue
                    if c == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            candidates.append(response[i:j+1])
                            break

        for candidate in reversed(candidates):  # 优先尝试最长的
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                # Strategy 4: 修复常见问题
                fixed = self._try_fix_json(candidate)
                if fixed is not None:
                    return fixed

        # Strategy 5: 同样处理 [...] 数组
        arr_match = re.search(r'\[.*\]', response, re.DOTALL)
        if arr_match:
            try:
                return json.loads(arr_match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        self.logger.warning(f"无法从响应中提取JSON (长度={len(response)})")
        return None

    def _try_fix_json(self, text: str) -> Optional[Dict[str, Any]]:
        """尝试修复常见JSON格式问题"""
        import json
        import re

        if not text:
            return None

        # Fix 1: 移除尾部逗号 (trailing comma)
        fixed = re.sub(r',\s*([}\]])', r'\1', text)
        try:
            return json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fix 2: 修复单引号为双引号
        fixed = fixed.replace("'", '"')
        try:
            return json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fix 3: 移除注释 // 和 /* */
        fixed = re.sub(r'//.*?\n', '\n', text)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        try:
            return json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fix 4: 截断修复 — 找到最后一个完整的 key:value 后补 }
        last_colon = fixed.rfind(':')
        if last_colon > 0:
            # 找到这个值之后的最后一个合法结束位置
            for end_char in ['"', ']', '}', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                pos = fixed.rfind(end_char, 0, last_colon + 100)
            # 简单粗暴：截到倒数第一个逗号或冒号后补 }
            truncated = fixed.rstrip()
            # 计算缺少的 }
            open_braces = truncated.count('{') - truncated.count('}')
            if open_braces > 0:
                truncated += '}' * open_braces
                try:
                    return json.loads(truncated)
                except (json.JSONDecodeError, ValueError):
                    pass

        return None
