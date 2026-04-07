"""
工具函数
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def load_json(file_path: str) -> Dict:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_timestamp() -> str:
    """获取当前时间戳"""
    return datetime.now().isoformat()


def format_case_data(raw_data: Dict) -> Dict:
    """格式化案例数据"""
    return {
        "case_id": raw_data.get("id", ""),
        "case_title": raw_data.get("title", ""),
        "case_text": raw_data.get("description", ""),
        "case_type": raw_data.get("type", "modern"),
        "created_at": get_timestamp()
    }


def truncate_text(text: str, max_length: int = 100) -> str:
    """截断文本"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def calculate_similarity(text1: str, text2: str) -> float:
    """计算文本相似度（简单版本）"""
    # TODO: 实现更复杂的相似度计算
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1 & words2
    union = words1 | words2
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)
