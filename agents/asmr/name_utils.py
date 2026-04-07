"""
姓名归一化工具
处理 "王福来（管家）" → "王福来"，"陈明辉/侄子" → "陈明辉" 等变体
处理 "罗伊洛特医生" → "罗伊洛特"（剥离头衔）
处理 "格里姆斯比·罗伊洛特" → 保持完整（中间点不是多人分隔符）
"""
import re
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# 头衔（Title / Honorific）剥离
# ============================================================
# 中文常见头衔，按长度降序排列以优先匹配长串
_CHINESE_TITLES = [
    # 双字头衔（需紧跟在姓名后面，不能误拆普通词）
    "医生", "博士", "先生", "女士", "夫人", "太太",
    "教授", "律师", "警官", "警长", "探长", "局长",
    "处长", "科长", "队长", "经理", "董事长", "总裁",
    "总督", "总监", "院长", "校长", "班长", "组长",
    "主任", "法官", "神父", "牧师", "修女", "护士长",
    # 单字头衔
    "爷", "哥", "姐", "叔", "婶", "伯", "舅", "姨",
]

# 编译正则: 头衔在名字末尾
# 例如 "罗伊洛特医生" → "罗伊洛特"
_TITLE_SUFFIX_RE = re.compile(
    r'(' + '|'.join(re.escape(t) for t in _CHINESE_TITLES) + r')$'
)

# 英文常见头衔（不区分大小写）
_ENGLISH_TITLES_PATTERN = re.compile(
    r'\b(Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Miss|Prof\.?|Professor|Sir|Lord|Lady|'
    r'Rev\.?|Reverend|Capt\.?|Captain|Lt\.?|Lieutenant|Sgt\.?|Sergeant|'
    r'Inspector|Detective|Attorney|Esq\.?)\b',
    re.IGNORECASE
)


# ============================================================
# 中文括号和分隔符模式
# ============================================================
_STRIP_PATTERNS = [
    re.compile(r'[（(][^）)]*[）)]'),   # 去掉（管家）(侄子)等
    re.compile(r'[/／][^/／]*$'),        # 去掉 /侄子 等后缀
    # 注意: 这里不再剥离 · — - 等间隔符，因为它们可能是全名的一部分
    # 例如 "格里姆斯比·罗伊洛特" 是一个人名，不应该被拆开
]

# ============================================================
# 无效结论 — 这些不算具体嫌疑人
# ============================================================
_INVALID_CULPRITS = {
    "未知", "无法确定", "无法判断", "不确定", "无结论", "不明",
    "unknown", "uncertain", "n/a", "none", "",
}

# ============================================================
# 多人分隔符（用于 split_multiple_names）
# 真正用于列举多个人名的分隔符
# ============================================================
# 中文顿号、逗号、分号、"和"、"与"、"及"、"同"、"以及"
_MULTI_NAME_SPLIT_RE = re.compile(
    r'[、，,；;]'          # 顿号、中英文逗号、分号
    r'|和(?=者\b)'        # "和" 只在 "X和Y者" 等极少数上下文才拆
    r'|(?:、|以及)\s*'     # 以及
)

# 更宽松的多人拆分（只用于明确的多候选列表）
_EXPLICIT_MULTI_SPLIT_RE = re.compile(
    r'[、，,；;]'          # 标点分隔
)


def _strip_title_suffix(name: str) -> str:
    """
    剥离姓名末尾的中文头衔。
    "罗伊洛特医生" → "罗伊洛特"
    "约翰博士" → "约翰"
    "史密斯教授" → "史密斯"
    但不会误拆 "王福来" (最后一个字不是已知头衔)
    """
    # 反复剥离，处理 "某某医生先生" 这种罕见叠加
    prev = None
    while prev != name:
        prev = name
        name = _TITLE_SUFFIX_RE.sub('', name).strip()
    return name


def _strip_english_title(name: str) -> str:
    """
    剥离英文头衔（前缀形式）。
    "Dr. Roylott" → "Roylott"
    "Mr. Smith" → "Smith"
    """
    return _ENGLISH_TITLES_PATTERN.sub('', name).strip()


def normalize_name(name: str) -> str:
    """
    清洗姓名：去括号、去后缀、去空格、去头衔
    
    处理逻辑（按顺序）:
    1. 去空格
    2. 去括号标注 (如 "（管家）")
    3. 去斜杠后缀 (如 "/侄子")
    4. 去中文头衔后缀 (如 "医生")
    5. 去英文头衔前缀 (如 "Dr.")
    6. 去残余空格和标点
    """
    if not name or not isinstance(name, str):
        return ""
    name = name.strip()
    
    # 去括号标注
    for pat in _STRIP_PATTERNS:
        name = pat.sub("", name)
    
    # 去中文头衔后缀
    name = _strip_title_suffix(name)
    
    # 去英文头衔
    name = _strip_english_title(name)
    
    return name.strip()


def is_valid_suspect(name: str) -> bool:
    """判断是否是有效的嫌疑人名（非"无法确定"等）"""
    if not name or not isinstance(name, str):
        return False
    return name.strip().lower() not in _INVALID_CULPRITS


def split_multiple_names(text: str) -> List[str]:
    """
    从一段文本中拆分出多个人名。
    
    关键规则:
    - 顿号、逗号、分号 → 多人分隔
    - "和"、"与"、"及"、"以及" → 多人分隔（仅在独立词组间）
    - 中间点(·)、连字符(-、—)、句点(.) → **不拆分**，这是全名的一部分
      例如 "格里姆斯比·罗伊洛特" 是一个人名
      例如 "让-保罗·萨特" 是一个人名
    
    Args:
        text: 可能包含多个人名的文本
        
    Returns:
        拆分后的人名列表
    """
    if not text or not isinstance(text, str):
        return []
    
    text = text.strip()
    
    # 如果包含明确的标点分隔符，按标点拆
    if _EXPLICIT_MULTI_SPLIT_RE.search(text):
        parts = [p.strip() for p in _EXPLICIT_MULTI_SPLIT_RE.split(text) if p.strip()]
        return [normalize_name(p) for p in parts]
    
    # 如果包含 "和"、"与"、"以及" 且两边都是较长的词（>=2字符），则拆分
    for conj in ['以及', '和', '与', '及']:
        # 只在 conjunction 两侧都有内容且各 >= 2 字符时拆分
        # 避免把 "赫子和" 拆成 "赫子" + "和"
        idx = text.find(conj)
        if idx > 0 and idx < len(text) - len(conj):
            left = text[:idx].strip()
            right = text[idx + len(conj):].strip()
            if len(left) >= 2 and len(right) >= 2:
                return [normalize_name(left), normalize_name(right)]
    
    # 否则不拆分，当作一个人名
    return [normalize_name(text)]


def merge_name_variants(scores: Dict[str, float]) -> Dict[str, float]:
    """
    合并同一人的不同写法变体
    例: {"王福来": 0.9, "王福来（管家）": 0.8} → {"王福来": 0.9}
    例: {"罗伊洛特": 0.8, "罗伊洛特医生": 0.7} → {"罗伊洛特": 0.8}
    取每个归一化名字下的最高分
    """
    merged: Dict[str, float] = {}
    for name, score in scores.items():
        clean = normalize_name(name)
        if not clean:
            continue
        if clean not in merged or score > merged[clean]:
            merged[clean] = score
    return merged


def merge_name_details(details: Dict[str, list]) -> Dict[str, list]:
    """
    合并投票详情中的同名变体
    """
    merged: Dict[str, list] = {}
    for name, detail_list in details.items():
        clean = normalize_name(name)
        if not clean:
            continue
        if clean not in merged:
            merged[clean] = []
        merged[clean].extend(detail_list)
    return merged


def are_same_person(name_a: str, name_b: str) -> bool:
    """
    判断两个名字是否指向同一个人。
    通过归一化后比较，处理头衔差异、括号标注等。
    
    "罗伊洛特医生" vs "罗伊洛特" → True
    "王福来（管家）" vs "王福来" → True
    "格里姆斯比·罗伊洛特" vs "罗伊洛特" → 比较归一化结果 (不同，需要别名映射)
    """
    a = normalize_name(name_a)
    b = normalize_name(name_b)
    if not a or not b:
        return False
    return a == b


def build_name_alias_map(names: List[str]) -> Dict[str, str]:
    """
    构建姓名别名映射表。
    将同一人的不同写法映射到同一个规范名。
    
    规则:
    1. 归一化后相同 → 映射到最长那个
    2. 短名是长名的后缀/子串 → 映射到长名
       (例如 "罗伊洛特" 是 "格里姆斯比·罗伊洛特" 的后缀部分)
    
    Args:
        names: 所有出现的人名列表
        
    Returns:
        别名映射: {"罗伊洛特": "格里姆斯比·罗伊洛特", "罗伊洛特医生": "格里姆斯比·罗伊洛特", ...}
    """
    # 先归一化所有名字
    normalized_map = {}  # norm_name -> list of original names
    for name in names:
        norm = normalize_name(name)
        if not norm or not is_valid_suspect(norm):
            continue
        if norm not in normalized_map:
            normalized_map[norm] = []
        normalized_map[norm].append(name)
    
    alias_map = {}
    
    # 同一归一化结果内，取最长的作为规范名
    for norm, originals in normalized_map.items():
        canonical = max(originals, key=len)
        for orig in originals:
            alias_map[orig] = canonical
            alias_map[canonical] = canonical  # 自映射
    
    # 跨组检查：短名是长名的后缀（去掉中间点后）
    all_norms = list(normalized_map.keys())
    for i, norm_a in enumerate(all_norms):
        for j, norm_b in enumerate(all_norms):
            if i == j:
                continue
            # 如果 norm_a 是 norm_b 去掉"名"部分后的姓（或反之）
            # 例如 "罗伊洛特" 是 "格里姆斯比·罗伊洛特" 的后缀
            # 处理含间隔点的情况
            a_parts = re.split(r'[·•\-—.]', norm_a)
            b_parts = re.split(r'[·•\-—.]', norm_b)
            
            if len(a_parts) >= 2 and len(b_parts) >= 2:
                # 两者都是全名，不做合并（除非完全一致）
                continue
            
            if len(a_parts) == 1 and len(b_parts) >= 2:
                # a 是短名，检查是否匹配 b 的某一部分
                for part in b_parts:
                    if part == norm_a and len(norm_a) >= 2:
                        # a 是 b 的一部分，合并到 b 的规范名
                        b_canonical = alias_map.get(norm_b, norm_b)
                        a_canonical = alias_map.get(norm_a, norm_a)
                        # 保留更长的
                        final = b_canonical if len(b_canonical) >= len(a_canonical) else a_canonical
                        alias_map[norm_a] = final
                        # 也更新原始名字的映射
                        for orig in normalized_map.get(norm_a, []):
                            alias_map[orig] = final
                        break
            
            if len(b_parts) == 1 and len(a_parts) >= 2:
                # b 是短名，检查是否匹配 a 的某一部分
                for part in a_parts:
                    if part == norm_b and len(norm_b) >= 2:
                        a_canonical = alias_map.get(norm_a, norm_a)
                        b_canonical = alias_map.get(norm_b, norm_b)
                        final = a_canonical if len(a_canonical) >= len(b_canonical) else b_canonical
                        alias_map[norm_b] = final
                        for orig in normalized_map.get(norm_b, []):
                            alias_map[orig] = final
                        break
    
    return alias_map
