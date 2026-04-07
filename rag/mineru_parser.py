"""
MinerU文档解析适配器
使用MinerU CLI (v3.0.7) 替代docling进行文档解析
支持PDF/PNG/JPG → Markdown，为RAG系统提供结构化chunk

依赖: mineru>=3.0.7 (pip安装，CLI方式调用)
后端: pipeline (通用模式，不依赖GPU/VLM)
"""

import os
import re
import json
import hashlib
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

from loguru import logger

# ============================================================
#  常量
# ============================================================


SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

MINERU_CLI = "mineru"

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

CACHE_FILE_NAME = ".mineru_parse_cache.json"


# ============================================================
#  数据结构
# ============================================================


@dataclass
class ParsedDocument:
    """解析后的文档结构"""

    id: str
    content: str  # 完整Markdown内容
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    chunks: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
#  MinerU解析器
# ============================================================


class MinerUParser:
    """
    MinerU文档解析适配器

    - 封装mineru CLI调用
    - 支持pipeline后端（通用，无需GPU）
    - 批量处理目录下的文档
    - 自动切分chunk供RAG系统索引
    - 缓存机制避免重复解析
    - 与RAGAnything.index_documents()无缝对接

    用法::

        parser = MinerUParser(output_dir="./data/parsed")
        docs = parser.parse_file("case_report.pdf")
        chunks = docs.chunks

        # 批量
        all_docs = parser.parse_directory("./cases/")
        # 直接喂给RAG
        rag.index_documents([doc for d in all_docs for doc in d.chunks])
    """

    def __init__(
        self,
        output_dir: str = "./data/mineru_output",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        lang: str = "ch",
        backend: str = "pipeline",
        method: str = "auto",
        enable_formula: bool = True,
        enable_table: bool = True,
        use_cache: bool = True,
        clean_output: bool = True,
    ):
        """
        初始化MinerU解析器

        Args:
            output_dir:  MinerU输出根目录
            chunk_size:  每个chunk的最大字符数
            chunk_overlap: chunk之间的重叠字符数
            lang:        文档语言 (ch/en/...)
            backend:     解析后端 (pipeline通用, vlm-*需GPU)
            method:      解析方法 (auto/txt/ocr)
            enable_formula: 是否启用公式解析
            enable_table:   是否启用表格解析
            use_cache:   是否启用缓存（已解析文件不重复解析）
            clean_output: 解析完成后是否清理MinerU中间文件
        """
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.lang = lang
        self.backend = backend
        self.method = method
        self.enable_formula = enable_formula
        self.enable_table = enable_table
        self.use_cache = use_cache
        self.clean_output = clean_output

        self.logger = logger.bind(module="mineru_parser")

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 缓存: {file_path_abs: {hash, chunks_count, parsed_at}}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_path = self.output_dir / CACHE_FILE_NAME

        if self.use_cache:
            self._load_cache()

        self.logger.info(
            f"MinerU解析器初始化 | backend={backend} lang={lang} "
            f"chunk_size={chunk_size} overlap={chunk_overlap} cache={use_cache}"
        )

    # ----------------------------------------------------------
    #  公共接口
    # ----------------------------------------------------------

    def parse_file(self, file_path: str) -> Optional[ParsedDocument]:
        """
        解析单个文件

        Args:
            file_path: 文件路径 (PDF/PNG/JPG)

        Returns:
            ParsedDocument 或 None（解析失败时）
        """
        path = Path(file_path).resolve()
        if not path.exists():
            self.logger.error(f"文件不存在: {path}")
            return None

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            self.logger.warning(f"不支持的文件格式: {path.suffix} ({path.name})")
            return None

        # 缓存检查
        file_hash = self._file_hash(path)
        cache_key = str(path)
        if self.use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached.get("hash") == file_hash:
                self.logger.info(f"命中缓存，跳过解析: {path.name}")
                # 重新从磁盘读取已解析的Markdown
                md_content = self._read_cached_markdown(path)
                if md_content is not None:
                    return self._build_document(
                        path, md_content, from_cache=True
                    )

        # 调用MinerU CLI
        self.logger.info(f"开始解析: {path.name}")
        success = self._run_mineru(path)
        if not success:
            return None

        # 读取输出Markdown
        md_content = self._read_output_markdown(path)
        if md_content is None:
            return None

        # 构建文档
        doc = self._build_document(path, md_content)

        # 更新缓存
        if self.use_cache:
            self._cache[cache_key] = {
                "hash": file_hash,
                "chunks_count": len(doc.chunks),
                "parsed_at": self._now_iso(),
            }
            self._save_cache()

        self.logger.info(
            f"解析完成: {path.name} | 全文 {len(doc.content)} 字符 | "
            f"{len(doc.chunks)} chunks"
        )

        # 清理中间文件
        if self.clean_output:
            self._cleanup_intermediate(path)

        return doc

    def parse_directory(
        self,
        dir_path: str,
        recursive: bool = True,
    ) -> List[ParsedDocument]:
        """
        批量解析目录下的所有文档

        Args:
            dir_path:    目录路径
            recursive:   是否递归子目录

        Returns:
            解析后的文档列表（跳过失败的文件）
        """
        dir_path = Path(dir_path).resolve()
        if not dir_path.is_dir():
            self.logger.error(f"目录不存在: {dir_path}")
            return []

        # 收集文件
        if recursive:
            files = [
                f
                for f in dir_path.rglob("*")
                if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()
            ]
        else:
            files = [
                f
                for f in dir_path.iterdir()
                if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()
            ]

        if not files:
            self.logger.warning(f"目录下无支持的文档: {dir_path}")
            return []

        self.logger.info(f"发现 {len(files)} 个文档待解析: {dir_path}")

        results = []
        for i, f in enumerate(files, 1):
            self.logger.info(f"[{i}/{len(files)}] 解析: {f.name}")
            doc = self.parse_file(str(f))
            if doc is not None:
                results.append(doc)

        self.logger.info(
            f"批量解析完成: {len(results)}/{len(files)} 成功"
        )
        return results

    def parse_to_chunks(
        self,
        file_path: str,
    ) -> List[Dict[str, Any]]:
        """
        解析文件并直接返回chunk列表（可直接喂给RAGAnything）

        Args:
            file_path: 文件路径

        Returns:
            chunk列表，每个chunk格式:
            {
                "id": str,
                "content": str,
                "metadata": dict,
                "source_file": str,
            }
        """
        doc = self.parse_file(file_path)
        if doc is None:
            return []
        return doc.chunks

    def parse_directory_to_chunks(
        self,
        dir_path: str,
        recursive: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量解析目录并返回所有chunk（扁平列表）

        Args:
            dir_path:  目录路径
            recursive: 是否递归

        Returns:
            所有文档的chunk扁平列表
        """
        docs = self.parse_directory(dir_path, recursive=recursive)
        all_chunks = []
        for doc in docs:
            all_chunks.extend(doc.chunks)
        self.logger.info(f"共生成 {len(all_chunks)} 个chunks")
        return all_chunks

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "cached_files": len(self._cache),
            "cache_path": str(self._cache_path),
            "cache_enabled": self.use_cache,
        }

    def clear_cache(self):
        """清除解析缓存"""
        self._cache = {}
        if self._cache_path.exists():
            self._cache_path.unlink()
        self.logger.info("缓存已清除")

    # ----------------------------------------------------------
    #  MinerU CLI 调用
    # ----------------------------------------------------------

    def _run_mineru(self, file_path: Path) -> bool:
        """
        调用mineru CLI解析文件

        Args:
            file_path: 文件绝对路径

        Returns:
            是否成功
        """
        cmd = [
            MINERU_CLI,
            "-p", str(file_path),
            "-o", str(self.output_dir),
            "-b", self.backend,
            "-l", self.lang,
            "-m", self.method,
            "-f", str(self.enable_formula).lower(),
            "-t", str(self.enable_table).lower(),
        ]

        self.logger.debug(f"执行命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10分钟超时
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                self.logger.error(
                    f"MinerU解析失败 (exit={result.returncode}): {stderr[:500]}"
                )
                return False

            self.logger.debug(f"MinerU stdout: {result.stdout[:200]}")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error(f"MinerU解析超时(600s): {file_path.name}")
            return False
        except FileNotFoundError:
            self.logger.error(
                f"未找到mineru命令，请确认已安装: pip install mineru"
            )
            return False
        except Exception as e:
            self.logger.error(f"MinerU调用异常: {e}")
            return False

    # ----------------------------------------------------------
    #  输出读取
    # ----------------------------------------------------------

    def _get_output_subdir(self, file_path: Path) -> Path:
        """
        获取MinerU为该文件生成的输出子目录

        MinerU输出结构: <output_dir>/<filename_without_ext>/
        """
        stem = file_path.stem
        return self.output_dir / stem

    def _read_output_markdown(self, file_path: Path) -> Optional[str]:
        """
        从MinerU输出目录中读取Markdown内容

        输出目录结构: <output_dir>/<filename>/ 下有 .md 文件
        """
        output_subdir = self._get_output_subdir(file_path)

        if not output_subdir.exists():
            self.logger.error(f"MinerU输出目录不存在: {output_subdir}")
            return None

        # 查找 .md 文件
        md_files = list(output_subdir.rglob("*.md"))
        if not md_files:
            self.logger.error(f"输出目录中无.md文件: {output_subdir}")
            return None

        # 合并所有md文件内容
        contents = []
        for md_file in sorted(md_files):
            try:
                text = md_file.read_text(encoding="utf-8")
                if text.strip():
                    contents.append(text.strip())
            except Exception as e:
                self.logger.warning(f"读取Markdown失败: {md_file} - {e}")

        if not contents:
            self.logger.error(f"所有Markdown文件为空: {output_subdir}")
            return None

        return "\n\n---\n\n".join(contents)

    def _read_cached_markdown(self, file_path: Path) -> Optional[str]:
        """读取已缓存的Markdown（用于缓存命中时）"""
        return self._read_output_markdown(file_path)

    # ----------------------------------------------------------
    #  文档构建 & Chunk切分
    # ----------------------------------------------------------

    def _build_document(
        self,
        file_path: Path,
        md_content: str,
        from_cache: bool = False,
    ) -> ParsedDocument:
        """构建ParsedDocument，包含自动chunk切分"""
        doc_id = self._generate_doc_id(file_path)
        file_hash = self._file_hash(file_path)

        metadata = {
            "filename": file_path.name,
            "file_ext": file_path.suffix.lower(),
            "file_size": file_path.stat().st_size,
            "file_hash": file_hash[:16],
            "parser": "mineru",
            "parser_backend": self.backend,
            "lang": self.lang,
            "from_cache": from_cache,
        }

        # 切分chunks
        chunks = self._split_chunks(
            md_content,
            doc_id=doc_id,
            source_file=str(file_path),
            metadata=metadata,
        )

        return ParsedDocument(
            id=doc_id,
            content=md_content,
            metadata=metadata,
            source_file=str(file_path),
            chunks=chunks,
        )

    def _split_chunks(
        self,
        text: str,
        doc_id: str,
        source_file: str,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        将Markdown文本切分为chunks

        策略:
        1. 按Markdown标题(#/##/###)切分为段落
        2. 段落内按chunk_size + overlap滑动窗口
        3. 每个chunk带有上下文元数据
        """
        chunks = []
        chunk_index = 0

        # 按标题或分隔符拆分段落
        sections = self._split_by_heading(text)

        for section_title, section_text in sections:
            if not section_text.strip():
                continue

            # 如果段落小于chunk_size，作为一个chunk
            if len(section_text) <= self.chunk_size:
                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                chunks.append(
                    self._make_chunk(
                        chunk_id=chunk_id,
                        chunk_index=chunk_index,
                        content=section_text.strip(),
                        section_title=section_title,
                        doc_id=doc_id,
                        source_file=source_file,
                        metadata=metadata,
                    )
                )
                chunk_index += 1
            else:
                # 滑动窗口切分
                window_chunks = self._sliding_window_split(
                    section_text,
                    chunk_size=self.chunk_size,
                    overlap=self.chunk_overlap,
                )

                for wc in window_chunks:
                    chunk_id = f"{doc_id}_chunk_{chunk_index}"
                    chunks.append(
                        self._make_chunk(
                            chunk_id=chunk_id,
                            chunk_index=chunk_index,
                            content=wc.strip(),
                            section_title=section_title,
                            doc_id=doc_id,
                            source_file=source_file,
                            metadata=metadata,
                        )
                    )
                    chunk_index += 1

        return chunks

    @staticmethod
    def _split_by_heading(text: str) -> List[Tuple[str, str]]:
        """
        按Markdown标题切分段落

        Returns:
            [(section_title, section_text), ...]
        """
        # 匹配Markdown标题行
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        sections = []
        last_end = 0
        last_title = ""

        for match in heading_pattern.finditer(text):
            start = match.start()

            # 标题之前的内容
            if start > last_end:
                pre_text = text[last_end:start].strip()
                if pre_text:
                    sections.append((last_title, pre_text))

            last_title = match.group(2).strip()
            last_end = match.end()

        # 最后一段
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sections.append((last_title, remaining))

        # 如果没有标题，整段返回
        if not sections and text.strip():
            sections.append(("", text.strip()))

        return sections

    @staticmethod
    def _sliding_window_split(
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> List[str]:
        """
        滑动窗口切分长文本

        优先在句号/换行处切分
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            # 在chunk末尾附近寻找自然断句点
            best_break = end
            search_range = text[end - min(overlap, 50): end + 50]

            for sep in ["\n\n", "。", "！", "？", ".", "！", "\n"]:
                pos = search_range.rfind(sep)
                if pos != -1:
                    best_break = end - min(overlap, 50) + pos + len(sep)
                    break

            chunks.append(text[start:best_break])
            start = best_break - overlap
            if start <= 0 or start >= len(text):
                if start < len(text):
                    start = best_break
                else:
                    break

        return chunks

    @staticmethod
    def _make_chunk(
        chunk_id: str,
        chunk_index: int,
        content: str,
        section_title: str,
        doc_id: str,
        source_file: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构造单个chunk字典"""
        chunk_meta = {
            **metadata,
            "chunk_index": chunk_index,
            "section_title": section_title,
            "content_length": len(content),
        }
        return {
            "id": chunk_id,
            "content": content,
            "metadata": chunk_meta,
            "source_file": source_file,
        }

    # ----------------------------------------------------------
    #  缓存管理
    # ----------------------------------------------------------

    def _load_cache(self):
        """从磁盘加载缓存"""
        if self._cache_path.exists():
            try:
                data = json.loads(self._cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._cache = data
                    self.logger.info(
                        f"加载解析缓存: {len(self._cache)} 条记录"
                    )
            except Exception as e:
                self.logger.warning(f"缓存文件损坏，跳过: {e}")
                self._cache = {}

    def _save_cache(self):
        """将缓存写入磁盘"""
        try:
            self._cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            self.logger.warning(f"缓存写入失败: {e}")

    # ----------------------------------------------------------
    #  工具方法
    # ----------------------------------------------------------

    @staticmethod
    def _file_hash(path: Path) -> str:
        """计算文件SHA256哈希"""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _generate_doc_id(path: Path) -> str:
        """生成文档唯一ID"""
        raw = f"{path.stem}_{path.stat().st_size}_{path.stat().st_mtime}"
        return f"doc_{hashlib.md5(raw.encode()).hexdigest()[:12]}"

    @staticmethod
    def _now_iso() -> str:
        """当前时间ISO格式"""
        from datetime import datetime

        return datetime.now().isoformat()

    def _cleanup_intermediate(self, file_path: Path):
        """
        清理MinerU中间文件（保留.md，删除其余）

        MinerU输出目录中可能包含: images/, layout/ 等中间文件
        """
        output_subdir = self._get_output_subdir(file_path)
        if not output_subdir.exists():
            return

        # 保留.md文件和目录本身，删除其他
        for item in output_subdir.iterdir():
            if item.is_dir():
                try:
                    shutil.rmtree(item)
                except Exception as e:
                    self.logger.debug(f"清理目录失败: {item} - {e}")
            elif item.suffix not in (".md",):
                try:
                    item.unlink()
                except Exception as e:
                    self.logger.debug(f"清理文件失败: {item} - {e}")


# ============================================================
#  便捷函数
# ============================================================


def create_mineru_parser(
    output_dir: str = "./data/mineru_output",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    **kwargs,
) -> MinerUParser:
    """
    创建MinerU解析器实例

    Args:
        output_dir:    输出目录
        chunk_size:    chunk大小
        chunk_overlap: chunk重叠
        **kwargs:      其他MinerUParser参数

    Returns:
        MinerUParser实例
    """
    return MinerUParser(
        output_dir=output_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )


def parse_for_rag(
    file_path: str,
    output_dir: str = "./data/mineru_output",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    一站式解析：解析文件并返回可直接喂给RAGAnything的chunk列表

    用法::

        chunks = parse_for_rag("case.pdf")
        rag.index_documents(chunks, embeddings=embeddings)

    Args:
        file_path:     文件路径
        output_dir:    输出目录
        chunk_size:    chunk大小
        chunk_overlap: chunk重叠

    Returns:
        chunk列表
    """
    parser = MinerUParser(
        output_dir=output_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return parser.parse_to_chunks(file_path)


# ============================================================
#  测试
# ============================================================

if __name__ == "__main__":
    import sys
    import tempfile

    logging_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.configure(handlers=[{"sink": sys.stderr, "format": logging_format}])

    print("\n" + "=" * 70)
    print("🧪 MinerU文档解析适配器 测试")
    print("=" * 70)

    # 1. 创建解析器
    print("\n1️⃣ 创建MinerU解析器...")
    parser = MinerUParser(
        output_dir="./test_mineru_output",
        chunk_size=256,
        chunk_overlap=30,
    )
    print(f"   ✅ 解析器已创建: backend={parser.backend}, lang={parser.lang}")

    # 2. 检查MinerU版本
    print("\n2️⃣ 检查MinerU CLI...")
    try:
        result = subprocess.run(
            [MINERU_CLI, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = result.stdout.strip()
        print(f"   ✅ {version}")
    except Exception as e:
        print(f"   ❌ MinerU未安装或不可用: {e}")

    # 3. 测试chunk切分逻辑
    print("\n3️⃣ 测试chunk切分...")
    test_md = """# 案件报告：密室杀人案

## 案件概述
2024年1月15日晚，在东城区某高档公寓12层发生一起密室杀人案。死者张某（男，35岁）被发现死于书房内，门窗反锁，现场无他人痕迹。

## 嫌疑人分析
### 嫌疑人A：李某（妻子）
李某案发当晚声称在卧室睡觉，但监控显示她在案发时段曾前往走廊。她与死者近期因财产问题多次争吵，动机充分。

### 嫌疑人B：王某（商业伙伴）
王某与死者在生意上存在纠纷，案发前曾威胁过死者。但有不在场证明——当晚在公司加班，多名员工证实。

## 关键证据
1. 书房门锁为电子锁，只有3人持有密码
2. 现场发现一封未寄出的信件，内容涉及保险受益人变更
3. 死者手机最后一条消息发送给李某，内容为"我知道了"
"""

    chunks = parser._split_chunks(
        test_md,
        doc_id="test_doc_001",
        source_file="test_case.md",
        metadata={"parser": "mineru", "test": True},
    )
    print(f"   ✅ 切分出 {len(chunks)} 个chunks")
    for c in chunks[:3]:
        title = c["metadata"].get("section_title", "(无标题)")
        print(f"      [{c['id']}] 标题='{title}' 长度={c['metadata']['content_length']}")

    # 4. 测试标题切分
    print("\n4️⃣ 测试标题切分...")
    sections = parser._split_by_heading(test_md)
    print(f"   ✅ 切分出 {len(sections)} 个段落")
    for title, text in sections:
        print(f"      标题='{title}' 内容长度={len(text)}")

    # 5. 缓存统计
    print("\n5️⃣ 缓存状态...")
    stats = parser.get_cache_stats()
    print(f"   {stats}")

    print("\n✅ 测试完成!")
    print("=" * 70)
