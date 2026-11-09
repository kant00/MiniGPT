"""
统一解析、加载、记录和校验 Hugging Face tokenizer。

解析顺序：
  1. 调用方显式传入的本地目录或 Hugging Face ID。
  2. MINIGPT_TOKENIZER_PATH 环境变量。
  3. 项目附近常见的 bert-base-chinese 本地目录。
  4. 在线的 bert-base-chinese。

本文件是内部工具模块，不提供命令行入口。Python 使用示例：

  from src.tokenizer_utils import load_tokenizer
  tokenizer, metadata = load_tokenizer("bert-base-chinese")

metadata 包含词表大小、SHA-256 指纹和特殊 token。数据、checkpoint 与生成
阶段必须使用相同指纹；切换 tokenizer 后需要重新预处理并从头训练。
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER_ID = "bert-base-chinese"
TOKENIZER_ENV_VAR = "MINIGPT_TOKENIZER_PATH"


# 用途：寻找可用本地分词器来源，找不到时返回显式 Hub ID 或默认 ID。
# 调用时机：load_tokenizer 实际加载之前调用。
# 输入输出与注意：按显式路径、环境变量、常见目录检查存在性；所有本地候选先于 Hub 回退，故本地目录可能优先于显式在线 ID，需核对实际 source。
def resolve_tokenizer_source(source: Optional[str] = None) -> str:
    """按显式参数、环境变量、常用本地目录、Hub ID 的顺序解析 tokenizer。"""
    candidates = []
    if source:
        candidates.append(Path(source).expanduser())

    env_source = os.environ.get(TOKENIZER_ENV_VAR)
    if env_source:
        candidates.append(Path(env_source).expanduser())

    candidates.extend(
        [
            BASE_DIR.parent.parent / "pretrain_models" / DEFAULT_TOKENIZER_ID,
            BASE_DIR.parent / "pretrain_models" / DEFAULT_TOKENIZER_ID,
            BASE_DIR / "pretrain_models" / DEFAULT_TOKENIZER_ID,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    # 显式传入了一个不存在的路径时直接报错，避免悄悄换用另一套词表。
    if source and (
        Path(source).is_absolute() or source.startswith((".", "~"))
    ):
        raise FileNotFoundError(f"Tokenizer 路径不存在：{Path(source).expanduser()}")
    return source or DEFAULT_TOKENIZER_ID


# 用途：为已加载 tokenizer 生成可保存的身份信息。
# 调用时机：load_tokenizer 返回分词器时调用，预处理与 checkpoint 用其记录词表身份。
# 输入输出与注意：返回来源、类名、词表大小、按 ID 排序映射的 SHA-256 和特殊符号；指纹不覆盖完整 normalizer/merges 行为。
def tokenizer_metadata(
    tokenizer: PreTrainedTokenizerBase, source: str
) -> Dict[str, Any]:
    """生成可写入数据文件/checkpoint 的 tokenizer 身份信息。"""
    vocab = tokenizer.get_vocab()
    canonical_vocab = json.dumps(
        # 匿名排序键：sorted 为每个 (token,ID) 调用，取 ID 排序，让指纹不依赖词典插入顺序。
        sorted(vocab.items(), key=lambda item: item[1]),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "source": source,
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "vocab_sha256": hashlib.sha256(canonical_vocab).hexdigest(),
        "special_tokens_map": json.loads(
            json.dumps(tokenizer.special_tokens_map, ensure_ascii=False, default=str)
        ),
    }


# 用途：解析来源并加载 Hugging Face tokenizer，再返回身份元数据。
# 调用时机：预处理编码文章、生成编码提示词、分词器分析比较候选时调用。
# 输入输出与注意：返回 (tokenizer,metadata)，use_fast=True；可能读取缓存或联网下载，本函数不训练分词器、不加载 MiniGPT 权重。
def load_tokenizer(
    source: Optional[str] = None,
) -> Tuple[PreTrainedTokenizerBase, Dict[str, Any]]:
    resolved_source = resolve_tokenizer_source(source)
    tokenizer = AutoTokenizer.from_pretrained(resolved_source, use_fast=True)
    return tokenizer, tokenizer_metadata(tokenizer, resolved_source)


# 用途：拒绝模型记录与实际加载词表的身份不一致。
# 调用时机：生成及生成基准加载 tokenizer 后调用，其他调用方也可用于接口校验。
# 输入输出与注意：比较 vocab_size 和 vocab_sha256；缺 expected 或不匹配时抛 ValueError，context 加入错误说明，成功返回 None。
def validate_tokenizer_metadata(
    expected: Optional[Dict[str, Any]], actual: Dict[str, Any], context: str
) -> None:
    """拒绝使用不同词表处理数据或加载模型。旧文件没有元数据时给出明确错误。"""
    if not expected:
        raise ValueError(
            f"{context} 缺少 tokenizer 元数据。请重新运行 prepare_dataset.py "
            "或使用兼容的新 checkpoint。"
        )
    for key in ("vocab_size", "vocab_sha256"):
        if expected.get(key) != actual.get(key):
            raise ValueError(
                f"{context} 的 tokenizer 不匹配：{key} "
                f"expected={expected.get(key)!r}, actual={actual.get(key)!r}"
            )
