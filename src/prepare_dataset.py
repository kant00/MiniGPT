"""
将中文 JSONL 语料流式编码为可内存映射的 token 数据集。

主要功能：
  1. 对正文执行质量过滤、精确去重和 SimHash 近似去重。
  2. 按文档稳定划分 train、val、test，避免同一文档跨集合。
  3. 使用指定 tokenizer 编码，并以 uint16/uint32 连续写入磁盘。
  4. 为每个 split 生成包含 tokenizer 指纹和处理统计的元数据。

输入：
  data/wiki_zh.jsonl，每行包含 id、title、text。

输出：
  data/<split>_tokens_seq<长度>.bin
  data/<split>_seq<长度>.meta.json

使用方式（在项目根目录执行）：
  python -m src.prepare_dataset
  python -m src.prepare_dataset --seq_len 256 --max_tokens 2000000
  python -m src.prepare_dataset --tokenizer <本地目录或 Hugging-Face-ID>
  python -m src.prepare_dataset --no_near_dedup --min_cjk_ratio 0.05
  python -m src.prepare_dataset --config configs/default.json

运行前请先执行 python -m src.download_data 获取语料。切换 tokenizer 后必须
重新生成数据，并从头训练模型。未显式传入的参数从配置文件的
prepare_dataset 分区读取。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import logging
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch

try:
    from .config_utils import parse_args_with_config
    from .tokenizer_utils import load_tokenizer
except ImportError:  # 兼容在 src 目录中直接运行脚本
    from config_utils import parse_args_with_config
    from tokenizer_utils import load_tokenizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# 用途：把内存中的 token 列表按固定窗口切成旧式 train/val 张量。
# 调用时机：切块单元测试和小数据演示直接调用；正式 build_dataset 不依赖这条内存路径。
# 输入输出与注意：返回两个二维张量，每行 seq_len+1，步长 seq_len；至少两个窗口，val_ratio 必须在 (0,1)，不提供文档级三路隔离。
def split_token_stream(all_ids, seq_len: int, val_ratio: float):
    """测试和小数据场景使用的内存版切块函数。"""
    if seq_len < 1:
        raise ValueError("seq_len 必须大于 0")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio 必须位于 (0, 1) 区间")
    n_chunks = (len(all_ids) - 1) // seq_len
    if n_chunks < 2:
        raise ValueError(
            f"至少需要 2 个完整切块，当前只有 {n_chunks} 个；请增加数据量或减小 seq_len"
        )
    val_size = max(1, int(n_chunks * val_ratio))
    train_size = n_chunks - val_size
    ids_tensor = torch.tensor(all_ids[: n_chunks * seq_len + 1], dtype=torch.long)
    train_data = ids_tensor[: train_size * seq_len + 1].unfold(
        0, seq_len + 1, seq_len
    )
    val_data = ids_tensor[train_size * seq_len : n_chunks * seq_len + 1].unfold(
        0, seq_len + 1, seq_len
    )
    if len(train_data) != train_size or len(val_data) != val_size:
        raise RuntimeError("数据切块数量校验失败")
    return train_data.contiguous(), val_data.contiguous()


# 用途：规范化 Unicode、换行和首尾空白，并按需拼接标题。
# 调用时机：build_dataset 分别准备过滤用正文和编码用文本时调用。
# 输入输出与注意：输入 title/text 字符串，返回 NFC 文本；include_title 时用换行连接，不做简繁转换或全面内容清洗。
def normalize_document(title: str, text: str, include_title: bool = True) -> str:
    """执行保守的 Unicode/换行规范化，并可把标题加入训练文本。"""
    title = unicodedata.normalize("NFC", title).strip()
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()
    if include_title and title:
        return f"{title}\n{text}"
    return text


# 用途：为规范化正文生成精确内容指纹。
# 调用时机：预处理精确去重时调用，缺文章 ID 时指纹也作为稳定划分身份。
# 输入输出与注意：返回 UTF-8 正文的 SHA-256 十六进制字符串；正文相同但标题不同仍相同，不测语义相似。
def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 用途：依据种子和文档 ID 稳定分配 train/val/test。
# 调用时机：预处理文章通过质量检查与去重后调用，划分测试也直接调用。
# 输入输出与注意：将哈希映射到 [0,1)，先 test 再 val，其余 train；同 ID/seed 结果固定，但实际比例只是概率近似。
def choose_split(
    document_id: str, val_ratio: float, test_ratio: float, split_seed: int
) -> str:
    """按文档稳定哈希划分，数据顺序改变时同一文章仍落在同一集合。"""
    digest = hashlib.sha256(f"{split_seed}:{document_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


# 用途：检查正文长度、汉字比例、异常控制字符与连续重复，返回首个拒绝原因。
# 调用时机：build_dataset 编码前逐文档调用；质量过滤回归测试直接构造文本调用。
# 输入输出与注意：通过返回 None；所有比例的分母按各规则定义，正常换行/回车/制表符不算异常控制字符，只统计首个命中原因。
def quality_rejection_reason(
    text: str,
    min_chars: int,
    min_cjk_ratio: float,
    max_control_ratio: float,
    max_repeat_run_ratio: float,
):
    """返回拒绝原因；返回 None 表示文档通过基础质量检查。"""
    if len(text) < min_chars:
        return "too_short"
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return "empty"
    cjk_count = sum("\u3400" <= char <= "\u9fff" for char in visible)
    if cjk_count / len(visible) < min_cjk_ratio:
        return "low_cjk_ratio"
    # 换行、回车和制表符虽然属于 Unicode Cc，但它们是正常文本排版字符，
    # 不能计入异常控制字符。否则多段落 Wikipedia 正文会被大面积误杀。
    allowed_formatting_controls = {"\n", "\r", "\t"}
    control_count = sum(
        unicodedata.category(char) == "Cc" and char not in allowed_formatting_controls
        for char in text
    )
    if control_count / len(text) > max_control_ratio:
        return "control_chars"
    longest_run = 1
    current_run = 1
    for previous, current in zip(text, text[1:]):
        current_run = current_run + 1 if current == previous else 1
        longest_run = max(longest_run, current_run)
    if longest_run / len(text) > max_repeat_run_ratio:
        return "repeated_character_run"
    return None


# 用途：用抽样字符 5-gram 构建 64 位近似内容指纹。
# 调用时机：近似去重启用时，build_dataset 将规范化正文送入此函数。
# 输入输出与注意：返回整数；先去空白，特征哈希逐位投票，短文本直接哈希；是近似字面特征而非语义向量。
def simhash64(text: str, max_features: int = 256) -> int:
    """使用均匀采样的字符 5-gram 生成轻量 64 位近似指纹。"""
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 5:
        return int.from_bytes(hashlib.blake2b(compact.encode(), digest_size=8).digest(), "big")
    feature_count = len(compact) - 4
    step = max(1, math.ceil(feature_count / max_features))
    weights = [0] * 64
    for index in range(0, feature_count, step):
        feature_hash = int.from_bytes(
            hashlib.blake2b(compact[index : index + 5].encode(), digest_size=8).digest(),
            "big",
        )
        for bit in range(64):
            weights[bit] += 1 if feature_hash & (1 << bit) else -1
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


# 用途：维护四段指纹桶，快速召回可能重复的文章候选。
# 调用时机：一次 build_dataset 建立一个索引，遍历文档时持续查询和添加。
# 输入输出与注意：内存随保留指纹增长；默认距离 3 可利用四段保证候选召回，阈值更大不保证召回全部近邻。
class NearDuplicateIndex:
    """四段 LSH 候选召回，再用汉明距离做保守近似去重。"""

    # 用途：记录允许的汉明距离并初始化空桶字典。
    # 调用时机：NearDuplicateIndex(max_distance) 实例化时自动调用。
    # 输入输出与注意：仅存参数，不验证输入范围；build_dataset 的入口负责范围检查，每次新建索引不继承历史语料状态。
    def __init__(self, max_distance: int = 3):
        self.max_distance = max_distance
        self.buckets = {}

    # 用途：判断指纹是否与已有候选近似重复，未命中才加入索引。
    # 调用时机：预处理开启近似去重时每篇文章调用，也用于索引回归测试。
    # 输入输出与注意：返回 True 表示重复且不添加，False 表示新增；先查四个 16 位桶，再计算异或 bit_count 距离。
    def check_and_add(self, fingerprint: int) -> bool:
        candidates = set()
        keys = []
        for band in range(4):
            value = (fingerprint >> (band * 16)) & 0xFFFF
            key = (band, value)
            keys.append(key)
            candidates.update(self.buckets.get(key, ()))
        if any((fingerprint ^ candidate).bit_count() <= self.max_distance for candidate in candidates):
            return True
        for key in keys:
            self.buckets.setdefault(key, []).append(fingerprint)
        return False


# 用途：按词表大小选择足够容纳 token ID 的二进制无符号整数类型。
# 调用时机：build_dataset 获取 tokenizer 元数据后、打开 token 文件前调用。
# 输入输出与注意：词表至多 65536 用小端 uint16，更大用 uint32，超出表示范围抛错；训练取样后才转 int64。
def token_dtype(vocab_size: int):
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype("<u2")
    if vocab_size <= np.iinfo(np.uint32).max + 1:
        return np.dtype("<u4")
    raise ValueError(f"词表过大，无法使用 uint32 存储：{vocab_size}")


# 用途：阻止质量规则误杀绝大多数语料后仍发布数据。
# 调用时机：预处理结束、替换正式 bin/meta 之前调用，保护阈值测试也调用。
# 输入输出与注意：返回拒绝数/已读文章数；超过 max_ratio 抛 ValueError，重复去除不属于质量拒绝，分母至少按 1 处理。
def validate_quality_rejection_ratio(
    input_articles: int, rejection_counts: dict, max_ratio: float
) -> float:
    """防止配置或规则错误导致绝大多数输入文章被静默过滤。"""
    ratio = sum(rejection_counts.values()) / max(1, input_articles)
    if ratio > max_ratio:
        raise ValueError(
            f"质量过滤拒绝了 {ratio:.1%} 的输入文章，超过上限 "
            f"{max_ratio:.1%}；已停止发布数据，请检查过滤规则"
        )
    return ratio


# 用途：统一生成某个 split 的 bin 和 meta 文件名。
# 调用时机：build_dataset 为各集合规划输出路径时调用。
# 输入输出与注意：输入 split 与 seq_len，返回 (data_path,meta_path)；只计算路径，不创建或打开文件。
def dataset_paths(split: str, seq_len: int):
    data_path = DATA_DIR / f"{split}_tokens_seq{seq_len}.bin"
    meta_path = DATA_DIR / f"{split}_seq{seq_len}.meta.json"
    return data_path, meta_path


# 用途：将数据集描述写入临时 JSON 再替换正式元数据。
# 调用时机：build_dataset 校验 token 文件并发布对应 bin 后调用。
# 输入输出与注意：输入 Path 和字典，输出 UTF-8 JSON；单文件替换不等于三套数据整体事务。
def write_metadata(path: Path, metadata: dict) -> None:
    part_path = Path(str(path) + ".part")
    with open(part_path, "w", encoding="utf-8") as fout:
        json.dump(metadata, fout, ensure_ascii=False, indent=2)
    part_path.replace(path)


# 用途：将 JSONL 文章过滤、去重、按文档划分并流式编码成训练数据。
# 调用时机：预处理 main 或使用临时目录/tokenizer 的流水线测试调用。
# 输入输出与注意：返回各 split 的 meta 路径元组；追加 EOS/SEP、按总 max_tokens 控制预算，写 bin/meta 并检查完整样本数量，会替换同名输出。
def build_dataset(
    seq_len: int = 256,
    val_ratio: float = 0.04,
    test_ratio: float = 0.01,
    max_tokens: int = None,
    tokenizer_source: str = None,
    split_seed: int = 42,
    deduplicate: bool = True,
    near_deduplicate: bool = True,
    near_dedup_distance: int = 3,
    include_title: bool = True,
    min_chars: int = 50,
    min_cjk_ratio: float = 0.1,
    max_control_ratio: float = 0.01,
    max_repeat_run_ratio: float = 0.2,
    max_quality_rejection_ratio: float = 0.8,
    max_document_tokens: int = None,
):
    if seq_len < 1:
        raise ValueError("seq_len 必须大于 0")
    if val_ratio <= 0.0 or test_ratio < 0.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio 必须大于 0，test_ratio 不能为负，且两者之和必须小于 1")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens 必须大于 0")
    if max_document_tokens is not None and max_document_tokens < 2:
        raise ValueError("max_document_tokens 至少为 2，包含文档结束符")
    if not 0 <= near_dedup_distance <= 64:
        raise ValueError("near_dedup_distance 必须位于 [0, 64]")
    if min_chars < 0:
        raise ValueError("min_chars 不能为负数")
    for name, value in (
        ("min_cjk_ratio", min_cjk_ratio),
        ("max_control_ratio", max_control_ratio),
        ("max_repeat_run_ratio", max_repeat_run_ratio),
        ("max_quality_rejection_ratio", max_quality_rejection_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} 必须位于 [0, 1]")

    jsonl_path = DATA_DIR / "wiki_zh.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"未找到数据文件 {jsonl_path}，请先运行 download_data.py")
    if jsonl_path.stat().st_size == 0:
        raise ValueError(f"数据文件为空：{jsonl_path}，请重新运行 download_data.py")

    tokenizer, tokenizer_info = load_tokenizer(tokenizer_source)
    vocab_size = tokenizer_info["vocab_size"]
    boundary_token_id = tokenizer.eos_token_id
    if boundary_token_id is None:
        boundary_token_id = tokenizer.sep_token_id
    if boundary_token_id is None:
        raise ValueError("Tokenizer 必须定义 eos_token_id 或 sep_token_id 作为文章边界")
    dtype = token_dtype(vocab_size)
    embedding_params = vocab_size * 384
    logger.info(
        f"Tokenizer：{tokenizer_info['source']}，vocab_size={vocab_size}，"
        f"fingerprint={tokenizer_info['vocab_sha256'][:12]}"
    )
    logger.info(
        f"Token 存储类型：{dtype.name}；按 d_model=384 估算 embedding="
        f"{embedding_params / 1e6:.1f}M 参数"
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    splits = ("train", "val", "test") if test_ratio > 0 else ("train", "val")
    final_paths = {split: dataset_paths(split, seq_len) for split in splits}
    part_paths = {
        split: Path(str(paths[0]) + ".part") for split, paths in final_paths.items()
    }
    for path in part_paths.values():
        path.unlink(missing_ok=True)

    token_counts = {split: 0 for split in splits}
    document_counts = {split: 0 for split in splits}
    duplicate_count = 0
    near_duplicate_count = 0
    rejection_counts = {}
    total_articles = 0
    input_articles = 0
    seen_hashes = set()
    near_duplicate_index = NearDuplicateIndex(near_dedup_distance)

    handles = {split: open(path, "wb") for split, path in part_paths.items()}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    title = str(obj.get("title") or "")
                    text = str(obj.get("text") or "")
                except (json.JSONDecodeError, AttributeError) as exc:
                    raise ValueError(f"JSONL 第 {line_no} 行无效：{exc}") from exc
                input_articles += 1

                normalized_text = normalize_document("", text, include_title=False)
                document = normalize_document(title, text, include_title)
                rejection_reason = quality_rejection_reason(
                    normalized_text,
                    min_chars,
                    min_cjk_ratio,
                    max_control_ratio,
                    max_repeat_run_ratio,
                )
                if rejection_reason:
                    rejection_counts[rejection_reason] = (
                        rejection_counts.get(rejection_reason, 0) + 1
                    )
                    continue
                # 标题不同但正文相同的镜像文章仍视为重复。
                content_hash = document_hash(normalized_text)
                if deduplicate and content_hash in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(content_hash)
                if near_deduplicate and near_duplicate_index.check_and_add(
                    simhash64(normalized_text)
                ):
                    near_duplicate_count += 1
                    continue

                document_id = str(obj.get("id") or content_hash)
                split = choose_split(document_id, val_ratio, test_ratio, split_seed)
                ids = tokenizer.encode(document, add_special_tokens=False)
                if max_document_tokens is not None:
                    ids = ids[:max_document_tokens - 1]
                ids.append(boundary_token_id)
                if ids and (min(ids) < 0 or max(ids) >= vocab_size):
                    raise ValueError(
                        f"第 {line_no} 行产生了超出词表范围的 token id"
                    )

                if max_tokens is not None:
                    remaining = max_tokens - sum(token_counts.values())
                    if remaining <= 0:
                        break
                    ids = ids[:remaining]
                np.asarray(ids, dtype=dtype).tofile(handles[split])
                token_counts[split] += len(ids)
                document_counts[split] += 1
                total_articles += 1

                if total_articles % 5000 == 0:
                    logger.info(
                        f"已处理 {total_articles:,} 篇，token={sum(token_counts.values()):,}，"
                        f"精确去重={duplicate_count:,}，近似去重={near_duplicate_count:,}"
                    )
                if max_tokens is not None and sum(token_counts.values()) >= max_tokens:
                    break
    except Exception:
        for handle in handles.values():
            handle.close()
        for path in part_paths.values():
            path.unlink(missing_ok=True)
        raise
    else:
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    quality_rejected = sum(rejection_counts.values())
    try:
        quality_rejection_ratio = validate_quality_rejection_ratio(
            input_articles, rejection_counts, max_quality_rejection_ratio
        )
    except ValueError:
        for path in part_paths.values():
            path.unlink(missing_ok=True)
        raise

    metadata_by_split = {}
    for split in splits:
        num_tokens = token_counts[split]
        num_samples = max(0, (num_tokens - 1) // seq_len)
        if num_samples < 1:
            for path in part_paths.values():
                path.unlink(missing_ok=True)
            raise ValueError(
                f"{split} 集只有 {num_tokens} 个 token，不足一个完整样本；"
                "请增加数据量或调整 val_ratio"
            )
        expected_bytes = num_tokens * dtype.itemsize
        if part_paths[split].stat().st_size != expected_bytes:
            raise RuntimeError(f"{split} token 文件大小校验失败")
        data_path, _ = final_paths[split]
        metadata_by_split[split] = {
            "format_version": 2,
            "storage_format": "memmap",
            "split": split,
            "data_file": data_path.name,
            "dtype": dtype.name,
            "num_tokens": num_tokens,
            "num_samples": num_samples,
            "num_documents": document_counts[split],
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "tokenizer": tokenizer_info,
            "boundary_token_id": boundary_token_id,
            "source_file": jsonl_path.name,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "split_seed": split_seed,
            "deduplicate": deduplicate,
            "near_deduplicate": near_deduplicate,
            "near_dedup_distance": near_dedup_distance,
            "include_title": include_title,
            "quality_filters": {
                "min_chars": min_chars,
                "min_cjk_ratio": min_cjk_ratio,
                "max_control_ratio": max_control_ratio,
                "max_repeat_run_ratio": max_repeat_run_ratio,
            },
            "rejection_counts": rejection_counts,
            "input_documents": input_articles,
            "retained_documents": total_articles,
            "retention_ratio": total_articles / max(1, input_articles),
            "quality_rejection_ratio": quality_rejection_ratio,
        }
        if max_document_tokens is not None:
            metadata_by_split[split]["max_document_tokens"] = max_document_tokens

    # 数据文件通过大小校验后再替换；元数据最后发布。
    for split in splits:
        data_path, meta_path = final_paths[split]
        part_paths[split].replace(data_path)
        write_metadata(meta_path, metadata_by_split[split])
        logger.info(
            f"{split}：{metadata_by_split[split]['num_documents']:,} 篇，"
            f"{metadata_by_split[split]['num_tokens']:,} tokens，"
            f"{metadata_by_split[split]['num_samples']:,} samples → {meta_path}"
        )

    logger.info(
        f"完成：输入={input_articles:,}，有效文章={total_articles:,}，"
        f"保留率={total_articles / max(1, input_articles):.1%}，重复={duplicate_count:,}，"
        f"近似重复={near_duplicate_count:,}，质量过滤={quality_rejected:,}"
    )
    return tuple(final_paths[split][1] for split in splits)


# 用途：解析预处理配置与 CLI，把反向开关转换后交给 build_dataset。
# 调用时机：python -m src.prepare_dataset 或直接执行脚本时调用。
# 输入输出与注意：no_dedup/no_near_dedup/no_title 用 not 转成启用参数；实际过滤和写文件在 build_dataset，参数解释见下方注释。
def main():
    # 参数分为：样本与划分、分词器、去重、标题处理、质量过滤。
    # 下方 default 是解析器的兜底值；实际优先级为：
    # 显式命令行参数 > 指定 JSON 的 prepare_dataset 分区 > 默认 JSON > 此处 default。
    parser = argparse.ArgumentParser()

    # 每条训练样本的输入长度（token 数），必须 > 0，并与训练时的 seq_len 一致。
    # bin 保存连续 token；读取样本时取 seq_len+1 个，构成右移一位的输入和目标。
    parser.add_argument("--seq_len", type=int, default=256)
    # 按文档哈希分到验证集的概率，必须 > 0；0.04 约为 4%，不是精确 token 占比。
    parser.add_argument("--val_ratio", type=float, default=0.04)
    # 测试集划分概率，必须 >= 0；0 表示本次不生成 test 集。
    # val_ratio + test_ratio 必须 < 1，剩余文档分给训练集；默认约为 95%/4%/1%。
    parser.add_argument("--test_ratio", type=float, default=0.01)
    # 所有 split 合计最多写入多少 token（含写入的文章结束符），适合小规模试跑。
    # None 表示不限；指定时必须 > 0。预算可能截断最后一篇文章，过小会导致某个集不足一条样本。
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument(
        "--max_document_tokens", type=int, default=None,
        help="单篇文档保留的最大 token 数，包含文档结束符",
    )
    # 文档划分用的种子：相同文档 ID 和种子会分到相同集合；修改它会改变划分。
    # 这是数据划分种子，不是模型初始化或训练打乱顺序的随机种子。
    parser.add_argument("--split_seed", type=int, default=42)
    # 分词器来源：已有的本地目录或 Hugging Face ID；None 时交给公共加载器自动查找。
    # 更换词表后要重新生成数据，并使用匹配词表的模型；此参数不会训练新分词器。
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer 本地目录或 Hugging Face ID；也可设置 MINIGPT_TOKENIZER_PATH",
    )
    # 默认执行精确去重：规范化后的正文完全相同就只保留一篇，标题不参与比较。
    # 传入此开关会令 no_dedup=True，从而关闭该步骤（近似去重仍由独立开关控制）。
    parser.add_argument("--no_dedup", action="store_true", help="关闭精确文档去重")
    # 反向开关：令同一个 no_dedup=False，可覆盖 JSON 中的关闭设置，重新开启精确去重。
    parser.add_argument("--dedup", dest="no_dedup", action="store_false", help=argparse.SUPPRESS)
    # 关闭 SimHash 近似去重；默认开启，用于过滤正文相似但不完全相同的文章。
    parser.add_argument("--no_near_dedup", action="store_true", help="关闭 SimHash 近似去重")
    # 反向开关：令 no_near_dedup=False，重新开启近似去重，可覆盖 JSON 配置。
    parser.add_argument("--near_dedup", dest="no_near_dedup", action="store_false", help=argparse.SUPPRESS)
    # 近似去重候选的 64 位指纹最多允许多少位不同（汉明距离），范围为 0~64。
    # 候选内距离 <= 此值即判重复；默认 3。四段索引在阈值 > 3 时可能漏召回，非全量比较。
    parser.add_argument("--near_dedup_distance", type=int, default=3)
    # 默认编码“标题 + 换行 + 正文”；传入此开关后仅编码正文，过滤和去重始终只看正文。
    parser.add_argument("--no_title", action="store_true", help="不把文章标题加入训练文本")
    # 反向开关：令 no_title=False，恢复加入标题，可覆盖 JSON 中的 no_title=true。
    parser.add_argument("--include_title", dest="no_title", action="store_false", help=argparse.SUPPRESS)
    # 规范化后的正文至少需要多少字符，必须 >= 0；默认少于 50 字符就丢弃。
    # 这里统计字符长度，不是分词后的 token 数，也不计入标题长度。
    parser.add_argument("--min_chars", type=int, default=50)
    # 非空白字符中，代码指定汉字范围（U+3400~U+9FFF）的字符占比下限，范围 0~1。
    # 默认 0.1 即至少 10%；调低可保留更多外文/混合文本，设为 0 可放开此比例要求。
    parser.add_argument("--min_cjk_ratio", type=float, default=0.1)
    # 异常 Unicode Cc 控制字符 / 正文全部字符的占比上限，范围 0~1；默认超过 1% 就拒绝。
    # 正常换行、回车、制表符不算异常；例如空字符 \x00 会被计入。
    parser.add_argument("--max_control_ratio", type=float, default=0.01)
    # 最长连续相同字符段 / 正文字符数的占比上限，范围 0~1；默认超过 20% 就拒绝。
    # 检测的是“啊啊啊啊”式连续重复，不是重复句子或段落。
    parser.add_argument("--max_repeat_run_ratio", type=float, default=0.2)
    # 整批质量过滤的保护阈值，范围 0~1；质量拒绝篇数 / 已读篇数超过默认 80% 时，
    # 抛错并停止发布数据，提示检查过滤规则；精确/近似去重丢弃的篇数不计入质量拒绝数。
    parser.add_argument("--max_quality_rejection_ratio", type=float, default=0.8)
    # 公共解析器还会添加 --config，用于选择 JSON 配置文件；未指定时读取 configs/default.json。
    args = parse_args_with_config(parser, "prepare_dataset")
    # 将命令行参数交给实际处理函数；no_* 表示“关闭”，所以用 not 转成函数需要的“启用”布尔值。
    build_dataset(
        seq_len=args.seq_len,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_tokens=args.max_tokens,
        max_document_tokens=args.max_document_tokens,
        tokenizer_source=args.tokenizer,
        split_seed=args.split_seed,
        deduplicate=not args.no_dedup,
        near_deduplicate=not args.no_near_dedup,
        near_dedup_distance=args.near_dedup_distance,
        include_title=not args.no_title,
        min_chars=args.min_chars,
        min_cjk_ratio=args.min_cjk_ratio,
        max_control_ratio=args.max_control_ratio,
        max_repeat_run_ratio=args.max_repeat_run_ratio,
        max_quality_rejection_ratio=args.max_quality_rejection_ratio,
    )


if __name__ == "__main__":
    main()
