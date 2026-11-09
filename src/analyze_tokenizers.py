"""
在同一份中文语料样本上比较多个本地或在线 tokenizer。

比较指标：
  词表大小、embedding 参数量、tokens/字符、字符/token、tokens/UTF-8 字节、
  UNK 比例，以及是否具有可作为文档边界的 EOS/SEP。

输入：
  默认读取 data/wiki_zh.jsonl；每个 --tokenizer 参数可以是本地目录或
  Hugging Face 模型 ID。该参数可重复传入。

输出：
  默认写入 outputs/tokenizer_comparison.json，并在终端打印摘要。

使用方式（在项目根目录执行）：
  python -m src.analyze_tokenizers --tokenizer bert-base-chinese
  python -m src.analyze_tokenizers --tokenizer bert-base-chinese --tokenizer <BPE-ID>
  python -m src.analyze_tokenizers --tokenizer <本地目录> --max_articles 5000
  python -m src.analyze_tokenizers --config <已填写tokenizer数组的配置文件>

未显式传入的参数从配置文件的 analyze_tokenizers 分区读取。注意：此工具
比较切分和词表成本，不等价于训练后模型效果评测。
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from .config_utils import parse_args_with_config
    from .experiment_utils import atomic_json_dump
    from .tokenizer_utils import load_tokenizer
except ImportError:
    from config_utils import parse_args_with_config
    from experiment_utils import atomic_json_dump
    from tokenizer_utils import load_tokenizer


BASE_DIR = Path(__file__).resolve().parent.parent


# 用途：从文章 JSONL 提取一份可供所有候选复用的正文样本。
# 调用时机：compare_tokenizers 开始时调用。
# 输入输出与注意：返回前 max_articles 篇非空正文列表；非随机取样，不带标题和结束符，也不执行正式质量过滤，坏 JSON 带行号报错。
def load_text_sample(path: Path, max_articles: int = 1000):
    if max_articles < 1:
        raise ValueError("max_articles 必须大于 0")
    texts = []
    with open(path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_no} 行无效：{exc}") from exc
            text = str(row.get("text") or "").strip()
            if text:
                texts.append(text)
            if len(texts) >= max_articles:
                break
    if not texts:
        raise ValueError(f"语料中没有有效正文：{path}")
    return texts


# 用途：统计某个 tokenizer 的切分长度、UNK 比例和词表参数成本。
# 调用时机：compare_tokenizers 为每个候选调用，指标测试也用模拟 tokenizer 调用。
# 输入输出与注意：输入同一 texts 和假设 d_model，返回比率与身份字典；embedding=vocab_size*d_model，不代表完整模型参数或训练后能力。
def analyze_tokenizer(tokenizer, metadata: dict, texts, d_model: int = 384):
    if d_model < 1:
        raise ValueError("d_model 必须大于 0")
    total_chars = sum(len(text) for text in texts)
    total_utf8_bytes = sum(len(text.encode("utf-8")) for text in texts)
    token_count = 0
    unknown_count = 0
    unk_id = tokenizer.unk_token_id
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_count += len(ids)
        if unk_id is not None:
            unknown_count += sum(token_id == unk_id for token_id in ids)
    if token_count < 1:
        raise ValueError(f"Tokenizer {metadata['source']} 没有产生任何 token")
    vocab_size = int(metadata["vocab_size"])
    return {
        "source": metadata["source"],
        "class": metadata["class"],
        "vocab_size": vocab_size,
        "vocab_sha256": metadata["vocab_sha256"],
        "embedding_parameters": vocab_size * d_model,
        "articles": len(texts),
        "characters": total_chars,
        "utf8_bytes": total_utf8_bytes,
        "tokens": token_count,
        "tokens_per_character": token_count / total_chars,
        "characters_per_token": total_chars / token_count,
        "tokens_per_utf8_byte": token_count / total_utf8_bytes,
        "unknown_tokens": unknown_count,
        "unknown_rate": unknown_count / token_count,
        "has_document_boundary_token": (
            tokenizer.eos_token_id is not None or tokenizer.sep_token_id is not None
        ),
    }


# 用途：加载一次正文样本，再逐来源加载分词器进行相同口径比较。
# 调用时机：main 取得候选列表后调用。
# 输入输出与注意：返回含 input/d_model/sample_articles/results 的报告；实际加载来源需核对 metadata，不训练分词器、不直接保存文件。
def compare_tokenizers(sources, input_path: Path, max_articles=1000, d_model=384):
    texts = load_text_sample(input_path, max_articles)
    results = []
    for source in sources:
        tokenizer, metadata = load_tokenizer(source)
        results.append(analyze_tokenizer(tokenizer, metadata, texts, d_model))
    return {
        "input": str(Path(input_path).resolve()),
        "d_model": d_model,
        "sample_articles": len(texts),
        "results": results,
    }


# 用途：解析候选 tokenizer、样本与输出配置，运行比较并保存 JSON 摘要。
# 调用时机：python -m src.analyze_tokenizers 时调用。
# 输入输出与注意：至少指定一个候选；可重复 --tokenizer，报告保存用 atomic_json_dump，默认数组为空会报错。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", action="append",
        help="可重复传入本地目录或 Hugging Face ID",
    )
    parser.add_argument("--input", type=Path, default=BASE_DIR / "data" / "wiki_zh.jsonl")
    parser.add_argument("--max_articles", type=int, default=1000)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument(
        "--output", type=Path, default=BASE_DIR / "outputs" / "tokenizer_comparison.json"
    )
    args = parse_args_with_config(parser, "analyze_tokenizers")
    if not args.tokenizer:
        parser.error("请通过配置文件或 --tokenizer 至少指定一个 tokenizer")
    report = compare_tokenizers(
        args.tokenizer, args.input, args.max_articles, args.d_model
    )
    atomic_json_dump(report, args.output)
    for result in report["results"]:
        print(
            f"{result['source']}: vocab={result['vocab_size']:,}, "
            f"tokens/char={result['tokens_per_character']:.4f}, "
            f"UNK={result['unknown_rate']:.4%}, "
            f"embedding={result['embedding_parameters'] / 1e6:.2f}M"
        )
    print(f"报告已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
