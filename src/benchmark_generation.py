"""
在固定提示词和随机种子下执行 MiniGPT 生成回归基准。

主要功能：
  保存每条 completion，并统计生成 token 数、延迟、tokens/s、
  Distinct-1/2 和 3-gram 重复率。自动指标只是质量代理，不能替代人工评审。

输入：
  默认使用最近实验的 best_model.pt 和内置提示词；--prompts 可接收
  纯文本逐行提示词，或每行包含 {"prompt": "..."} 的 JSONL 文件。

输出：
  默认写入 outputs/runs/<latest>/generation_benchmark.json。

使用方式（在项目根目录执行）：
  python -m src.benchmark_generation
  python -m src.benchmark_generation --strategy greedy --max_new_tokens 40
  python -m src.benchmark_generation --prompts prompts.jsonl
  python -m src.benchmark_generation --checkpoint <checkpoint路径> --output report.json
  python -m src.benchmark_generation --config configs/default.json

未显式传入的参数从配置文件的 benchmark_generation 分区读取。
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

try:
    from .config_utils import parse_args_with_config
    from .experiment_utils import atomic_json_dump, infer_run_dir, resolve_checkpoint
    from .generate import (
        decode_text,
        generate,
        load_model,
        resolve_generation_token_ids,
    )
    from .tokenizer_utils import load_tokenizer, validate_tokenizer_metadata
except ImportError:
    from config_utils import parse_args_with_config
    from experiment_utils import atomic_json_dump, infer_run_dir, resolve_checkpoint
    from generate import decode_text, generate, load_model, resolve_generation_token_ids
    from tokenizer_utils import load_tokenizer, validate_tokenizer_metadata


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = [
    "中国的首都是",
    "人工智能的发展",
    "长江是中国",
    "在一个宁静的夜晚",
    "科学研究最重要的是",
]


# 用途：计算不同 n-gram 数占全部相邻 n-gram 的比例，衡量表面多样性。
# 调用时机：benchmark 为新生成 token 计算 Distinct-1/2，指标测试也调用。
# 输入输出与注意：返回 0~1 浮点数，长度不足 n 返回 0，n 必须正；不衡量事实正确性或语义质量。
def distinct_n(token_ids, n: int) -> float:
    if n < 1:
        raise ValueError("n 必须大于 0")
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


# 用途：计算相邻 n-gram 的重复比例，默认统计三元组。
# 调用时机：benchmark 每条生成结束后调用，重复率测试也调用。
# 输入输出与注意：返回 1-唯一数量/总数量，长度不足返回 0；调用方需传正 n，统计 token 组合而非句子语义重复。
def repetition_ratio(token_ids, n: int = 3) -> float:
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


# 用途：加载默认或文件中的固定提示词集合。
# 调用时机：基准 main 调用 benchmark 前使用。
# 输入输出与注意：返回字符串列表；JSONL 对象取 prompt，JSON 字符串直接用，解析失败按普通文本行处理，全部为空时报错。
def load_prompts(path: Path = None):
    if path is None:
        return DEFAULT_PROMPTS
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            prompt = value.get("prompt") if isinstance(value, dict) else value
        except json.JSONDecodeError:
            prompt = line
        if prompt:
            prompts.append(str(prompt))
    if not prompts:
        raise ValueError(f"提示词文件为空：{path}")
    return prompts


# 用途：对多条提示词逐条生成，测量延迟/吞吐并统计重复和多样性。
# 调用时机：main 加载模型、tokenizer 并设置种子后调用，也可由外部比较脚本调用。
# 输入输出与注意：返回 summary/samples，计时包含 generate 但不含分词解码；CUDA 前后同步，统计仅新 token 且可能含结束符，调用方固定种子与顺序。
def benchmark(
    model, tokenizer, prompts, device, max_new_tokens=80, strategy="top_p",
    temperature=0.7, top_k=50, top_p=0.85, use_kv_cache=True,
    repetition_penalty=1.1, no_repeat_ngram_size=4, min_new_tokens=0,
):
    rows = []
    eos_ids, banned_ids = resolve_generation_token_ids(tokenizer)
    for prompt in prompts:
        input_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = generate(
            model, input_ids, max_new_tokens=max_new_tokens, strategy=strategy,
            temperature=temperature, top_k=top_k, top_p=top_p,
            eos_token_ids=eos_ids, use_kv_cache=use_kv_cache,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            banned_token_ids=banned_ids,
            min_new_tokens=min_new_tokens,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - started, 1e-9)
        generated_ids = output[input_ids.shape[1] :].tolist()
        rows.append({
            "prompt": prompt,
            "completion": decode_text(tokenizer, output[input_ids.shape[1] :]),
            "generated_token_ids": generated_ids,
            "stopped_on_boundary": bool(generated_ids and generated_ids[-1] in eos_ids),
            "generated_tokens": len(generated_ids),
            "latency_seconds": elapsed,
            "tokens_per_second": len(generated_ids) / elapsed,
            "distinct_1": distinct_n(generated_ids, 1),
            "distinct_2": distinct_n(generated_ids, 2),
            "repetition_3gram": repetition_ratio(generated_ids, 3),
        })
    total_tokens = sum(row["generated_tokens"] for row in rows)
    total_time = sum(row["latency_seconds"] for row in rows)
    return {
        "summary": {
            "prompts": len(rows),
            "generated_tokens": total_tokens,
            "latency_seconds": total_time,
            "tokens_per_second": total_tokens / max(total_time, 1e-9),
            "mean_repetition_3gram": sum(r["repetition_3gram"] for r in rows) / len(rows),
        },
        "samples": rows,
    }


# 用途：解析生成基准设置、验证词表并将固定提示词结果保存为 JSON。
# 调用时机：python -m src.benchmark_generation 或消融评估调用此模块时执行。
# 输入输出与注意：默认输出所属实验 generation_benchmark.json；设置 PyTorch 种子一次，报告只记录部分采样配置，完整复现还需保留源配置。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--strategy", choices=("greedy", "temperature", "top_k", "top_p"), default="top_p")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--no_kv_cache", action="store_true")
    parser.add_argument("--kv_cache", dest="no_kv_cache", action="store_false", help=argparse.SUPPRESS)
    args = parse_args_with_config(parser, "benchmark_generation")

    checkpoint_path = resolve_checkpoint(args.checkpoint, BASE_DIR / "outputs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(checkpoint_path, device)
    tokenizer_source = args.tokenizer or (checkpoint.get("tokenizer") or {}).get("source")
    tokenizer, metadata = load_tokenizer(tokenizer_source)
    validate_tokenizer_metadata(checkpoint.get("tokenizer"), metadata, "生成基准")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    report = benchmark(
        model, tokenizer, load_prompts(args.prompts), device,
        args.max_new_tokens, args.strategy, args.temperature, args.top_k,
        args.top_p, not args.no_kv_cache,
        args.repetition_penalty, args.no_repeat_ngram_size, args.min_new_tokens,
    )
    report.update({
        "checkpoint": str(checkpoint_path),
        "strategy": args.strategy,
        "seed": args.seed,
        "kv_cache": not args.no_kv_cache,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "min_new_tokens": args.min_new_tokens,
    })
    output = args.output or infer_run_dir(checkpoint_path) / "generation_benchmark.json"
    atomic_json_dump(report, output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"报告已保存：{Path(output).resolve()}")


if __name__ == "__main__":
    main()
