"""
加载 MiniGPT checkpoint 进行自回归文本生成。

主要功能：
  支持 greedy、temperature、top-k、top-p 采样；默认启用 KV Cache，
  遇到训练使用的 EOS/SEP 时提前停止；支持重复惩罚、n-gram 防重复、无效
  特殊 token 屏蔽和中文 WordPiece 解码清理，并校验 tokenizer 指纹。

输入：
  默认使用 outputs/latest_run.json 指向的 best_model.pt，可通过
  --checkpoint 和 --tokenizer 显式覆盖。

输出：
  在终端打印“提示词 + 模型续写”；本文件不会修改 checkpoint。

使用方式（在项目根目录执行）：
  python -m src.generate --prompt "人工智能的发展"
  python -m src.generate --strategy top_p --temperature 0.8
  python -m src.generate --compare --prompt "中国的首都是"
  python -m src.generate --no_kv_cache --prompt "自然语言处理"
  python -m src.generate --config configs/default.json

未显式传入的参数从配置文件的 generate 分区读取。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import re
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .config_utils import parse_args_with_config
    from .experiment_utils import resolve_checkpoint
    from .model import build_model
    from .tokenizer_utils import load_tokenizer, validate_tokenizer_metadata
except ImportError:
    from config_utils import parse_args_with_config
    from experiment_utils import resolve_checkpoint
    from model import build_model
    from tokenizer_utils import load_tokenizer, validate_tokenizer_metadata


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE_DIR / "outputs" / "checkpoints"
CJK_AND_PUNCTUATION = (
    r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"，。！？；：、（）《》〈〉【】“”‘’…—"
)


# 用途：按 checkpoint 内的结构配置重建 MiniGPT 并加载权重。
# 调用时机：生成 main 和生成基准入口在处理提示词前调用。
# 输入输出与注意：返回 (eval 模型,checkpoint 字典)，使用 weights_only=True；缺 model_config 报错，不根据当前 JSON 猜旧模型结构。
def load_model(ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    if not checkpoint.get("model_config"):
        raise ValueError("Checkpoint 缺少 P1 model_config，请使用 P1 版本重新训练")
    model = build_model(
        vocab_size=checkpoint["vocab_size"],
        seq_len=checkpoint["seq_len"],
        model_config=checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# 用途：从单个位置的一维词表 logits 选择下一个 token。
# 调用时机：generate 每轮应用约束后调用。
# 输入输出与注意：greedy 取 argmax，其余按温度及 top-k/top-p 筛选后 multinomial；返回一个 token 张量，非 greedy 要求正温度。
def sample_next_token(
    next_logits: torch.Tensor,
    strategy: str,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if strategy == "greedy":
        return next_logits.argmax(dim=-1, keepdim=True)
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    next_logits = next_logits / temperature
    if strategy == "top_k":
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        top_k = min(top_k, next_logits.numel())
        threshold = torch.topk(next_logits, top_k).values[-1]
        next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
    elif strategy == "top_p":
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p 必须位于 (0, 1]")
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[1:] = sorted_remove[:-1].clone()
        sorted_remove[0] = False
        next_logits[sorted_indices[sorted_remove]] = float("-inf")
    elif strategy != "temperature":
        raise ValueError(f"不支持的生成策略：{strategy}")
    return torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)


# 用途：确定与预处理一致的文章结束符和禁止生成的其他特殊 ID。
# 调用时机：单次生成入口、策略比较和固定提示词基准开始前调用。
# 输入输出与注意：优先 EOS、否则 SEP，两者没有时报错；返回 (停止 ID 列表,禁用 ID 列表)，只选一个边界符。
def resolve_generation_token_ids(tokenizer):
    """与数据预处理保持一致：优先 EOS，没有 EOS 时以 SEP 结束文章。"""
    boundary_token_id = tokenizer.eos_token_id
    if boundary_token_id is None:
        boundary_token_id = tokenizer.sep_token_id
    if boundary_token_id is None:
        raise ValueError("Tokenizer 必须定义 eos_token_id 或 sep_token_id")
    banned_token_ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
    banned_token_ids.discard(boundary_token_id)
    return [int(boundary_token_id)], sorted(int(token) for token in banned_token_ids)


# 用途：在下一 token 分数上应用历史重复惩罚、n-gram 禁止项和特殊 ID 屏蔽。
# 调用时机：generate 每次采样前调用，约束测试也直接检查结果。
# 输入输出与注意：返回 float32 副本，历史包括提示词；正分数除惩罚系数、负分数乘系数，禁止项设 -inf，全部屏蔽时报错。
def apply_generation_constraints(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    banned_token_ids=None,
):
    """对下一 token logits 应用重复惩罚、n-gram 禁止项和特殊 token 屏蔽。"""
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty 必须大于或等于 1")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size 不能为负数")
    constrained = logits.float().clone()
    history = generated_ids.reshape(-1).tolist()

    if repetition_penalty > 1.0 and history:
        seen = torch.tensor(sorted(set(history)), device=constrained.device)
        values = constrained[seen]
        constrained[seen] = torch.where(
            values < 0, values * repetition_penalty, values / repetition_penalty
        )

    blocked = set(int(token) for token in (banned_token_ids or ()))
    n = no_repeat_ngram_size
    if n == 1:
        blocked.update(history)
    elif n > 1 and len(history) >= n - 1:
        prefix = tuple(history[-(n - 1) :])
        for index in range(len(history) - n + 1):
            if tuple(history[index : index + n - 1]) == prefix:
                blocked.add(history[index + n - 1])

    valid_blocked = [token for token in blocked if 0 <= token < constrained.numel()]
    if valid_blocked:
        constrained[valid_blocked] = float("-inf")
    if not torch.isfinite(constrained).any():
        raise ValueError("生成约束屏蔽了整个词表，请放宽约束")
    return constrained


# 用途：基于提示词逐 token 续写，支持缓存、采样约束和文章边界停止。
# 调用时机：CLI、策略比较、基准和模型测试需要生成完整 token 序列时调用。
# 输入输出与注意：输入非空 (1,T)，返回含提示词与新内容的一维张量；inference_mode 不代替 eval，窗口满时重建缓存，max_new_tokens=0 不前向。
@torch.inference_mode()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 80,
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_ids=None,
    use_kv_cache: bool = True,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    banned_token_ids=None,
    min_new_tokens: int = 0,
) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("input_ids 必须是形状为 (1, T) 的非空张量")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能为负数")
    if not 0 <= min_new_tokens <= max_new_tokens:
        raise ValueError("min_new_tokens 必须位于 [0, max_new_tokens]")
    eos_token_ids = set(eos_token_ids or [])
    generated = input_ids.clone()
    seq_len = model.seq_len
    cache = None
    cached_logits = None

    if use_kv_cache and max_new_tokens:
        cached_logits, cache = model(generated[:, -seq_len:], use_cache=True)

    for generated_count in range(max_new_tokens):
        if use_kv_cache:
            next_logits = cached_logits[0, -1]
        else:
            next_logits = model(generated[:, -seq_len:])[0, -1]
        current_banned = set(banned_token_ids or ())
        if generated_count < min_new_tokens:
            current_banned.update(eos_token_ids)
        next_logits = apply_generation_constraints(
            next_logits,
            generated,
            repetition_penalty,
            no_repeat_ngram_size,
            current_banned,
        )
        next_token = sample_next_token(
            next_logits, strategy, temperature, top_k, top_p
        )
        generated = torch.cat((generated, next_token.view(1, 1)), dim=1)
        if int(next_token.item()) in eos_token_ids:
            break

        if use_kv_cache:
            cache_len = cache[0][0].shape[-2]
            if cache_len >= seq_len:
                cached_logits, cache = model(
                    generated[:, -seq_len:], use_cache=True
                )
            else:
                cached_logits, cache = model(
                    next_token.view(1, 1),
                    past_key_values=cache,
                    use_cache=True,
                )
    return generated[0]


# 用途：将 ID 解码成文字并清理中文字符/标点附近的多余空白。
# 调用时机：生成输出打印、基准记录 completion、中文解码测试时调用。
# 输入输出与注意：返回 strip 后字符串，默认跳过特殊 token；保留通常英文空格，但中文之间的换行也可能被清理。
def decode_text(tokenizer, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
    text = tokenizer.decode(ids.tolist(), skip_special_tokens=skip_special_tokens)
    # BertTokenizer.decode 会在中文字符之间插入空格；仅清理 CJK/中文标点附近
    # 的空格，保留英文单词之间真正需要的空格。
    text = re.sub(
        rf"(?<=[{CJK_AND_PUNCTUATION}])\s+(?=[{CJK_AND_PUNCTUATION}])",
        "",
        text,
    )
    text = re.sub(rf"\s+(?=[，。！？；：、）》〉】])", "", text)
    text = re.sub(rf"(?<=[（《〈【])\s+", "", text)
    return text.strip()


# 用途：对相同提示词依次展示四种采样策略的续写。
# 调用时机：生成 main 收到 compare 开关时调用。
# 输入输出与注意：编码一次 prompt，分别调用 generate 并打印；内部有固定温度/top-k/top-p，未为每种策略重置 RNG，不返回报告文件。
def compare_strategies(
    model, tokenizer, prompt, max_new_tokens, device, use_kv_cache,
    repetition_penalty, no_repeat_ngram_size, min_new_tokens,
):
    input_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    prompt_len = input_ids.shape[1]
    eos_ids, banned_ids = resolve_generation_token_ids(tokenizer)
    strategies = [
        ("Greedy", {"strategy": "greedy"}),
        ("Temperature", {"strategy": "temperature", "temperature": 0.8}),
        ("Top-K", {"strategy": "top_k", "temperature": 0.8, "top_k": 50}),
        ("Top-P", {"strategy": "top_p", "temperature": 0.8, "top_p": 0.9}),
    ]
    for name, kwargs in strategies:
        output = generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_ids,
            use_kv_cache=use_kv_cache,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            banned_token_ids=banned_ids,
            min_new_tokens=min_new_tokens,
            **kwargs,
        )
        print(f"\n【{name}】\n{prompt}{decode_text(tokenizer, output[prompt_len:])}")


# 用途：解析生成设置、定位权重和词表、设置随机种子后输出续写。
# 调用时机：python -m src.generate 或脚本执行时调用。
# 输入输出与注意：默认选择最近实验 best_model；验证词表后进入单次或比较模式，不训练权重，不实现持续聊天历史。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="中国的首都是")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument(
        "--strategy",
        choices=("greedy", "temperature", "top_k", "top_p"),
        default="top_p",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--no_compare", dest="compare", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--no_kv_cache", action="store_true")
    parser.add_argument("--kv_cache", dest="no_kv_cache", action="store_false", help=argparse.SUPPRESS)
    args = parse_args_with_config(parser, "generate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_checkpoint(args.checkpoint, BASE_DIR / "outputs")
    model, checkpoint = load_model(ckpt_path, device)
    tokenizer_source = args.tokenizer or (checkpoint.get("tokenizer") or {}).get("source")
    tokenizer, tokenizer_info = load_tokenizer(tokenizer_source)
    validate_tokenizer_metadata(
        checkpoint.get("tokenizer"), tokenizer_info, f"Checkpoint {ckpt_path.name}"
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(
        f"模型加载完成：seq_len={model.seq_len}，KV Cache={not args.no_kv_cache}"
    )

    if args.compare:
        compare_strategies(
            model, tokenizer, args.prompt, args.max_new_tokens, device,
            not args.no_kv_cache,
            args.repetition_penalty,
            args.no_repeat_ngram_size,
            args.min_new_tokens,
        )
        return
    input_ids = tokenizer.encode(
        args.prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    prompt_len = input_ids.shape[1]
    eos_ids, banned_ids = resolve_generation_token_ids(tokenizer)
    output = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_ids=eos_ids,
        use_kv_cache=not args.no_kv_cache,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        banned_token_ids=banned_ids,
        min_new_tokens=args.min_new_tokens,
    )
    print(f"\n{args.prompt}{decode_text(tokenizer, output[prompt_len:])}\n")


if __name__ == "__main__":
    main()
