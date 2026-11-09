"""
评估 MiniGPT checkpoint，并可绘制对应实验的训练曲线。

评估指标：
  token 级交叉熵、PPL、bits/token、next-token accuracy 和 tokens/s。
  val 用于选模，test 应只在模型和超参数确定后用于最终报告。

输入：
  默认读取 outputs/latest_run.json 指向的 best_model.pt，以及匹配的
  data/val_seq*.meta.json 或 data/test_seq*.meta.json。

输出：
  默认在当前实验的 evaluations/ 目录保存 JSON 指标；使用 --plot 时在当前
  实验目录生成 training_curve.png。--output 可覆盖评估 JSON 的保存路径。

使用方式（在项目根目录执行）：
  python -m src.evaluate
  python -m src.evaluate --split test
  python -m src.evaluate --checkpoint outputs/runs/baseline/checkpoints/best_model.pt
  python -m src.evaluate --max_batches 20 --plot
  python -m src.evaluate --config configs/default.json --split test

未显式传入的参数从配置文件的 evaluate 分区读取。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

try:
    from .config_utils import parse_args_with_config
    from .experiment_utils import atomic_json_dump, infer_run_dir, resolve_checkpoint
    from .model import build_model
    from .train import (
        TokenDataset,
        autocast_context,
        make_dataloader,
        resolve_amp_dtype,
        resolve_dataset_path,
    )
except ImportError:
    from config_utils import parse_args_with_config
    from experiment_utils import atomic_json_dump, infer_run_dir, resolve_checkpoint
    from model import build_model
    from train import (
        TokenDataset,
        autocast_context,
        make_dataloader,
        resolve_amp_dtype,
        resolve_dataset_path,
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_PATH = OUTPUT_DIR / "training_log.jsonl"


# 用途：从保存的架构及权重重建评估模型并关闭 dropout。
# 调用时机：evaluate_checkpoint 开始时调用。
# 输入输出与注意：输入 checkpoint Path 和目标设备，返回 (模型,checkpoint)；读取整个 payload，eval 不等于关闭梯度，循环另用 inference_mode。
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


# 用途：在真实上下文的 val/test 上计算 token 级语言模型指标。
# 调用时机：评估 main、微型训练回归或外部脚本需要检查已训练权重时调用。
# 输入输出与注意：返回 loss/PPL/bits_per_token/accuracy/token 数和吞吐；校验词表指纹，按总 NLL 加权，可取前 max_batches 批，不更新参数。
def evaluate_checkpoint(
    ckpt_path: Path,
    batch_size: int = 32,
    amp: str = "auto",
    max_batches: int = None,
    split: str = "val",
):
    if split not in {"val", "test"}:
        raise ValueError("split 必须是 val 或 test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(amp, device)
    model, checkpoint = load_model(ckpt_path, device)
    dataset = TokenDataset(resolve_dataset_path(split, checkpoint["seq_len"]))
    checkpoint_tokenizer = checkpoint.get("tokenizer")
    if (
        not checkpoint_tokenizer
        or not dataset.tokenizer_info
        or checkpoint_tokenizer.get("vocab_sha256")
        != dataset.tokenizer_info.get("vocab_sha256")
    ):
        raise ValueError("Checkpoint 与验证集使用了不同 tokenizer")
    loader = make_dataloader(dataset, batch_size, False, 0, device)

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_tokens = 0
    correct_tokens = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (input_ids, targets) in enumerate(loader, 1):
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast_context(device, amp_dtype):
                logits = model(input_ids)
                rows, seq_len, vocab_size = logits.shape
                loss = loss_fn(
                    logits.reshape(rows * seq_len, vocab_size),
                    targets.reshape(rows * seq_len),
                )
            total_nll += loss.float().item()
            total_tokens += targets.numel()
            correct_tokens += (logits.argmax(-1) == targets).sum().item()
            if max_batches is not None and batch_index >= max_batches:
                break
    elapsed = max(time.perf_counter() - start, 1e-6)
    avg_loss = total_nll / total_tokens
    metrics = {
        "loss": avg_loss,
        "ppl": math.exp(min(avg_loss, 20)),
        "bits_per_token": avg_loss / math.log(2),
        "token_accuracy": correct_tokens / total_tokens,
        "tokens": total_tokens,
        "tokens_per_second": total_tokens / elapsed,
    }
    metrics["split"] = split
    logger.info(f"{split} 集评估结果")
    for key, value in metrics.items():
        if key == "split":
            logger.info(f"  {key}: {value}")
        elif key == "tokens":
            logger.info(f"  {key}: {value:,}")
        else:
            logger.info(f"  {key}: {value:.4f}")
    return metrics


# 用途：把指标连同 checkpoint 路径和评估设置保存为可追溯 JSON。
# 调用时机：评估 main 得到指标后调用，报告路径测试也直接调用。
# 输入输出与注意：返回报告 Path；默认在所属实验 evaluations 中，抽样文件带批数后缀，相对自定义路径基于项目根目录，同名会替换。
def save_evaluation_report(
    metrics: dict,
    checkpoint_path: Path,
    batch_size: int,
    amp: str,
    max_batches: int = None,
    output_path: Path = None,
):
    """保存可追溯的评估结果；完整评估和抽样评估使用不同文件名。"""
    checkpoint_path = Path(checkpoint_path).resolve()
    if output_path is None:
        run_dir = infer_run_dir(checkpoint_path)
        suffix = "" if max_batches is None else f"_max{max_batches}batches"
        output_path = run_dir / "evaluations" / f"{metrics['split']}_metrics{suffix}.json"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = BASE_DIR / output_path
    report = {
        "format_version": 1,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "batch_size": batch_size,
        "amp": amp,
        "max_batches": max_batches,
        "metrics": metrics,
    }
    atomic_json_dump(report, output_path)
    logger.info(f"评估结果已保存：{output_path}")
    return output_path


# 用途：从逐轮训练日志绘制训练/验证 Loss 与 PPL 曲线。
# 调用时机：评估 main 的 plot 分支调用，也可单独调用以避免重新评估模型。
# 输入输出与注意：输入 JSONL 路径，返回同目录 training_curve.png；Agg 无窗口绘图，缺日志/空日志报错，每轮一个点，不能从权重还原曲线。
def plot_training_curve(log_path: Path = None):
    log_path = Path(log_path or LOG_PATH)
    if not log_path.exists():
        raise FileNotFoundError(f"未找到训练日志：{log_path}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise ValueError("训练日志为空")
    epochs = [entry["epoch"] for entry in entries]
    train_ppl = [entry["train_ppl"] for entry in entries]
    val_ppl = [entry["val_ppl"] for entry in entries]
    train_loss = [entry["train_loss"] for entry in entries]
    val_loss = [entry.get("val_loss", math.log(entry["val_ppl"])) for entry in entries]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_ppl, "b-o", label="Train PPL")
    axes[0].plot(epochs, val_ppl, "r-o", label="Val PPL")
    axes[0].set(xlabel="Epoch", ylabel="PPL", title="Perplexity")
    axes[1].plot(epochs, train_loss, "b-o", label="Train Loss")
    axes[1].plot(epochs, val_loss, "r-o", label="Val Loss")
    axes[1].set(xlabel="Epoch", ylabel="Cross-Entropy", title="Loss")
    for axis in axes:
        axis.legend()
        axis.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = log_path.parent / "training_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"训练曲线已保存：{out_path}")
    return out_path


# 用途：解析评估参数，执行指标计算、报告保存和可选绘图。
# 调用时机：python -m src.evaluate 或直接执行脚本时调用。
# 输入输出与注意：默认最近实验 best_model 和 val；plot 是评估后绘图，不是只画图，test 用于方案确定后的独立测量。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--amp", choices=("auto", "off", "bf16", "fp16"), default="auto")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no_plot", dest="plot", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=None)
    args = parse_args_with_config(parser, "evaluate")
    ckpt_path = resolve_checkpoint(args.checkpoint, OUTPUT_DIR)
    metrics = evaluate_checkpoint(
        ckpt_path, args.batch_size, args.amp, args.max_batches, args.split
    )
    save_evaluation_report(
        metrics,
        ckpt_path,
        args.batch_size,
        args.amp,
        args.max_batches,
        args.output,
    )
    if args.plot:
        plot_training_curve(infer_run_dir(ckpt_path) / "training_log.jsonl")


if __name__ == "__main__":
    main()
