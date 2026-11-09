"""从已完成实验的记录生成报告图表与来源清单，不训练、不重新评估模型。

项目根目录运行：python experiments/build_experiment_figures.py
输入：消融协议、各组训练/评估日志、历史 baseline 日志。
输出：docs/experiment_report/ 下的 PNG、曲线采样 JSON、证据 SHA-256 清单。
详细报告正文位于项目根目录 EXPERIMENT_REPORT.md。
"""

import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs" / "experiment_report"
SPEC = importlib.util.spec_from_file_location("ablation", ROOT / "experiments/ablation/run.py")
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


# 用途：校验已完成实验记录，从控制台与轮末日志生成对比图、曲线点和证据清单。
# 调用时机：直接运行本图表脚本、需要重建已有实验报告图时调用。
# 输入输出与注意：先 check_lock/collect，再核对配置、完整验证 token 数和日志；写 docs/experiment_report，包含单独历史 baseline 图，不训练也不重新评估模型。
def main():
    protocol = ablation.check_lock()
    rows = ablation.collect()
    evidence = [ablation.LOCK, ROOT / "configs/default.json"]
    curves = {}
    pattern = re.compile(r"Epoch (\d+)/(\d+) step=(\d+) loss=([\d.]+).*?lr=([\deE.+-]+).*?tokens/s≈([\d,]+)")
    for row in rows:
        group = row["group"]
        if row["status"] != "completed" or "val" not in row:
            raise ValueError(f"{group} 尚未完整训练并评估")
        directory = ablation.run_dir(group)
        manifest = ablation.read(directory / "run_manifest.json")
        config = ablation.configuration(group)
        for key, value in manifest["training_config"].items():
            if config["train"].get(key) != value:
                raise ValueError(f"{group} 实际训练配置不符：{key}")
        for key in ("d_model", "n_heads", "n_layers", "d_ff", "position_encoding", "norm_type", "activation", "attention_impl"):
            if config["train"][key] != manifest["model_config"][key]:
                raise ValueError(f"{group} 模型配置不符：{key}")
        if abs(row["val"]["loss"] - row["training"]["val_loss"]) > 1e-6:
            raise ValueError(f"{group} 独立验证与训练日志不同")
        report = ablation.read(directory / "evaluations/val_metrics.json")
        if report["max_batches"] is not None or row["val"]["tokens"] != 7048448:
            raise ValueError(f"{group} 不是本批完整验证")
        console = ROOT / f"outputs/ablation/{group}/train_console.log"
        points = [{"epoch": int(m[1]), "step": int(m[3]), "cumulative_train_loss": float(m[4]),
                   "lr": float(m[5]), "interval_tokens_per_second": int(m[6].replace(",", ""))}
                  for m in pattern.finditer(console.read_text(encoding="utf-8"))]
        if not points or any(p["epoch"] != 1 for p in points):
            raise ValueError(f"{group} 控制台曲线记录异常")
        curves[group] = points
        evidence.extend([console, directory / "run_manifest.json", directory / "source_config.json",
                         directory / "training_log.jsonl", directory / "evaluations/val_metrics.json",
                         directory / "generation_benchmark.json", ROOT / f"outputs/ablation/{group}/execution.json",
                         ROOT / f"outputs/ablation/{group}/evaluation_val_provenance.json"])
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for group, points in curves.items():
        steps = [p["step"] for p in points]
        axes[0].plot(steps, [p["cumulative_train_loss"] for p in points], label=group)
        axes[1].plot(steps, [p["lr"] for p in points], label=group)
        axes[2].plot(steps, [p["interval_tokens_per_second"] / 1000 for p in points], label=group, alpha=.8)
    for ax, title in zip(axes, ("Cumulative train loss (not batch loss)", "Learning rate", "Logged interval throughput (k tokens/s)")):
        ax.set(xlabel="Optimizer step", title=title)
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(DEST / "ablation_training.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [("Validation PPL", [r["val"]["ppl"] for r in rows]),
               ("Training time (minutes)", [r["training"]["train_seconds"] / 60 for r in rows]),
               ("Peak allocated CUDA memory (GiB)", [r["training"]["peak_cuda_allocated_bytes"] / 2**30 for r in rows])]
    for ax, (title, values) in zip(axes, metrics):
        bars = ax.bar([r["group"] for r in rows], values, color=["#64748b", "#0f766e", "#2563eb"])
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set_title(title)
        ax.margins(y=.18)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(DEST / "ablation_results.png", dpi=170)
    plt.close(fig)
    baseline_dir = ROOT / "outputs/runs/baseline"
    baseline_log = baseline_dir / "training_log.jsonl"
    baseline = [json.loads(line) for line in baseline_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("loss", "ppl")):
        for split in ("train", "val"):
            ax.plot([r["epoch"] for r in baseline], [r[f"{split}_{metric}"] for r in baseline], "o-", label=split)
        ax.set(xlabel="Epoch", ylabel=metric, title="Historical baseline (separate budget)")
        ax.set_xticks([r["epoch"] for r in baseline])
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(DEST / "baseline_training.png", dpi=170)
    plt.close(fig)
    evidence += [baseline_log, baseline_dir / "run_manifest.json", baseline_dir / "evaluations/test_metrics.json"]
    evidence += [ROOT / f"data/{split}_seq256.meta.json" for split in ("train", "val", "test")]
    ablation.atomic_json_dump({"created_at": datetime.now(timezone.utc).isoformat(), "protocol_id": protocol,
                              "files": {p.relative_to(ROOT).as_posix(): ablation.digest(p) for p in evidence},
                              "verification": "Protocol, completed budgets, manifests and full validation checked; no training performed."}, DEST / "evidence.json")
    ablation.atomic_json_dump(curves, DEST / "training_curve_points.json")
    print(json.dumps({"curve_points": {g: len(p) for g, p in curves.items()},
                      "B_vs_A_ppl_reduction_pct": (1-rows[1]["val"]["ppl"]/rows[0]["val"]["ppl"])*100,
                      "C_vs_B_speed_gain_pct": (rows[2]["training"]["train_tokens_per_second"]/rows[1]["training"]["train_tokens_per_second"]-1)*100,
                      "C_vs_B_memory_reduction_pct": (1-rows[2]["training"]["peak_cuda_allocated_bytes"]/rows[1]["training"]["peak_cuda_allocated_bytes"])*100,
                      "C_vs_B_ppl_increase_pct": (rows[2]["val"]["ppl"]/rows[1]["val"]["ppl"]-1)*100,
                      "memory_gib": [r["training"]["peak_cuda_allocated_bytes"]/2**30 for r in rows],
                      "stopped": [sum(s["stopped_on_boundary"] for s in r["generation"]["samples"]) for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
