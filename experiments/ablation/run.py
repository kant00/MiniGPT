"""管理 A/B/C 消融实验：检查、训练启动、评估、报告与绘图。

在已安装依赖的环境中，从项目根目录运行：
  python experiments/ablation/run.py prepare
  python experiments/ablation/run.py check
  python experiments/ablation/run.py train A
  python experiments/ablation/run.py evaluate A
  python experiments/ablation/run.py report
只有 train 子命令会启动真实训练；report 只读取已有结果。
prepare 固定数据/代码/配置 SHA-256；后续命令拒绝内容漂移。
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiment_utils import atomic_json_dump

BASE = ROOT / "experiments" / "ablation"
LOCK = BASE / "protocol_lock.json"
OUT = ROOT / "outputs" / "ablation"
IDS = ("A", "B", "C")


# 用途：读取实验工具所需的 JSON 配置、协议或结果字典。
# 调用时机：configuration、协议检查、结果收集等多个辅助函数需要结构化文件时调用。
# 输入输出与注意：输入路径并以 UTF-8 解析，返回 JSON 值；文件缺失或无效直接抛错，不自行补默认或写回。
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# 用途：分块计算文件内容 SHA-256，避免一次载入大型数据或权重。
# 调用时机：实验协议固定、漂移检查和评估来源验证时调用。
# 输入输出与注意：返回十六进制指纹；每次读取 8MiB，注释改动也改变源码内容哈希，不是仅比较可执行语义。
def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        # 无参匿名读取回调：iter 每轮调用以取得下一块 8MiB 字节，返回 b"" 时结束。
        # 闭包持有当前打开的 stream，避免将整份大型 bin/checkpoint 一次加载进内存。
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


# 用途：读取指定 A/B/C 组的配置文件。
# 调用时机：校验协议、解析实验路径、启动训练和记录评估来源时调用。
# 输入输出与注意：输入组名，返回 configs/ablation/<组>.json 字典；本函数只读，组名选项由 CLI 约束。
def configuration(group):
    return read(ROOT / "configs" / "ablation" / f"{group}.json")


# 用途：由组配置中的 train.run_name 推导标准实验目录。
# 调用时机：训练启动前防覆盖检查、评估定位权重和收集日志时调用。
# 输入输出与注意：返回 ROOT/outputs/runs/<名称> 的 Path，不创建目录，不通过 latest 指针猜测本组实验。
def run_dir(group):
    return ROOT / "outputs" / "runs" / configuration(group)["train"]["run_name"]


# 用途：检查三组公共训练条件相同且架构符合固定消融协议。
# 调用时机：prepare/check_lock 在启动或读取实验结果前调用，协议测试也直接调用。
# 输入输出与注意：允许预定结构字段不同，要求完整一轮、无 resume；不符抛 ValueError，成功无返回值，不启动训练。
def validate_configs():
    configs = [configuration(group)["train"] for group in IDS]
    variable = {"run_name", "d_ff", "position_encoding", "norm_type", "activation", "attention_impl"}
    common = [{k: v for k, v in config.items() if k not in variable} for config in configs]
    if any(item != common[0] for item in common[1:]):
        raise ValueError("三组公共训练参数不一致")
    expected = [(2304, "learned", "layernorm", "gelu", "manual"),
                (1536, "rope", "rmsnorm", "swiglu", "manual"),
                (1536, "rope", "rmsnorm", "swiglu", "sdpa")]
    keys = ("d_ff", "position_encoding", "norm_type", "activation", "attention_impl")
    if [tuple(c[k] for k in keys) for c in configs] != expected:
        raise ValueError("架构与 A/B/C 实验协议不一致")
    if configs[0]["epochs"] != 1 or configs[0]["resume"] is not None:
        raise ValueError("本协议要求随机初始化、完整 1 epoch")


# 用途：列出协议需要追踪的源码、配置、提示词和三套实际数据文件。
# 调用时机：fingerprint 计算当前实验输入身份时调用。
# 输入输出与注意：先校验数据长度、词表、指纹、样本数、字节大小和目录范围，再返回去重排序路径；会读元数据但不加载完整张量。
def tracked_paths():
    paths = list((ROOT / "src").glob("*.py"))
    paths += list((ROOT / "configs" / "ablation").glob("*.json"))
    paths += [BASE / "prompts.txt", Path(__file__).resolve(), ROOT / "configs" / "default.json"]
    tokenizer_hash = None
    for split in ("train", "val", "test"):
        meta_path = ROOT / "data" / f"{split}_seq256.meta.json"
        meta = read(meta_path)
        if meta["seq_len"] != 256 or meta["vocab_size"] != 21128 or meta["dtype"] not in ("uint16", "uint32"):
            raise ValueError(f"{split} 数据格式不符合本协议")
        current_hash = meta["tokenizer"]["vocab_sha256"]
        if tokenizer_hash is not None and current_hash != tokenizer_hash:
            raise ValueError("数据划分使用了不同 tokenizer")
        tokenizer_hash = current_hash
        if meta["num_samples"] != (meta["num_tokens"] - 1) // 256 or meta["num_samples"] < 1:
            raise ValueError(f"{split} 样本数无效")
        data = (meta_path.parent / meta["data_file"]).resolve()
        data.relative_to(ROOT / "data")
        if data.stat().st_size != meta["num_tokens"] * (2 if meta["dtype"] == "uint16" else 4):
            raise ValueError(f"数据文件大小不符：{split}")
        paths.extend([meta_path, data])
    return sorted(set(paths))


# 用途：为所有协议追踪文件建立相对路径到 SHA-256 的映射。
# 调用时机：首次 prepare 固定协议或 check_lock 检查漂移时调用。
# 输入输出与注意：返回字典，通过 tracked_paths 和 digest 读取真实文件；大 bin 也参与哈希，因此校验有磁盘开销。
def fingerprint():
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in tracked_paths()}


# 用途：获取用于关联历史执行记录的稳定协议身份。
# 调用时机：check_lock 返回身份、collect 对照执行与评估来源时调用。
# 输入输出与注意：优先返回锁内 protocol_id，否则哈希锁文件；读取已有 LOCK，不会自动修订协议或迁移历史记录。
def protocol_id():
    """协议修订可保留原实验身份，历史执行记录保持原样。"""
    return read(LOCK).get("protocol_id") or digest(LOCK)


# 用途：判断各组是否完整覆盖实际入口的默认配置字段。
# 调用时机：check_lock 决定默认配置内容变化是否可以忽略时调用。
# 输入输出与注意：检查 train/evaluate/benchmark_generation 三分区键集合，返回 bool；新增未覆盖默认键会令检查失败。
def defaults_fully_overridden():
    """只有每组完整覆盖三个实际入口的默认字段时，默认值变化才无影响。"""
    defaults = read(ROOT / "configs" / "default.json")
    for group in IDS:
        config = configuration(group)
        for section in ("train", "evaluate", "benchmark_generation"):
            if not isinstance(defaults.get(section), dict) or not isinstance(config.get(section), dict):
                return False
            if set(defaults[section]) - set(config[section]):
                return False
    return True


# 用途：确认当前配置、数据和源码与固定实验协议一致。
# 调用时机：训练/评估前后及报告收集前调用，是消融工具的入口守卫。
# 输入输出与注意：返回协议身份；缺锁或内容变化报错，仅对完整覆盖情况下的默认配置有例外，源码新增注释仍会被判为变化。
def check_lock():
    validate_configs()
    if not LOCK.exists():
        raise ValueError("尚未固定实验协议，请先运行 prepare")
    saved = read(LOCK)
    current = fingerprint()
    if ("configs/default.json" in saved["sha256"]
            and defaults_fully_overridden()):
        # 配置解析先浅合并默认分区，再由完整的组配置覆盖所有字段。
        current["configs/default.json"] = saved["sha256"]["configs/default.json"]
    if saved["sha256"] != current:
        changed = sorted(k for k in set(current) | set(saved["sha256"])
                         if current.get(k) != saved["sha256"].get(k))
        raise ValueError(f"数据/代码/配置已变化，不应混合比较：{changed}")
    return protocol_id()


# 用途：检查当前运行环境满足单进程、CUDA 和 BF16 的实验约束。
# 调用时机：check 命令、真实训练和评估启动前调用。
# 输入输出与注意：返回 Python/PyTorch/CUDA/GPU 信息字典；不满足抛错，设置 OpenMP 兼容变量后延迟导入 torch。
def runtime():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("需要支持 BF16 的 CUDA GPU，并安装 CUDA 版 PyTorch")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("本协议只支持单进程单卡")
    return {"python": sys.version, "executable": sys.executable,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "amp": "bf16"}


# 用途：用当前 Python 启动子命令，同时把合并输出写到终端和日志。
# 调用时机：消融 train/evaluate 需要运行 src 模块时调用。
# 输入输出与注意：输入参数列表和日志 Path；等待完成，异常时终止子进程，非零退出抛 RuntimeError；会覆盖该日志并实际执行业务。
def execute(arguments, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", *arguments], cwd=ROOT,
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"子进程失败，exit={code}，日志：{log_path}")


# 用途：在锁定协议和环境下启动指定组的一轮真实训练，记录执行状态。
# 调用时机：CLI action=train 且已指定 A/B/C 时调用。
# 输入输出与注意：防止覆盖已有实验，写 execution.json 与控制台日志；成功后再次核验锁，异常记 failed_or_interrupted，finally 记录耗时。
def train(group):
    lock_hash = check_lock()
    environment = runtime()
    directory = run_dir(group)
    if directory.exists():
        raise ValueError(f"实验目录已存在，避免覆盖：{directory}")
    state_path = OUT / group / "execution.json"
    state = {"group": group, "status": "running", "protocol_sha256": lock_hash,
             "environment": environment, "started_at": datetime.now(timezone.utc).isoformat()}
    atomic_json_dump(state, state_path)
    started = time.perf_counter()
    try:
        execute(["-m", "src.train", "--config", f"configs/ablation/{group}.json"], OUT / group / "train_console.log")
        check_lock()
        state["status"] = "completed"
    except BaseException as exc:
        state["status"] = "failed_or_interrupted"
        state["error"] = str(exc)
        raise
    finally:
        state["wall_seconds"] = time.perf_counter() - started
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json_dump(state, state_path)


# 用途：评估指定组已完成模型并保存结果来源证明。
# 调用时机：CLI action=evaluate 指定组和 split 时调用。
# 输入输出与注意：要求训练状态和协议一致；val 还运行生成基准，前后比较 checkpoint 哈希，保存配置/环境/来源，不更新模型。
def evaluate(group, split):
    lock_hash = check_lock()
    environment = runtime()
    execution = read(OUT / group / "execution.json")
    if execution["status"] != "completed" or execution["protocol_sha256"] != lock_hash:
        raise ValueError("必须先完整训练本组且协议一致")
    checkpoint = run_dir(group) / "checkpoints" / "best_model.pt"
    before = digest(checkpoint)
    config = f"configs/ablation/{group}.json"
    execute(["-m", "src.evaluate", "--config", config, "--split", split], OUT / group / f"evaluate_{split}.log")
    if split == "val":
        execute(["-m", "src.benchmark_generation", "--config", config], OUT / group / "generation.log")
    if before != digest(checkpoint):
        raise ValueError("评估期间 checkpoint 改变")
    atomic_json_dump({"checkpoint_sha256": before, "protocol_sha256": lock_hash,
                      "environment": environment, "configuration": configuration(group),
                      "completed_at": datetime.now(timezone.utc).isoformat()},
                     OUT / group / f"evaluation_{split}_provenance.json")


# 用途：汇总各组真实训练、评估和生成结果并检查可比性。
# 调用时机：报告生成和实验图表脚本读取现有实验时调用。
# 输入输出与注意：返回行字典列表；缺失保持 pending，校验来源、权重指纹、完整 token 预算及环境一致，不补造指标或执行新训练。
def collect():
    rows = []
    for group in IDS:
        row = {"group": group, "status": "pending", "parameters": 38251520 if group == "A" else 38089216}
        execution_path = OUT / group / "execution.json"
        if execution_path.exists():
            row.update(read(execution_path))
            if row.get("protocol_sha256") != protocol_id():
                raise ValueError(f"{group} 训练来源与本协议不一致")
        log = run_dir(group) / "training_log.jsonl"
        if log.exists():
            entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
            row["history"] = entries
            if entries:
                row["training"] = entries[-1]
        for split in ("val", "test"):
            report = run_dir(group) / "evaluations" / f"{split}_metrics.json"
            provenance = OUT / group / f"evaluation_{split}_provenance.json"
            if report.exists() and provenance.exists():
                proof = read(provenance)
                checkpoint = run_dir(group) / "checkpoints" / "best_model.pt"
                if proof["protocol_sha256"] != protocol_id() or proof["checkpoint_sha256"] != digest(checkpoint):
                    raise ValueError(f"{group}/{split} 结果来源过期")
                row[split] = read(report)["metrics"]
        generation = run_dir(group) / "generation_benchmark.json"
        if generation.exists() and "val" in row:
            row["generation"] = read(generation)
        if row["status"] == "completed" and row.get("training", {}).get("epoch") != 1:
            raise ValueError(f"{group} 训练预算不完整")
        rows.append(row)
    completed = [r for r in rows if r["status"] == "completed"]
    if completed:
        budgets = {r["training"]["tokens_seen"] for r in completed}
        expected_tokens = read(ROOT / "data" / "train_seq256.meta.json")["num_samples"] * 256
        if budgets != {expected_tokens}:
            raise ValueError("已完成实验未达到完整一轮 token 预算")
        environments = {json.dumps(r["environment"], sort_keys=True) for r in completed}
        if len(budgets) > 1 or len(environments) > 1:
            raise ValueError("已完成实验的 token 预算或运行环境不同，不应直接比较")
    return rows


# 用途：把已有或测试传入的实验记录保存为比较 JSON、PNG 和 Markdown。
# 调用时机：report 命令在 collect 后调用，报告测试用临时目录调用。
# 输入输出与注意：输入 rows 和目标目录；缺指标显示待运行，曲线按已见 token 绘制；会覆盖同名产物，不自行检查锁或加载权重。
def make_report(rows, destination):
    """从真实或测试传入的记录绘图；缺失值保留为空，不填造结果。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json_dump({"experiments": rows}, destination / "comparison.json")
    # 下列四个 lambda 都是绘图取值回调：输入一组记录 r，在后面的 value(r) 中调用。
    # 验证 PPL 回调返回 val.ppl；缺少评估时返回 None，使该组不绘制柱子。
    panels = [("Validation PPL", lambda r: r.get("val", {}).get("ppl")),
              # 训练吞吐回调读取整轮平均 tokens/s；不从日志中某次区间吞吐推算。
              ("Train tokens/s (epoch average)", lambda r: r.get("training", {}).get("train_tokens_per_second")),
              # 显存回调将 allocated 峰值字节数换算为 GiB；缺失返回 None，实测 0 保持 0。
              ("Peak CUDA allocated (GiB)", lambda r: (r.get("training", {}).get("peak_cuda_allocated_bytes") or 0) / 2**30 if r.get("training", {}).get("peak_cuda_allocated_bytes") is not None else None),
              # 生成吞吐回调读取基准 summary 的总 token/总时间；未运行基准时返回 None。
              ("Generation tokens/s", lambda r: r.get("generation", {}).get("summary", {}).get("tokens_per_second"))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (title, value) in zip(axes.flat, panels):
        available = [(r["group"], value(r)) for r in rows if value(r) is not None]
        if available:
            bars = ax.bar([p[0] for p in available], [p[1] for p in available], color="#377eb8")
            ax.bar_label(bars, fmt="%.2f", padding=3)
            ax.margins(y=0.18)
        else:
            ax.text(.5, .5, "Awaiting experiment results", ha="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(destination / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for row in rows:
        history = row.get("history", [])
        for axis, key in zip(axes, ("train_loss", "val_loss")):
            if history:
                axis.plot([e["tokens_seen"] / 1e6 for e in history], [e[key] for e in history], "o-", label=row["group"])
            axis.set(xlabel="Training tokens seen (millions)", ylabel=key)
            axis.grid(alpha=.2)
    for axis in axes:
        if axis.lines:
            axis.legend()
    fig.tight_layout()
    fig.savefig(destination / "loss_by_tokens.png", dpi=160)
    plt.close(fig)
    lines = ["# A/B/C 消融实验结果", "", "缺失指标表示尚未运行。单轮实验只有一个轮末点，不代表完整收敛曲线。", "", "| 组 | 状态 | Val PPL | Test PPL | 训练 tokens/s |", "|---|---|---:|---:|---:|"]
    for row in rows:
        values = [row.get("val", {}).get("ppl"), row.get("test", {}).get("ppl"), row.get("training", {}).get("train_tokens_per_second")]
        formatted = ["待运行" if v is None else f"{v:.4f}" for v in values]
        lines.append(f"| {row['group']} | {row['status']} | " + " | ".join(formatted) + " |")
    lines += ["", "![指标对比](comparison.png)", "", "![同 token 预算下的 Loss](loss_by_tokens.png)", "", "A→B 是架构组合对比；B→C 是 SDPA 对比。本实验固定 BF16，不单独证明 AMP 收益。", "", "训练吞吐为整轮平均，包含数据读取、反向传播及更新，不包含轮末验证和 checkpoint 保存。显存为 PyTorch allocated 峰值，不是整卡总占用。", "", "## 固定提示词输出"]
    for row in rows:
        for sample in row.get("generation", {}).get("samples", []):
            lines.extend(["", f"### {row['group']} · {sample['prompt']}", "", *["> " + line for line in sample["completion"].splitlines()]])
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# 用途：解析消融操作并分派准备协议、检查、训练、评估或报告。
# 调用时机：直接运行 experiments/ablation/run.py 时调用。
# 输入输出与注意：train/evaluate 必须有组名；prepare 不覆盖不兼容旧锁，report 先检查锁，只有对应执行分支会运行模型任务。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "train", "evaluate", "report"))
    parser.add_argument("group", nargs="?", choices=IDS)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.action in ("train", "evaluate") and not args.group:
        parser.error("请指定 A、B 或 C")
    if args.action == "prepare":
        validate_configs()
        if LOCK.exists():
            check_lock()
        else:
            atomic_json_dump({"created_at": datetime.now(timezone.utc).isoformat(), "sha256": fingerprint()}, LOCK)
        OUT.mkdir(parents=True, exist_ok=True)
        print(f"实验协议已固定：{LOCK}")
    elif args.action == "check":
        check_lock()
        print(json.dumps(runtime(), ensure_ascii=False, indent=2))
        for group in IDS:
            print(f"{group}: {run_dir(group)} ({'已存在' if run_dir(group).exists() else '可新建'})")
        print("数据、代码、配置检查通过；未启动训练。")
    elif args.action == "train":
        train(args.group)
    elif args.action == "evaluate":
        evaluate(args.group, args.split)
    else:
        check_lock()
        make_report(collect(), OUT)
        print(f"报告：{OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
