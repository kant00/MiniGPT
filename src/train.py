"""
训练 Decoder-only MiniGPT，支持单进程和 torchrun DDP。

主要功能：
  memmap 数据读取、AMP、梯度累积、warmup + cosine、梯度裁剪、
  原子 checkpoint、epoch 级续训、实验目录隔离和分布式指标汇总。

输入：
  data/train_seq<长度>.meta.json
  data/val_seq<长度>.meta.json

输出：
  outputs/runs/<run-name>/run_manifest.json
  outputs/runs/<run-name>/training_log.jsonl
  outputs/runs/<run-name>/checkpoints/*.pt
  outputs/latest_run.json

使用方式（在项目根目录执行）：
  python -m src.train --run_name baseline
  python -m src.train --run_name smoke --epochs 1 --batch_size 16
  python -m src.train --config configs/default.json --run_name baseline
  python -m src.train --resume latest
  torchrun --standalone --nproc_per_node=2 -m src.train --run_name ddp-2gpu

运行前必须先用 src.prepare_dataset 生成与 seq_len、tokenizer 一致的数据。
未显式传入的参数从配置文件的 train 分区读取，CLI 参数优先。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import inspect
import json
import math
import random
import time
from datetime import datetime, timezone
from contextlib import nullcontext
import logging
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    from .config_utils import load_config, parse_args_with_config
    from .experiment_utils import (
        atomic_json_dump,
        create_run_dir,
        infer_run_dir,
        resolve_latest_epoch_checkpoint,
        set_latest_run,
    )
    from .model import MiniGPTConfig, build_model
except ImportError:  # 兼容在 src 目录中直接运行脚本
    from config_utils import load_config, parse_args_with_config
    from experiment_utils import (
        atomic_json_dump,
        create_run_dir,
        infer_run_dir,
        resolve_latest_epoch_checkpoint,
        set_latest_run,
    )
    from model import MiniGPTConfig, build_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_PATH = OUTPUT_DIR / "training_log.jsonl"


# 用途：把连续 token 文件包装为 PyTorch 数据集，按需提供下一 token 监督样本。
# 调用时机：训练和独立评估创建 DataLoader 前实例化；测试用于验证预处理输出能读回。
# 输入输出与注意：新版读取 meta+memmap，也兼容带元数据的旧 pt；每条样本返回长度 seq_len 的输入与右移目标。
class TokenDataset(Dataset):
    """加载新版 memmap 数据，兼容带 tokenizer 元数据的旧版 .pt 数据。"""

    # 用途：解析数据描述并打开对应存储，检查文件大小、样本数或张量形状。
    # 调用时机：TokenDataset(path) 构造时自动调用，path 通常是 split 的 meta.json。
    # 输入输出与注意：memmap 只读映射，旧 pt 加载张量；元信息暴露词表、长度和 tokenizer 身份，不在此处逐 token 校验全部内容。
    def __init__(self, path: Path):
        self.path = Path(path)
        self.storage_format = "pt"
        if self.path.name.endswith(".meta.json"):
            metadata = json.loads(self.path.read_text(encoding="utf-8"))
            if metadata.get("storage_format") != "memmap":
                raise ValueError(f"不支持的数据格式：{metadata.get('storage_format')}")
            self.storage_format = "memmap"
            self.vocab_size = int(metadata["vocab_size"])
            self.seq_len = int(metadata["seq_len"])
            self.tokenizer_info = metadata.get("tokenizer")
            self.boundary_token_id = metadata.get("boundary_token_id")
            self.num_tokens = int(metadata["num_tokens"])
            self.num_samples = int(metadata["num_samples"])
            dtype = np.dtype(metadata["dtype"])
            data_path = self.path.parent / metadata["data_file"]
            expected_bytes = self.num_tokens * dtype.itemsize
            if not data_path.exists() or data_path.stat().st_size != expected_bytes:
                raise ValueError(f"Token 文件缺失或大小不匹配：{data_path}")
            expected_samples = max(0, (self.num_tokens - 1) // self.seq_len)
            if self.num_samples != expected_samples or self.num_samples < 1:
                raise ValueError(f"元数据中的样本数无效：{self.path}")
            self.data = np.memmap(
                data_path, dtype=dtype, mode="r", shape=(self.num_tokens,)
            )
        else:
            payload = torch.load(self.path, weights_only=True)
            self.data = payload["data"]
            self.vocab_size = int(payload["vocab_size"])
            self.seq_len = int(payload["seq_len"])
            self.tokenizer_info = payload.get("tokenizer")
            self.boundary_token_id = payload.get("boundary_token_id")
            if self.data.ndim != 2 or self.data.shape[1] != self.seq_len + 1:
                raise ValueError(f"数据形状无效：{tuple(self.data.shape)}")
            self.num_samples = len(self.data)
            self.num_tokens = self.num_samples * self.seq_len + 1

        if self.num_samples < 1:
            raise ValueError(f"数据集为空：{self.path}")
        logger.info(
            f"加载数据集：{self.path.name}，格式={self.storage_format}，"
            f"samples={self.num_samples:,}，tokens={self.num_tokens:,}"
        )

    # 用途：返回可读取的完整窗口数量。
    # 调用时机：len(dataset)、DataLoader 或 sampler 计算迭代批数时自动调用。
    # 输入输出与注意：返回 num_samples；不等于文章数，样本数通常为 floor((num_tokens-1)/seq_len)。
    def __len__(self):
        return self.num_samples

    # 用途：按样本下标读取窗口并拆成输入/目标。
    # 调用时机：DataLoader 取一个 batch 时反复调用，也可通过 dataset[index] 手动访问。
    # 输入输出与注意：memmap 从 index*seq_len 取 seq_len+1 项，复制成 int64；返回 chunk[:-1] 与 chunk[1:]，相邻窗口共用边界 token。
    def __getitem__(self, index):
        if self.storage_format == "memmap":
            start = index * self.seq_len
            chunk = torch.from_numpy(
                np.array(
                    self.data[start : start + self.seq_len + 1],
                    dtype=np.int64,
                    copy=True,
                )
            )
        else:
            chunk = self.data[index]
        return chunk[:-1], chunk[1:]

    # 用途：释放底层 mmap 文件句柄，避免数据文件持续被占用。
    # 调用时机：测试结束或调用方不再使用数据集时主动调用，析构也会尝试调用。
    # 输入输出与注意：无返回值；仅有 mmap 句柄时关闭，关闭后不要继续读取该数据集，Windows 清理临时文件前尤其需要。
    def close(self):
        mmap_handle = getattr(getattr(self, "data", None), "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    # 用途：在数据集对象销毁时尝试关闭映射文件。
    # 调用时机：Python 回收对象时自动触发，不是每读取一条样本就调用。
    # 输入输出与注意：委托 close；析构时机不确定，需要及时释放资源时应显式 close。
    def __del__(self):
        self.close()


# 用途：依据 split 与 seq_len 找到新版或兼容旧版数据路径。
# 调用时机：训练读取 train/val、评估读取 val/test 之前调用。
# 输入输出与注意：优先 meta.json，其次 pt；都没有则抛 FileNotFoundError；使用本模块 DATA_DIR。
def resolve_dataset_path(split: str, seq_len: int) -> Path:
    memmap_path = DATA_DIR / f"{split}_seq{seq_len}.meta.json"
    legacy_path = DATA_DIR / f"{split}_seq{seq_len}.pt"
    if memmap_path.exists():
        return memmap_path
    if legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(
        f"未找到 {split} 数据集（检查过 {memmap_path.name} 和 {legacy_path.name}），"
        "请先运行 prepare_dataset.py"
    )


# 用途：确保 train 和 val 的长度、词表规模及 tokenizer 指纹兼容。
# 调用时机：train 构造两个数据集后、建立模型之前调用。
# 输入输出与注意：不匹配或缺身份元数据时报错；成功返回 None，不检查原始文章泄漏或语料内容哈希。
def validate_dataset_pair(train_ds: TokenDataset, val_ds: TokenDataset) -> None:
    if train_ds.seq_len != val_ds.seq_len:
        raise ValueError("训练集和验证集的 seq_len 不一致")
    if train_ds.vocab_size != val_ds.vocab_size:
        raise ValueError("训练集和验证集的 vocab_size 不一致")
    if not train_ds.tokenizer_info or not val_ds.tokenizer_info:
        raise ValueError("数据集缺少 tokenizer 元数据，请重新运行 prepare_dataset.py")
    if (
        train_ds.tokenizer_info.get("vocab_sha256")
        != val_ds.tokenizer_info.get("vocab_sha256")
    ):
        raise ValueError("训练集和验证集使用了不同 tokenizer")


# 用途：设置 Python、NumPy 和 PyTorch 的随机起点，可选确定性算子设置。
# 调用时机：训练初始化时以 seed+rank 调用，随机状态测试也直接调用。
# 输入输出与注意：会修改进程全局随机状态；确定性使用 warn_only=True，不能据此保证跨环境逐位复现。
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# 用途：捕获当前 Python 和 PyTorch CPU/CUDA 随机状态，供续训恢复。
# 调用时机：每轮 checkpoint 保存前及训练状态测试中调用。
# 输入输出与注意：返回可保存字典；此实现未收集 NumPy RNG 状态，多卡训练另将各进程字典 gather。
def capture_rng_state():
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


# 用途：恢复 checkpoint 记录的随机序列位置，使后续抽样接着之前继续。
# 调用时机：续训加载优化状态后、下一轮训练开始前调用。
# 输入输出与注意：修改全局 RNG，缺 state 时警告并返回；CPU 状态先转回 CPU，CUDA 可用时恢复对应列表。
def restore_rng_state(state) -> None:
    if not state:
        logger.warning("Checkpoint 缺少 RNG 状态，续训结果可能无法完全复现")
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


# 用途：先保存临时 checkpoint，再替换正式目标文件。
# 调用时机：训练轮末写 epoch/best_model 或测试保存状态时调用。
# 输入输出与注意：输入可由 torch.save 序列化的 payload 和 Path；建立父目录并写 .tmp，成功后 replace，不提供并发锁或整轮事务。
def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


# 用途：把续训参数转换为 checkpoint 文件路径。
# 调用时机：train 收到非空 resume 时调用；DDP 非主进程也解析自己的恢复路径。
# 输入输出与注意：latest 使用最近实验中修改时间最新的 epoch 文件；普通相对路径基于项目根目录，不是默认 best_model。
def resolve_resume_path(resume: str, output_dir: Path = None) -> Path:
    output_dir = Path(output_dir or OUTPUT_DIR)
    if resume == "latest":
        return resolve_latest_epoch_checkpoint(output_dir)
    path = Path(resume)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"未找到续训 checkpoint：{path}")
    return path


# 用途：为优化器建立先预热、后余弦衰减的学习率调度器。
# 调用时机：train 创建 AdamW 后调用，续训时还会加载已保存 scheduler 状态。
# 输入输出与注意：输入总更新步、预热步、最低倍率，返回 LambdaLR；预热步会限制在 total_steps-1 内，total_steps 必须为正。
def build_warmup_cosine_scheduler(
    optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float
):
    if total_steps < 1:
        raise ValueError("total_steps 必须大于 0")
    warmup_steps = min(max(0, warmup_steps), max(0, total_steps - 1))

    # 用途：根据当前调度步计算基础学习率应乘的倍率。
    # 调用时机：LambdaLR 初始化和 scheduler.step 时作为闭包回调，不由每个前向批次直接调用。
    # 输入输出与注意：读取外层预热步/总步/最低倍率；预热线性增加，之后余弦下降，返回倍率而非绝对学习率。
    def lr_lambda(step: int):
        if warmup_steps and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# 用途：按请求精度与实际设备决定 autocast 使用的数据类型。
# 调用时机：训练和评估开始时、创建 scaler 与进入计算循环之前调用。
# 输入输出与注意：CPU 或 off 返回 None；CUDA auto 优先 BF16，否则 FP16；显式 BF16 不支持时抛错。
def resolve_amp_dtype(mode: str, device: torch.device):
    if device.type != "cuda" or mode == "off":
        return None
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("当前 CUDA 设备不支持 BF16")
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# 用途：创建兼容不同 PyTorch AMP 接口的梯度缩放器。
# 调用时机：训练确定 amp_dtype 后调用；FP16 启用，其他精度传 enabled=False。
# 输入输出与注意：返回 scaler 对象，兼容分支处理构造签名 TypeError；缩放用于减少 FP16 小梯度下溢。
def create_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # PyTorch 2.2 兼容
        return torch.cuda.amp.GradScaler(enabled=enabled)


# 用途：统一提供混合精度上下文或无操作上下文。
# 调用时机：训练前向/交叉熵和评估循环的 with 语句调用。
# 输入输出与注意：输入设备和 dtype；dtype=None 返回 nullcontext，否则 torch.autocast，不负责关闭梯度或切换 eval。
def autocast_context(device: torch.device, dtype):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


# 用途：根据 torchrun 环境变量初始化单进程或 DDP 设备与通信。
# 调用时机：train 开始时调用，普通 python 启动通常进入单进程分支。
# 输入输出与注意：返回 enabled/world_size/rank/local_rank/is_main/device 字典；多进程选择 CUDA/NCCL 或 CPU/Gloo，会建立通信组。
def initialize_distributed():
    """由 torchrun 环境变量自动启用 DDP，普通 python 启动保持单进程。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("当前 PyTorch 不支持 distributed")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "enabled": world_size > 1,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
        "device": device,
    }


# 用途：遍历验证 loader，按总 token 加权计算平均交叉熵和 PPL。
# 调用时机：新模型初始化基线及每轮训练结束后的验证调用。
# 输入输出与注意：临时切 eval 和 inference_mode，结束恢复原训练模式；DDP 汇总 NLL/token，PPL=exp(min(loss,20))，可限制前 max_batches 批。
def compute_ppl(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype=None,
    max_batches: int = None,
):
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for batch_index, (input_ids, targets) in enumerate(loader, 1):
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast_context(device, amp_dtype):
                logits = model(input_ids)
                batch_size, seq_len, vocab_size = logits.shape
                loss = loss_fn(
                    logits.reshape(batch_size * seq_len, vocab_size),
                    targets.reshape(batch_size * seq_len),
                )
            total_nll += loss.float().item()
            total_tokens += targets.numel()
            if max_batches is not None and batch_index >= max_batches:
                break
    if was_training:
        model.train()
    if dist.is_available() and dist.is_initialized():
        totals = torch.tensor(
            [total_nll, total_tokens], dtype=torch.float64, device=device
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_nll, total_tokens = totals.tolist()
    avg_loss = total_nll / total_tokens
    return {"loss": avg_loss, "ppl": math.exp(min(avg_loss, 20)), "tokens": total_tokens}


# 用途：统一设置批量读取、采样、进程数和数据搬运选项。
# 调用时机：训练建立 train/val loader、独立评估建立 loader 时调用。
# 输入输出与注意：返回 DataLoader；有 sampler 时不另开 shuffle，CUDA 使用 pin_memory，workers>0 启用预取与持久 worker，不丢弃尾批。
def make_dataloader(dataset, batch_size, shuffle, num_workers, device, sampler=None):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


# 用途：组织数据检查、建模、优化、梯度累积、轮末验证、日志和完整 checkpoint 保存。
# 调用时机：main 接收 CLI 后调用；微型流水线测试也可直接传参数调用。
# 输入输出与注意：返回实验 run_dir；新训练建立目录，续训恢复 epoch 级状态并校验关键设置；会写文件和更新权重，直接调用不自动读取 JSON 默认值。
def train(
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    seq_len: int = 256,
    num_workers: int = 0,
    seed: int = 42,
    deterministic: bool = False,
    resume: str = None,
    grad_accum_steps: int = 1,
    warmup_ratio: float = 0.02,
    min_lr_ratio: float = 0.1,
    amp: str = "auto",
    compile_model: bool = False,
    log_interval: int = 100,
    baseline_batches: int = 20,
    model_overrides: dict = None,
    run_name: str = None,
    output_dir: Path = None,
    config_path: Path = None,
):
    if epochs < 1 or batch_size < 1 or grad_accum_steps < 1 or num_workers < 0:
        raise ValueError("epochs、batch_size、grad_accum_steps 必须大于 0")
    if not 0.0 <= warmup_ratio < 1.0 or not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("warmup_ratio 必须位于 [0,1)，min_lr_ratio 必须位于 [0,1]")

    distributed = initialize_distributed()
    is_main = distributed["is_main"]
    output_dir = Path(output_dir or OUTPUT_DIR)
    resume_path = resolve_resume_path(resume, output_dir) if resume else None
    run_dir = None
    if is_main:
        if resume_path:
            run_dir = infer_run_dir(resume_path)
            set_latest_run(output_dir, run_dir)
        else:
            run_dir = create_run_dir(output_dir, run_name)
    if distributed["enabled"]:
        run_dir_holder = [str(run_dir) if is_main else None]
        dist.broadcast_object_list(run_dir_holder, src=0)
        run_dir = Path(run_dir_holder[0])
        if resume and not is_main:
            resume_path = resolve_resume_path(resume, output_dir)
    checkpoint_dir = run_dir / "checkpoints"
    log_path = run_dir / "training_log.jsonl"

    set_seed(seed + distributed["rank"], deterministic)
    device = distributed["device"]
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_dtype = resolve_amp_dtype(amp, device)
    logger.info(
        f"设备={device}，AMP={amp_dtype}，seed={seed}，"
        f"grad_accum_steps={grad_accum_steps}，rank={distributed['rank']}/"
        f"{distributed['world_size']}"
    )

    train_ds = TokenDataset(resolve_dataset_path("train", seq_len))
    val_ds = TokenDataset(resolve_dataset_path("val", seq_len))
    validate_dataset_pair(train_ds, val_ds)
    if train_ds.seq_len != seq_len:
        raise ValueError("命令行 seq_len 与数据集不一致")
    vocab_size = train_ds.vocab_size

    train_sampler = DistributedSampler(
        train_ds, shuffle=True, seed=seed
    ) if distributed["enabled"] else None
    val_sampler = DistributedSampler(
        val_ds, shuffle=False
    ) if distributed["enabled"] else None
    train_loader = make_dataloader(
        train_ds, batch_size, True, num_workers, device, train_sampler
    )
    val_loader = make_dataloader(
        val_ds, batch_size, False, num_workers, device, val_sampler
    )

    checkpoint = (
        torch.load(resume_path, map_location=device, weights_only=True)
        if resume_path
        else None
    )
    if checkpoint and checkpoint.get("model_config"):
        model_config = checkpoint["model_config"]
    else:
        model_config = MiniGPTConfig(
            vocab_size=vocab_size,
            seq_len=seq_len,
            **(model_overrides or {}),
        ).to_dict()
    raw_model = build_model(
        vocab_size=vocab_size, seq_len=seq_len, model_config=model_config
    ).to(device)
    logger.info(
        f"模型参数量={raw_model.count_parameters() / 1e6:.2f}M，"
        f"架构={raw_model.config.to_dict()}"
    )

    decay_params = [parameter for parameter in raw_model.parameters() if parameter.dim() >= 2]
    no_decay_params = [parameter for parameter in raw_model.parameters() if parameter.dim() < 2]
    optimizer_kwargs = {"lr": lr, "betas": (0.9, 0.95)}
    if device.type == "cuda" and "fused" in inspect.signature(AdamW).parameters:
        optimizer_kwargs["fused"] = True
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        **optimizer_kwargs,
    )

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = round(total_steps * warmup_ratio)
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_steps, warmup_steps, min_lr_ratio
    )
    scaler = create_grad_scaler(amp_dtype == torch.float16)
    loss_fn = nn.CrossEntropyLoss()

    training_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "seq_len": seq_len,
        "seed": seed,
        "deterministic": deterministic,
        "grad_accum_steps": grad_accum_steps,
        "warmup_ratio": warmup_ratio,
        "min_lr_ratio": min_lr_ratio,
        "amp": amp,
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_ppl = float("inf")
    global_step = 0
    tokens_seen = 0
    start_epoch = 1

    if checkpoint:
        if checkpoint.get("vocab_size") != vocab_size or checkpoint.get("seq_len") != seq_len:
            raise ValueError("Checkpoint 与当前数据集的 vocab_size 或 seq_len 不一致")
        checkpoint_tokenizer = checkpoint.get("tokenizer")
        if (
            not checkpoint_tokenizer
            or checkpoint_tokenizer.get("vocab_sha256")
            != train_ds.tokenizer_info.get("vocab_sha256")
        ):
            raise ValueError("Checkpoint 与当前数据集使用了不同 tokenizer")
        saved_config = checkpoint.get("training_config") or {}
        for key in (
            "epochs", "batch_size", "lr", "weight_decay", "grad_clip",
            "grad_accum_steps", "warmup_ratio", "min_lr_ratio", "amp",
        ):
            if key in saved_config and saved_config[key] != training_config[key]:
                raise ValueError(
                    f"续训参数 {key} 不一致：checkpoint={saved_config[key]!r}, "
                    f"current={training_config[key]!r}"
                )
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        best_val_ppl = float(
            checkpoint.get("best_val_ppl", checkpoint.get("val_ppl", float("inf")))
        )
        rng_states = checkpoint.get("rng_states")
        rng_state = (
            rng_states[distributed["rank"]]
            if rng_states and distributed["rank"] < len(rng_states)
            else checkpoint.get("rng_state")
        )
        restore_rng_state(rng_state)
        logger.info(
            f"从 {resume_path.name} 续训：epoch={start_epoch}，step={global_step}"
        )
    elif is_main:
        log_path.write_text("", encoding="utf-8")
        if config_path is not None:
            atomic_json_dump(load_config(config_path), run_dir / "source_config.json")
        atomic_json_dump(
            {
                "format_version": 1,
                "run_name": run_dir.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_config": (
                    str(Path(config_path).resolve()) if config_path is not None else None
                ),
                "device": str(device),
                "torch_version": torch.__version__,
                "model_config": raw_model.config.to_dict(),
                "training_config": training_config,
                "tokenizer": train_ds.tokenizer_info,
                "train_dataset": str(train_ds.path.resolve()),
                "val_dataset": str(val_ds.path.resolve()),
            },
            run_dir / "run_manifest.json",
        )

    model = raw_model
    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("当前 PyTorch 不支持 torch.compile")
        model = torch.compile(raw_model)
        logger.info("已启用 torch.compile")

    if distributed["enabled"]:
        model = DistributedDataParallel(
            model,
            device_ids=[distributed["local_rank"]] if device.type == "cuda" else None,
        )

    if not checkpoint and baseline_batches > 0:
        baseline = compute_ppl(
            model, val_loader, device, amp_dtype, max_batches=baseline_batches
        )
        logger.info(
            f"初始化基线（{baseline['tokens']:,} tokens）："
            f"loss={baseline['loss']:.4f}，PPL={baseline['ppl']:.1f}"
        )

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        else:
            # 固定每轮样本顺序，使不同架构的初始化随机数消耗不影响消融数据顺序。
            train_loader.sampler.generator = torch.Generator().manual_seed(seed + epoch)
        model.train()
        epoch_nll = 0.0
        epoch_tokens = 0
        interval_tokens = 0
        interval_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for micro_step, (input_ids, targets) in enumerate(train_loader, 1):
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_tokens = targets.numel()
            window_start = ((micro_step - 1) // grad_accum_steps) * grad_accum_steps
            current_window = min(grad_accum_steps, len(train_loader) - window_start)
            should_step = (
                micro_step % grad_accum_steps == 0 or micro_step == len(train_loader)
            )
            sync_context = (
                model.no_sync()
                if distributed["enabled"] and not should_step
                else nullcontext()
            )

            with sync_context:
                with autocast_context(device, amp_dtype):
                    logits = model(input_ids)
                    batch_rows, batch_seq_len, current_vocab = logits.shape
                    raw_loss = loss_fn(
                        logits.reshape(batch_rows * batch_seq_len, current_vocab),
                        targets.reshape(batch_rows * batch_seq_len),
                    )
                    scaled_loss = raw_loss / current_window
                scaler.scale(scaled_loss).backward()

            epoch_nll += raw_loss.detach().float().item() * batch_tokens
            epoch_tokens += batch_tokens
            interval_tokens += batch_tokens

            if not should_step:
                continue
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if is_main and global_step % log_interval == 0:
                elapsed = max(time.perf_counter() - interval_start, 1e-6)
                avg_loss = epoch_nll / epoch_tokens
                logger.info(
                    f"Epoch {epoch}/{epochs} step={global_step} "
                    f"loss={avg_loss:.4f} ppl={math.exp(min(avg_loss, 20)):.1f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} grad={float(grad_norm):.3f} "
                    f"tokens/s≈{interval_tokens * distributed['world_size'] / elapsed:,.0f}"
                )
                interval_tokens = 0
                interval_start = time.perf_counter()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        peak_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        if distributed["enabled"]:
            train_totals = torch.tensor(
                [epoch_nll, epoch_tokens], dtype=torch.float64, device=device
            )
            dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
            epoch_nll, epoch_tokens = train_totals.tolist()
        validation = compute_ppl(model, val_loader, device, amp_dtype)
        epoch_tokens = int(epoch_tokens)
        tokens_seen += epoch_tokens
        train_loss = epoch_nll / epoch_tokens
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = validation["ppl"]
        if is_main:
            logger.info(
                f"Epoch {epoch} 完成 | train PPL={train_ppl:.1f} | "
                f"val PPL={val_ppl:.1f} | tokens_seen={tokens_seen:,}"
            )

        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "val_loss": validation["loss"],
            "val_ppl": val_ppl,
            "lr": scheduler.get_last_lr()[0],
            "train_seconds": train_seconds,
            "train_tokens_per_second": epoch_tokens / max(train_seconds, 1e-9),
            "peak_cuda_allocated_bytes": peak_memory_bytes,
        }
        if is_main:
            with open(log_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        local_rng_state = capture_rng_state()
        if distributed["enabled"]:
            rng_states = [None] * distributed["world_size"] if is_main else None
            dist.gather_object(local_rng_state, rng_states, dst=0)
        else:
            rng_states = [local_rng_state]

        checkpoint_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_ppl": val_ppl,
            "best_val_ppl": min(best_val_ppl, val_ppl),
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "tokenizer": train_ds.tokenizer_info,
            "model_config": raw_model.config.to_dict(),
            "training_config": training_config,
            "rng_state": rng_states[0] if is_main else None,
            "rng_states": rng_states if is_main else None,
            "world_size": distributed["world_size"],
        }
        if is_main:
            ckpt_path = checkpoint_dir / f"epoch{epoch}_ppl{val_ppl:.1f}.pt"
            atomic_torch_save(checkpoint_payload, ckpt_path)
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                checkpoint_payload["best_val_ppl"] = best_val_ppl
                atomic_torch_save(checkpoint_payload, checkpoint_dir / "best_model.pt")

    if distributed["enabled"]:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        logger.info(f"训练完成，最优 val PPL={best_val_ppl:.1f}，实验目录={run_dir}")
    return run_dir


# 用途：声明训练与模型参数，合并 JSON 配置和 CLI，再启动 train。
# 调用时机：python -m src.train 或脚本作为主程序执行时调用，导入模块不会自动训练。
# 输入输出与注意：无业务返回值；结构字段组成 model_overrides，config_path 用于来源记录，各参数的用途见下方逐项注释。
def main():
    # 参数分为：训练规模与优化、数据加载与复现、续训与加速、日志与实验、模型结构。
    # 实际优先级：显式命令行参数 > 指定 JSON 的 train 分区 > 默认 JSON > 此处 default。
    # 因此只修改下面的 default，不一定会改变运行参数；通常应修改 configs/default.json。
    parser = argparse.ArgumentParser()

    # 训练计划的总轮数，必须 >= 1；每轮遍历训练集，并在轮末验证、保存 checkpoint。
    # 续训时表示原计划总轮数，不是额外训练几轮；当前实现要求与 checkpoint 的已保存值一致。
    parser.add_argument("--epochs", type=int, default=3)
    # 每个进程一次前向/反向处理的样本数，必须 >= 1；每条样本包含 seq_len 个输入 token。
    # 增大通常会增加显存占用；多卡时这是每个进程的 batch，不是所有卡合计。
    parser.add_argument("--batch_size", type=int, default=32)
    # AdamW 的基础学习率，3e-4 即 0.0003；控制参数更新幅度。
    # 实际学习率还会经过 warmup 和余弦调度，训练过程中并非始终等于此值。
    parser.add_argument("--lr", type=float, default=3e-4)
    # AdamW 权重衰减系数，用于抑制权重过度增大；0 表示不衰减。
    # 本实现只对二维及以上参数（如权重矩阵）衰减，偏置和归一化的一维参数不衰减。
    parser.add_argument("--weight_decay", type=float, default=0.1)
    # 全部参数梯度的总范数上限，超过时整体缩放梯度，缓解梯度爆炸；通常使用正数。
    # 裁剪发生在更新前、FP16 梯度还原后；这里设为 0 不会关闭裁剪，而会把有限梯度缩至零。
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # 每条样本的输入 token 数，也作为模型上下文上限；必须与预处理数据的 seq_len 一致。
    # 用于寻找 train/val_seq<长度>.meta.json；改大后需要对应数据，注意力计算与显存开销也会增加。
    parser.add_argument("--seq_len", type=int, default=256)
    # 每个 DataLoader 的数据加载子进程数，必须 >= 0；0 表示由训练主进程加载。
    # 大于 0 时启用预取和持久 worker；能否加速取决于磁盘与 CPU，并非越多越好。
    parser.add_argument("--num_workers", type=int, default=0)
    # Python、NumPy、PyTorch 的随机种子，影响模型初始化和训练打乱顺序等。
    # 多进程训练以 seed+rank 初始化各进程；这不会重新划分已经生成的 train/val/test 数据。
    parser.add_argument("--seed", type=int, default=42)
    # 开启确定性相关设置，尽量提高重复运行的一致性，可能影响性能。
    # 实现使用 warn_only=True，遇到不支持确定性的算子仅警告，不能保证跨设备/版本逐位一致。
    parser.add_argument("--deterministic", action="store_true")
    # 反向开关：令 deterministic=False，覆盖 JSON 中的开启设置。
    # 它使本次初始化跳过确定性设置，不会主动撤销同一 Python 进程此前已设置的全局状态。
    parser.add_argument("--non_deterministic", dest="deterministic", action="store_false", help=argparse.SUPPRESS)
    # 续训 checkpoint 路径；None 表示新建实验，相对路径按项目根目录解析。
    # latest 选择最近实验中按修改时间最新的 epoch 文件，并从所存 epoch 的下一轮继续。
    # 会恢复模型、优化器、调度器等状态；不是只加载权重，也不是恢复某轮中间的样本位置。
    parser.add_argument("--resume", type=str, default=None)
    # 累积多少个小批次的梯度再更新一次参数，必须 >= 1；用于控制单次前向的显存需求。
    # 通常有效全局 batch ≈ batch_size × grad_accum_steps × 进程数；最后不足的窗口也会更新。
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    # 总更新步数中用于学习率预热的比例，范围 [0,1)；默认 0.02 即约前 2% 步逐渐升温。
    # 设为 0 不预热；比例换算为整数步，小实验可能四舍五入为 0，之后使用余弦下降。
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    # 余弦调度的最低学习率相对基础 lr 的比例，范围 [0,1]。
    # 例如 lr=0.0003、比例=0.1 时下限为 0.00003；这是倍率，不是绝对学习率。
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    # 混合精度模式：off 关闭；CUDA 上 auto 优先 BF16，不支持时使用 FP16。
    # bf16 显式要求设备支持；fp16 启用梯度缩放。CPU 路径不启用 AMP，不受所选模式影响。
    parser.add_argument("--amp", choices=("auto", "off", "bf16", "fp16"), default="auto")
    # 使用 torch.compile 编译模型计算，可能提高后续吞吐，但首次编译有开销，依赖运行环境支持。
    # CLI 名称为 --compile，JSON 和 train() 中对应名称是 compile_model；默认关闭。
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    # 反向开关：令 compile_model=False，可覆盖 JSON 中的开启设置。
    parser.add_argument("--no_compile", dest="compile_model", action="store_false", help=argparse.SUPPRESS)
    # 每隔多少个 global_step 在主进程打印训练信息；应设为正整数，0 会导致取模错误。
    # global_step 按梯度累积的更新边界计数，不是每个小批次；每轮 JSON 日志另行保存。
    parser.add_argument("--log_interval", type=int, default=100)
    # 新训练开始前，用随机初始化模型评估前多少个验证 batch，作为初始 Loss/PPL 参照。
    # 0 关闭此检查；续训时跳过。它不限制正式训练批数，也不限制每轮结束时的完整验证。
    parser.add_argument("--baseline_batches", type=int, default=20)
    # 新实验目录名，保存在 outputs/runs/<名称>/；None 使用时间戳，显式重名会报错。
    # 名称限 1~80 字符，以英文字母/数字开头，后续还允许 ._-；续训沿用原实验目录。
    parser.add_argument("--run_name", type=str, default=None)

    # 以下控制新模型结构，经 model_overrides 传入；续训通常使用 checkpoint 已保存的架构。
    # 当前默认 JSON 是 512 维、8 头、8 层、FFN 1536，与下面解析器的兜底值不同。
    # token embedding 及各层隐藏向量的维度；增大通常增加模型参数、计算量和显存占用。
    # 必须能被 n_heads 整除；词表大小由数据集提供，无需在这里另设 vocab_size。
    parser.add_argument("--d_model", type=int, default=384)
    # 每层注意力头数；每个头的维度 = d_model/n_heads。
    # 使用 RoPE 时单头维度必须为偶数；固定 d_model 时增加头数并不等于增加隐藏宽度。
    parser.add_argument("--n_heads", type=int, default=6)
    # Transformer Block 的数量；每个块包含注意力和前馈网络，层数越多模型越深。
    # 增加层数通常增加参数量、训练开销和表达能力，但不保证最终效果必然更好。
    parser.add_argument("--n_layers", type=int, default=6)
    # 每个块中前馈网络的中间维度，最后再投影回 d_model；增大可增加前馈层容量。
    # 相同 d_ff 下 SwiGLU 有三组权重矩阵，GELU 有两组，因此两种激活的参数量不相同。
    parser.add_argument("--d_ff", type=int, default=1024)
    # 训练时随机丢弃部分中间值的概率，用于正则化；0 表示关闭，通常取 [0,1) 内的小值。
    # 应用于 embedding、注意力和前馈相关分支；model.eval() 时关闭，不会删除模型参数。
    parser.add_argument("--dropout", type=float, default=0.1)
    # 位置编码：rope 对注意力 Q/K 做位置相关旋转；learned 在输入上加可训练位置向量。
    # 两者都受本项目 seq_len 上限约束；选择 RoPE 不会自动获得无限上下文。
    parser.add_argument("--position_encoding", choices=("rope", "learned"), default="rope")
    # 归一化方式：rmsnorm 按均方根缩放、不减均值；layernorm 按均值和方差归一化。
    # 用于各块子层前及模型最终输出前，影响数值尺度和模型结构。
    parser.add_argument("--norm_type", choices=("rmsnorm", "layernorm"), default="rmsnorm")
    # 前馈网络形式：swiglu 使用 SiLU 门控分支乘以另一投影；gelu 使用普通 GELU 激活。
    # 这是网络结构选择，不是优化器或生成时的采样方式。
    parser.add_argument("--activation", choices=("swiglu", "gelu"), default="swiglu")
    # 注意力实现：sdpa 调用 PyTorch 的 scaled_dot_product_attention；manual 手写分数、掩码和 softmax。
    # 两者实现相同因果注意力，便于性能/正确性对照；SDPA 实际内核取决于设备、精度和形状。
    parser.add_argument("--attention_impl", choices=("sdpa", "manual"), default="sdpa")
    # RoPE 旋转频率的基数，通常保持 10000.0；仅 position_encoding=rope 时使用。
    # 它控制不同维度的旋转频率分布，不是位置个数，调大也不会改变 seq_len 的硬性上限。
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    # 公共解析器还会添加 --config：选择 JSON 配置文件，默认 configs/default.json，读取 train 分区。
    args = parse_args_with_config(parser, "train")
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        seed=args.seed,
        deterministic=args.deterministic,
        resume=args.resume,
        grad_accum_steps=args.grad_accum_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        amp=args.amp,
        compile_model=args.compile_model,
        log_interval=args.log_interval,
        baseline_batches=args.baseline_batches,
        run_name=args.run_name,
        # 配置来源路径用于保存 source_config.json 和记录实验清单；CLI 覆盖值另在实际配置中记录。
        config_path=args.config,
        # 将模型结构参数集中传入；其余参数控制训练过程，而不是网络层的定义。
        model_overrides={
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "position_encoding": args.position_encoding,
            "norm_type": args.norm_type,
            "activation": args.activation,
            "attention_impl": args.attention_impl,
            "rope_theta": args.rope_theta,
        },
    )


if __name__ == "__main__":
    main()
