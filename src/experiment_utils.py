"""
统一管理训练实验目录、latest 指针、checkpoint 解析和原子 JSON 写入。

目录约定：
  outputs/runs/<run-name>/checkpoints/
  outputs/latest_run.json

本文件是内部工具模块，不提供命令行入口。典型 Python 用法：

  from pathlib import Path
  from src.experiment_utils import create_run_dir, resolve_checkpoint
  run_dir = create_run_dir(Path("outputs"), "baseline")
  checkpoint = resolve_checkpoint(None, Path("outputs"))

显式 run 名称已存在时不会覆盖；latest 指针只允许指向 output_dir 内部，
并使用临时文件替换方式原子发布，避免路径越界和半写入 JSON。
"""

import json
import re
from datetime import datetime
from pathlib import Path


RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


# 用途：统一以 UTF-8 格式保存 JSON，并用临时文件替换避免半写入正式文件。
# 调用时机：训练清单、latest 指针、评估及基准报告保存时调用。
# 输入输出与注意：自动建立父目录，写 .part 再 replace；无返回值，会覆盖同名目标，没有跨文件事务或并发写锁。
def atomic_json_dump(payload, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_suffix(path.suffix + ".part")
    part_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    part_path.replace(path)


# 用途：验证实验名能作为受约束的单层目录名称。
# 调用时机：create_run_dir 建目录前调用，实验目录测试也直接调用。
# 输入输出与注意：返回原字符串；仅允许字母/数字开头和后续 ._-，总长 1~80，拒绝中文、斜杠和路径穿越形式。
def validate_run_name(run_name: str) -> str:
    if not RUN_NAME_PATTERN.fullmatch(run_name or "") or run_name in {".", ".."}:
        raise ValueError(
            "run_name 只能包含字母、数字、点、下划线和连字符，且长度不超过 80"
        )
    return run_name


# 用途：为新实验创建独立目录和 checkpoints，并更新最近实验指针。
# 调用时机：新训练启动时由主进程调用，也用于实验目录测试。
# 输入输出与注意：返回 Path；显式重名报错，自动时间戳碰撞则加后缀；latest 在训练成功前就更新，不代表已有 best_model。
def create_run_dir(output_dir: Path, run_name: str = None) -> Path:
    """创建互不覆盖的实验目录；未命名时使用时间戳。"""
    output_dir = Path(output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    base_name = validate_run_name(
        run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    candidate = runs_dir / base_name
    if run_name and candidate.exists():
        raise FileExistsError(f"实验目录已存在：{candidate}")
    suffix = 1
    while candidate.exists():
        candidate = runs_dir / f"{base_name}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=False)
    (candidate / "checkpoints").mkdir()
    set_latest_run(output_dir, candidate)
    return candidate


# 用途：把最近选定的实验位置写成输出目录内的相对路径。
# 调用时机：新建实验或从原实验恢复训练时调用。
# 输入输出与注意：输入 output_dir/run_dir，要求后者位于前者内；写 latest_run.json，不检查实验是否完成。
def set_latest_run(output_dir: Path, run_dir: Path) -> None:
    output_dir = Path(output_dir).resolve()
    run_dir = Path(run_dir).resolve()
    try:
        relative = run_dir.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError("run_dir 必须位于 output_dir 内") from exc
    atomic_json_dump({"run_dir": relative.as_posix()}, output_dir / "latest_run.json")


# 用途：读取最近实验指针并校验目录位置。
# 调用时机：默认找 checkpoint 或续训 latest 时调用。
# 输入输出与注意：返回绝对 Path；拒绝绝对指针、包含 ..、解析越界和不存在目录；不比较多个实验的模型效果。
def latest_run_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir).resolve()
    pointer = output_dir / "latest_run.json"
    if not pointer.exists():
        raise FileNotFoundError(f"未找到最近实验指针：{pointer}")
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    relative = Path(payload.get("run_dir", ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"最近实验指针无效：{pointer}")
    run_dir = (output_dir / relative).resolve()
    try:
        run_dir.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError(f"最近实验指针越界：{pointer}") from exc
    if not run_dir.is_dir():
        raise FileNotFoundError(f"最近实验目录不存在：{run_dir}")
    return run_dir


# 用途：从 checkpoints 下的权重路径推导所属实验目录。
# 调用时机：续训定位原实验、评估保存报告和绘图、生成基准确定输出目录时调用。
# 输入输出与注意：要求父目录名为 checkpoints，返回其父目录；不读取权重内容，不要求一定放在标准 outputs/runs 内。
def infer_run_dir(checkpoint_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError("checkpoint 必须位于实验的 checkpoints 目录中")
    return checkpoint_path.parent.parent


# 用途：为评估或生成解析显式权重路径或默认最佳模型。
# 调用时机：evaluate、generate、benchmark_generation 的 main 加载模型前调用。
# 输入输出与注意：省略时取 latest 实验下 filename，默认 best_model.pt；相对显式路径按 output_dir 父目录处理，字符串 latest 在此不是特殊参数。
def resolve_checkpoint(
    checkpoint: str = None, output_dir: Path = Path("outputs"), filename: str = "best_model.pt"
) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = Path(output_dir).resolve().parent / path
    else:
        path = latest_run_dir(output_dir) / "checkpoints" / filename
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"未找到 checkpoint：{path}")
    return path


# 用途：在最近实验中找修改时间最新的轮末 checkpoint。
# 调用时机：train 的 resume=latest 分支调用。
# 输入输出与注意：匹配 epoch*_ppl*.pt 后按 mtime 取最大值，缺文件报错；不是按 PPL 最低或文件名 epoch 数字排序。
def resolve_latest_epoch_checkpoint(output_dir: Path) -> Path:
    checkpoint_dir = latest_run_dir(output_dir) / "checkpoints"
    candidates = list(checkpoint_dir.glob("epoch*_ppl*.pt"))
    if not candidates:
        raise FileNotFoundError(f"{checkpoint_dir} 中没有 epoch checkpoint")
    # 匿名排序键：max 为每个候选调用，输入 Path、返回修改时间，用来选最近写入的 epoch 文件。
    # 文件复制或手动改时间会影响选择；这里没有比较权重内容、epoch 数字或验证 PPL。
    return max(candidates, key=lambda path: path.stat().st_mtime)
