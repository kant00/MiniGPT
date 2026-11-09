# experiment_utils.py 讲解

源码：[src/experiment_utils.py](../../src/experiment_utils.py)。

## 1. 总体描述

本文件规定实验目录如何创建、如何找到最近实验、如何解析 checkpoint，以及如何完整写入 JSON。训练、评估和生成共同使用它，避免各自猜测文件位置。

它不读取模型参数，也不比较模型效果。输入主要是 Path、实验名和 JSON 数据；输出是目录或文件 Path，部分函数会创建目录、更新 latest 指针或写 JSON。

```text
outputs/
  latest_run.json
  runs/
    baseline/
      run_manifest.json
      source_config.json
      training_log.jsonl
      checkpoints/
        best_model.pt
        epoch3_ppl22.2.pt
      evaluations/
        val_metrics.json
```

本模块只建立部分基础目录；清单、日志、checkpoint 与评估报告由调用方写入。

## 2. atomic_json_dump(payload, path)

自动建立父目录，将 JSON 用 UTF-8 写入带 `.part` 后缀的临时文件，再 replace 到正式路径。ensure_ascii=False 保留可读中文，indent=2 方便查看。

它避免读到正在写入的半份正式 JSON，但没有对整个实验实现事务，也没有 fsync 持久化保证。多个进程同时写同一目标会共用临时文件名；训练通过仅主进程写入减少冲突，并非此函数提供锁。

## 3. validate_run_name(run_name)

使用 RUN_NAME_PATTERN 限定名称：首字符是英文字母或数字，其后允许英文字母、数字、点、下划线、连字符，总长度不超过 80。空字符串、中文、斜杠和目录穿越形式被拒绝。

通过时返回原字符串，便于直接组合进路径。名称不是显示标题的任意文本，而是路径组成部分。

## 4. create_run_dir(output_dir, run_name)

先建立 runs 目录；有显式名称就验证该名称，没有则用本地时间戳。显式名称已存在会抛 FileExistsError，不覆盖旧实验。自动名称碰撞时尝试 `-1`、`-2` 等后缀。

随后创建实验目录与 checkpoints 子目录，调用 set_latest_run，再返回实验目录。latest 在训练开始阶段就更新，所以它可能指向尚未产出权重、甚至后来失败的实验。

## 5. set_latest_run 与 latest_run_dir

`set_latest_run(output_dir, run_dir)` 把两个目录解析为绝对路径，要求 run_dir 在 output_dir 内，保存相对 POSIX 路径，例如 `{"run_dir":"runs/baseline"}`。

`latest_run_dir(output_dir)` 读取该 JSON，拒绝绝对路径与包含 `..` 的路径，再解析真实路径并检查仍在输出目录内，最后要求它是存在的目录。

这既让输出目录整体移动后更容易复用，也约束 latest 不能跳到外部路径。但 latest 表示最近选定的实验，不表示所有实验中 PPL 最低的模型。

## 6. infer_run_dir(checkpoint_path)

把路径解析成绝对形式，要求父目录名称为 checkpoints，返回再上一层。该约定让评估能找到同实验的日志并确定报告目录。

函数本身不检查模型文件内容，也不要求目录一定处于 outputs/runs。随意把权重移动到一个不叫 checkpoints 的目录，可能导致评估保存或绘图阶段报错。

## 7. resolve_checkpoint(checkpoint, output_dir, filename)

显式给出 checkpoint 时优先使用；相对路径基于 output_dir 的父目录解析，标准项目中就是项目根目录。没有显式路径时，去 latest 实验的 checkpoints 中取 filename，默认 best_model.pt。

最后解析绝对路径并检查存在。字符串 `latest` 在此处不是特殊关键字；想选最新实验的 best 模型通常省略 checkpoint。它与训练模块的 `--resume latest` 不是同一种参数语义。

## 8. resolve_latest_epoch_checkpoint(output_dir)

在 latest 实验 checkpoints 中查找 `epoch*_ppl*.pt`，按修改时间最大值返回。它不读取 checkpoint 的 epoch 字段，也不按文件名中的数字排序。

训练续跑要恢复最近训练状态，因此找 epoch 文件；推理通常使用验证集选出的 best_model。两个路径可能指向不同轮数。复制文件改变修改时间可能影响 latest epoch 的选择。

## 9. 使用示例和常见问题

```python
from pathlib import Path
from src.experiment_utils import resolve_checkpoint

root = Path(r".")
checkpoint = resolve_checkpoint(None, root / "outputs")
print(checkpoint)
```

如果最近实验还没有 best_model，函数会报错，不会退回上一个成功实验。做消融或复现报告时明确指定 checkpoint 路径，可以避免 latest 指针变化导致评估错模型。
