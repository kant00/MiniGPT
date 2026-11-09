# train.py 讲解

源码：[src/train.py](../../src/train.py)。

## 1. 总体描述

这是项目的训练协调模块。它读取已编码数据，构建 MiniGPT，计算下一 token 预测损失，反向传播更新权重，每轮验证并保存训练状态。数据生产由 prepare_dataset 完成，网络结构由 model 完成，本文件把它们组织成可运行实验。

输入为 train/val 的 bin 和 meta、训练与模型配置，以及可选恢复 checkpoint。输出包括独立 run 目录中的源配置、实际配置清单、逐轮日志、epoch checkpoint 和 best_model.pt，并更新 latest 指针。

主流程可以记成：解析参数 → 初始化设备与实验 → 检查数据 → 建模和优化器 → 恢复或初始化 → 逐批更新 → 逐轮验证 → 保存。当前是全 token 的语言模型预训练，不是只监督回答部分的 SFT。

## 2. TokenDataset：磁盘整数怎样成为训练样本

### __init__(path)

新版路径以 `.meta.json` 结尾。函数读取 storage_format、vocab_size、seq_len、tokenizer_info、boundary_token_id、num_tokens、num_samples 和 dtype，定位同目录 bin，检查字节数是否等于 num_tokens×dtype.itemsize，并重新计算样本数。

之后用只读 `np.memmap` 映射连续整数，按需访问文件。memmap 不等于所有数据从来不进入内存：操作系统会缓存页面，但不需要程序先构造一个全量 Python token 列表。

旧版 `.pt` 分支读取已保存的二维张量，要求每行长度为 seq_len+1。保留这个分支是格式兼容，不意味着所有更早的无 tokenizer 元数据文件都可以继续训练。

### __len__ 与 __getitem__

`__len__` 返回 num_samples。第 i 条样本从 i×seq_len 读取 seq_len+1 个 token，复制成 NumPy int64，再转为 torch.Tensor；返回 `chunk[:-1]` 与 `chunk[1:]`。

例如 `[5,7,9,2]` 返回输入 `[5,7,9]` 和目标 `[7,9,2]`。原磁盘数据是 uint16，模型索引和交叉熵目标需要整数张量，因此读样本时转换类型。

### close 与 __del__

close 尝试关闭底层 mmap；析构时调用它，减少 Windows 文件仍被映射而无法替换的问题。正常对象生命周期仍比依赖不确定的垃圾回收时机更容易管理。

## 3. 数据路径与一致性检查

`resolve_dataset_path(split, seq_len)` 优先寻找新版 meta，没有则寻找旧 pt，两者都没有给出错误。路径由源文件计算出的项目 data 目录决定，不从 CLI 任意切换数据目录。

`validate_dataset_pair(train_ds, val_ds)` 检查 seq_len、词表大小、tokenizer 元数据与词表指纹一致。它检查格式与词表兼容性，不检查源语料内容哈希、训练/验证语义泄漏或数据是否已被重新处理。

`make_dataloader(...)` 设置 batch、shuffle、sampler、workers。传 sampler 时禁用独立 shuffle；CUDA 时启用 pin_memory，workers>0 时启用 persistent_workers 和预取。没有设置 drop_last，所以最后一批可以较小。

## 4. 随机数、保存与续训路径工具

`set_seed(seed, deterministic)` 设置 Python、NumPy、PyTorch CPU/CUDA 随机种子。deterministic 分支还开启确定性算法，但使用 warn_only=True，因此不能理解为任何算子、设备和版本下都逐位一致。

`capture_rng_state()` 保存 Python 和 PyTorch CPU/CUDA 的当前随机状态；`restore_rng_state(state)` 恢复它们，缺失时警告。当前虽设置 NumPy seed，却没有把 NumPy RNG 状态放入 checkpoint。

种子相同只表示从同一个随机起点开始，恢复 RNG 状态则表示接着中间的位置继续抽样，两者不同。

`atomic_torch_save(payload, path)` 先写 `.pt.tmp`，再 replace。`resolve_resume_path(resume, output_dir)` 对字符串 latest 使用最近实验的 epoch checkpoint，否则按项目根目录解析相对路径。此处 latest 的含义与推理默认 best_model 不同。

## 5. 学习率与混合精度工具

### build_warmup_cosine_scheduler

返回 LambdaLR。内部闭包 `lr_lambda(step)` 根据当前步数返回学习率倍率，并读取外层的 total_steps、warmup_steps、min_lr_ratio；调度器将倍率乘到基础学习率上。warmup 阶段按步数逐渐增加到基础学习率，之后按余弦下降到 base_lr×min_lr_ratio。代码限制 warmup_steps 不超过 total_steps-1，避免全部步数都处在 warmup。

总步数来自 `ceil(len(train_loader)/grad_accum_steps) × epochs`。因此 epochs 不只是控制循环次数，还决定完整学习率日程。计划 8 轮跑到第 3 轮，与完整计划 3 轮的学习率轨迹不同。

### resolve_amp_dtype

CPU 或 amp=off 返回 None；bf16 显式检查支持；fp16 返回 float16；auto 在 CUDA 上优先 BF16，否则 FP16。自动选择 CPU 不会主动报错，这解释了用错 Python 环境时训练或评估可能很慢。

### create_grad_scaler 与 autocast_context

create_grad_scaler 兼容不同 torch.amp API 调用方式。训练只在 FP16 时启用缩放，避免小梯度数值下溢；BF16 默认无需该缩放。

autocast_context 在启用 AMP 时返回 autocast，否则返回 nullcontext。AMP 控制部分算子的运算 dtype，不等于把所有模型参数永久转换成半精度。

## 6. initialize_distributed：单进程与 DDP

读取 WORLD_SIZE、RANK、LOCAL_RANK。world_size=1 时正常选择 cuda/cpu；多进程时设置当前 GPU，CUDA 路径选 NCCL，CPU 路径选 Gloo，再初始化通信组。

返回字典包含设备、全局进程编号、机器内编号、总进程数和 is_main。训练数据使用 DistributedSampler，每轮 set_epoch(epoch) 改变确定的打乱顺序；主进程创建目录并把目录广播给其他进程。

当前代码选择 NCCL 并不保证你的 Windows 安装提供该后端；这里是实现的 DDP 路径，不代表本机已完成多 GPU 实测。分布式验证使用默认 DistributedSampler，长度不能整除进程数时可能补齐重复样本，汇总结果并非严格无重复遍历。

## 7. compute_ppl：训练期间验证

记住模型原来是否处于 training，切换 eval，在 inference_mode 下遍历验证 loader。交叉熵使用 reduction=sum，累加全部负对数似然和 token 数，再用总和除以总 token 数；DDP 时 all_reduce 两个总量。

平均 Loss=L 时返回 PPL=`exp(min(L,20))`，其中 20 是数值截断，不是另一种评估公式。max_batches 可用于抽样检查。结束后若原来在训练状态，就恢复 train。

该函数只返回 loss/ppl/tokens，不保存独立评估报告，也不计算准确率。完整离线报告在 evaluate 模块。

## 8. train：建立实验和优化器

函数先验证基本训练参数，然后创建或恢复 run、设置 seed+rank、选择 AMP、设置 CUDA TF32，读取并检查 train/val 数据。新实验根据 model_overrides 构建模型；恢复实验使用 checkpoint 的 model_config。

参数按维数分组：二维及以上矩阵使用 weight_decay，其余参数不衰减。AdamW 的 betas 固定为 (0.9,0.95)，CUDA 且 API 支持时启用 fused。

接着创建调度器、GradScaler、交叉熵，并建立 training_config。新实验保存源 JSON 副本和 run_manifest；源副本不包含 CLI 覆盖，实际架构和已记录训练参数应查看 manifest。

可选 torch.compile 包装 raw_model，随后可选 DDP 再包装。raw_model 始终保留用于统一参数访问和 state_dict 保存，避免把包装层前缀混入模型定义。

baseline_batches 默认 20，只对随机初始化模型抽样验证，不是只训练 20 批，也不是你已训练的 baseline 实验。

## 9. 一次梯度更新完整解释

设 grad_accum_steps=4，表示通常四个 micro-batch 才更新一次权重。有效全局 batch 约等于 batch_size×grad_accum_steps×world_size，尾批和最后一个累积窗口可能更小。

```text
取 x/y 并搬到设备
  → autocast 中前向得到 [B,T,V]
  → reshape 为 [B*T,V]，目标为 [B*T]
  → 交叉熵得到 raw_loss
  → 除以当前累积窗口的 micro-batch 数
  → scaler.scale(...).backward()
  → 到更新边界时 unscale、裁剪、step、update、清零
```

`window_start` 和 `current_window` 处理末尾不足四批的窗口，防止还除以四导致梯度额外缩小。DDP 在非更新批使用 no_sync，避免每个 micro-batch 都做梯度通信。

先 unscale 再 clip_grad_norm_，否则会裁剪被放大的梯度。随后 scaler.step(optimizer)、scaler.update()、scheduler.step()、zero_grad(set_to_none=True)，并增加 global_step。

统计训练 Loss 时使用 raw_loss×batch_tokens 累加，因此日志按 token 加权。梯度累积则按 micro-batch 平均；若同一累积窗口里尾批大小不同，它不严格等于按所有 token 一次性求平均的梯度。

FP16 溢出时 GradScaler 可能跳过优化器更新，而当前代码仍推进 scheduler 和 global_step。因此 global_step 更精确地说是代码的更新边界计数，不应无条件视为成功参数更新次数。

## 10. 日志、验证与 checkpoint

每到 log_interval 的更新边界，主进程输出当前轮累计平均 Loss、PPL、学习率、裁剪前梯度范数和近似 tokens/s。这个平均 Loss 不是仅最近一个区间的 Loss；tokens/s 才使用区间计数。

每轮结束，汇总训练 token 与 NLL，计算验证 PPL，增加 tokens_seen，向 training_log.jsonl 追加一行。基本字段为 epoch、global_step、tokens_seen、train_loss、train_ppl、val_loss、val_ppl、lr。消融实验准备时新增 train_seconds、train_tokens_per_second、peak_cuda_allocated_bytes，分别记录训练循环耗时、整轮平均吞吐和训练阶段 PyTorch allocated 显存峰值；计时不含轮末验证与 checkpoint 保存，仍不包含逐步 Loss 曲线。

单进程训练的 RandomSampler 现在每轮使用独立的 seed+epoch 生成器。不同模型初始化消耗不同数量随机数时，样本打乱顺序仍保持一致。该更改面向新实验；它与旧版本使用全局随机数的样本顺序不同，旧 checkpoint 跨版本恢复不保证复现原顺序。

checkpoint 主要字段：

| 类别 | 内容 | 用途 |
|---|---|---|
| 模型 | model_state_dict、model_config | 重建和恢复网络 |
| 优化 | optimizer、scheduler、scaler 状态 | 接着原训练日程继续 |
| 进度 | epoch、global_step、tokens_seen | 知道训练到哪里 |
| 评估 | val_ppl、best_val_ppl | 比较当前与历史最好 |
| 数据身份 | vocab_size、seq_len、tokenizer | 防止接口错配 |
| 随机/分布式 | rng_state、rng_states、world_size | 恢复各进程随机状态 |

每轮保存 epoch 文件；验证 PPL 严格变小时更新 best_model.pt。checkpoint 包含优化器等状态，所以大小明显大于单纯模型权重文件。

## 11. 恢复训练的约束

恢复时检查词表、上下文长度、tokenizer 指纹，并要求 epochs、batch、lr、weight_decay、grad_clip、梯度累积、warmup、最低学习率比例、AMP 等已保存配置一致。接着恢复模型、优化器、调度器、scaler 与 RNG，从 epoch+1 开始。

这是 epoch 级恢复，不会恢复某轮中途的样本游标。它也不是 SFT 的“加载权重、新建训练阶段”。数据源指纹、world_size、seed 等没有都纳入相同强度的恢复校验，不能宣称跨任意环境严格复现。

## 12. main、使用方式与阅读检查

main 先声明 parser，再调用统一配置解析，最后把模型参数放入 model_overrides，把训练参数逐项传入 train。CLI 的 --compile 对应配置键 compile_model。

```powershell
python -m src.train --run_name reading-demo --epochs 1
python -m src.train --resume latest
```

第二条恢复的是最近实验的日程，参数必须与它兼容。第一条的一轮仍然会遍历完整训练集，不是几十步的 smoke test。

读完应能从一个 token 窗口讲到 logits、Loss、backward、AdamW、验证和 checkpoint，并能区分 micro_step、global_step、epoch、tokens_seen 四种进度单位。
