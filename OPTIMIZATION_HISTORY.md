# MiniGPT 优化记录

本文记录项目从初始教学版本到 P0、P1、P2 优化后的变化、设计原因、兼容性影响和验证结果。

## 1. 优化前基线

初始项目已经包含 Wikipedia 数据下载、BERT 中文 tokenizer、Decoder-only Transformer、训练、PPL 评估和四种采样策略，默认模型约 18.85M 参数。

主要限制包括：

- 数据切块误删最后一个有效窗口。
- 预处理与生成脚本使用不同的硬编码 tokenizer 路径。
- 数据、checkpoint 没有 tokenizer 身份信息，容易误混用。
- checkpoint 无法恢复 scheduler 和随机状态。
- 全部 token 同时存放在 Python 列表和 int64 Tensor 中，大数据时内存占用很高。
- 手写注意力显式创建平方级注意力矩阵。
- 没有混合精度、梯度累积、warmup、吞吐量统计或 KV Cache。
- 数据集在拼接后按 token 位置划分，且没有去重。
- 模型结构固定，checkpoint 没有保存完整架构。

## 2. P0：正确性与可运行性

### 2.1 修复数据切块

旧实现对 `Tensor.unfold` 的结果再次执行 `[:-1]`，导致训练集和验证集各丢失一个完整窗口。新实现显式计算期望样本数，并校验切块结果。

结果：

- 不再浪费有效 token。
- 小数据集不会因为额外切片意外变为空集。
- 增加了输入/目标右移和样本数量回归测试。

### 2.2 统一 tokenizer

新增 `src/tokenizer_utils.py`，加载顺序为：

1. `--tokenizer` 显式参数。
2. `MINIGPT_TOKENIZER_PATH` 环境变量。
3. 项目附近常见的 `pretrain_models/bert-base-chinese` 目录。
4. Hugging Face 模型 ID。

每个 tokenizer 都会生成以下元数据：

- 来源路径或 Hub ID。
- tokenizer 类名。
- 实际词表大小。
- 按 token ID 排序后的词表 SHA-256 指纹。
- 特殊 token 映射。

数据集、checkpoint、生成和评估会比较词表指纹，不匹配时立即停止。

### 2.3 数据文件保护

下载器新增 JSONL 完整性检查：

- 文件必须存在且非空。
- 每个非空行必须是合法 JSON。
- 必须包含 `id`、`title`、`text`。
- 正文必须是非空字符串。
- 写入行数必须与预期一致。

在线下载和本地导入都先写 `.part` 文件，通过校验后才替换正式文件。此前遗留的 0 字节 `.part` 文件已清理，它没有包含可恢复数据。

### 2.4 可复现与断点续训

训练新增：

- Python、NumPy、PyTorch、CUDA 随机种子。
- 可选确定性算法模式。
- 原子 checkpoint 写入。
- `--resume latest` 或指定 checkpoint 续训。

checkpoint 保存：

- 模型、优化器、学习率调度器状态。
- AMP GradScaler 状态。
- epoch、global step、累计 token 数。
- 最优验证 PPL。
- tokenizer 元数据。
- 完整模型配置和训练配置。
- Python、PyTorch、CUDA RNG 状态。

为了保证精确续训，batch size、epoch 总数、学习率、梯度累积等关键配置必须与原实验一致。

### 2.5 工程基础

- 新增 `requirements.txt`。
- 新增 `src/__init__.py`，支持 `python -m src.xxx`。
- 保留从 `src` 目录直接运行脚本的兼容导入。
- 新增 `.gitignore`，排除数据、checkpoint、缓存和临时文件。
- 修正文档中的模型参数量和过拟合判断错误。

## 3. P1：训练和推理性能

### 3.1 SDPA / Flash Attention 路径

默认注意力实现改为 PyTorch `scaled_dot_product_attention`，传入 `is_causal=True`。PyTorch 会根据设备和数据类型选择可用的高效内核。

仍保留 `--attention_impl manual`，用于教学对照和正确性测试。测试确认手写实现与 SDPA 在相同权重下输出一致。

### 3.2 混合精度

新增 `--amp auto|off|bf16|fp16`：

- CUDA 支持 BF16 时，`auto` 优先 BF16。
- 其他 CUDA 设备使用 FP16 和 GradScaler。
- CPU 自动关闭 AMP。

验证过程使用相同 autocast 策略，损失累加转回 FP32。

### 3.3 梯度累积与有效 batch

新增 `--grad_accum_steps`。最后一个不足完整累积窗口的 batch 会使用实际窗口大小缩放损失，避免梯度被错误缩小。

有效 batch token 数为：

```text
batch_size × seq_len × grad_accum_steps
```

学习率调度以 optimizer step 而不是 micro-batch 为单位。

### 3.4 Warmup + Cosine

新增线性 warmup 和余弦衰减：

- `--warmup_ratio` 默认 0.02。
- `--min_lr_ratio` 默认 0.1。
- 总步数按梯度累积后的 optimizer step 计算。

### 3.5 训练运行时优化

- CUDA 上启用 TF32。
- 支持时自动使用 fused AdamW。
- 数据传输使用 pinned memory 和 non-blocking copy。
- `num_workers > 0` 时启用 persistent workers 和预取。
- `optimizer.zero_grad(set_to_none=True)` 减少清零开销。
- 新增可选 `--compile`。
- 日志记录 tokens/s、累计训练 token、梯度范数和学习率。
- 初始化 PPL 默认只抽样 20 个 batch，可用 `--baseline_batches 0` 跳过。

### 3.6 KV Cache

模型 forward 支持每层 past key/value。生成脚本默认启用 KV Cache：

- prompt 只完整计算一次。
- 后续步骤只计算最新 token。
- 达到上下文窗口上限时，使用最后一个完整窗口重新建立缓存。
- 可用 `--no_kv_cache` 做速度和正确性对照。

测试确认缓存生成与非缓存 greedy 生成结果一致。

## 4. P1：数据质量和存储

### 4.1 文档级稳定划分

在 tokenize 和 packing 之前，使用 `split_seed + document_id` 的 SHA-256 哈希划分训练集/验证集。

优势：

- 同一文章不会跨集合。
- 输入文件顺序变化不会改变文章归属。
- 划分可以通过 `--split_seed` 复现。

### 4.2 精确文档去重

正文经过 NFC 规范化后计算 SHA-256。标题不同但正文相同的文章仍会被识别为重复。默认开启，可用 `--no_dedup` 关闭。

这是精确去重，不包括相似文章或段落级 MinHash 去重；近似去重留作后续扩展。

### 4.3 标题和文档边界

默认训练文本为 `标题 + 换行 + 正文`，充分利用已下载的标题信息。可用 `--no_title` 关闭。

文档末尾优先插入 tokenizer 的 EOS；没有 EOS 时使用 SEP。token ID 会校验是否落在词表范围内。

### 4.4 内存映射数据格式

新格式由一个连续 token 文件和一个元数据文件组成：

```text
data/train_tokens_seq256.bin
data/train_seq256.meta.json
data/val_tokens_seq256.bin
data/val_seq256.meta.json
```

- 词表不超过 65536 时使用 uint16，否则使用 uint32。
- 预处理按文档编码并直接追加到磁盘，不保存全量 Python token 列表。
- 训练通过 NumPy memmap 按需读取窗口。
- 相邻训练样本在逻辑上共享一个边界 token，但磁盘中只存一份连续 token 流。
- 元数据会校验 token 文件大小、token 数和样本数。

旧 `.pt` 数据仍可读取，但必须包含 P0 tokenizer 元数据。新实验推荐重新生成 memmap 数据。

## 5. P1：模型架构

默认模型由以下配置组成：

| 配置               |         值 |
| ------------------ | ---------: |
| vocab size         |      21128 |
| context length     |        256 |
| hidden size        |        384 |
| layers             |          6 |
| heads              |          6 |
| SwiGLU hidden size |       1024 |
| position           |       RoPE |
| normalization      |    RMSNorm |
| attention          |       SDPA |
| 参数量             | 18,734,976 |

变化说明：

- Learned position embedding → RoPE。
- LayerNorm → RMSNorm。
- GELU FFN → SwiGLU。
- SwiGLU 使用 1024 中间维度，使 FFN 参数量与旧版 1536 GELU 基本相当。
- residual output projection 使用按层数缩放的初始化标准差。

所有结构均可通过命令行切换，完整配置会写入 checkpoint。旧版 P0 checkpoint 没有 P1 的结构字段和参数命名，不能直接加载到 P1 模型。

## 6. P1：评估指标

评估脚本现在输出：

- token 级交叉熵。
- PPL。
- bits per token。
- next-token accuracy。
- 评估 token 数。
- tokens/s。

支持 `--max_batches` 快速评估和 `--amp`。训练日志同时记录 train/val loss、PPL、学习率、global step 和累计 token 数。

## 7. P2：数据治理与独立测试

### 7.1 三路文档划分

文档在 tokenize 前按稳定哈希划分为 train、val、test，默认比例为 95%/4%/1%。验证集用于训练过程中的模型选择，测试集不参与训练和选模，只用于最终一次评估。数据顺序变化不会改变同一文档的归属。

### 7.2 基础质量过滤

预处理新增以下流式过滤，不需要把全量语料载入内存：

- 最短正文长度，默认 50 字符。
- 最低中文字符比例，默认 10%。
- 最大控制字符比例，默认 1%。
- 最大连续重复字符比例，默认 20%。

每类拒绝数量写入三个 split 的元数据，便于审计阈值影响。所有阈值都有命令行开关。

后续实测发现 Unicode `Cc` 分类包含正常换行符，旧规则因此把 100,000 篇
Wikipedia 文章中的 99,562 篇误判为控制字符噪声。现已将换行、回车和制表符
排除在异常控制字符之外，并增加默认 80% 的质量拒绝率熔断保护。修复后同一
原始语料保留 99,883 篇，训练 token 从 2,145,703 增至 164,575,686；元数据
新增输入文章数、保留率和质量拒绝率，便于训练前审计。

### 7.3 SimHash 近似去重

在精确 SHA-256 去重后，对规范化正文构造采样字符 5-gram 的 64 位 SimHash。使用 4×16 位 LSH 桶召回候选，再按汉明距离确认，默认阈值为 3。该实现的目标是低内存、保守过滤常见镜像或轻微编辑副本；可用 `--no_near_dedup` 关闭，或通过 `--near_dedup_distance` 调整。

## 8. P2：可比较实验

### 8.1 隔离的 run 目录

每次从头训练不再覆盖统一的 `outputs/checkpoints`，而是写入：

```text
outputs/runs/<run-name>/
├── run_manifest.json
├── training_log.jsonl
└── checkpoints/
```

`run_manifest.json` 固化模型配置、训练配置、数据文件、tokenizer 指纹、PyTorch 版本和设备。`outputs/latest_run.json` 是原子更新的相对路径指针，评估、生成、生成基准和 `--resume latest` 默认使用它。显式命名的目录已存在时会拒绝启动，防止误覆盖。

### 8.2 Tokenizer 对比工具

新增 `src/analyze_tokenizers.py`，可在完全相同的语料样本上加载多个本地或 Hugging Face tokenizer，输出：

- 词表大小及在指定 hidden size 下的 embedding 参数量。
- tokens/字符、字符/token、tokens/UTF-8 字节。
- UNK 数量与比例。
- 是否具备可作为文档边界的 EOS/SEP。

这解决了“网上现成 BPE 能否替换”的选型问题：可以替换，但必须先量化压缩率与模型参数成本，随后重新预处理并从头训练。不同 tokenizer 的 token-PPL 不应直接横向比较。

### 8.3 固定提示词生成基准

新增 `src/benchmark_generation.py`。它在固定 seed、提示词与采样参数下保存每条 completion，并统计生成 token 数、端到端延迟、tokens/s、Distinct-1/2 和 3-gram 重复率。默认结果跟随当前 run 保存，便于架构或训练配置之间回归对比。这些自动指标是质量代理，不能替代人工事实性与连贯性评审。

### 8.4 单机 DDP

训练入口读取 `torchrun` 注入的环境变量自动启用 DistributedDataParallel：

- CUDA 使用 NCCL，CPU 调试使用 Gloo。
- train/val 使用分布式采样，训练指标跨 rank 汇总。
- checkpoint 和日志只由 rank 0 写入。
- checkpoint 保存每个 rank 的 RNG 状态，续训时分别恢复。
- 普通 `python -m src.train` 的单卡/CPU 行为和参数无需改变。

### 8.5 统一项目配置

新增 `configs/default.json`，集中管理数据下载、预处理、模型、训练、评估和
生成相关参数。七个可执行模块统一支持 `--config`，解析优先级为：

```text
显式 CLI 参数 > 自定义配置 > configs/default.json
```

自定义配置允许只填写需要覆盖的字段；未知字段、非法 choices 和错误类型会
在任务启动前报错。训练 run 会保存原始配置副本，同时继续在 manifest 和
checkpoint 中记录最终生效的训练与模型参数，避免配置文件后来被修改后无法
追溯实验。

### 8.6 推理质量修复

针对 BERT WordPiece 基础模型的实际生成结果，修复并增加以下推理约束：

- 生成停止符与预处理一致：优先 EOS，无 EOS 时使用 SEP。
- 屏蔽 PAD、CLS、MASK、UNK 等未作为正文边界训练的特殊 token。
- 新增 repetition penalty 和 no-repeat n-gram，降低固定句式循环。
- 支持最少生成 token 数，避免过早结束。
- 清理 BertTokenizer.decode 在中文字符和标点之间插入的展示空格，同时保留
  英文单词所需空格。
- 默认采样调整为 temperature=0.7、top-p=0.85，并让生成基准记录原始 token
  ID 和是否命中文档边界。

这些修改改善可读性和重复问题，但不会把 Wikipedia 基础语言模型变成指令
模型；问答、对话和 Agent 能力仍需后续 SFT/LoRA 数据与专门评测。

## 9. 当前推荐流程

先做小规模冒烟测试：

```bash
python -m src.download_data --max_articles 5000
python -m src.prepare_dataset --max_tokens 2000000
python -m src.train --run_name smoke --epochs 1 --batch_size 16 --grad_accum_steps 2
python -m src.evaluate --max_batches 20
python -m src.evaluate --split test
python -m src.generate --prompt "人工智能的发展"
```

正式训练示例：

```bash
python -m src.download_data --max_articles 100000
python -m src.prepare_dataset --seq_len 256
python -m src.train --epochs 5 --batch_size 16 --grad_accum_steps 4 --amp auto
```

续训：

```bash
python -m src.train --epochs 5 --batch_size 16 --grad_accum_steps 4 --amp auto --resume latest
```

如使用网上现成 tokenizer：

```bash
python -m src.prepare_dataset --tokenizer <Hugging-Face-ID>
```

切换 tokenizer 后必须重新生成数据并从头训练。对于当前小模型，建议把词表控制在 8K～32K；现代通用大模型的十几万词表会显著扩大 embedding 和输出层。

## 10. 验证覆盖

测试覆盖以下场景：

- 数据切块数量及 input/target 右移。
- JSONL 完整性检查。
- memmap 数据生成、读取和形状。
- causal attention 不泄露未来 token。
- SDPA 与手写注意力输出一致。
- KV Cache 与完整前向输出一致。
- cached/uncached greedy 生成一致。
- RNG 状态恢复。
- checkpoint 原子写入和安全加载。
- 微型端到端数据准备、训练、验证与完整 checkpoint 保存。
- train/val/test 三路 memmap 数据生成。
- 质量拒绝、SimHash 指纹和近似重复索引。
- run 目录隔离、latest 指针及路径越界保护。
- tokenizer 成本指标与生成多样性/重复率指标。

## 11. 尚未包含的后续方向

- sequence packing 中的跨文档 loss mask。
- 大规模 MinHash/语义去重和段落级语言质量打分。
- FSDP、ZeRO 和多机容错训练。
- 标准中文下游任务与 bits/字符模型指标。
- 从语料训练自有 BPE tokenizer。
- KV Cache 预分配和批量生成。
- ONNX、量化和部署。
