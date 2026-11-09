# MiniGPT 预训练项目架构

## 项目定位

这是一个使用 PyTorch 实现的小型中文自回归语言模型项目，覆盖语料获取、分词、预训练、验证和生成。命令行默认模型约 38.09M 参数，适合单卡实验。

## 配置流

`configs/default.json` 是命令行流程的默认参数源，按下载、预处理、训练、
评估和生成等模块分区。自定义 JSON 配置只需填写需要覆盖的字段，显式 CLI
参数拥有最高优先级。配置字段会根据对应命令行参数执行名称、类型和 choices
校验；新训练 run 同时保存源配置副本和最终生效配置。

## 数据流

```text
中文 Wikipedia JSONL
        │
        ▼
质量过滤、精确/SimHash 近似去重、稳定三路哈希划分
        │
        ▼
Tokenizer + EOS/SEP 文档边界
        │
        ▼
uint16/uint32 连续 token 文件 + JSON 元数据
        │
        ▼
NumPy memmap 按需读取固定长度窗口
        │
        ▼
MiniGPT 单卡/DDP 预训练 → 隔离实验目录 → val/test 评估 / 生成基准
```

## 默认模型

```text
Token Embedding (vocab=21128, dim=512)
        │
        ├─ RoPE
        ▼
8 × Pre-Norm Transformer Block
        ├─ RMSNorm
        ├─ Causal SDPA (8 heads × 64 dim)
        ├─ RMSNorm
        └─ SwiGLU FFN (512 → 1536 → 512)
        │
        ▼
RMSNorm → tied LM Head
```

`configs/default.json` 的参数量为 38,089,216。直接调用 `build_model()` 时使用 `MiniGPTConfig` 的类默认值，参数量为 18,734,976。模型支持以下对照配置。

- `rope` / `learned` position embedding。
- `rmsnorm` / `layernorm`。
- `swiglu` / `gelu`。
- `sdpa` / `manual` attention。

完整模型配置随 checkpoint 保存，加载时不依赖代码中的默认值猜测架构。

## 训练策略

- AdamW，矩阵参数 weight decay，bias/norm 不衰减。
- 线性 warmup + cosine decay。
- BF16 优先的自动混合精度；FP16 使用 GradScaler。
- 梯度裁剪和梯度累积。
- CUDA TF32、可用时 fused AdamW、可选 `torch.compile`。
- 按 token 加权计算训练 loss。
- 每个 epoch 计算 token 级验证 PPL。
- 原子 checkpoint，支持恢复 optimizer、scheduler、scaler 和 RNG。
- 使用 `torchrun` 时自动启用 DDP、分布式采样和跨 rank 指标汇总。
- 每次新训练写入独立 run 目录及实验清单，避免覆盖其他实验。

## 数据格式

每个 split 包含：

```text
<split>_tokens_seq256.bin
<split>_seq256.meta.json
```

元数据记录数据类型、token 数、样本数、文档数、上下文长度、tokenizer 指纹、划分 seed、质量过滤和去重统计。训练时每个样本从连续 token 流读取 `seq_len + 1` 个 token：前 `seq_len` 个作为输入，后 `seq_len` 个作为目标。

## 推理

支持 greedy、temperature、top-k、top-p。KV Cache 默认开启；达到最大上下文后，对最后一个完整窗口重新建立缓存。生成遇到 tokenizer EOS 后停止。

## 评估

验证集用于选模型，独立测试集只用于最终报告。评估输出 loss、PPL、bits/token、next-token accuracy、token 数和 tokens/s。`analyze_tokenizers.py` 额外报告 tokens/字符、tokens/UTF-8 字节、UNK 率及 embedding 成本。

## 可靠性边界

- 数据、checkpoint、生成 tokenizer 必须拥有相同词表 SHA-256。
- 文档划分在 tokenize 前完成，避免同一文档跨集合。
- SimHash 是轻量文档级近似去重，不等价于大规模 MinHash/语义去重。
- P1 架构与早期 checkpoint 参数命名不同，需要重新训练。

完整变更和迁移说明见 [OPTIMIZATION_HISTORY.md](OPTIMIZATION_HISTORY.md)。
