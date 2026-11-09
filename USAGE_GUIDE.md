# MiniGPT 使用指南

## 1. 安装

先按 [README](README.md#快速开始) 创建并激活虚拟环境，再在项目根目录安装依赖。

```bash
python -m pip install -r requirements.txt
```

推荐始终使用 `python -m src.xxx`，这样包内导入路径最稳定。

## 2. 统一配置

项目的默认参数集中在 `configs/default.json`，按执行模块分为
`download_data`、`prepare_dataset`、`train`、`evaluate`、`generate`、
`analyze_tokenizers` 和 `benchmark_generation` 七个分区。

直接运行时自动读取默认配置：

```bash
python -m src.train
```

建议复制默认文件创建实验配置，然后只修改需要变化的字段：

```bash
python -m src.train --config configs/my_experiment.json
```

命令行参数优先于配置文件，适合做一次性覆盖：

```bash
python -m src.train --config configs/my_experiment.json --batch_size 8
```

自定义配置可以只保留当前命令需要覆盖的分区和字段，其余参数继承
`configs/default.json`。配置中的未知参数会直接报错。新训练还会把配置文件
复制为当前 run 的 `source_config.json`，并在 `run_manifest.json` 记录最终生效值。

## 3. Tokenizer

默认依次查找：显式 `--tokenizer`、`MINIGPT_TOKENIZER_PATH`、项目附近的本地 `bert-base-chinese`、Hugging Face 上的 `bert-base-chinese`。

显式使用本地 tokenizer：

```bash
python -m src.prepare_dataset --tokenizer ./local-models/bert-base-chinese
```

使用 Hugging Face tokenizer：

```bash
python -m src.prepare_dataset --tokenizer google-bert/bert-base-chinese
```

切换 tokenizer 后必须重新预处理并从头训练。

先在同一语料样本上比较两个现成 tokenizer：

```bash
python -m src.analyze_tokenizers --tokenizer bert-base-chinese --tokenizer another-org/chinese-bpe
```

重点比较 `tokens_per_character`、`unknown_rate` 和 `embedding_parameters`，不要只看词表大小。

## 4. 获取数据

快速测试：

```bash
python -m src.download_data --max_articles 5000
```

正式数据：

```bash
python -m src.download_data --max_articles 100000
```

备用数据源：

```bash
python -m src.download_data --source wikimedia
python -m src.download_data --hf_mirror
python -m src.download_data --from_local
```

本地模式读取 `data/raw/*.txt`，使用 UTF-8 编码，以空行分隔文档。准备足量的不同文档，让训练、验证和测试集都能产生完整样本。所有来源先写临时文件并通过 JSONL 校验后再发布。

如果已有本地语料和分词器，可以离线运行小样本流程。把分词器文件放在 `local-models/bert-base-chinese/`，把语料放在 `data/raw/`，然后在项目根目录执行以下命令。若目录已有正式实验语料，使用单独的工作副本，以免本地导入替换 `data/wiki_zh.jsonl`。

```sh
python -m src.download_data --config configs/quickstart.json --from_local
python -m src.prepare_dataset --config configs/quickstart.json --tokenizer ./local-models/bert-base-chinese
python -m src.train --config configs/quickstart.json
python -m src.evaluate --config configs/quickstart.json
python -m src.generate --config configs/quickstart.json --tokenizer ./local-models/bert-base-chinese
```

## 5. 数据预处理

```bash
python -m src.prepare_dataset --seq_len 256
```

快速限制 token 数：

```bash
python -m src.prepare_dataset --max_tokens 2000000
```

常用选项：

```bash
python -m src.prepare_dataset --split_seed 42 --val_ratio 0.04 --test_ratio 0.01
python -m src.prepare_dataset --no_dedup
python -m src.prepare_dataset --no_near_dedup
python -m src.prepare_dataset --no_title
```

默认执行基础中文质量过滤、精确去重和保守的 SimHash 近似去重，并输出 train/val/test 三套 memmap 数据。可用 `--min_chars`、`--min_cjk_ratio`、`--max_control_ratio`、`--max_repeat_run_ratio` 调整质量阈值。换行、回车和制表符被视为正常排版，不计入异常控制字符；若质量过滤拒绝率超过 `--max_quality_rejection_ratio`（默认 80%），预处理会停止且不会替换已有数据。

## 6. 训练

默认训练：

```bash
python -m src.train --run_name baseline
```

8GB 显存可从以下配置开始：

```bash
python -m src.train --batch_size 16 --grad_accum_steps 2 --amp auto
```

有效 batch token 数为 `batch_size × seq_len × grad_accum_steps`。

每次新训练创建 `outputs/runs/<run-name>/`，其中包含实验清单、日志和 checkpoint；`outputs/latest_run.json` 指向最近实验。已有名称不会被静默覆盖。

其他常用选项：

```bash
python -m src.train --epochs 5 --warmup_ratio 0.02
python -m src.train --compile
python -m src.train --baseline_batches 0
python -m src.train --num_workers 2
```

模型消融示例：

```bash
python -m src.train --position_encoding learned --norm_type layernorm \
  --activation gelu --d_ff 1536 --attention_impl manual
```

Windows PowerShell 中可以写成单行。

### 断点续训

```bash
python -m src.train --resume latest
```

如果原训练使用了非默认参数，续训时必须重复这些参数：

```bash
python -m src.train --epochs 5 --batch_size 16 --grad_accum_steps 4 \
  --amp auto --resume latest
```

### 多卡训练

单机双卡示例（无需修改原单卡命令参数）：

```bash
torchrun --standalone --nproc_per_node=2 -m src.train --run_name ddp-2gpu --batch_size 16
```

`batch_size` 是每张卡的 batch；全局有效 batch token 数还要乘以 GPU 数。checkpoint 只由 rank 0 写入，并保存各 rank 的随机状态。

## 7. 评估

```bash
python -m src.evaluate
python -m src.evaluate --split test
python -m src.evaluate --max_batches 20
python -m src.evaluate --plot
```

输出包括 loss、PPL、bits/token、token accuracy 和 tokens/s。

完整评估默认保存到当前实验的 `evaluations/val_metrics.json` 或
`evaluations/test_metrics.json`。使用 `--max_batches 20` 时会保存为
`val_metrics_max20batches.json`，避免抽样结果覆盖完整结果；也可以通过
`--output <路径>` 显式指定位置。训练期间每个 epoch 的验证指标仍保存在
`training_log.jsonl`。

训练调参期间使用默认 `val`；模型配置确定后再运行 `--split test`，避免把测试集变成调参集。

## 8. 生成

```bash
python -m src.generate --prompt "人工智能的发展"
python -m src.generate --compare --prompt "中国的首都是"
python -m src.generate --strategy top_k --top_k 30
```

默认推理参数针对当前小模型设置为 `temperature=0.7`、`top_p=0.85`、
`repetition_penalty=1.1`、`no_repeat_ngram_size=4`。生成会以训练使用的
EOS/SEP 作为文档结束符，屏蔽其余特殊 token，并自动清理 BERT WordPiece
解码产生的中文字符间空格。可按需覆盖：

```bash
python -m src.generate --prompt "人工智能的发展" \
  --temperature 0.7 --top_p 0.85 \
  --repetition_penalty 1.1 --no_repeat_ngram_size 4
```

KV Cache 默认开启。对照无缓存速度或结果：

```bash
python -m src.generate --no_kv_cache --prompt "人工智能"
```

固定提示词生成基准：

```bash
python -m src.benchmark_generation --strategy top_p --max_new_tokens 80
```

结果保存到当前实验目录，包含逐样本输出、原始 token ID、是否命中文档边界、延迟、tokens/s、Distinct-1/2 和 3-gram 重复率。也可用 `--prompts prompts.jsonl` 提供自定义提示词。

## 9. 测试

```bash
python -m unittest discover -s tests -v
```

测试包含数据切块、JSONL 校验、memmap 往返、causal mask、SDPA 对照、KV Cache、随机状态和微型端到端训练。

## 10. 产物

```text
data/train_tokens_seq256.bin
data/train_seq256.meta.json
data/val_tokens_seq256.bin
data/val_seq256.meta.json
data/test_tokens_seq256.bin
data/test_seq256.meta.json
outputs/latest_run.json
outputs/runs/<run-name>/run_manifest.json
outputs/runs/<run-name>/source_config.json
outputs/runs/<run-name>/checkpoints/best_model.pt
outputs/runs/<run-name>/training_log.jsonl
outputs/runs/<run-name>/training_curve.png
outputs/runs/<run-name>/evaluations/val_metrics.json
outputs/runs/<run-name>/evaluations/test_metrics.json
outputs/runs/<run-name>/generation_benchmark.json
outputs/tokenizer_comparison.json
```

详细优化记录见 [OPTIMIZATION_HISTORY.md](OPTIMIZATION_HISTORY.md)。
