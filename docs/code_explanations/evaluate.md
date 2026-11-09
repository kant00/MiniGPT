# evaluate.py 讲解

源码：[src/evaluate.py](../../src/evaluate.py)。

## 1. 总体描述

本文件读取已有 checkpoint，在 val 或 test token 数据上计算语言模型指标，并可根据该实验训练日志生成曲线。它不更新模型参数，也不让模型自由生成整段文本。

输入是 checkpoint、数据 split、batch size、AMP 设置和可选抽样批数；输出是终端指标、同实验 evaluations 目录中的 JSON，以及启用 plot 后的 training_curve.png。执行次序是“评估 → 保存指标 → 绘图”，所以 --plot 不是只画图开关。

## 2. load_model(ckpt_path, device)

torch.load 使用 weights_only=True，并把张量映射到目标设备。要求 checkpoint 存在 model_config，然后用已保存的 vocab_size、seq_len、model_config 重建 MiniGPT，加载 model_state_dict，切换 eval，返回模型与整个 checkpoint 字典。

eval 关闭 dropout 等训练行为，但本身不关闭梯度记录；真正评估循环另外使用 inference_mode。checkpoint 中还有优化器状态，该加载路径读取整个 payload，而非只从磁盘取模型权重。

## 3. evaluate_checkpoint：初始化

只接受 val/test。通过 torch.cuda.is_available 自动选择设备，再由训练模块的 resolve_amp_dtype 选择精度。随后加载与 checkpoint.seq_len 匹配的数据，并比较 tokenizer 指纹。

loader 不打乱数据，num_workers=0，batch size 默认 32。当前没有显式 device 参数，也没有“检测不到 GPU 就退出”的要求，所以错误环境可能默默在 CPU 上长时间运行。

## 4. evaluate_checkpoint：计算循环

对于每批 x/y，模型一次返回所有 T 个位置的 logits，将其从 B×T×V 展成 (B*T)×V，与展平目标计算 reduction=sum 的交叉熵。

累计三个量：总负对数似然 total_nll、有效目标 token 数 total_tokens、argmax 与目标相等的次数 correct_tokens。max_batches 非空时读到该批数停止；默认 None 表示完整遍历。

这里输入的上下文来自真实文章，即 teacher forcing。模型自由生成时后续上下文来自自己的输出，错误会累积，因此低 PPL 不保证长回答可靠。

循环没有逐批日志或进度条，最后才输出总指标。这是“进度一直不更新”的一个直接原因；是否真的卡住还要结合 Python 环境、CPU/GPU 活动判断。

## 5. 指标公式与含义

| 字段 | 公式/定义 | 如何理解 |
|---|---|---|
| loss | total_nll / total_tokens | 平均负对数似然，越低越好 |
| ppl | exp(min(loss,20)) | 同词表同数据下比较预测不确定性 |
| bits_per_token | loss / ln(2) | 平均每 token 的信息损失 |
| token_accuracy | correct_tokens / total_tokens | 贪心第一候选恰好正确的比例 |
| tokens | 实际评估的目标数量 | 受尾窗口舍弃与抽样上限影响 |
| tokens_per_second | tokens / elapsed | 批量评估吞吐 |
| split | val 或 test | 指标来源集合 |

PPL 有防溢出截断，极高 Loss 时不会继续指数增长。不同 tokenizer 的 token 粒度不同，不宜直接用 token PPL 做公平横向比较。

准确率不是问答成功率；评估 tokens/s 也不是单条自回归生成 tokens/s。计时从 loader 循环前开始，到循环后结束，包含读取、传输、前向和统计，不包含之前的模型加载，也不包含之后的报告写入与绘图。逐批 .item() 会使 GPU 结果同步到 CPU，影响吞吐。

## 6. save_evaluation_report(...)

默认通过 infer_run_dir 找到实验根目录，完整评估写 `evaluations/val_metrics.json` 或 test 对应文件。抽样评估带 `_maxNbatches` 后缀，避免默认情况下覆盖完整结果。

报告包含 format_version、UTC evaluated_at、绝对 checkpoint 路径、batch_size、amp、max_batches、metrics。使用 atomic_json_dump 发布；相同默认目标再次评估会替换先前报告，不保留历史版本。

自定义 output 为相对路径时按项目根目录处理。报告的 amp 记录的是请求的 auto/bf16 等字符串，不是完整实测设备和最终 dtype 清单；比较性能时应另外记环境。

## 7. plot_training_curve(log_path)

读取 JSONL 日志的所有非空行，取 epoch、train_ppl、val_ppl、train_loss 和 val_loss；旧日志缺少 val_loss 时用 log(val_ppl) 回推。

使用 Matplotlib 的 Agg 后端，不需要交互桌面窗口。创建左右两张子图，分别绘制训练/验证 PPL 与 Loss，增加图例和网格，以 150 DPI 保存到日志同目录 training_curve.png，关闭 figure 后返回路径。

每轮只有一个点。训练 Loss 是轮内不同时间模型的平均，验证 Loss 是轮末模型的测量，两条曲线口径并不完全相同。没有日志时不能仅凭 checkpoint 还原完整训练曲线。

## 8. main 与两种用法

main 读取 evaluate 配置分区，解析 checkpoint；默认使用 latest 实验的 best_model。计算和保存后，plot 为真才调用曲线函数，并显式传该实验日志，避免使用旧版全局 LOG_PATH。

```powershell
python -m src.evaluate --checkpoint 'outputs/runs/baseline/checkpoints/best_model.pt' --split val --plot
python -m src.evaluate --max_batches 20
```

第二条只是前 20 批的抽样结果，不是随机采样整个集合。max_batches 当前缺少正数范围校验，不要用 0 表示不评估。

如果只想从日志画图，可以调用已有函数，不需要重新跑完整评估：

```python
from pathlib import Path
from src.evaluate import plot_training_curve

plot_training_curve(Path(r"outputs/runs/baseline/training_log.jsonl"))
```

## 9. 阅读检查

val 用于开发与选模型，test 留到方案确定后评估。读完应能解释为什么逐批平均 PPL 不正确、为什么按总 token 数加权，以及为什么生成效果差时不能只靠这个文件的准确率判断原因。
