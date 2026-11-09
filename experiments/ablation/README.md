# A/B/C 公平消融实验操作指南

本目录准备固定 1 epoch、seed=42、单卡的三组实验。训练必须由你执行 train 命令启动。prepare/check/report 不会训练；evaluate 只使用已经训练好的权重评估和生成。

## 1. 实验方案

| 组 | 架构 | 注意力 | d_ff | 参数量（词表 21128） |
|---|---|---|---:|---:|
| A | learned + LayerNorm + GELU | manual | 2304 | 38,251,520 |
| B | RoPE + RMSNorm + SwiGLU | manual | 1536 | 38,089,216 |
| C | RoPE + RMSNorm + SwiGLU | SDPA | 1536 | 38,089,216 |

固定：d_model=512、8 层、8 头、seq_len=256、batch_size=32、累积=1、学习率 3e-4、weight_decay=0.1、warmup=0.02、min_lr_ratio=0.1、dropout=0.1、compile 关闭。

三组显式使用 BF16，等同当前 RTX 4070 上原方案 auto 的选择，但避免无意切换环境后采用不同精度。A 与 B 参数量相差约 0.43%，报告时注明不是完全相等。

训练采样器每轮用独立的 seed+epoch 随机生成器，使三种架构即使初始化消耗不同数量的随机数，也使用相同的样本顺序；按 epoch 恢复时同样可重建顺序。

A→B 只能说明整套架构组合的差异；B→C 用来观察 SDPA 实现的影响。本批不单独隔离 AMP 收益，也不声称每个架构组件单独有效。后续若做逐组件消融或多种子复验，应另建协议目录，不修改已经开始的本批配置。

## 2. 准备数据与运行环境

本实验需要支持 BF16 的 CUDA GPU。先按 [项目说明](../../README.md#快速开始) 安装依赖，再从项目根目录执行以下命令。

```sh
python -m src.download_data --max_articles 100000
python -m src.prepare_dataset --tokenizer google-bert/bert-base-chinese
python experiments/ablation/run.py prepare
python experiments/ablation/run.py check
```

`prepare` 根据当前语料、代码和配置生成本地 `protocol_lock.json`。`check` 校验协议哈希、CUDA、BF16 和单进程环境。脚本使用当前 Python 解释器启动子进程。

协议锁记录 `src/`、默认配置、三组配置、提示词、执行脚本、train/val/test 元数据及 bin 的 SHA-256。训练或评估前会复核。已经准备好当前数据的本地副本可从 `prepare` 开始。

同一组实验开始后修改上述文件将被拒绝。若确需改变协议，应先保留旧实验并建立新实验批次，不要为了绕过检查而删除锁文件。

公开仓库中的 [历史协议](protocol_lock.original.json) 用于对照 [历史实验报告](../../EXPERIMENT_REPORT.md)。新运行的输入、环境和结果可能与历史实验不同。默认配置中已被所有组显式覆盖的值允许变化，数据、源码和组配置仍按哈希校验。

## 3. 真实训练：你需要执行的命令

建议每组训练后立即评估，依次运行；避免同时抢占一张 GPU。

```sh
python experiments/ablation/run.py train A
python experiments/ablation/run.py evaluate A

python experiments/ablation/run.py train B
python experiments/ablation/run.py evaluate B

python experiments/ablation/run.py train C
python experiments/ablation/run.py evaluate C

python experiments/ablation/run.py report
```

每个 `train` 都从随机初始化完整运行一轮。一轮时间取决于实现、显卡负载和环境。历史数据每组处理了 164,575,488 个训练 token、20,090 次更新。对照实验使用相同的 batch、epoch 和数据。

evaluate 默认做完整验证集评估、绘制本组训练曲线，并执行同一份 20 条提示词的 greedy 生成测试。生成固定最大 80 token、KV Cache 开、重复惩罚 1.1、禁重复 4-gram；其输出可供人工比较，不是问答准确率。

每完成一组都可以运行 report 查看部分结果。未完成组显示 pending/待运行，不会填充模拟指标。

## 4. 结果保存位置

以下路径均相对于项目根目录。

| 内容 | 路径 |
|---|---|
| 三组配置 | configs/ablation/A.json、B.json、C.json |
| A 权重、清单、训练日志 | outputs/runs/ablation-A-e1-s42/ |
| B 权重、清单、训练日志 | outputs/runs/ablation-B-e1-s42/ |
| C 权重、清单、训练日志 | outputs/runs/ablation-C-e1-s42/ |
| 每组验证报告 | 对应 run/evaluations/val_metrics.json |
| 每组生成报告 | 对应 run/generation_benchmark.json |
| 每组训练曲线 | 对应 run/training_curve.png |
| 控制台日志、状态、环境、评估来源证明 | outputs/ablation/A/、B/、C/ |
| 汇总 JSON | outputs/ablation/comparison.json |
| 汇总说明与生成案例 | outputs/ablation/REPORT.md |
| 四项指标对比图 | outputs/ablation/comparison.png |
| 按累计训练 token 对比 Loss | outputs/ablation/loss_by_tokens.png |

训练会更新项目 latest 指针，但本流程始终用明确的组配置和 checkpoint 路径，避免混用旧 baseline。旧 baseline 目录不会被覆盖。相同组的 run 目录已存在时拒绝重训，避免覆盖结果。

## 5. 如何读图与计时

Val PPL 越低通常表示同一数据、同一词表上的预测效果更好；train tokens/s 与生成 tokens/s 分别对应不同工作负载。显存图为 PyTorch allocated 峰值，不是 nvidia-smi 的整卡显存。

训练新增 train_seconds、train_tokens_per_second、peak_cuda_allocated_bytes，每轮写入 training_log。计时包含该轮数据读取、前向、反向、优化器更新和训练日志开销，排除轮末验证与 checkpoint 保存。它是整轮平均吞吐，不是独立预热后的稳态微基准。

execution.json 的 wall_seconds 包含进程启动、加载、初始验证、训练、轮末验证、checkpoint 保存，表示端到端时间。每组环境也会记录，汇总时检查完成组的 token 预算和训练环境相同。

只训练一轮时每组只有一个轮末 Loss 点，图上点之间不存在更细的训练证据。单个种子的差异应当作为初步结果，不作显著性结论。比较表不会自动宣称架构 B 或 C 一定更好。

## 6. 最终测试集

先根据验证集决定方案。方案确定后才对选中组跑 test，例如选 C：

```sh
python experiments/ablation/run.py evaluate C --split test
python experiments/ablation/run.py report
```

如果论文式报告需要全部组的 test 数值，可在超参数冻结后分别评估 A/B/C；不能根据 test 结果反复调参。测试输出独立保存，汇总表允许未评估组的 Test PPL 为空。

## 7. 失败与重跑

启动失败或 Ctrl+C 会记录 failed_or_interrupted。当前 wrapper 故意没有自动 resume，因为一轮预算中途恢复的样本游标没有保存。不要用旧 baseline 续训替代本组。

如果一组留下不完整目录，先检查 outputs/ablation/<组>/train_console.log 并保留失败记录；需要重新安排实验名和锁文件时，应为整批重新建协议，避免混合旧结果。评估失败可以直接重跑 evaluate，它会替换本组同名评估报告与控制台日志，不重新训练。

## 8. 前置流程验证

```sh
python -m unittest discover -s experiments/ablation -p test_tools.py -v
```

测试检查公共参数漂移、文件哈希漂移、缺失结果与部分结果绘图。使用临时模拟记录，不调用训练或模型前向，不把模拟数值写进正式结果。
