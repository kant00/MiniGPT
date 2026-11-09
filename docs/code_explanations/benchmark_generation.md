# benchmark_generation.py 讲解

源码：[src/benchmark_generation.py](../../src/benchmark_generation.py)。

## 1. 总体描述

本文件把单次生成包装成固定提示词的回归基准：加载一次模型，对多条提示词逐条生成，保存文本、token、耗时和重复指标，再汇总为 JSON。它用于比较解码改动、缓存开关和模型 checkpoint 的表现。

输入为 checkpoint、tokenizer、提示词集合、随机种子和生成设置；输出默认是该 checkpoint 所属实验的 generation_benchmark.json。它复用 generate 中的实际生成实现，因此停止符、重复约束和窗口重建规则保持一致。

它不是问答正确性评测。输出越多样、越快并不意味着内容越真实，需要同时查看样例和任务评分。

## 2. DEFAULT_PROMPTS 与 load_prompts(path)

没有外部文件时使用五条内置提示词，包含首都、人工智能、长江、故事和科学研究主题。它们适合快速发现明显回归，数量不足以代表全面能力。

load_prompts 逐行处理文件，跳过空行。能解析为 JSON 对象时取 prompt 字段；JSON 字符串直接使用；不能解析成 JSON 时把原行当提示词。最后过滤空值并转字符串，完全没有提示词时报错。

因此格式损坏的 JSON 行可能被当成普通文本继续运行，而不是严格格式报错。多行提示词应在 JSONL 的单行字符串中使用转义换行；普通文本文件一行就是一个独立 prompt。

## 3. distinct_n(token_ids, n)

先验证 n>0，序列短于 n 时返回 0。否则构造所有相邻 n-gram，计算唯一 n-gram 数除以总 n-gram 数。

例如 `[1,2,1,2]` 的 2-gram 是 `(1,2),(2,1),(1,2)`，Distinct-2=2/3。Distinct-1 统计单个 token 的多样性，Distinct-2 统计相邻组合多样性。

## 4. repetition_ratio(token_ids, n=3)

同样建立 n-gram，返回 `1 - 唯一数量/总数量`；长度不足 n 时返回 0。默认用于 3-gram 重复率，反映重复组合占比。

它不衡量语义重复，换个措辞说同一件事可能得低重复率；随机乱码也可能非常多样。与 distinct_n 不同，本函数没有显式检查 n>0，调用方应使用正整数。

## 5. benchmark：每条样本如何测试

先得到结束与禁用特殊 ID。每条 prompt 单独分词并送设备，CUDA 环境先 synchronize 等待之前的 GPU 工作完成，再开启计时。

调用 generate 后再次 synchronize，测得完整生成调用延迟；按照原提示词 token 长度切出 generated_ids，最后在计时之外解码文本并计算指标。

这段计时包含 prefill、后续逐 token 生成、约束和缓存重建，不包括分词、模型加载、最终解码及报告保存。它没有首 token 延迟、逐 token 延迟分位数或独立 warmup 阶段，因此第一条可能受到首次运行开销影响。

## 6. samples 字段

| 字段 | 解释 |
|---|---|
| prompt | 原提示词 |
| completion | 仅新生成文本，经过显示清理 |
| generated_token_ids | 仅新生成 ID，便于复查 |
| stopped_on_boundary | 最后一个新 ID 是否为选中的结束符 |
| generated_tokens | 新生成 ID 数量 |
| latency_seconds | 该次 generate 完整调用时间 |
| tokens_per_second | 新 token 数 / 本条耗时 |
| distinct_1、distinct_2 | token 与二元组多样性 |
| repetition_3gram | 三元组重复率 |

结束符虽在显示文本中通常被跳过，仍计入 generated_tokens 和 n-gram 指标。max_new_tokens=0 时新输出为空，停止标志为 false。

## 7. summary 的聚合口径

总吞吐=`所有新 token 数之和 / 所有生成延迟之和`，不是逐样本 tokens/s 的算术平均。mean_repetition_3gram 则是每条重复率的简单平均，短回答与长回答权重相同。

benchmark 本身不固定种子，main 在调用前固定一次；如果自行在 Python 中调用，要负责设置种子。随机策略下更改提示词顺序也会改变后续随机数消费，固定 seed 之外还应固定提示词顺序。

## 8. main：加载、报告与保存

读取 benchmark_generation 配置后解析 checkpoint、选择设备、加载模型与 tokenizer、验证词表，设置种子，然后运行 benchmark。

报告额外记录 checkpoint、strategy、seed、kv_cache、重复惩罚、n-gram 约束和最小输出长度。当前没有保存 temperature、top_k、top_p、max_new_tokens、实际设备等全部测试设置；正式复现时需额外保留配置与命令，不能只靠这份 JSON 还原所有条件。

默认按显式 checkpoint 的所属实验保存，省略 checkpoint 时才使用 latest。指定 output 时直接交给 atomic_json_dump，相对路径按当前工作目录处理。相同输出路径再次运行会替换已有报告。

## 9. 使用与公平比较

```powershell
python -m src.benchmark_generation --strategy greedy --max_new_tokens 40
python -m src.benchmark_generation --strategy greedy --max_new_tokens 40 --no_kv_cache --output 'outputs/benchmark_no_cache.json'
```

第一条会保存到默认实验报告，第二条单独输出。比较缓存开关时固定 checkpoint、提示词、顺序、解码设置和硬件，最好预热并重复测量；若两次结束长度不同，应同时解释长度对吞吐的影响。

长提示词或生成超过上下文上限后，generate 会重建完整窗口缓存，不能把短文本速度外推到所有长度。读完应能区分“生成性能”“表面重复”“任务质量”三种结论。
