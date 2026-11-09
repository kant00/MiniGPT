# generate.py 讲解

源码：[src/generate.py](../../src/generate.py)。

## 1. 总体描述

本文件让已有模型根据提示词逐 token 续写：文本编码 → 模型前向 → 取最后位置 logits → 应用约束 → 选择一个 token → 拼回上下文 → 重复，直到结束符或长度上限。

输入为 checkpoint、tokenizer、提示词及解码参数。核心 generate 返回包含提示词和新内容的一维 token 张量；CLI 将新内容解码后拼接原始提示词打印，不默认保存报告。固定多提示词报告由 benchmark_generation 负责。

它控制如何使用模型已有的概率分布，不更新模型知识。重复惩罚、中文空格修正和正确结束符可以改善输出表现，但不会自动获得事实核验、指令遵循或聊天能力。

## 2. load_model 与 tokenizer 检查

load_model 从 checkpoint 的 model_config、vocab_size、seq_len 重建网络，加载权重，切换 eval，返回模型与 checkpoint。它与 evaluate 中的同名函数目前是两份相似实现。

main 优先取 CLI tokenizer，否则取 checkpoint 中记录的 source，加载后比较词表大小和指纹。然后设置 PyTorch CPU/CUDA 种子。随机种子控制抽样序列，不保证所有设备和运行方式下输出逐位一致。

## 3. sample_next_token：四种选择方式

本函数面向单条样本的一维词表 logits。

| strategy | 操作 | 特点 |
|---|---|---|
| greedy | argmax | 每次选最大分数，忽略 temperature/top_k/top_p |
| temperature | logits 除以温度后采样 | 温度小于 1 更集中，大于 1 更分散 |
| top_k | 保留分数最高的一组，再采样 | k 限定候选规模 |
| top_p | 按概率累计保留前缀，再采样 | 候选数量随分布变化 |

非 greedy 路径要求 temperature>0，最后 softmax 再 multinomial。top_k 按第 k 大分数作为阈值，存在并列时可能保留超过 k 个 token。

top_p 先排序并累计概率，把越过阈值的移除掩码右移一位，从而保留第一个使累计概率达到阈值的 token；最高概率 token 始终保留。它不是要求单个 token 的概率大于 p。

## 4. resolve_generation_token_ids(tokenizer)

优先选择 eos_token_id，没有则选 sep_token_id，二者都没有报错。返回一个结束 token 列表，以及“所有特殊 token 减去选中结束符”的禁用列表。

这与预处理的文章结束符策略一致。它不是同时让所有 EOS 与 SEP 都合法停止；当前选中一个。BERT tokenizer 没有 EOS 时用 SEP，可避免模型永远无法通过训练中的文章边界停止。

## 5. apply_generation_constraints

先复制 logits 并转 float32，再基于完整 generated 历史应用约束。该历史包括提示词，而不只是新生成部分。

### 重复惩罚

对出现过的 token 去重后，正分数除以 penalty，负分数乘以 penalty，让它们都变得不那么有竞争力。要求 penalty>=1，等于 1 不惩罚。不能简单把所有分数都除以 penalty，否则负数反而更接近零。

这是“出现过就惩罚一次”，不是按出现次数不断相乘的频率惩罚。

### no_repeat_ngram_size

n=0 关闭；n=1 禁止所有出现过的 token；n>1 取历史最后 n-1 个 token，在过去窗口中寻找相同前缀，把其曾经跟随的 token 禁掉。

例如 n=3，历史出现过 `[甲,乙,丙]`，当前末尾为 `[甲,乙]`，则下一步禁止丙，防止相同三元组再次出现。该操作只比较 token 序列，不理解句子意义。

### 特殊 token 屏蔽与检查

把合法范围内的禁用 ID 分数设为负无穷。如果整个词表没有有限分数，抛错提示放宽约束。强约束可能损伤正常语言表达，不能只根据重复率下降就认为质量更好。

## 6. generate：输入输出与主循环

函数用 inference_mode 装饰，要求 input_ids 为非空 `(1,T)`，只支持一次生成一条文本。max_new_tokens 不能为负，min_new_tokens 必须在 0 到最大值之间。

它复制输入到 generated。每次取最后位置 logits，应用重复和禁止项，再采样一个 token 拼接。若是选中的结束符立即停止；返回 generated[0]，保留结束符 ID 和整个提示词。

min_new_tokens 的实现是在前若干步临时禁止结束符，达到最小长度后再允许。max_new_tokens=0 时不执行模型前向，直接返回原输入。

直接调用 generate 时仍应由调用方执行 model.eval()，inference_mode 不会自动关闭 dropout。CLI 的 load_model 已执行 eval。

## 7. KV Cache：首次与后续前向有什么不同

有缓存时，第一次把提示词最后 seq_len 个 token 一起输入模型，得到 cached_logits 和每层 K/V，称为 prefill。采样后如果缓存未满，仅把新 token 及已有缓存输入模型，避免重新计算全部历史的投影。

没有缓存时，每一步都把 generated 的最后 seq_len 个 token 重新输入。两条路径使用同一个模型，只是复用中间结果的程度不同。

缓存达到上限后，代码用最新的完整窗口重建全部缓存。它没有实现持续增长的无限上下文，也不是只删除一个旧 K/V 然后永远做常量大小更新。窗口满后缓存加速收益会明显下降，当前位置从新窗口重新计算。

generated 本身保留全部历史，注意力只看尾部窗口；重复约束仍查看全部历史，两者的上下文范围不同。循环最后一次非结束采样后还可能多做一次缓存前向，随后因达到长度上限退出；这部分开销也进入现有基准计时。

## 8. decode_text

调用 tokenizer.decode 后，用正则去掉相邻中文字符/中文标点之间的空白，以及部分括号、标点附近的空白，最后 strip。这样能缓解 WordPiece 解码得到“人 工 智 能”的显示问题，同时保留通常的英文单词间空格。

它是显示清理，不是重新分词。使用 `\s+` 意味着相关中文位置间的换行也可能被移除；因此不能用它逐字符还原原始排版。

## 9. compare_strategies

对同一个 prompt 依次运行 Greedy、Temperature、Top-K、Top-P，保留共同的缓存与重复约束设置，并打印原提示词加新内容。

比较分支内部固定温度 0.8、Top-K=50、Top-P=0.9；不是直接套用主 CLI 的全部采样参数。种子在 main 中只设置一次，没有每种策略单独重置，所以后续策略消费的是继续前进的随机数序列。

## 10. main、默认值和用法

main 使用 generate 配置分区，默认 top_p、temperature=0.7、top_p=0.85、penalty=1.1、no_repeat_ngram_size=4、max_new_tokens=80、启用 KV Cache。

核心 Python generate 函数默认 greedy、penalty=1.0、n=0，与 CLI 默认不同。编写脚本调用时应显式传入需要的参数。

```powershell
python -m src.generate --prompt '人工智能的发展'
python -m src.generate --prompt '人工智能的发展' --strategy greedy
python -m src.generate --compare --max_new_tokens 60
python -m src.generate --no_kv_cache
```

主入口仅解码 output 中超过原 prompt 长度的部分，再拼接用户原文，因此提示词显示保留原来的排版。它没有聊天历史管理、system/user/assistant 模板或逐 token 流式显示。

## 11. 阅读检查

读完应能解释为什么只取 logits 的最后一个位置、为什么缓存的是 K/V、为什么 EOS 需要与预处理保持一致，以及为什么重复约束同时可能改善重复和损害准确表达。
