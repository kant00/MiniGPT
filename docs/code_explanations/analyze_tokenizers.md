# analyze_tokenizers.py 讲解

源码：[src/analyze_tokenizers.py](../../src/analyze_tokenizers.py)。

## 1. 总体描述

本文件在同一批原始正文上运行多个 tokenizer，比较分词长度、未知 token 比例、词表大小和 embedding 参数成本。它帮助判断一种词表在当前中文语料上是否过度切碎、是否丢失字符、是否占用过多模型参数。

输入为文章 JSONL、候选 tokenizer 来源列表、取样文章数和假设的 d_model；输出为比较报告与终端摘要，默认写 outputs/tokenizer_comparison.json。整个过程不训练 tokenizer、不加载 MiniGPT checkpoint，也不测量训练后的 PPL。

## 2. load_text_sample(path, max_articles)

要求 max_articles>0，按文件顺序读取非空 JSONL 行，提取 text 并 strip，只保留非空正文，达到数量上限停止。JSON 语法错误会带上行号；没有有效正文会报错。

它取的是前 N 篇有效正文，不是随机样本，不按主题分层，也不执行 prepare_dataset 的质量过滤或去重。样本会保存在 Python 列表中供所有候选复用，所以候选比较相同文本，但内存随采样规模增长。

这里只使用 text，不包括 title 和手动结束符，因此 token 数与正式预处理可能不同。字符数使用 Python len，包含正文内部空白和换行，不是只统计汉字。

## 3. analyze_tokenizer(tokenizer, metadata, texts, d_model)

先检查 d_model>0，计算所有文本的字符数与 UTF-8 字节数。逐篇调用 encode(add_special_tokens=False)，累计 token_count；若 tokenizer 有 unk_token_id，则数出等于该 ID 的输出数量。

没有生成任何 token 时报错；否则返回来源、实际类、词表指纹和以下指标：

| 字段 | 计算 | 用途 |
|---|---|---|
| vocab_size | metadata 中的词表大小 | 词表规模 |
| embedding_parameters | vocab_size × d_model | 假设隐藏维度下的输入词表矩阵参数量 |
| articles / characters / utf8_bytes | 实际样本规模 | 确认对比材料一致 |
| tokens | 不含手动文档结束符的总 ID 数 | 分词后的长度 |
| tokens_per_character | tokens / characters | 每字符平均拆成多少 token |
| characters_per_token | characters / tokens | 一个 token 平均覆盖多少字符 |
| tokens_per_utf8_byte | tokens / utf8_bytes | 相对于字节长度的分词成本 |
| unknown_tokens / unknown_rate | UNK 数量 / token 数量 | 未知 ID 发生情况 |
| has_document_boundary_token | 有 EOS 或 SEP | 能否满足当前预处理边界要求 |

词表 21128、d_model=512 时 embedding 为 10,817,536 参数。MiniGPT 的输出 head 与输入 embedding 共享权重，因此不能无条件再把这部分乘二；这个数字也不是完整模型参数量。

## 4. 一个小例子

假设同一份 1000 字符的文本，A 编码为 1100 tokens，B 编码为 700 tokens，则 A 的 tokens/character=1.1，B=0.7。固定 256 token 上下文时，B 通常能容纳更多原始字符，但这只是该样本上的长度推断。

如果 B 的词表大很多，embedding 与输出词表计算也更贵；如果较低 token 数来自把复杂字符折叠成 UNK，则还损失了信息。因此不能只挑 tokens/character 最低的候选。

UNK 比例为零也不证明完全无信息损失：normalizer 等处理可能改变原文；本文件没有检查 encode/decode 的往返保真度。

## 5. compare_tokenizers(sources, input_path, max_articles, d_model)

只读取一次样本，然后按来源顺序调用公共 load_tokenizer，再交给 analyze_tokenizer，最后返回 input 绝对路径、d_model、sample_articles 和 results 数组。

公共 tokenizer 来源解析当前会先扫描本地候选，显式在线 ID 可能被已存在的本地 tokenizer 选中。对比时务必核对每行 source、class、vocab_sha256，确保实际加载的是不同候选；相同结果不一定说明两种分词算法相同。

本文件不实现 BPE 训练。比较现成 BPE 时需要其可加载配置、词表和算法文件，并确保公共加载函数真正选到了该来源。

## 6. main 与配置

main 的 --tokenizer 使用 action=append，允许重复给出来源，也可以在 JSON 的 analyze_tokenizers 分区填写数组。当前默认数组为空，未指定候选会报错，不会自动运行一份比较。

函数参数默认 d_model=384，而当前 JSON 分区为 512，CLI 经过配置解析后通常采用 512。input 和 output 的配置相对路径按当前工作目录使用，不按配置文件目录重新定位。

```powershell
python -m src.analyze_tokenizers --tokenizer 'google-bert/bert-base-chinese' --max_articles 1000 --d_model 512
```

候选本地目录准备好后，可重复添加 --tokenizer。main 用 atomic_json_dump 保存完整报告，再打印词表、tokens/char、UNK 与 embedding 摘要；重复运行相同 output 会替换原报告。

## 7. 如何把结果用于实验

报告适合筛选候选和估算上下文/参数成本。最终判断还需要同数据预算、合理模型规模下的训练与任务评测；不同词表下的 token PPL 不能直接当作统一标尺。

当前报告没有编码速度、内存峰值、分词长度分位数、来源文件内容哈希，也没有完整 tokenizer 行为指纹。需要这些结果时应新增实际测量，不能从现有比率直接推导。

读完应能解释：小词表节省 embedding，却可能使文本更长；大词表可能压缩序列，却增加词表计算；分词成本与模型最终能力需要分别验证。
