# prepare_dataset.py 讲解

源码：[src/prepare_dataset.py](../../src/prepare_dataset.py)。

## 1. 总体描述

本文件把文章 JSONL 变成训练可直接读取的连续 token 文件。核心流程是：正文规范化 → 质量过滤 → 精确去重 → 近似去重 → 按文档确定 split → 加标题并编码 → 追加结束符 → 写二进制文件 → 校验并发布元数据。

输入为项目 data 目录中的 `wiki_zh.jsonl`。默认输出 train、val、test 三组文件，每组包含 `*_tokens_seq256.bin` 和 `*_seq256.meta.json`。bin 只保存整数，meta 描述这些整数的类型、长度、词表、样本数与处理统计。

这里不把全部 token 放进一个 Python 列表，也不保存完整训练张量。文章逐条处理，训练时再用 memmap 读取所需窗口。不过去重集合和近似索引会随文章数量增长，不能说整个过程内存恒定。

## 2. 为什么一条样本需要 seq_len + 1 个 token

模型学习“已知前文预测下一项”。假设 seq_len=4，连续流为 `[10,11,12,13,14,15,16,17,18]`：

| 样本 | 读取窗口 | 输入 x | 目标 y |
|---|---|---|---|
| 0 | 10,11,12,13,14 | 10,11,12,13 | 11,12,13,14 |
| 1 | 14,15,16,17,18 | 14,15,16,17 | 15,16,17,18 |

窗口步长为 seq_len，相邻窗口共享一个边界 token，但目标位置没有因此重复。N 个 token 可提供 `floor((N-1)/seq_len)` 个完整样本，余下不足窗口的尾部不会成为样本。

实际主流程不在此处展开窗口，只记录样本数，由 train.TokenDataset 在读取时切片。

## 3. split_token_stream(all_ids, seq_len, val_ratio)

这是保留给测试和小数据的内存版工具。它计算完整窗口数，要求至少两个窗口，先按比例分配 train/val，再通过 `Tensor.unfold` 建立长度 seq_len+1、步长 seq_len 的窗口，最后返回连续张量。

它按 token 流位置分割，不具备主流程的文档级三路划分，不应把它当成当前生产预处理入口。函数中的数量校验用于防止窗口边界少一条样本等错误。

## 4. 文档规范化、哈希与划分

### normalize_document(title, text, include_title)

对标题和正文做 Unicode NFC 规范化，统一正文 CRLF 为 LF，清理首尾空白。include_title 为真且标题非空时返回“标题 + 换行 + 正文”。NFC 用于规范等价字符表示，不会统一简繁体，也不会清除所有内部空白。

### document_hash(text)

对规范化正文计算 SHA-256。精确去重故意不包含标题，避免正文相同但标题不同的镜像文章逃过去重。空白或内容有差别的正文仍可能得到不同哈希，需要后续近似去重。

### choose_split(document_id, val_ratio, test_ratio, split_seed)

对 `seed:id` 计算 SHA-256，取前 8 字节映射到 [0,1)。先分配 test 区间，再分配 val，其余为 train。默认比例约 95%/4%/1%，这是文档的概率划分，不保证文章数或 token 数精确达到比例。

有 id 时使用原始 id，没有时用正文哈希。同一个 id 和 seed 总会进入同一个集合。但按“先遇到先保留”的去重会受输入顺序影响，因此稳定划分不等于任意重排后完整输出逐字节相同。

## 5. quality_rejection_reason(...)

按顺序返回第一个拒绝原因，通过时返回 None。统计中的原因互斥，并不是一篇文章同时累计所有违规项。

| 原因 | 判断方法 | 默认阈值 |
|---|---|---|
| too_short | 正文字符数太少 | min_chars=50 |
| empty | 没有非空白字符 | 无 |
| low_cjk_ratio | 指定字符范围内字符 / 非空白字符过少 | 0.1 |
| control_chars | 异常 Cc 字符 / 全部字符过多 | 0.01 |
| repeated_character_run | 最长连续相同字符长度 / 正文长度过大 | 0.2 |

中文判定采用代码中的 `\u3400` 到 `\u9fff` 范围，是粗略语言启发式。连续重复检测指“啊啊啊啊”这样的字符串，不是重复句子或重复段落。

换行、回车、制表符虽属于 Unicode Cc，但被明确排除在异常控制字符之外。此前把正常换行计作异常会大面积过滤百科文章，这也是后面增加整体拒绝比例保护的原因。

## 6. simhash64(text, max_features=256)

先移除空白，然后使用字符 5-gram 特征。长文章通过 step 均匀抽样，控制特征数量；每个特征用 8 字节 BLAKE2b 哈希，逐比特加一或减一；最终每位权重非负则设置为 1，得到 64 位指纹。

两篇文本共享较多特征时，指纹通常更接近。短于 5 字符时直接哈希全文。SimHash 是近似方法，字符抽样可能遗漏局部差异，也不等于语义相似度。

### NearDuplicateIndex.__init__ 与 check_and_add

构造函数保存最大汉明距离和 buckets 字典。查询时把 64 位拆成四个 16 位片段，以 `(band, value)` 查找候选，再用 `(a ^ b).bit_count()` 计算不同位数。

存在距离不大于阈值的候选时返回 True，表示重复；否则把指纹加入四个桶并返回 False。名称中的 add 只在未命中时发生。

默认阈值 3 时，至多三位不同必然留有一个完全相同的片段，候选召回有相应保证。虽然参数允许到 64，超过 3 后四片段索引不再保证召回所有符合阈值的指纹。因此不能把调大阈值理解为严格的全量距离搜索。

## 7. 存储和保护函数

`token_dtype(vocab_size)`：词表至多 65536 时选小端 uint16，再大则选小端 uint32，超出 uint32 表示范围报错。词表 21128 使用每 token 两字节；训练读取后才转换为 PyTorch 需要的 int64。

`dataset_paths(split, seq_len)`：统一 bin 和 meta 文件名，返回两个 Path。文件名包含 seq_len，bin 本身仍然是连续 token 流。

`write_metadata(path, metadata)`：先写 JSON 临时文件，再 replace 发布。它不是在原 JSON 后追加。

`validate_quality_rejection_ratio(input_articles, rejection_counts, max_ratio)`：计算质量拒绝数 / 已读文章数，超过默认 0.8 时抛错。精确重复与近似重复不算质量拒绝。保护发生在处理结束、正式发布前，能够防止静默发布极低保留率数据，但不是实时中止开销的机制。

## 8. build_dataset(...) 完整控制流

1. 验证长度、划分比例、token 上限、过滤阈值，检查源文件存在且非空。
2. 加载 tokenizer，优先使用 EOS，否则 SEP；两者都没有则拒绝处理。
3. 确定存储 dtype。日志中的 embedding 成本仍按硬编码 d_model=384 估算，不代表 JSON 当前的 512 维模型。
4. 为每个 split 建立临时 bin、token 计数、文档计数，以及去重集合。
5. 逐行解析 JSON，分别准备“用于过滤去重的正文”和“用于训练的标题加正文”。
6. 通过过滤、去重后确定 split，编码时关闭自动特殊 token，再手动追加文章边界，并检查 ID 范围。
7. 如果有 max_tokens，按所有 split 的总 token 预算裁剪，再通过 NumPy tofile 写入对应文件。
8. 正常结束后 flush、fsync、close；普通异常时关闭文件并清理临时输出。
9. 检查整体质量拒绝比例、各 split 至少一个样本、文件字节大小。
10. 逐 split 替换 bin，再发布 meta，返回各 meta 路径组成的元组。

max_tokens 不是只限制 train，也不是每个 split 分别限制。它可能把最后一篇文章及其结束符截断；小预算还可能让 test 不够一个窗口而报错。

## 9. 元数据怎样使用

训练必须依赖的字段有 dtype、data_file、num_tokens、num_samples、seq_len、vocab_size 和 tokenizer。num_documents 是当前 split 的文档数；input_documents、retained_documents、retention_ratio、rejection_counts 等是本次处理全局统计，复制在各 split meta 中。

精确和近似去重数量当前主要输出在日志中，没有各自独立的计数字段写入 meta。source_file 记录文件名，不是原始语料内容哈希。

## 10. 文档边界与可靠性边界

文档先分配到集合，因此一篇正常完整文章不会被拆到 train 和 val。每个 split 内文章 token 仍连续拼接，一个训练窗口可以横跨文章结束符；当前没有按文章构造隔离 attention mask，模型可以注意到窗口中前一篇文章的内容。

bin 与 meta 是逐文件替换，不是全部 split 的整体事务。发布中途断电仍可能得到部分新、部分旧文件。重新预处理会替换同名数据，test_ratio=0 也不会自动清理以前的 test 文件。

## 11. 使用方式与阅读检查

```powershell
python -m src.prepare_dataset --seq_len 256
python -m src.prepare_dataset --max_tokens 2000000
```

main 把 `no_dedup`、`no_near_dedup`、`no_title` 反转成 build_dataset 的正向布尔参数。运行前确认是否要替换当前数据。读完应能说明：为什么先按文档划分、为什么保存 N 个 token 只能得到 floor((N-1)/T) 条样本、为什么词表不同必须重新准备数据。
