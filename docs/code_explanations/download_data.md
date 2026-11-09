# download_data.py 讲解

源码：[src/download_data.py](../../src/download_data.py)。

## 1. 总体描述

本文件把外部文章统一变成 JSONL，是数据流水线第一站。它有三条路径：Hugging Face 已清洗 Wikipedia、Wikimedia 原始压缩 XML、用户提供的本地 TXT。三者最终都输出项目 data 目录中的 `wiki_zh.jsonl`，每行包含 `id`、`title`、`text`。

它只完成文章获取和初步格式清理，不分词，不划分验证集，不构建训练张量。深一层的质量过滤、去重和划分在 prepare_dataset 中。

写入先使用 `wiki_zh.jsonl.part`，校验成功后再替换正式文件，避免普通下载失败直接毁掉旧数据。已有有效文件达到在线下载目标时会复用，但不会检查它是否来自本次指定数据源。

## 2. 主要常量

`HF_DATASET` 为 `wikimedia/wikipedia`，`HF_CONFIG` 为代码固定的 `20231101.zh`；不是运行当天的百科快照。`HF_MIRROR_ENDPOINT` 用于可选镜像。`DUMP_URL` 指向原始 dump 的一个分片，不能理解为所有中文页面的完整下载地址。`MEDIAWIKI_NS` 用于定位 XML 标签的命名空间。

## 3. clean_wikitext(text)

这个函数只服务原始 dump 路径。处理包含注释、脚注、模板、表格、文件与分类链接、普通链接、外链、HTML、章节标题、粗斜体标记和多余空白。

典型转换：`[[人工智能|AI]]` 保留 `AI`；`[https://example.com 描述]` 保留描述；脚注和表格整体移除。模板最多迭代去除五轮，遇到无变化提前停止。

顺序有意义：先去掉大块噪声，再清理普通标签，避免大量格式内容残留。它仍是正则启发式清洗器，不能完整解析所有嵌套 Wikitext；“保留纯中文”也不是严格的语言过滤保证，英文正文仍可能存在。

## 4. validate_jsonl(path, expected_rows)

返回 `(valid, rows, reason)`，而不是直接抛出所有格式错误。依次检查文件存在、非空、逐行能解析为对象、包含三个字段、正文为非空字符串，最后可检查文章数。

空白行会跳过；不要求 title 和 id 必须是某种精确类型，也不检查文章事实正确性、去重情况或中文比例。下载过程利用此函数决定复用旧文件还是发布新文件。

## 5. reset_part_file 与 parse_version_triplet

`reset_part_file(part_path)` 删除上次未完成的临时文件并记录日志。这说明下载是重新开始，不是断点续传。

`parse_version_triplet(version)` 使用正则提取版本开头的三段数字，返回整数元组；无法匹配返回 None。`download_hf_articles` 使用它执行代码内预设的依赖版本检查。该检查是一条现有代码规则，不代表本文重新验证了所有相关库版本。

## 6. download_hf_articles(max_articles, hf_mirror)

执行顺序如下：设置镜像环境变量，延迟导入 datasets/huggingface_hub，检查版本，检查已有文件，打开流式数据集，逐文章过滤并写临时文件，校验后替换。

镜像变量要在 Hugging Face 库导入前设置，否则已初始化的客户端可能不会采用新地址。`streaming=True` 让程序逐项取得数据，不要求先加载完整数据集。

正文去首尾空白后不足 50 字符会跳过。文章数计数器只统计写入的样本，每 1000 篇输出速度和预计剩余时间。达到目标停止；源提前结束但产生了有效文章时，可以发布较小文件并警告。

普通异常会清理临时文件、保留旧文件并给出备用方案。这里的异常处理不等价于对所有中断和系统故障的事务保证；后续运行会清理遗留 part 文件。

## 7. download_wiki_articles(max_articles)

这一分支连接四种表示：HTTP 压缩字节 → BZ2 解压字节 → XML 元素 → 清洗后的正文。

`requests.get(stream=True)` 和 `iter_content` 按约 1MB 压缩块读取；`BZ2Decompressor` 保留跨块解压状态；`XMLPullParser` 累积字节，只在完整 page 结束后处理。

只保留命名空间 `ns=0` 的正文页面，跳过重定向，提取 title 与 revision 中的 text，调用 clean_wikitext，过滤清洗后少于 50 字符的文章，再保存为 JSONL。达到上限时停止网络读取。

`elem.clear()` 释放已处理页面的内容，降低内存压力。但“任何规模下只占一篇文章内存”不能作为严格保证：解析器树结构和空元素引用仍可能有额外开销。当前实现也不应被当成支持任意多成员 BZ2 文件的通用解压器。

## 8. from_local_txt(raw_dir)

只查找目录直接包含的 `*.txt`，不递归子目录。以 UTF-8、忽略非法字节读取；非空行加入 buffer，空行触发段落结束，文件末尾还会处理最后一个 buffer。

段落内部使用 `"".join(buffer)`，因此原行间没有自动加入空格或换行。段落至少 30 字符才写入；标题采用文件名，ID 使用递增计数。这里没有使用 max_articles，上限参数不控制本地段落数。

本地初筛 30 字符与下游默认 min_chars=50 不同，因此写进 JSONL 的短段落仍可能被下游拒绝。读取非法字节被忽略也可能损失文字，需要自行检查数据质量。

## 9. main 与用法

main 解析 `download_data` 配置分区，并验证 max_articles 大于 0。分支优先级为 from_local，然后 wikimedia，最后 huggingface。

```powershell
python -m src.download_data --max_articles 100000
python -m src.download_data --hf_mirror
python -m src.download_data --source wikimedia
python -m src.download_data --from_local
```

新数据成功发布后可能替换旧 JSONL，因此它是数据生产命令。下载得到文章不代表它们已经通过训练级质量检查，下一步阅读 prepare_dataset。
