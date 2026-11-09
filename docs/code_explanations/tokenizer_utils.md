# tokenizer_utils.py 讲解

源码：[src/tokenizer_utils.py](../../src/tokenizer_utils.py)。

## 1. 总体描述

本模块统一寻找 tokenizer、加载 tokenizer、生成身份元数据，并检查不同阶段的词表是否一致。它是文本和模型之间的接口：模型只看到数字 ID，不知道这些数字最初对应哪个汉字或子词。

主要输入为本地 tokenizer 目录或 Hugging Face ID；`load_tokenizer` 返回 `(tokenizer, metadata)`。预处理使用它编码文章，生成使用它编码提示词并解码结果，分析工具使用它统计分词成本。本模块没有 CLI，也不训练新 tokenizer。

当前项目使用中文 BERT 的 WordPiece 词表。使用 BERT tokenizer 不代表模型是 BERT：网络仍然是本项目定义的因果 Decoder-only MiniGPT。

## 2. resolve_tokenizer_source(source)

代码先建立本地候选列表：显式 source、`MINIGPT_TOKENIZER_PATH` 环境变量、项目附近三处常见的 `pretrain_models/bert-base-chinese` 目录。逐个检查是否存在，返回第一个存在路径的绝对形式。

如果候选均不存在，显式给出的绝对路径或以 `.`、`~` 开头的路径会报错；其他 source 作为 Hub ID 返回；没有 source 时返回 `bert-base-chinese`。

这里有一个需要按真实控制流理解的边界：代码先检查所有本地候选，再返回显式 Hub ID。因此存在自动发现的本地目录时，它可能优先于显式在线 ID；不存在的显式路径也可能在轮到报错前被另一个本地候选替代。环境变量当前主要按本地路径使用，并没有独立的 Hub ID 回退逻辑。

所以比较不同 tokenizer 时要核对返回的 `metadata["source"]` 和指纹，不能只看你输入的名字。明确传入存在的本地目录最容易验证。

## 3. tokenizer_metadata(tokenizer, source)

`get_vocab()` 返回 token 到 ID 的映射。函数按 ID 排序，用稳定 JSON 分隔格式序列化为 UTF-8 字节，再计算 SHA-256。

例如两份词表都包含“北”和“京”，但 ID 顺序相反，模型对同一个数字的理解也相反。比较词表大小无法发现这种错误，映射指纹可以。

返回字段：

| 字段 | 含义 |
|---|---|
| source | 实际解析到的来源 |
| class | tokenizer 的实际 Python 类 |
| vocab_size | `len(tokenizer)`，包含计入长度的新增 token |
| vocab_sha256 | token 与 ID 映射的哈希 |
| special_tokens_map | UNK、SEP、PAD 等特殊符号定义 |

特殊 token 通过 JSON 往返转换成可保存的基本类型。指纹目前只覆盖词表映射，不覆盖完整 normalizer、pre-tokenizer、BPE merges 或特殊 token 角色配置；它是重要的一致性检查，但不是 tokenizer 全部行为的证明。

## 4. load_tokenizer(source)

先解析来源，再调用 `AutoTokenizer.from_pretrained(..., use_fast=True)`，最后计算元数据。`use_fast=True` 表达优先使用快速实现，实际返回类仍应查看 metadata。

加载来源可以触发文件读取、库缓存或网络访问。本项目的词表身份检查发生在加载后，不能因为目录同名就跳过。

## 5. validate_tokenizer_metadata(expected, actual, context)

expected 通常来自 checkpoint，actual 来自刚加载的 tokenizer。缺少 expected 会立即报错；接着比较 `vocab_size` 和 `vocab_sha256`。

`context` 只是用于生成可理解的错误信息，例如“生成基准”或 checkpoint 名称。函数成功时返回 None，不修改对象，也不尝试自动修补不匹配词表。

## 6. 特殊 token 与模型边界

此模块记录特殊 token，但“选择哪个结束符”由预处理和生成模块完成：优先 EOS，没有时使用 SEP。当前中文 BERT tokenizer 常用 SEP 作为文章边界。

编码正文使用 `add_special_tokens=False`，避免自动添加 BERT 风格的 CLS/SEP，再由预处理手动追加文章结束符。分词分析则不追加结束符，所以它统计的 token 总数不等于训练数据总数。

## 7. Python 使用方式

```python
from src.tokenizer_utils import load_tokenizer, validate_tokenizer_metadata

tokenizer, info = load_tokenizer(r"google-bert/bert-base-chinese")
ids = tokenizer.encode("人工智能的发展", add_special_tokens=False)
print(ids, info["source"], info["vocab_sha256"])
validate_tokenizer_metadata(info, info, "示例")
```

切换到另一个 BPE tokenizer 后，必须重新准备数据，并使用匹配词表的模型权重。它不会让已有 WordPiece checkpoint 自动适配新的 ID 语义。
