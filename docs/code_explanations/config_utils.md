# config_utils.py 讲解

源码：[src/config_utils.py](../../src/config_utils.py)。

## 1. 总体描述

本模块把 JSON 配置接入 argparse，使下载、预处理、训练、评估和生成等入口使用同一套参数规则。它读取配置、检查字段、转换类型，再让命令行覆盖配置；不负责训练或解释各个参数的业务含义。

输入是一个已经声明参数的 `ArgumentParser`、目标分区名和可选 argv；输出是包含最终参数的 `Namespace`，并附带绝对路径形式的 `args.config`。读取文件是主要外部行为，本模块不保存配置。

完整优先级是：显式 CLI > 自定义配置对应分区 > 默认配置对应分区 > argparse 原始默认值。最后一层只在前面的配置没有该字段时起作用。

## 2. 路径与常量

`BASE_DIR` 由当前源文件位置计算，`DEFAULT_CONFIG_PATH` 固定指向项目的 [configs/default.json](../../configs/default.json)。因此默认配置位置不依赖终端在哪个目录。

但是用户给出的相对 `--config` 路径按当前工作目录解析；配置中普通 Path 字段只转换成 Path，不会统一变成“相对于配置文件”的路径。后续由各调用模块处理，使用项目根目录或绝对路径最清楚。

## 3. load_config(path)

函数将输入转换为 Path，解析相对位置，检查文件存在，再以 UTF-8 读取并解析 JSON。它要求根节点是对象，数组、字符串等根节点会报错。

返回普通字典，不会检查所有业务分区，也不会验证 `format_version` 的取值。文件顶部提到的严格校验主要发生在调用模块选中的分区中。

## 4. _convert_config_value(action, value)

此函数复用 argparse 已有的参数定义作为校验依据，避免再维护一份字段类型表。

| 参数类型 | 转换规则 |
|---|---|
| null | 直接返回 None，是否允许由后续业务决定 |
| store_true / store_false | JSON 必须是真正的布尔值 |
| append | JSON 必须是数组，逐项转换，并检查 choices |
| int | 必须是整数；显式排除 Python 中也属于 int 的 bool |
| float | 接受整数或小数，拒绝 bool，再转 float |
| str | 必须是字符串 |
| Path | 转换为 Path 对象 |
| choices | 转换后必须属于预设选项 |

例如 `"epochs": "8"` 会被拒绝，`"epochs": 8` 才正确；`"amp": "bf16"` 合法与否由 parser 的 choices 决定。类型正确不代表数值范围正确，负 batch size 等仍由训练函数检查。

## 5. parse_args_with_config(...)

这个函数采用两次解析：

1. 给主 parser 添加 `--config`。
2. 用一个只认识 `--config` 的 bootstrap parser 找到配置位置，其他参数暂时保留。
3. 要求所选配置有目标分区对象，例如 `train`。
4. 读取默认分区，做浅层字典合并；自定义字段覆盖默认字段。
5. 遍历 parser 的 actions，以 `dest` 建立字段表，拒绝目标分区中的未知字段。
6. 调用转换函数，然后用 `parser.set_defaults` 注入配置。
7. 正式解析 CLI，得到最高优先级的覆盖结果。

配置键使用 `dest`。例如训练开关的 CLI 是 `--compile`，JSON 字段是 `compile_model`。

这是分区内的浅合并，不是任意层级的递归合并。一个仅有 `train` 分区的配置可用于训练，但拿去运行 evaluate 会因缺少 `evaluate` 分区而失败。

## 6. 布尔开关与列表覆盖

JSON 已经设为 true 时，只提供 `--xxx` 无法关闭它，所以业务入口还声明了反向参数，例如 `--no_plot`、`--no_compile`。反向参数映射到同一个 dest；配置里仍只保留一份布尔字段。

对于重复使用的 `--tokenizer`，append 原本可能把 CLI 值追加到配置数组。这里检测到标准的独立 CLI 选项后先清空默认列表，实现替换。当前检测按 argv 中完整字符串匹配，使用 `--tokenizer value` 最明确；不要假定 `--tokenizer=value` 也经过相同清空分支。

## 7. 使用例子

```powershell
python -m src.train --epochs 1 --batch_size 16 --run_name config-demo
```

这会采用默认 JSON 的模型架构，只覆盖命令行指定的训练轮数、batch size 和实验名。Python 内部直接调用 `train()` 或 `build_model()` 不会经过本函数，因此不会自动获得这些 JSON 默认值。

## 8. 阅读检查

遇到参数与预期不符时，按 CLI、选中分区、默认分区、函数入口的顺序检查。配置解析结果不是实验结果；训练模块还会把源配置与实际生效参数分别保存。
