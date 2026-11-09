"""
加载项目级 JSON 配置，并把命令行参数作为最高优先级覆盖项。

默认配置：
  configs/default.json

配置按可执行模块分区，例如 train、prepare_dataset、generate。调用方式：

  from src.config_utils import parse_args_with_config
  args = parse_args_with_config(parser, "train")

命令行使用方式：
  python -m src.train                         # 使用 configs/default.json
  python -m src.train --config configs/smoke.json
  python -m src.train --config configs/smoke.json --batch_size 8

优先级固定为：显式命令行参数 > --config 指定文件 > 默认配置文件。
未知分区、未知参数和类型/choices 不匹配会立即报错，避免拼写错误被静默忽略。
"""

import argparse
import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "default.json"


# 用途：读取 UTF-8 JSON 配置并要求根节点为对象。
# 调用时机：统一参数解析读取默认/自定义配置，训练复制来源配置时调用。
# 输入输出与注意：输入 Path，返回字典；相对路径按当前工作目录解析，缺文件或无效 JSON 报错，不在此处验证全部业务分区。
def load_config(path: Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"配置文件不是有效 JSON：{path}（{exc}）") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件根节点必须是 JSON 对象：{path}")
    return payload


# 用途：复用 argparse 参数定义校验并转换 JSON 字段。
# 调用时机：parse_args_with_config 将配置注入 parser 默认值之前逐字段调用。
# 输入输出与注意：输入 action/value，返回转换值；严格区分 bool/int，处理 append/Path/choices，null 直接返回 None，业务范围由调用方校验。
def _convert_config_value(action, value):
    if value is None:
        return None
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if not isinstance(value, bool):
            raise ValueError(f"配置项 {action.dest} 必须是 true 或 false")
        return value
    if isinstance(action, argparse._AppendAction):
        if not isinstance(value, list):
            raise ValueError(f"配置项 {action.dest} 必须是数组")
        converted = [action.type(item) if action.type else item for item in value]
        if action.choices is not None and any(
            item not in action.choices for item in converted
        ):
            raise ValueError(f"配置项 {action.dest} 包含无效选项")
        return converted
    if action.type:
        try:
            if action.type is Path:
                value = value if isinstance(value, Path) else Path(value)
            elif action.type is int:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError
            elif action.type is float:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError
                value = float(value)
            elif action.type is str:
                if not isinstance(value, str):
                    raise TypeError
            else:
                value = action.type(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"配置项 {action.dest} 的类型无效：{value!r}") from exc
    if action.choices is not None and value not in action.choices:
        raise ValueError(
            f"配置项 {action.dest}={value!r} 无效，可选值：{tuple(action.choices)}"
        )
    return value


# 用途：合并默认分区、自定义分区和命令行，提供统一配置优先级。
# 调用时机：各 src 命令的 main 完成 add_argument 后调用，配置测试也传人工 argv 调用。
# 输入输出与注意：返回 Namespace 并附 args.config 绝对路径；先找到 --config，再浅合并分区，拒绝未知参数，CLI 最后覆盖；不启动业务任务。
def parse_args_with_config(
    parser: argparse.ArgumentParser,
    section: str,
    argv=None,
    default_config_path: Path = DEFAULT_CONFIG_PATH,
):
    """把配置分区设为 argparse 默认值，再解析 CLI 覆盖。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(default_config_path),
        help="项目 JSON 配置文件；命令行参数会覆盖其中的同名配置",
    )
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=Path(default_config_path))
    config_path = bootstrap.parse_known_args(argv)[0].config
    config = load_config(config_path)
    selected_values = config.get(section)
    if not isinstance(selected_values, dict):
        raise ValueError(f"配置文件缺少对象分区：{section}")
    default_config = load_config(default_config_path)
    default_values = default_config.get(section, {})
    if not isinstance(default_values, dict):
        raise ValueError(f"默认配置中的 {section} 分区必须是对象")
    values = {**default_values, **selected_values}

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    unknown = sorted(set(values) - set(actions))
    if unknown:
        raise ValueError(f"配置分区 {section} 包含未知参数：{', '.join(unknown)}")
    defaults = {
        key: _convert_config_value(actions[key], value)
        for key, value in values.items()
    }

    # append 参数一旦在 CLI 出现，应替换配置数组，而不是附加到配置数组。
    cli_options = set(argv)
    for action in actions.values():
        if isinstance(action, argparse._AppendAction) and any(
            option in cli_options for option in action.option_strings
        ):
            defaults[action.dest] = None
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.config = Path(config_path).resolve()
    return args
