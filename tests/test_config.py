"""
测试统一 JSON 配置的加载、类型转换、CLI 覆盖和未知参数拒绝行为。

使用方式（在项目根目录执行）：
  python -m unittest tests.test_config -v
  python -m unittest discover -s tests -v       # 运行全部测试

测试只使用内存参数和临时配置文件，不会修改 configs/default.json。
"""

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.config_utils import DEFAULT_CONFIG_PATH, load_config, parse_args_with_config


TEST_TMP_ROOT = Path(os.environ.get("MINIGPT_TEST_TMPDIR", tempfile.gettempdir()))


# 用途：集中验证配置合并、类型转换和覆盖行为。
# 调用时机：unittest 测试发现阶段加载本类，再为 test_ 方法创建实例并执行。
# 输入输出与注意：继承 TestCase，测试输入由各方法构造；临时文件/替身用于隔离正式数据，测试通过不代表所有输入或环境均已覆盖。
class ConfigTests(unittest.TestCase):
    # 用途：读取正式默认配置，断言版本为 1 且七个命令分区都是字典；保护入口配置组织，不逐字段验证所有取值。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_default_config_contains_all_entrypoint_sections(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(config["format_version"], 1)
        for section in (
            "download_data",
            "prepare_dataset",
            "train",
            "evaluate",
            "generate",
            "analyze_tokenizers",
            "benchmark_generation",
        ):
            self.assertIsInstance(config[section], dict)

    # 用途：临时配置 count=3/item=[a]，CLI 改为 7/[b]，同时断言 input 转为 Path；验证覆盖、列表替换和路径类型转换。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_cli_overrides_config_and_converts_paths(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(
                json.dumps(
                    {"demo": {"count": 3, "input": "sample.txt", "item": ["a"]}}
                ),
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            parser.add_argument("--count", type=int, default=1)
            parser.add_argument("--input", type=Path, default=None)
            parser.add_argument("--item", action="append", default=None)
            args = parse_args_with_config(
                parser,
                "demo",
                argv=["--config", str(path), "--count", "7", "--item", "b"],
            )
            self.assertEqual(args.count, 7)
            self.assertEqual(args.input, Path("sample.txt"))
            self.assertEqual(args.item, ["b"])

    # 用途：临时 JSON 写入 parser 未声明的 typo，要求抛出含“未知参数”的 ValueError，避免拼错配置却静默执行。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_unknown_config_key_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text('{"demo": {"typo": 1}}', encoding="utf-8")
            parser = argparse.ArgumentParser()
            parser.add_argument("--count", type=int, default=1)
            with self.assertRaisesRegex(ValueError, "未知参数"):
                parse_args_with_config(
                    parser, "demo", argv=["--config", str(path)]
                )

    # 用途：配置先启用 enabled，再传同 dest 的反向开关，断言结果 False；验证命令行能够关闭 JSON 已开启的布尔项。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_boolean_cli_can_override_true_config_value(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text('{"demo": {"enabled": true}}', encoding="utf-8")
            parser = argparse.ArgumentParser()
            parser.add_argument("--enabled", action="store_true")
            parser.add_argument("--disabled", dest="enabled", action="store_false")
            args = parse_args_with_config(
                parser,
                "demo",
                argv=["--config", str(path), "--disabled"],
            )
            self.assertFalse(args.enabled)


if __name__ == "__main__":
    unittest.main()
