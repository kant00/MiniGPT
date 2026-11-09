"""检查消融协议与报告；只用模拟记录，不调用训练或模型前向。

使用：python -m unittest discover -s experiments/ablation -p test_tools.py -v
输出仅位于临时目录；模拟数值不进入正式实验报告。
"""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("ablation_run", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


# 用途：集中验证消融协议的漂移检查与报告呈现。
# 调用时机：unittest 测试发现阶段加载本类，再为 test_ 方法创建实例并执行。
# 输入输出与注意：继承 TestCase，测试输入由各方法构造；临时文件/替身用于隔离正式数据，测试通过不代表所有输入或环境均已覆盖。
class AblationTests(unittest.TestCase):
    # 用途：先检查各消融配置覆盖默认字段，再用 mock 向默认配置注入新字段，要求覆盖检查失败；不修改真实 JSON。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_default_values_do_not_override_complete_group_config(self):
        self.assertTrue(runner.defaults_fully_overridden())
        real_read = runner.read
        # 用途：模拟默认配置新增一项未被各组覆盖的字段。
        # 调用时机：该测试 patch runner.read 的 side_effect 回调时执行。
        # 输入输出与注意：先调用保存的真实 read，仅对 default.json 返回字典插入新键；不写文件，用于让 defaults_fully_overridden 返回 False。
        def changed(path):
            value = real_read(path)
            if Path(path).name == "default.json":
                value["train"]["new_inherited_option"] = 1
            return value
        with patch.object(runner, "read", side_effect=changed):
            self.assertFalse(runner.defaults_fully_overridden())

    # 用途：复制 A/B/C 配置并把 B 的学习率翻倍，要求公共参数校验抛错；仅模拟不同训练条件的漂移。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_common_parameter_drift_rejected(self):
        configs = {group: runner.configuration(group) for group in runner.IDS}
        runner.validate_configs()
        configs["B"] = copy.deepcopy(configs["B"])
        configs["B"]["train"]["lr"] *= 2
        # mock 回调：校验函数每次读取某组 g 时返回修改后的内存配置 configs[g]，不改真实文件。
        with patch.object(runner, "configuration", side_effect=lambda g: configs[g]):
            with self.assertRaisesRegex(ValueError, "公共训练参数"):
                runner.validate_configs()

    # 用途：临时锁记录旧指纹，mock 当前指纹为新值，要求 check_lock 报内容变化；验证拒绝混合实验来源。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "lock.json"
            lock.write_text(json.dumps({"sha256": {"data": "old"}}), encoding="utf-8")
            with patch.object(runner, "LOCK", lock), patch.object(runner, "fingerprint", return_value={"data": "new"}):
                with self.assertRaisesRegex(ValueError, "变化"):
                    runner.check_lock()

    # 用途：用临时模拟记录先写全 pending 报告，再加一组指标，检查缺失值保留、数值呈现和 PNG 非空；不把模拟数据写进正式实验。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_pending_and_partial_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"group": g, "status": "pending"} for g in runner.IDS]
            runner.make_report(rows, root)
            saved = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
            self.assertNotIn("val", saved["experiments"][0])
            rows[0].update(val={"ppl": 25.0}, history=[{"tokens_seen": 1000, "train_loss": 3.4, "val_loss": 3.2}])
            runner.make_report(rows, root)
            self.assertIn("25.0000", (root / "REPORT.md").read_text(encoding="utf-8"))
            for name in ("comparison.png", "loss_by_tokens.png"):
                self.assertGreater((root / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
