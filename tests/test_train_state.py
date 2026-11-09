"""
测试随机状态恢复和 checkpoint 原子写入。

使用方式（在项目根目录执行）：
  python -m unittest tests.test_train_state -v
  python -m unittest discover -s tests -v       # 运行全部测试

所有 checkpoint 都写入临时目录；可用 MINIGPT_TEST_TMPDIR 指定其根路径。
"""

import os
import random
import tempfile
import unittest
from pathlib import Path

from src.train import (
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    set_seed,
)
import torch


TEST_TMP_ROOT = Path(os.environ.get("MINIGPT_TEST_TMPDIR", tempfile.gettempdir()))


# 用途：集中验证随机状态恢复和 checkpoint 保存。
# 调用时机：unittest 测试发现阶段加载本类，再为 test_ 方法创建实例并执行。
# 输入输出与注意：继承 TestCase，测试输入由各方法构造；临时文件/替身用于隔离正式数据，测试通过不代表所有输入或环境均已覆盖。
class TrainStateTests(unittest.TestCase):
    # 用途：固定种子并捕获状态，抽取期望值后再推进 RNG，恢复并比较 Python/CPU PyTorch 随机数；不验证完整续训或 NumPy 恢复。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_rng_state_can_be_restored(self):
        set_seed(123)
        state = capture_rng_state()
        expected_python = random.random()
        expected_torch = torch.rand(3)

        random.random()
        torch.rand(3)
        restore_rng_state(state)

        self.assertEqual(random.random(), expected_python)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))

    # 用途：临时保存 epoch 与 RNG 字典，用 weights_only=True 读回并确认 .tmp 消失；验证原子保存成功路径，不模拟磁盘故障。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_checkpoint_write_is_loadable(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmp_dir:
            path = Path(tmp_dir) / "checkpoint.pt"
            atomic_torch_save({"epoch": 2, "rng_state": capture_rng_state()}, path)
            loaded = torch.load(path, weights_only=True)
            self.assertEqual(loaded["epoch"], 2)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
