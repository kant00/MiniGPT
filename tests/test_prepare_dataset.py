"""
测试旧版内存切块辅助函数的窗口数量、右移边界和参数校验。

使用方式（在项目根目录执行）：
  python -m unittest tests.test_prepare_dataset -v
  python -m unittest discover -s tests -v       # 运行全部测试

本文件只构造内存 token，不读取或修改项目 data 目录。
"""

import unittest

from src.prepare_dataset import split_token_stream


# 用途：集中验证旧内存切块的数量与边界。
# 调用时机：unittest 测试发现阶段加载本类，再为 test_ 方法创建实例并执行。
# 输入输出与注意：继承 TestCase，测试输入由各方法构造；临时文件/替身用于隔离正式数据，测试通过不代表所有输入或环境均已覆盖。
class SplitTokenStreamTests(unittest.TestCase):
    # 用途：用 21 个 token、长度 4 得到训练 (4,5) 与验证 (1,5)，检查首尾窗口及共享边界；保护旧内存切块而非正式文档三路划分。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_keeps_every_complete_window(self):
        tokens = list(range(21))
        train, val = split_token_stream(tokens, seq_len=4, val_ratio=0.2)

        self.assertEqual(tuple(train.shape), (4, 5))
        self.assertEqual(tuple(val.shape), (1, 5))
        self.assertEqual(train[0].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(train[-1].tolist(), [12, 13, 14, 15, 16])
        self.assertEqual(val[0].tolist(), [16, 17, 18, 19, 20])

    # 用途：只有一个完整窗口时要求含“至少需要 2 个完整切块”的 ValueError，避免 train/val 无法各分样本。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_rejects_too_little_data(self):
        with self.assertRaisesRegex(ValueError, "至少需要 2 个完整切块"):
            split_token_stream(list(range(5)), seq_len=4, val_ratio=0.2)

    # 用途：传 val_ratio=1.0 要求含参数名的 ValueError；测试比例上界，不穷举全部非法参数。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_validates_ratio(self):
        with self.assertRaisesRegex(ValueError, "val_ratio"):
            split_token_stream(list(range(21)), seq_len=4, val_ratio=1.0)


if __name__ == "__main__":
    unittest.main()
