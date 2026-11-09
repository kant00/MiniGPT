"""
测试 JSONL 完整性检查及 train/val/test memmap 数据往返读取。

使用方式（在项目根目录执行）：
  python -m unittest tests.test_data_validation -v
  python -m unittest discover -s tests -v       # 运行全部测试

测试使用临时目录和 FakeTokenizer，不会覆盖项目中的正式数据。
可用 MINIGPT_TEST_TMPDIR 指定临时目录根路径。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.download_data import validate_jsonl
from src.prepare_dataset import build_dataset
from src.train import TokenDataset


TEST_TMP_ROOT = Path(os.environ.get("MINIGPT_TEST_TMPDIR", tempfile.gettempdir()))


# 用途：集中验证文章完整性与 memmap 读写接口。
# 调用时机：unittest 测试发现阶段加载本类，再为 test_ 方法创建实例并执行。
# 输入输出与注意：继承 TestCase，测试输入由各方法构造；临时文件/替身用于隔离正式数据，测试通过不代表所有输入或环境均已覆盖。
class JsonlValidationTests(unittest.TestCase):
    # 用途：写一条含 id/title/text 的临时记录，要求 validate_jsonl 返回 (True,1,OK)，同时验证状态、行数和说明。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_accepts_complete_jsonl(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmp_dir:
            path = Path(tmp_dir) / "wiki.jsonl"
            path.write_text(
                json.dumps({"id": "1", "title": "标题", "text": "有效正文"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(validate_jsonl(path, expected_rows=1), (True, 1, "OK"))

    # 用途：分别创建空文件和缺 title/text 的记录，要求有效标志 False；不等于覆盖全部编码及 JSON 损坏情况。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_rejects_empty_or_incomplete_jsonl(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmp_dir:
            empty_path = Path(tmp_dir) / "empty.jsonl"
            empty_path.write_text("", encoding="utf-8")
            self.assertFalse(validate_jsonl(empty_path)[0])

            invalid_path = Path(tmp_dir) / "invalid.jsonl"
            invalid_path.write_text('{"id": "1"}\n', encoding="utf-8")
            self.assertFalse(validate_jsonl(invalid_path)[0])

    # 用途：替换数据目录和 tokenizer，真实处理 100 篇模拟文章，再读回三个 split；断言均非空、输入长度为 8 且目标右移，最后关闭映射。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_memmap_dataset_round_trip(self):
        # 用途：提供本测试需要的最小分词器接口，替代真实外部分词器。
        # 调用时机：所在测试构造数据或调用分词/解码工具时实例化，避免下载模型文件。
        # 输入输出与注意：只实现测试使用的属性和方法，不是可用于中文训练的真实 tokenizer；固定元数据用于接口一致性断言。
        class FakeTokenizer:
            eos_token_id = 2
            sep_token_id = None

            # 用途：为模拟文本生成可预测的整数 ID。
            # 调用时机：真实预处理或指标分析调用替身 tokenizer.encode 时执行。
            # 输入输出与注意：逐字符返回 3+ord(char)%20，普通 ID 在 3~22；忽略 add_special_tokens，边界符由预处理追加。 无文件读写，不反映实际分词质量。
            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return [3 + (ord(char) % 20) for char in text]

        tokenizer_info = {
            "source": "fake",
            "class": "FakeTokenizer",
            "vocab_size": 32,
            "vocab_sha256": "test-fingerprint",
            "special_tokens_map": {"eos_token": "<eos>"},
        }
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmp_dir:
            data_dir = Path(tmp_dir)
            jsonl_path = data_dir / "wiki_zh.jsonl"
            rows = [
                {
                    "id": str(index),
                    "title": f"标题{index}",
                    "text": ("正文内容" * 5) + str(index),
                }
                for index in range(100)
            ]
            jsonl_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            with patch("src.prepare_dataset.DATA_DIR", data_dir), patch(
                "src.prepare_dataset.load_tokenizer",
                return_value=(FakeTokenizer(), tokenizer_info),
            ):
                train_meta, val_meta, test_meta = build_dataset(
                    seq_len=8,
                    val_ratio=0.2,
                    test_ratio=0.1,
                    near_deduplicate=False,
                    min_chars=1,
                )
            train_dataset = TokenDataset(train_meta)
            val_dataset = TokenDataset(val_meta)
            test_dataset = TokenDataset(test_meta)
            self.assertGreater(len(train_dataset), 0)
            self.assertGreater(len(val_dataset), 0)
            self.assertGreater(len(test_dataset), 0)
            inputs, targets = train_dataset[0]
            self.assertEqual(tuple(inputs.shape), (8,))
            self.assertTrue((inputs[1:] == targets[:-1]).all())
            train_dataset.close()
            val_dataset.close()
            test_dataset.close()


if __name__ == "__main__":
    unittest.main()
