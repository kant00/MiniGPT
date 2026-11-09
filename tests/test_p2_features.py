"""
测试 P2 的三路划分、质量过滤、近似去重、实验目录和评测指标。

使用方式（在项目根目录执行）：
  python -m unittest tests.test_p2_features -v
  python -m unittest discover -s tests -v       # 运行全部测试

测试不访问网络；临时实验目录可通过 MINIGPT_TEST_TMPDIR 指定根路径。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from src.analyze_tokenizers import analyze_tokenizer
from src.benchmark_generation import distinct_n, repetition_ratio
from src.experiment_utils import create_run_dir, latest_run_dir, validate_run_name
from src.evaluate import save_evaluation_report
from src.generate import (
    apply_generation_constraints,
    decode_text,
    resolve_generation_token_ids,
)
from src.prepare_dataset import (
    NearDuplicateIndex,
    choose_split,
    quality_rejection_reason,
    simhash64,
    validate_quality_rejection_ratio,
)


TEST_TMP_ROOT = Path(os.environ.get("MINIGPT_TEST_TMPDIR", tempfile.gettempdir()))


# 用途：集中验证三路数据治理、实验目录和生成指标。
# 调用时机：unittest 测试发现阶段加载本类，再为 test_ 方法创建实例并执行。
# 输入输出与注意：继承 TestCase，测试输入由各方法构造；临时文件/替身用于隔离正式数据，测试通过不代表所有输入或环境均已覆盖。
class P2FeatureTests(unittest.TestCase):
    # 用途：相同文档 ID、比例和种子连续划分两次，要求结果相同且属于 train/val/test；不验证全量比例严格一致。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_three_way_split_is_stable(self):
        first = choose_split("article-42", 0.04, 0.01, 42)
        self.assertEqual(first, choose_split("article-42", 0.04, 0.01, 42))
        self.assertIn(first, {"train", "val", "test"})

    # 用途：短正文须返回 too_short；同一 SimHash 首次加入为非重复、第二次为重复，验证索引基本查询语义。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_quality_filter_and_near_duplicate(self):
        self.assertEqual(
            quality_rejection_reason("短", 10, 0, 1, 1), "too_short"
        )
        original = "这是一个用于测试近似去重功能的中文文档，内容足够长。" * 4
        fingerprint = simhash64(original)
        index = NearDuplicateIndex(max_distance=3)
        self.assertFalse(index.check_and_add(fingerprint))
        self.assertTrue(index.check_and_add(fingerprint))

    # 用途：正常多段中文换行应通过，含大量空字符的正文须命中 control_chars，防止把排版字符误判为噪声。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_normal_newlines_are_not_control_character_noise(self):
        text = ("这是正常的中文段落。\n" * 20).strip()
        self.assertIsNone(quality_rejection_reason(text, 10, 0.1, 0.01, 0.2))
        corrupted = "正常中文" + ("\x00" * 20)
        self.assertEqual(
            quality_rejection_reason(corrupted, 1, 0.0, 0.01, 1.0),
            "control_chars",
        )

    # 用途：100 篇拒绝 81 篇超过 0.8 应报错，拒绝 5 篇应返回 0.05；验证批次发布前保护阈值。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_excessive_quality_rejection_is_blocked(self):
        with self.assertRaisesRegex(ValueError, "超过上限"):
            validate_quality_rejection_ratio(100, {"control_chars": 81}, 0.8)
        self.assertEqual(
            validate_quality_rejection_ratio(100, {"too_short": 5}, 0.8), 0.05
        )

    # 用途：临时创建两个实验，检查目录隔离与 latest 指向第二个，再拒绝 ../escape 名称；不写正式输出目录。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_run_directories_are_isolated_and_latest_is_resolved(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmp_dir:
            output = Path(tmp_dir) / "outputs"
            first = create_run_dir(output, "experiment-a")
            second = create_run_dir(output, "experiment-b")
            self.assertNotEqual(first, second)
            self.assertEqual(latest_run_dir(output), second.resolve())
            pointer = json.loads((output / "latest_run.json").read_text("utf-8"))
            self.assertEqual(pointer["run_dir"], "runs/experiment-b")
        with self.assertRaises(ValueError):
            validate_run_name("../escape")

    # 用途：替身分词器对“中文?”产生一个 UNK，检查 embedding=80、UNK=1/3，再用明确序列验证 Distinct-2 与三元组重复率。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_tokenizer_and_generation_metrics(self):
        # 用途：提供本测试需要的最小分词器接口，替代真实外部分词器。
        # 调用时机：所在测试构造数据或调用分词/解码工具时实例化，避免下载模型文件。
        # 输入输出与注意：只实现测试使用的属性和方法，不是可用于中文训练的真实 tokenizer；固定元数据用于接口一致性断言。
        class TinyTokenizer:
            unk_token_id = 0
            eos_token_id = 2
            sep_token_id = None

            # 用途：为模拟文本生成可预测的整数 ID。
            # 调用时机：真实预处理或指标分析调用替身 tokenizer.encode 时执行。
            # 输入输出与注意：问号映射 UNK=0，其他字符映射 3；忽略 add_special_tokens，返回每字符一个 ID，以便精确验证 UNK 比例。 无文件读写，不反映实际分词质量。
            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return [0 if char == "?" else 3 for char in text]

        metadata = {
            "source": "tiny",
            "class": "TinyTokenizer",
            "vocab_size": 10,
            "vocab_sha256": "fake",
        }
        report = analyze_tokenizer(TinyTokenizer(), metadata, ["中文?"], d_model=8)
        self.assertEqual(report["embedding_parameters"], 80)
        self.assertAlmostEqual(report["unknown_rate"], 1 / 3)
        self.assertEqual(distinct_n([1, 2, 1, 2], 2), 2 / 3)
        self.assertAlmostEqual(repetition_ratio([1, 2, 3, 1, 2, 3], 3), 0.25)

    # 用途：无 EOS 时应选 SEP=2，其他特殊 ID 禁用；将带字间空格的中文解码结果清理为连续正文。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_generation_boundary_special_tokens_and_chinese_decode(self):
        # 用途：提供本测试需要的最小分词器接口，替代真实外部分词器。
        # 调用时机：所在测试构造数据或调用分词/解码工具时实例化，避免下载模型文件。
        # 输入输出与注意：只实现测试使用的属性和方法，不是可用于中文训练的真实 tokenizer；固定元数据用于接口一致性断言。
        class Tokenizer:
            eos_token_id = None
            sep_token_id = 2
            all_special_ids = [0, 1, 2, 3]

            # 用途：提供固定的带空格中文解码文本供清理规则测试。
            # 调用时机：decode_text 接收本测试 Tokenizer 时间接调用。
            # 输入输出与注意：忽略 ids 和 skip_special_tokens，固定返回“人 工 智 能 ， 发 展”；只验证下游显示清理，不执行真正逆分词。
            @staticmethod
            def decode(ids, skip_special_tokens=True):
                del ids, skip_special_tokens
                return "人 工 智 能 ， 发 展"

        stop_ids, banned_ids = resolve_generation_token_ids(Tokenizer())
        self.assertEqual(stop_ids, [2])
        self.assertEqual(banned_ids, [0, 1, 3])
        self.assertEqual(decode_text(Tokenizer(), torch.tensor([4])), "人工智能，发展")

    # 用途：历史 [1,2,1,2,1] 下检查 ID 0 被禁、下一 ID 2 会重复三元组而被禁，已有 ID 1 的正分数被惩罚。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_generation_constraints_block_special_and_repeated_ngram(self):
        logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        history = torch.tensor([[1, 2, 1, 2, 1]])
        constrained = apply_generation_constraints(
            logits,
            history,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            banned_token_ids=[0],
        )
        self.assertTrue(torch.isneginf(constrained[0]))
        self.assertTrue(torch.isneginf(constrained[2]))
        self.assertLess(constrained[1], logits[1])

    # 用途：临时 checkpoints 下放占位文件，直接保存预造指标并检查 test_metrics.json 的位置与 split；不加载假模型或执行评估。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_evaluation_report_is_saved_in_run_directory(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmp_dir:
            run_dir = Path(tmp_dir) / "outputs" / "runs" / "demo"
            checkpoint = run_dir / "checkpoints" / "best_model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"placeholder")
            output = save_evaluation_report(
                {"split": "test", "loss": 1.0, "ppl": 2.718},
                checkpoint,
                batch_size=8,
                amp="off",
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(output, run_dir / "evaluations" / "test_metrics.json")
            self.assertEqual(payload["metrics"]["split"], "test")


if __name__ == "__main__":
    unittest.main()
