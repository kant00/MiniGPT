"""
执行微型端到端回归：准备数据、训练、保存 checkpoint、独立 test 集评估。

使用方式（在项目根目录执行）：
  python -m unittest tests.test_training_pipeline -v
  python -m unittest discover -s tests -v       # 运行全部测试

测试使用 FakeTokenizer、极小模型和临时目录，不下载语料，也不会改动正式产物。
可用 MINIGPT_TEST_TMPDIR 指定临时目录根路径。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.prepare_dataset import build_dataset
from src.evaluate import evaluate_checkpoint
from src.train import train
import torch


TEST_TMP_ROOT = Path(os.environ.get("MINIGPT_TEST_TMPDIR", tempfile.gettempdir()))


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


TOKENIZER_INFO = {
    "source": "fake",
    "class": "FakeTokenizer",
    "vocab_size": 32,
    "vocab_sha256": "training-test-fingerprint",
    "special_tokens_map": {"eos_token": "<eos>"},
}


# 用途：集中验证微型预处理、训练、保存与独立评估流水线。
# 调用时机：unittest 测试发现阶段加载本类，再为 test_ 方法创建实例并执行。
# 输入输出与注意：继承 TestCase，测试输入由各方法构造；临时文件/替身用于隔离正式数据，测试通过不代表所有输入或环境均已覆盖。
class TrainingPipelineTests(unittest.TestCase):
    # 用途：临时语料和 FakeTokenizer 驱动一轮微型真实训练，检查权重包关键状态与 token 计数，再做两批独立 test 评估；不要求生成质量或续训一致性。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_tiny_training_saves_complete_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "data"
            output_dir = root / "outputs"
            data_dir.mkdir()
            rows = [
                {
                    "id": str(index),
                    "title": f"t{index}",
                    "text": f"abcdefghij{index}",
                }
                for index in range(100)
            ]
            (data_dir / "wiki_zh.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with patch("src.prepare_dataset.DATA_DIR", data_dir), patch(
                "src.prepare_dataset.load_tokenizer",
                return_value=(FakeTokenizer(), TOKENIZER_INFO),
            ):
                build_dataset(
                    seq_len=8,
                    val_ratio=0.2,
                    test_ratio=0.1,
                    include_title=False,
                    near_deduplicate=False,
                    min_chars=1,
                    min_cjk_ratio=0,
                    max_document_tokens=8,
                )

            for split in ("train", "val", "test"):
                meta = json.loads((data_dir / f"{split}_seq8.meta.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["num_tokens"], meta["num_documents"] * 8)
                self.assertEqual(meta["max_document_tokens"], 8)

            with patch("src.train.DATA_DIR", data_dir):
                run_dir = train(
                    epochs=1,
                    batch_size=8,
                    seq_len=8,
                    amp="off",
                    grad_accum_steps=3,
                    baseline_batches=0,
                    log_interval=1000,
                    output_dir=output_dir,
                    run_name="tiny-test",
                    model_overrides={
                        "d_model": 16,
                        "n_heads": 4,
                        "n_layers": 1,
                        "d_ff": 32,
                        "dropout": 0.0,
                    },
                )

            checkpoint_dir = run_dir / "checkpoints"
            checkpoint = torch.load(
                checkpoint_dir / "best_model.pt", map_location="cpu", weights_only=True
            )
            for key in (
                "model_config",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "rng_state",
                "tokens_seen",
            ):
                self.assertIn(key, checkpoint)
            self.assertEqual(checkpoint["model_config"]["d_model"], 16)
            self.assertGreater(checkpoint["tokens_seen"], 0)
            with patch("src.train.DATA_DIR", data_dir):
                metrics = evaluate_checkpoint(
                    checkpoint_dir / "best_model.pt",
                    batch_size=8,
                    amp="off",
                    max_batches=2,
                    split="test",
                )
            self.assertGreater(metrics["tokens"], 0)
            self.assertGreater(metrics["ppl"], 0)
            self.assertEqual(metrics["split"], "test")


if __name__ == "__main__":
    unittest.main()
