"""
测试 MiniGPT 的因果性、注意力实现一致性、KV Cache 和生成一致性。

使用方式（在项目根目录执行）：
  python -m unittest tests.test_model -v
  python -m unittest discover -s tests -v       # 运行全部测试

测试使用极小随机模型，不需要真实语料、checkpoint 或 GPU。
"""

import unittest

from src.model import MiniGPT, MiniGPTConfig
from src.generate import generate
import torch


# 用途：创建快速模型测试用的极小架构配置。
# 调用时机：ModelTests 建立随机模型前调用，可用 overrides 切换注意力实现。
# 输入输出与注意：返回 MiniGPTConfig，词表32/长度8/宽16/4头/2层/FFN32/dropout0；这些值不读取训练 JSON。
def tiny_config(**overrides):
    values = {
        "vocab_size": 32,
        "seq_len": 8,
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 32,
        "dropout": 0.0,
        "position_encoding": "rope",
        "norm_type": "rmsnorm",
        "activation": "swiglu",
        "attention_impl": "sdpa",
    }
    values.update(overrides)
    return MiniGPTConfig(**values)


# 用途：集中验证因果性、注意力一致性和缓存生成。
# 调用时机：unittest 测试发现阶段加载本类，再为 test_ 方法创建实例并执行。
# 输入输出与注意：继承 TestCase，测试输入由各方法构造；临时文件/替身用于隔离正式数据，测试通过不代表所有输入或环境均已覆盖。
class ModelTests(unittest.TestCase):
    # 用途：替身模型固定选择 ID 2，提示词 [4,5] 最终必须是 [4,5,2]；关闭缓存以适配替身，验证遇结束符立即停。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_generation_stops_on_document_boundary(self):
        # 用途：构造永远偏向结束 ID 的最小模型替身。
        # 调用时机：结束符停止测试调用 generate 前实例化。
        # 输入输出与注意：只实现普通 forward，seq_len=8，不支持 KV Cache，所以测试显式关闭缓存。
        class BoundaryModel(torch.nn.Module):
            seq_len = 8

            # 用途：返回固定结束符分数最高的 logits，制造确定的停止条件。
            # 调用时机：结束符测试中的 generate 调用 BoundaryModel(input_ids) 时触发。
            # 输入输出与注意：输入 B×T，返回 B×T×8 零张量并将 ID2 分数设为10；greedy 必选2，无可训练参数。
            def forward(self, input_ids):
                logits = torch.zeros((*input_ids.shape, 8))
                logits[..., 2] = 10.0
                return logits

        prompt = torch.tensor([[4, 5]])
        output = generate(
            BoundaryModel(),
            prompt,
            max_new_tokens=10,
            strategy="greedy",
            eos_token_ids=[2],
            use_kv_cache=False,
        )
        self.assertEqual(output.tolist(), [4, 5, 2])

    # 用途：只改变输入末尾两项，比较前面三项的完整 logits；固定小模型并关闭 dropout，验证未来 token 不影响已有前缀。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_causal_attention_hides_future_tokens(self):
        torch.manual_seed(1)
        model = MiniGPT(tiny_config()).eval()
        left = torch.tensor([[1, 2, 3, 4, 5]])
        changed_future = torch.tensor([[1, 2, 3, 9, 10]])
        with torch.no_grad():
            first = model(left)
            second = model(changed_future)
        self.assertTrue(torch.allclose(first[:, :3], second[:, :3], atol=1e-6))

    # 用途：先完整前向五项，再预填充三项并逐项带缓存计算；以浮点容差比较对应 logits，验证缓存拼接和位置偏移。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_kv_cache_matches_full_forward(self):
        torch.manual_seed(2)
        model = MiniGPT(tiny_config()).eval()
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            full_logits = model(tokens)
            cached_logits, cache = model(tokens[:, :3], use_cache=True)
            self.assertTrue(
                torch.allclose(cached_logits, full_logits[:, :3], atol=1e-5)
            )
            for index in range(3, tokens.shape[1]):
                step_logits, cache = model(
                    tokens[:, index : index + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
                self.assertTrue(
                    torch.allclose(
                        step_logits[:, -1], full_logits[:, index], atol=1e-5
                    )
                )

    # 用途：给两种注意力模型加载同一权重、输入同一序列，以容差比较 logits；测数学一致性，不测速度。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_manual_attention_matches_sdpa(self):
        torch.manual_seed(3)
        sdpa_model = MiniGPT(tiny_config(attention_impl="sdpa")).eval()
        manual_model = MiniGPT(tiny_config(attention_impl="manual")).eval()
        manual_model.load_state_dict(sdpa_model.state_dict())
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            sdpa_logits = sdpa_model(tokens)
            manual_logits = manual_model(tokens)
        self.assertTrue(torch.allclose(sdpa_logits, manual_logits, atol=1e-5))

    # 用途：同一小模型分别开关缓存进行贪心续写，要求 ID 完全一致；总长度超过窗口，也覆盖缓存满后的窗口重建。
    # 调用时机：unittest 发现本文件测试类后自动调用；也可指定该测试方法单独运行。
    # 输入输出与注意：输入由方法内的字符串、张量或临时文件构造，不接收 CLI 参数；通过返回 None，断言不符则报告失败。测试只覆盖明确构造的案例，不代表穷举全部边界。
    def test_cached_generation_matches_uncached_greedy(self):
        torch.manual_seed(4)
        model = MiniGPT(tiny_config()).eval()
        prompt = torch.tensor([[1, 2, 3, 4]])
        cached = generate(model, prompt, max_new_tokens=6, use_kv_cache=True)
        uncached = generate(model, prompt, max_new_tokens=6, use_kv_cache=False)
        self.assertTrue(torch.equal(cached, uncached))


if __name__ == "__main__":
    unittest.main()
