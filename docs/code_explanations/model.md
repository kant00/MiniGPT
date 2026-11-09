# model.py 讲解

源码：[src/model.py](../../src/model.py)。

## 1. 总体描述

本文件定义 MiniGPT 的网络计算。输入是整数 token ID，输出是每个位置对整个词表的原始分数 logits；计算 Loss、更新参数和选择下一个 token 都由外部模块负责。

整体是 Pre-Norm Decoder-only Transformer：Embedding → 多个 Attention/FFN 残差块 → 最终归一化 → 共享权重的词表投影。它支持两套位置编码、两套归一化、两套 FFN 激活及两种注意力实现，用于实验对照；生成时可以额外输入和返回每层 KV Cache。

类默认值与 JSON 默认值不同：`MiniGPTConfig()` 是 384 维、6 层、6 头、FFN 1024；当前训练 JSON 是 512 维、8 层、8 头、FFN 1536。评估和生成使用 checkpoint 保存的架构，不能仅根据当前默认值猜模型。

## 2. 先看张量形状

下面以当前 JSON 配置为例，B 为 batch，T 为序列长度：

| 位置 | 形状 | 示例 |
|---|---|---|
| input_ids | B × T | 32 × 256 |
| token embedding | B × T × D | 32 × 256 × 512 |
| 合并 QKV 投影 | B × T × 3D | 32 × 256 × 1536 |
| 拆头后的 Q/K/V | B × H × T × Dh | 32 × 8 × 256 × 64 |
| manual attention 分数 | B × H × T × T | 每个查询对所有键的分数 |
| 合并注意力输出 | B × T × D | 32 × 256 × 512 |
| FFN 中间输出 | B × T × F | 32 × 256 × 1536 |
| 最终 logits | B × T × V | 32 × 256 × 21128 |

模型不会把 logits 自动 softmax。训练的交叉熵内部处理 logits；生成时根据策略选择 argmax 或概率采样。

## 3. MiniGPTConfig

dataclass 集中记录架构字段：vocab_size 是词表长度，seq_len 是允许的上下文上限，d_model 是向量维度，n_heads 是注意力头数，n_layers 是块数，d_ff 是 FFN 中间维度；dropout 控制训练时丢弃概率，rope_theta 控制旋转频率，rms_norm_eps 防止归一化分母为零。

`validate()` 检查 d_model 可被 n_heads 整除、架构字符串属于支持值、RoPE 单头维度为偶数。它没有穷尽所有正数和数值范围检查。

`to_dict()` 用 dataclasses.asdict 转成普通字典，用于 checkpoint 和实验清单。dataclass 本身不读取 JSON 文件。

## 4. RMSNorm 与 build_norm

`RMSNorm.__init__` 建立长度为 D 的可训练 weight 和 eps。`forward` 沿最后一维计算：

```text
y = x / sqrt(mean(x²) + eps) * weight
```

它不减均值，也没有 LayerNorm 中的独立 bias。代码用 float32 计算平方、均值和缩放，再转回输入 dtype，减少低精度归一化的数值问题。

`build_norm(config)` 根据 norm_type 返回 RMSNorm 或 nn.LayerNorm，块内和最终输出共用这个选择规则。

## 5. RotaryEmbedding：位置怎样进入注意力

`__init__` 建立随维度变化的逆频率 inv_freq，并注册为非持久 buffer。buffer 随模型迁移设备，但不是优化器参数；persistent=False 表示不必存入 state_dict，可由配置重建。

`forward(q, k, position_offset)` 根据位置和频率计算旋转角，再生成可以广播到 B、H 维的 sin/cos。position_offset 在增量生成时等于缓存长度，保证新 token 用正确的位置。

`apply_rotary` 把相邻维度视为一对：偶数维和奇数维组成 `(a,b)`，旋转辅助向量为 `(-b,a)`，最后返回 `x*cos + rotated*sin`。

RoPE 作用在 Q 和 K 上，V 不旋转；它不是在输入 embedding 上直接加一个位置向量。另一条 learned 路径则真的在输入 embedding 上加可训练位置 embedding。

## 6. CausalSelfAttention

### 初始化

`qkv_proj` 一次把 D 投影到 3D，再拆成 Q、K、V。`out_proj` 把合并后的多头表示映射回 D。out_proj 上的 residual_projection 标记供后续缩小初始化方差；resid_dropout 用在残差分支输出。

### forward 的共同部分

首先投影、split、reshape、transpose，把张量变成 B×H×T×Dh。随后按缓存长度计算新 Q/K 的 RoPE；有历史时只拼接 K 和 V。历史 K 已旋转过，不重复旋转。

缓存 `PastKeyValue` 是 `(K,V)` 元组，二者形状为 B×H×缓存长度×Dh。查询 Q 只属于本次输入位置，不需要保存过去 Q，因为后续 token 不会重新查询旧位置的输出。

### manual 分支

执行 `QKᵀ / sqrt(Dh)` 得分，把 key_position > query_position 的位置置为负无穷，再 softmax、dropout、乘 V。因果掩码保证训练时每个位置看不到未来目标。

### SDPA 分支的三种情况

| 情况 | 掩码策略 | 原因 |
|---|---|---|
| 没有缓存 | is_causal=True | 标准方形因果注意力 |
| 有缓存，本次只有一个 token | is_causal=False | 所有已有键都是历史或当前，均可见 |
| 有缓存，本次多个 token | 手工布尔 mask | 用绝对位置正确对齐非方形 Q/K |

单 token 增量场景不能盲目套用从左上角开始的因果三角形，否则可能错误限制历史可见范围。SDPA 是调用 PyTorch 注意力接口，实际内核选择取决于设备、dtype 和形状，不能仅凭这个字符串就声称一定运行某种 Flash 内核。

最后把头转回 B×T×D，经过 out_proj 和 dropout。use_cache=True 返回 `(output, present)`，否则只返回 output。

## 7. FeedForward

FFN 独立处理每个位置，位置之间的信息交换发生在 attention 中。

SwiGLU 分支建立 gate_proj、up_proj、down_proj 三个无 bias 矩阵：`down(SiLU(gate(x)) * up(x))`。GELU 分支建立带 bias 的 up/down：`down(GELU(up(x)))`。两种结果都经过 dropout。

SwiGLU 主要参数约 3DF，GELU 约 2DF，因此消融时相同 d_ff 并不等于相同参数量。down_proj 也标记为 residual_projection。

## 8. TransformerBlock

`__init__` 建立 ln1、attn、ln2、ffn；`forward` 按以下次序计算：

```text
x1 = x + Attention(Norm1(x))
x2 = x1 + FFN(Norm2(x1))
```

这是 Pre-Norm：先归一化再进入子层。两次相加要求所有分支返回相同的 D 维表示。缓存参数传给 attention，FFN 不维护跨 token 缓存。

## 9. MiniGPT 主模型

### __init__ 与 _init_weights

主模型建立 token_emb、可选 pos_emb、embedding dropout、ModuleList 中的各层块、ln_final、lm_head。ModuleList 确保子模块被 PyTorch 注册、保存并参与参数遍历。

`lm_head.weight = token_emb.weight` 是同一个 Parameter 的共享，不是复制初值。输入查表和输出词表打分因此共享 V×D 矩阵，参数统计不会重复计数。

`_init_weights` 对 Linear/Embedding 使用标准差 0.02 的正态分布，线性 bias 置零。随后把 residual_projection 标记的输出矩阵标准差改为 `0.02/sqrt(2*n_layers)`，控制多层残差分支初始尺度。

### forward

先从缓存取得 past_len，验证 past_len + current_len 不超过 seq_len；输入查 embedding，learned 路径叠加位置向量，再经过所有块及最终归一化和词表投影。

use_cache=False 返回 logits；为 True 时返回 `(logits, presents)`，其中 presents 是长度 n_layers 的缓存列表。模型本身不会自动截断超长序列，窗口裁剪由 generate 完成。

### count_parameters

统计 requires_grad=True 参数的元素数量。当前全参数模型可用它表示可训练模型规模；以后冻结参数时，这个值会小于总参数量。

## 10. build_model 与文件末尾示例

工厂函数先复制 model_config，再用 overrides 覆盖，最后显式 vocab_size 与 seq_len 再覆盖同名字段，构造 MiniGPTConfig 和 MiniGPT。因此传入字典里的长度不一定是最后生效值，要看显式实参。

文件虽无 argparse 命令行配置入口，末尾确实有 `__main__` 自检：建立类默认模型，构造随机 token，打印参数量和输入输出形状。运行 `python -m src.model` 不会训练，也不会读取 JSON 的 38M 架构。

## 11. Python 使用与能力边界

```python
import torch
from src.model import MiniGPT, MiniGPTConfig

model = MiniGPT(MiniGPTConfig(d_model=512, n_heads=8, n_layers=8, d_ff=1536))
model.eval()
with torch.inference_mode():
    logits, cache = model(torch.tensor([[101, 102, 103]]), use_cache=True)
print(logits.shape, len(cache))
```

这里的数字只演示张量接口，不代表合理提示词。当前模型没有 padding attention_mask 参数、分组查询注意力、SFT 标签逻辑或聊天角色模板。eval() 关闭 dropout，inference_mode() 关闭梯度记录，两者作用不同。
