"""
定义可配置的 Decoder-only MiniGPT 模型及其基础组件。

包含内容：
  MiniGPTConfig、RMSNorm、RoPE、因果自注意力、SwiGLU/GELU 前馈层、
  Transformer Block、MiniGPT 主模型和 build_model 工厂函数。

本文件是内部模型模块，不提供命令行入口。通常由 src.train、src.evaluate
和 src.generate 调用；也可以在 Python 代码中直接使用：

  from src.model import MiniGPT, MiniGPTConfig
  config = MiniGPTConfig(vocab_size=21128, seq_len=256)
  model = MiniGPT(config)
  logits = model(input_ids)  # input_ids: (batch, sequence)

生成场景可调用 model(..., use_cache=True) 获取每层 KV Cache。模型配置必须
随 checkpoint 保存，加载时不要依赖当前代码中的默认参数猜测架构。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PastKeyValue = Tuple[torch.Tensor, torch.Tensor]


# 用途：集中保存网络结构与数值超参数，便于按同一配置创建、保存和重建模型。
# 调用时机：新训练建立架构、build_model 从 checkpoint 重建模型或测试构造小模型时实例化。
# 输入输出与注意：输入词表大小、长度、宽度、头数和结构选项；dataclass 只保存字段，不读取 JSON，校验由模型构造时触发。
@dataclass
class MiniGPTConfig:
    vocab_size: int = 21128
    seq_len: int = 256
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    # SwiGLU 有三组矩阵；1024 与旧版 GELU d_ff=1536 的参数量基本相当。
    d_ff: int = 1024
    dropout: float = 0.1
    position_encoding: str = "rope"
    norm_type: str = "rmsnorm"
    activation: str = "swiglu"
    attention_impl: str = "sdpa"
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    # 用途：检查注意力维度和结构选项是否兼容，尽早发现无法计算的模型配置。
    # 调用时机：MiniGPT.__init__ 在建立层之前调用，也可在准备自定义配置时主动调用。
    # 输入输出与注意：成功返回 None；维度不能整除、RoPE 单头维度为奇数或选项不支持时抛 ValueError；未穷举全部数值范围。
    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        if self.position_encoding not in {"rope", "learned"}:
            raise ValueError("position_encoding 必须是 rope 或 learned")
        if self.norm_type not in {"rmsnorm", "layernorm"}:
            raise ValueError("norm_type 必须是 rmsnorm 或 layernorm")
        if self.activation not in {"swiglu", "gelu"}:
            raise ValueError("activation 必须是 swiglu 或 gelu")
        if self.attention_impl not in {"sdpa", "manual"}:
            raise ValueError("attention_impl 必须是 sdpa 或 manual")
        if self.position_encoding == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("使用 RoPE 时，每个 attention head 的维度必须为偶数")

    # 用途：将结构配置转为普通字典，供工厂函数和实验文件使用。
    # 调用时机：训练记录 model_config、保存 checkpoint，或调用方需要序列化配置时调用。
    # 输入输出与注意：返回 dataclasses.asdict 结果；只转换字段，不保存磁盘文件，也不包含模型权重。
    def to_dict(self) -> Dict:
        return asdict(self)


# 用途：按最后一维的均方根归一化隐藏向量，并学习每个通道的缩放系数。
# 调用时机：build_norm 在 norm_type=rmsnorm 时建立，块内及最终输出归一化时使用。
# 输入输出与注意：输入输出形状相同；不减均值、无独立偏置，内部转 float32 计算再转回输入 dtype。
class RMSNorm(nn.Module):
    # 用途：创建长度 dim 的可学习缩放向量和防止除零的 eps。
    # 调用时机：构造 RMSNorm 实例时由 Python 自动调用。
    # 输入输出与注意：weight 初始全为 1；dim 应等于隐藏维度，eps 加在均方值上，不是额外训练参数。
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    # 用途：计算 x/sqrt(mean(x²)+eps)，再逐通道乘 weight。
    # 调用时机：对 RMSNorm 对象调用 norm(x) 时由 PyTorch 分派执行，每次模型前向都会经过。
    # 输入输出与注意：沿最后一维归一化，保留 batch/序列维；返回与 x 相同 dtype，梯度仍可通过 float32 计算传播。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(dtype=x.dtype)


# 用途：按配置选择 RMSNorm 或 LayerNorm，统一各位置的归一化构造。
# 调用时机：TransformerBlock 创建 ln1/ln2、MiniGPT 创建 ln_final 时调用。
# 输入输出与注意：输入 MiniGPTConfig，返回一个新模块；各次调用独立创建参数，不在层之间共享归一化权重。
def build_norm(config: MiniGPTConfig) -> nn.Module:
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.d_model, config.rms_norm_eps)
    return nn.LayerNorm(config.d_model)


# 用途：给注意力 Q/K 注入旋转位置编码，使位置影响查询与键的相似度。
# 调用时机：CausalSelfAttention 在 position_encoding=rope 时构造，每次注意力前向时应用。
# 输入输出与注意：保存非持久 inv_freq buffer，随设备迁移但不参与优化；只旋转 Q/K，不旋转 V。
class RotaryEmbedding(nn.Module):
    # 用途：根据单头维度 dim 和频率基数 theta 预先建立逆频率。
    # 调用时机：创建旋转位置编码模块时调用。
    # 输入输出与注意：每对维度共享一个频率；buffer 不写入 state_dict，加载时由模型配置重建。
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # 用途：为本次查询位置生成 sin/cos，并对新 Q 和 K 应用旋转。
    # 调用时机：注意力前向在拼接历史 K/V 之前调用；有缓存时 position_offset 等于历史长度。
    # 输入输出与注意：Q/K 形状为 B×H×T×Dh；返回旋转后的二元组，位置偏移保证增量 token 不从位置 0 重新编码。
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(
            position_offset,
            position_offset + q.shape[-2],
            device=q.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.outer(positions, self.inv_freq.to(device=q.device))
        cos = freqs.cos().repeat_interleave(2, dim=-1)[None, None].to(q.dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1)[None, None].to(q.dtype)
        return self.apply_rotary(q, cos, sin), self.apply_rotary(k, cos, sin)

    # 用途：把相邻偶/奇维度组成二维向量，执行旋转公式。
    # 调用时机：RotaryEmbedding.forward 分别处理 Q 和 K 时调用该静态方法。
    # 输入输出与注意：由偶奇分量构造 (-odd,even)，返回 x*cos+rotated*sin；sin/cos 借助广播适配 batch 和头维度。
    @staticmethod
    def apply_rotary(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
        return x * cos + rotated * sin


# 用途：执行多头因果自注意力，并可复用历史键值缓存。
# 调用时机：每个 TransformerBlock 构造一个实例；训练、评估和生成前向均使用。
# 输入输出与注意：输入 B×T×D；输出相同形状，use_cache 时还返回 K/V；支持 SDPA 和手写两条计算路径。
class CausalSelfAttention(nn.Module):
    # 用途：创建 QKV 投影、输出投影、残差 dropout 和可选 RoPE。
    # 调用时机：构造 TransformerBlock.attn 时自动调用。
    # 输入输出与注意：单头维度为 d_model/n_heads；输出投影带 residual_projection 标记，供主模型调整初始化尺度。
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.attention_impl = config.attention_impl
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj.residual_projection = True
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = (
            RotaryEmbedding(self.d_head, config.rope_theta)
            if config.position_encoding == "rope"
            else None
        )

    # 用途：投影并拆分 Q/K/V，处理位置与缓存，再按因果掩码聚合历史信息。
    # 调用时机：块的前向通过 self.attn(...) 调用；生成携带 past_key_value 时进入增量路径。
    # 输入输出与注意：训练不能看未来；缓存后单 token 可看全部已有键，多 token 必须按绝对位置建掩码；返回值由 use_cache 决定。
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
    ):
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(channels, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        past_len = 0 if past_key_value is None else past_key_value[0].shape[-2]
        if self.rope is not None:
            q, k = self.rope(q, k, position_offset=past_len)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present = (k, v) if use_cache else None

        dropout_p = self.dropout if self.training else 0.0
        incremental_single_token = past_len > 0 and seq_len == 1
        if self.attention_impl == "sdpa":
            if past_len == 0:
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=dropout_p, is_causal=True
                )
            elif incremental_single_token:
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=dropout_p, is_causal=False
                )
            else:
                query_positions = torch.arange(
                    past_len, past_len + seq_len, device=x.device
                )[:, None]
                key_positions = torch.arange(k.shape[-2], device=x.device)[None, :]
                mask = key_positions <= query_positions
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=dropout_p
                )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            query_positions = torch.arange(
                past_len, past_len + seq_len, device=x.device
            )[:, None]
            key_positions = torch.arange(k.shape[-2], device=x.device)[None, :]
            scores = scores.masked_fill(key_positions > query_positions, float("-inf"))
            probs = F.softmax(scores, dim=-1)
            probs = F.dropout(probs, p=dropout_p, training=self.training)
            out = probs @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        out = self.resid_dropout(self.out_proj(out))
        return (out, present) if use_cache else out


# 用途：逐位置变换隐藏表示，提供注意力之外的非线性表达能力。
# 调用时机：每个 TransformerBlock 创建 ffn，并在注意力残差之后调用。
# 输入输出与注意：SwiGLU 用三个无偏置矩阵，GELU 用两个带偏置矩阵；输出回到 d_model，不混合不同序列位置。
class FeedForward(nn.Module):
    # 用途：按 activation 创建门控或普通前馈投影，并设置输出 dropout。
    # 调用时机：建立每层前馈网络时自动调用。
    # 输入输出与注意：中间宽度是 d_ff；down_proj 标记为残差投影，主模型会用较小标准差初始化。
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.activation = config.activation
        if self.activation == "swiglu":
            self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        else:
            self.up_proj = nn.Linear(config.d_model, config.d_ff)
            self.down_proj = nn.Linear(config.d_ff, config.d_model)
        self.down_proj.residual_projection = True
        self.dropout = nn.Dropout(config.dropout)

    # 用途：执行 SwiGLU 门控乘法或 GELU，再投影回隐藏维度。
    # 调用时机：TransformerBlock.forward 调用 self.ffn(...) 时运行。
    # 输入输出与注意：输入输出均为 B×T×D；训练模式输出经过 dropout，评估模式不随机丢弃。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        else:
            x = F.gelu(self.up_proj(x))
        return self.dropout(self.down_proj(x))


# 用途：组织一个 Pre-Norm Transformer 块，包含注意力与前馈两条残差分支。
# 调用时机：MiniGPT 根据 n_layers 创建多个块，前向时逐层串联。
# 输入输出与注意：先 x+Attention(Norm(x))，再 x+FFN(Norm(x))；只在注意力部分传递历史缓存。
class TransformerBlock(nn.Module):
    # 用途：建立两个独立归一化层、注意力模块和前馈模块。
    # 调用时机：MiniGPT 创建 blocks 列表时自动调用。
    # 输入输出与注意：所有子模块从同一结构配置读取维度；子层输出维度相同才能做残差相加。
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.ln1 = build_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln2 = build_norm(config)
        self.ffn = FeedForward(config)

    # 用途：按预归一化和残差顺序更新隐藏状态，并按需传回本层缓存。
    # 调用时机：MiniGPT.forward 遍历 blocks 时调用。
    # 输入输出与注意：输入 x 与可选本层 past_key_value；use_cache=False 返回张量，True 返回 (张量,present)。
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
    ):
        if use_cache:
            attn_out, present = self.attn(
                self.ln1(x), past_key_value=past_key_value, use_cache=True
            )
        else:
            attn_out = self.attn(self.ln1(x), past_key_value=past_key_value)
            present = None
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return (x, present) if use_cache else x


# 用途：定义完整 Decoder-only 语言模型，将 token ID 转成每个位置的词表 logits。
# 调用时机：训练创建新模型，评估/生成从 checkpoint 重建时实例化；不负责计算训练损失。
# 输入输出与注意：Embedding→多个块→最终归一化→词表投影；输入输出词表矩阵共享参数，支持缓存但不自动截断超长输入。
class MiniGPT(nn.Module):
    # 用途：校验配置、注册网络层、绑定词表共享权重并初始化参数。
    # 调用时机：build_model 或直接 MiniGPT(config) 时自动运行。
    # 输入输出与注意：残差输出矩阵使用 0.02/sqrt(2*n_layers) 标准差；此处不加载 checkpoint，加载权重由调用方负责。
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.seq_len = config.seq_len
        self.d_model = config.d_model
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = (
            nn.Embedding(config.seq_len, config.d_model)
            if config.position_encoding == "learned"
            else None
        )
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_final = build_norm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for module in self.modules():
            if getattr(module, "residual_projection", False):
                nn.init.normal_(module.weight, mean=0.0, std=residual_std)

    # 用途：为线性层与词嵌入设置正态初值，将线性偏置置零。
    # 调用时机：MiniGPT.__init__ 用 self.apply 递归遍历子模块时回调。
    # 输入输出与注意：输入单个模块，无返回值；其他模块不处理，带残差标记的权重随后还会重新缩放初始化。
    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # 用途：查词嵌入、按层传播并输出原始 logits，按需收集各层 K/V。
    # 调用时机：训练和评估传完整窗口，生成首次传提示词、后续传增量 token 时调用 model(...)。
    # 输入输出与注意：输入整数张量 B×T，输出 B×T×V；use_cache 返回 (logits,缓存列表)，历史长度加当前长度超过 seq_len 时抛错。
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[PastKeyValue]] = None,
        use_cache: bool = False,
    ):
        _, current_len = input_ids.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        if past_len + current_len > self.seq_len:
            raise ValueError(
                f"上下文长度 {past_len + current_len} 超过最大长度 {self.seq_len}"
            )

        x = self.token_emb(input_ids)
        if self.pos_emb is not None:
            positions = torch.arange(
                past_len, past_len + current_len, device=input_ids.device
            )
            x = x + self.pos_emb(positions)
        x = self.emb_dropout(x)

        presents = []
        for index, block in enumerate(self.blocks):
            layer_past = None if past_key_values is None else past_key_values[index]
            if use_cache:
                x, present = block(x, past_key_value=layer_past, use_cache=True)
                presents.append(present)
            else:
                x = block(x, past_key_value=layer_past)

        logits = self.lm_head(self.ln_final(x))
        return (logits, presents) if use_cache else logits

    # 用途：统计所有可训练参数元素数，用于显示模型规模。
    # 调用时机：训练日志或模块自检打印参数量时调用。
    # 输入输出与注意：只计 requires_grad=True，共享 embedding/head 不重复计数；返回整数，不估算激活显存或优化器状态。
    def count_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


# 用途：合并架构字典与覆盖项，创建 MiniGPT 实例。
# 调用时机：train、evaluate、generate 加载网络，以及模型模块的自检入口调用。
# 输入输出与注意：overrides 覆盖 model_config，显式 vocab_size/seq_len 最后覆盖；返回新初始化模型，恢复训练权重需另调 load_state_dict。
def build_model(
    vocab_size: int = 21128,
    seq_len: int = 256,
    model_config: Optional[Dict] = None,
    **overrides,
) -> MiniGPT:
    values = dict(model_config or {})
    values.update(overrides)
    values["vocab_size"] = vocab_size
    values["seq_len"] = seq_len
    return MiniGPT(MiniGPTConfig(**values))


if __name__ == "__main__":
    model = build_model()
    print(f"模型参数量：{model.count_parameters() / 1e6:.2f}M")
    dummy = torch.randint(0, model.config.vocab_size, (2, model.seq_len))
    logits = model(dummy)
    print(f"输入形状：{dummy.shape} → 输出 logits：{logits.shape}")
