import torch
from torch import nn
from transformers import T5Config, T5EncoderModel, T5Model
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerSelfAttention, T5Stack
import math


class T5Attention_LoRA(T5Attention):
    def __init__(self, config, has_relative_attention_bias=False, num_new_scales=3, pred_length=96, pred_mask_tokens=True):
        # 只有第一层的 SelfAttention 才需要相对位置编码
        if has_relative_attention_bias:
            has_relative_attention_bias = self.is_first_layer()  # 只有 block.0 设置 True，其他层 False

        super().__init__(config, has_relative_attention_bias)
        self.init_multi_scale_modules(num_new_scales=num_new_scales, r=8, alpha=16)
        self.num_new_scales = num_new_scales
        self.pred_length = pred_length
        self.pred_mask_tokens = pred_mask_tokens

        self._ctx_len_per_scale = [math.ceil(512 / 2 ** i) for i in range(1 + self.num_new_scales)]
        self._pred_len_per_scale = [math.ceil(pred_length / 2 ** i) for i in range(1 + self.num_new_scales)]
        self._num_ctx_tokens_per_scale = [math.ceil(self._ctx_len_per_scale[i] / 8) for i in range(1 + self.num_new_scales)]
        self._num_pred_tokens_per_scale = [math.ceil(self._pred_len_per_scale[i] / 8) for i in range(1 + self.num_new_scales)]

    def scale_index(self, scale):
        start = 0
        if self.pred_mask_tokens:
            for i in range(scale):
                start += self._num_ctx_tokens_per_scale[i] + self._num_pred_tokens_per_scale[i]
            end = start + self._num_ctx_tokens_per_scale[scale] + self._num_pred_tokens_per_scale[scale]
        else:
            for i in range(scale):
                start += self._num_ctx_tokens_per_scale[i]
            end = start + self._num_ctx_tokens_per_scale[scale]
        return list(range(start, end))

    def is_first_layer(self):
        """ 判断当前 SelfAttention 是否属于 block.0 """
        return getattr(self, "layer_idx", 0) == 0

    def init_multi_scale_modules(self, num_new_scales, r, alpha):
        self.r = r  # LoRA 的低秩维度
        self.alpha = alpha  # LoRA 缩放因子

        for i in range(1 + num_new_scales):
            self.register_parameter(f"Q_A_{i}", nn.Parameter(torch.randn((self.r, self.d_model)) * 0.01))
            self.register_parameter(f"Q_B_{i}", nn.Parameter(torch.zeros((self.d_model if self.d_model!=512 else 384, self.r))))
            self.register_parameter(f"K_A_{i}", nn.Parameter(torch.randn((self.r, self.d_model)) * 0.01))
            self.register_parameter(f"K_B_{i}", nn.Parameter(torch.zeros((self.d_model if self.d_model!=512 else 384, self.r))))
            self.register_parameter(f"V_A_{i}", nn.Parameter(torch.randn((self.r, self.d_model)) * 0.01))
            self.register_parameter(f"V_B_{i}", nn.Parameter(torch.zeros((self.d_model if self.d_model!=512 else 384, self.r))))

    def apply_lora(self, input: torch.Tensor, layer: nn.Linear, A: nn.Parameter, B: nn.Parameter):
        """
        在给定的线性层上应用 LoRA。
        """
        W_no_grad = layer.weight
        lora_update = (self.alpha / self.r) * (B @ A)  # (in_features, out_features)
        W_lora = W_no_grad + lora_update  # (in_features, out_features)
        out = torch.matmul(input, W_lora.T)
        return out.float() 

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):

        batch_size, seq_length, _ = hidden_states.shape

        if self.num_new_scales is not None:
            query_scales, key_scales, value_scales = [], [], []
            for i in range(1+self.num_new_scales):
                index = self.scale_index(i)
                query_scale = hidden_states[..., index, :]
                key_scale = hidden_states[..., index, :]
                value_scale = hidden_states[..., index, :]

                # 使用 getattr 动态访问属性
                Q_A = getattr(self, f"Q_A_{i}")
                Q_B = getattr(self, f"Q_B_{i}")
                K_A = getattr(self, f"K_A_{i}")
                K_B = getattr(self, f"K_B_{i}")
                V_A = getattr(self, f"V_A_{i}")
                V_B = getattr(self, f"V_B_{i}")

                query_scales.append(self.apply_lora(query_scale, self.q, Q_A, Q_B))
                key_scales.append(self.apply_lora(key_scale, self.k, K_A, K_B))
                value_scales.append(self.apply_lora(value_scale, self.v, V_A, V_B))

            updated_query = torch.cat(query_scales, dim=-2)
            updated_key = torch.cat(key_scales, dim=-2)
            updated_value = torch.cat(value_scales, dim=-2)


        else:
            # No LoRA, use original projections
            updated_query = self.q(hidden_states)
            updated_key = self.k(hidden_states)
            updated_value = self.v(hidden_states)

        # (batch_size, num_heads, seq_length, head_dim)，
        query_states = updated_query.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        key_states = updated_key.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        value_states = updated_value.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, seq_length), device=scores.device, dtype=scores.dtype
                )
            else:
                position_bias = self.compute_bias(seq_length, seq_length, device=scores.device)

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class T5LayerSelfAttention_LoRA(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.SelfAttention = T5Attention_LoRA(config, has_relative_attention_bias)


class T5Stack_LoRA(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        
        self.block = nn.ModuleList(
            [T5LayerSelfAttention_LoRA(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )


class T5EncoderModel_LoRA(T5EncoderModel):
    def __init__(self, config, num_new_scales=3, pred_length=96, pred_mask_tokens=True):
        super().__init__(config)
        for i, block in enumerate(self.encoder.block):
            block.layer[0].SelfAttention = T5Attention_LoRA(config,
                                                            has_relative_attention_bias=(i == 0),
                                                            num_new_scales=num_new_scales,
                                                            pred_length=pred_length,
                                                            pred_mask_tokens=pred_mask_tokens)
            block.layer[0].SelfAttention.layer_idx = i  # 确保 is_first_layer() 正常工作