import torch
from torch import nn
from transformers import T5Config, T5EncoderModel, T5Model
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerSelfAttention, T5Stack
class T5Attention_LoRA(T5Attention):
    def __init__(self, config, has_relative_attention_bias=False, pred_length=96):
        # 只有第一层的 SelfAttention 才需要相对位置编码
        if has_relative_attention_bias:
            has_relative_attention_bias = self.is_first_layer()  # 只有 block.0 设置 True，其他层 False

        super().__init__(config, has_relative_attention_bias)
        self.init_multi_scale_modules(num_new_scales=3, r=8, alpha=16)
        self.pred_length = pred_length
    def is_first_layer(self):
        """ 判断当前 SelfAttention 是否属于 block.0 """
        return getattr(self, "layer_idx", 0) == 0
    def init_multi_scale_modules(self, num_new_scales, r, alpha):
        self.num_new_scales = num_new_scales
        self.r = r  # LoRA 的低秩维度
        self.alpha = alpha  # LoRA 缩放因子
        for i in range(1 + self.num_new_scales):
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
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        mask=None,  # 添加 mask，没用
        layer_head_mask=None,  # 添加 layer_head_mask，没用
    ):

        batch_size, seq_length, _ = hidden_states.shape

        # 初始化 updated_query, updated_key, 和 updated_value
        updated_query = hidden_states.clone()
        updated_key = hidden_states.clone()
        updated_value = hidden_states.clone()

        if self.num_new_scales is not None:
            self.SCALE_INDEX = {
            96: [
                list(range(64, 76)),    # Scale 0: 12 patches
                list(range(108, 114)),   # Scale 1: 6 patches
                list(range(130, 133)),   # Scale 2: 3 patches
                list(range(141, 143))    # Scale 3: 2 patches
            ],
            192: [
                list(range(64, 88)),     # Scale 0: 24 patches
                list(range(120, 132)),   # Scale 1: 12 patches
                list(range(148, 154)),   # Scale 2: 6 patches
                list(range(162, 166))    # Scale 3: 4 patches
            ],
            336: [
                list(range(64, 106)),    # Scale 0: 42 patches
                list(range(138, 159)),    # Scale 1: 21 patches
                list(range(175, 187)),    # Scale 2: 12 patches
                list(range(195, 201))     # Scale 3: 6 patches
            ],
            720: [
                list(range(64, 154)),    # Scale 0: 90 patches
                list(range(186, 231)),    # Scale 1: 45 patches
                list(range(247, 270)),    # Scale 2: 23 patches
                list(range(278, 290))     # Scale 3: 12 patches
            ]}[self.pred_length]

            query_scales, key_scales, value_scales = [], [], []
            for i in range(1+self.num_new_scales):
                index = self.SCALE_INDEX[i]
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
    def __init__(self, config, pred_length=96):
        super().__init__(config)
        for i, block in enumerate(self.encoder.block):
            block.layer[0].SelfAttention = T5Attention_LoRA(config, has_relative_attention_bias=(i == 0), pred_length=pred_length)
            block.layer[0].SelfAttention.layer_idx = i  # 确保 is_first_layer() 正常工作