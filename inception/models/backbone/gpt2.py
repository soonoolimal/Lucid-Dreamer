import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        n_embd = config.n_embd
        n_head = config.n_head

        self.attn_pdrop = config.attn_pdrop
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.n_embd = n_embd
        self.n_head = n_head

    def forward(self, stacked, mask=None):
        # mask: (B, 1, 1, 3*T) padding mask; always full-ones — stride=1 windows
        # guarantee no padding regardless of offline/online usage
        q, k, v = self.c_attn(stacked).split(self.n_embd, dim=-1)

        q = rearrange(q, 'B tT (nh dh) -> B nh tT dh', nh=self.n_head)
        k = rearrange(k, 'B tT (nh dh) -> B nh tT dh', nh=self.n_head)
        v = rearrange(v, 'B tT (nh dh) -> B nh tT dh', nh=self.n_head)

        dropout_p = self.attn_pdrop if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p,
            is_causal=True,  # always full-ones mask — is_causal handles causal masking
        )

        out = rearrange(out, 'B nh tT dh -> B tT (nh dh)')

        return self.resid_dropout(self.c_proj(out))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model = config.n_embd
        d_ff = 4 * d_model if config.n_inner is None else config.n_inner

        self.c_fc = nn.Linear(d_model, d_ff)
        self.actvn = ACT2FN[config.activation_function]
        self.c_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.actvn(x)
        x = self.c_proj(x)
        return self.dropout(x)


class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_size = config.n_embd

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ff = FeedForward(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.ff(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.apply(self._init_weights)

    def forward(self, stacked, mask=None):
        x = self.drop(stacked)
        for block in self.h:
            x = block(x, mask=mask)
        return self.ln_f(x)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
