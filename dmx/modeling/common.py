import math

import flax.linen as nn
import jax
import jax.numpy as jnp

def _custom_initializer(label_dim: int)->nn.initializers.Initializer:
    """Custom initializer for the label embedding layer."""
    he = nn.initializers.kaiming_normal()

    def init(key, shape, dtype=jnp.float32):
        return he(key, shape, dtype) * jnp.sqrt(label_dim)

    return init


class LabelEmbedding(nn.Module):
    emb_channels: int
    label_dropout: float = 0.0
    endpoint: bool = False
    max_positions: int = 10000

    dtype:type|None = None 

    @nn.compact
    def __call__(self, x, class_labels=None, *, deterministic=True):
        freqs = jnp.arange(self.emb_channels//2)
        freqs = freqs / (self.emb_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.reshape(-1, 1) * freqs
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)

        x = nn.Dense(self.emb_channels, dtype=self.dtype)(x)
        x = jax.nn.silu(x)
        x = nn.Dense(self.emb_channels, dtype=self.dtype)(x)

        if class_labels:
            label_emb = nn.Dense(
                self.emb_channels,
                use_bias=False,
                kernel_init=_custom_initializer(class_labels.shape[-1]),
                dtype=self.dtype
            )(class_labels)

            if self.label_dropout > 0:
                label_emb = nn.Dropout(self.label_dropout, broadcast_dims=[-1])(label_emb, deterministic=deterministic)

            x += label_emb

        x = jax.nn.silu(x)

        return x


class Normalize(nn.Module):
    norm: str = "group"  # group|layer
    eps: float = 1e-5
    max_groups: int = 32
    use_bias: bool = True
    default_group_size: int = 4

    @nn.compact
    def __call__(self, x):
        if self.norm == "group":
            n_grps = min(self.max_groups, x.shape[-1] // self.default_group_size)
            return nn.GroupNorm(
                num_groups=n_grps, 
                epsilon=self.eps,
                use_bias=self.use_bias,
            )(x)

        elif self.norm == "layer":
            return nn.LayerNorm(
                epsilon=self.eps,
                use_bias=self.use_bias,
            )(x)

        else:
            raise ValueError(f"Invalid normalization type: {self.norm}")


class UNetBlock(nn.Module):
    """ Unified U-Net block with optional up/downsampling and self-attention.
    based on https://github.com/NVlabs/edm/
    """
    out_channels: int
    resize: str = "none" # up|down|none
    attention: bool = False
    adaptive_scale: bool = True
    residual: bool = False

    num_heads: int = 0
    dropout: float = 0
    eps: float = 1e-5
    norm: str = "group"  # group|layer

    dtype: type|None = None

    def normalize(self, x):
        return Normalize(norm=self.norm, eps=self.eps)(x)

    @nn.compact
    def __call__(self, x, emb, *, deterministic=True):
        B, H, W, _ = x.shape
        shortcut = x

        if self.residual:
            assert x.shape[-1] == self.out_channels, "Input and output channels must match for residual connection."

        x = jax.nn.silu(self.normalize(x))

        if self.resize == "up":
            x = nn.ConvTranspose(self.out_channels, 3, strides=2, dtype=self.dtype)(x)
        elif self.resize == "down":
            x = nn.Conv(self.out_channels, 3, strides=2, dtype=self.dtype)(x)
        elif self.resize == "none":            
            x = nn.Conv(self.out_channels, 3, dtype=self.dtype)(x)
        else:
            raise ValueError(f"Invalid resize option: {self.resize}")

        emb = emb.reshape(B, 1, 1, -1)
        if self.adaptive_scale:
            scale = nn.Dense(self.out_channels, dtype=self.dtype)(emb)
            shift = nn.Dense(self.out_channels, dtype=self.dtype)(emb)
            x = jax.nn.silu(self.normalize(x) * scale + shift)
        else:
            x = x + nn.Dense(self.out_channels, dtype=self.dtype)(emb)
            x = self.normalize(x)

        if self.dropout > 0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)

        x = nn.Conv(self.out_channels, 3, dtype=self.dtype)(x)

        if self.attention:
            x = x.reshape(B, H * W, self.out_channels)            
            x = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                dtype=self.dtype,
            )(x, deterministic=deterministic)
            x = x.reshape(B, H, W, self.out_channels)

        if self.residual:
            x = x + shortcut

        return x

