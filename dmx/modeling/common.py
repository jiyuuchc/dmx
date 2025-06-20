import math
from dataclasses import dataclass, InitVar

from flax import nnx
import jax
import jax.numpy as jnp

def _custom_initializer(label_dim: int)->nnx.initializers.Initializer:
    """Custom initializer for the label embedding layer."""
    he = nnx.initializers.kaiming_normal()

    def init(key, shape, dtype=jnp.float32):
        return he(key, shape, dtype) * jnp.sqrt(label_dim)

    return init


@dataclass(repr=False)
class LabelEmbedding(nnx.Module):
    emb_channels: int
    label_dim: int = 0
    label_dropout: float = 0.0
    endpoint: bool = False
    max_positions: int = 10000
    dtype:type|None = None

    rngs: InitVar[nnx.Rngs|None] = None

    def __post_init__(self, rngs):
        common_kwargs = dict(
            dtype=self.dtype,
            rngs=rngs
        )

        self._emb_linear_1 = nnx.Linear(self.emb_channels, self.emb_channels, **common_kwargs)
        self._emb_linear_2 = nnx.Linear(self.emb_channels, self.emb_channels, **common_kwargs)

        if self.label_dim > 0:
            self._label_linear = nnx.Linear(
                self.label_dim, self.emb_channels, 
                use_bias=False,
                kernel_init=_custom_initializer(self.label_dim),
                **common_kwargs,
            )
            self._label_dropout = nnx.Dropout(
                self.label_dropout, 
                broadcast_dims=[-1],
                rngs=rngs,
            )

    def __call__(self, noise_label, class_labels=None, *, deterministic=True):
        x = jnp.asarray(noise_label).reshape(-1, 1)

        freqs = jnp.arange(self.emb_channels//2)
        freqs = freqs / (self.emb_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs

        x = x * freqs
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)

        x = self._emb_linear_1(x)
        x = jax.nn.silu(x)
        x = self._emb_linear_2(x)

        if class_labels:
            label_emb = self._label_linear(class_labels)

            if self.label_dropout > 0:
                label_emb = self._label_dropout(label_emb, deterministic=deterministic)

            x += label_emb

        x = jax.nn.silu(x)

        return x


@dataclass(repr=False)
class EffConv(nnx.Module):
    """ A conv module that combines depthwise conv and a MLP for efficient computation.
    Used in models such as EfficientNet and ConvNeXt
    """
    features: int
    kernel_size: int = 7
    expansion_factor: int = 4

    dtype: type|None = None
    rngs: InitVar[nnx.Rngs|None] = None

    def __post_init__(self, rngs):
        kwargs = dict(
            dtype=self.dtype,
            rngs=rngs
        )

        self._depthwise = nnx.Conv(
            self.features, self.features, 
            (self.kernel_size, self.kernel_size), 
            feature_group_count=self.features,
            use_bias=False,
            **kwargs
        )

        self._norm = nnx.LayerNorm(self.features, **kwargs)

        self._linear_1 = nnx.Linear(self.features, self.features * self.expansion_factor, **kwargs),
        self._linear_2 = nnx.Linear(self.features * self.expansion_factor, self.out_channels, **kwargs)
    
    def __call__(self, x):
        B, H, W, C = x.shape

        assert C == self.features, f"Input channels {C} do not match expected {self.features}."

        x = self._depthwise(x)
        x = self._norm(x)
        x = self._linear_1(x)
        x = jax.nn.silu(x)
        x = self._linear_2(x)

        return x


@dataclass(repr=False)
class UNetBlock(nnx.Module):
    """ Unified U-Net block with optional up/downsampling and self-attention.
    based on https://github.com/NVlabs/edm/
    """
    in_channels: int
    out_channels: int
    emb_channels: int
    resize: str = "none" # up|down|none
    adaptive_scale: bool = True
    residual: bool = False
    dropout: float = 0
    attention: bool = False
    num_heads: int = 0
    norm: str = "group"  # group|layer
    conv: str = "standard"  # standard|eff
    max_groups: int = 32
    default_group_size: int = 4
    eps: float = 1e-5

    dtype: type|None = None

    rngs: InitVar[nnx.Rngs|None] = None

    def __post_init__(self, rngs):
        if self.resize not in ["up", "down", "none"]:
            raise ValueError(f"Invalid resize option: {self.resize}")

        if self.in_channels != self.out_channels and self.residual:
            raise ValueError("Input and output channels must match if residual connection is used.")

        kwargs = dict(
            dtype=self.dtype,
            rngs=rngs
        )

        # stem
        self._norm_in = self._get_norm_fn(self.in_channels, rngs)
        if self.resize == "up":
            self._stem =  nnx.ConvTranspose(
                self.in_channels, self.out_channels, (3,3), strides=2, **kwargs
            )
        elif self.resize == "down":
            self._stem = nnx.Conv(
                self.in_channels, self.out_channels, (3,3), strides=2, **kwargs
            )
        else:
            self._stem = nnx.Conv(
                self.in_channels, self.out_channels, (3,3), **kwargs
            )
        
        # emb mixing
        self._norm = self._get_norm_fn(self.out_channels, rngs)
        self._shift = nnx.Linear(self.emb_channels, self.out_channels, **kwargs)
        if self.adaptive_scale:
            self._scale = nnx.Linear(self.emb_channels, self.out_channels, **kwargs)
        
        # main
        self._dropout = nnx.Dropout(self.dropout, rngs=rngs)
        if self.conv == "eff":
            self._conv = EffConv(self.out_channels, **kwargs)
        else:
            self._conv = nnx.Conv(
                self.out_channels, self.out_channels, 
                (3, 3),
                **kwargs
            )

        # self attention
        if self.attention:
            self._attention = nnx.MultiHeadAttention(
                num_heads=self.num_heads,
                in_features=self.out_channels,
                decode=False,
                **kwargs
            )


    def _get_norm_fn(self, num_features, rngs):
        if self.norm == "group":
            n_grps = min(self.max_groups, self.in_channels // self.default_group_size)
            return nnx.GroupNorm(num_features, num_groups=n_grps, epsilon=self.eps, dtype=self.dtype, rngs=rngs)

        elif self.norm == "layer":
            return nnx.LayerNorm(num_features, epsilon=self.eps, dtype=self.dtype, rngs=rngs)

        else:
            raise ValueError(f"Invalid normalization type: {self.norm}")


    def __call__(self, x, emb, *, deterministic=True, rngs=None):
        B, H, W, C = x.shape

        assert C == self.in_channels, f"Input channels {C} do not match expected {self.in_channels}."

        shortcut = x

        if self.residual:
            assert x.shape[-1] == self.out_channels, "Input and output channels must match for residual connection."

        x = self._norm_in(x)
        x = self._stem(x)

        emb = emb.reshape(B, 1, 1, -1)
        if self.adaptive_scale:
            x = self._norm(x) * self._scale(emb) + self._shift(emb)
            x = jax.nn.silu(x)
        else:
            x = self._norm(x + self._shifte(emb))

        if self.dropout > 0:
            x = self._dropout(x, deterministic=deterministic, rngs=rngs)

        x = self._conv(x)

        if self.attention:
            x = x.reshape(B, H * W, self.out_channels)            
            x = self._attention(x, deterministic=deterministic, rngs=rngs)
            x = x.reshape(B, H, W, self.out_channels)

        if self.residual:
            x = x + shortcut

        return x

