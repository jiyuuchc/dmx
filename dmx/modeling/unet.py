import flax.linen as nn
import jax
import jax.numpy as jnp


class UNetBlock(nn.Module):
    """ Unified U-Net block with optional up/downsampling and self-attention.
    based on https://github.com/NVlabs/edm/
    """
    out_channels: int
    resize: str = "none" # up|down|none
    attention: bool = False
    num_heads: int = 0
    dropout: float = 0
    eps: float = 1e-5
    adaptive_scale: bool = True
    norm: str = "group"  # group|layer

    deterministic: bool = True
    dtype: type|None = None

    def _group_norm(self, x):
        n_grps = min(32, x.shape[-1] // 4)
        return nn.GroupNorm(num_groups=n_grps, epsilon=self.eps, dtype=self.dtype)(x)

    def _layer_norm(self, x):
        return nn.LayerNorm(epsilon=self.eps, dtype=self.dtype)(x)

    def normalize(self, x):
        if self.norm == "group":
            return self._group_norm(x)
        elif self.norm == "layer":
            return self._layer_norm(x)
        else:
            raise ValueError(f"Invalid normalization type: {self.norm}")

    @nn.compact
    def __call__(self, x, emb, *, deterministic=None):
        B, H, W, _ = x.shape

        if deterministic is None:
            deterministic = not self.deterministic

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

        x = nn.Dropout(rate=self.dropout)(x, deterministic=self.deterministic)

        x = nn.Conv(self.out_channels, 3, dtype=self.dtype)(x)

        if self.attention:
            x = x.reshape(B, H * W, self.out_channels)
            x = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                dtype=self.dtype,
            )(x, deterministic=deterministic)
            x = x.reshape(B, H, W, self.out_channels)

        return x


class PositionalEmbedding(nn.Module):
    num_channels: int
    max_positions: int = 10000
    endpoint: bool = False
    dtype: type|None = None

    @nn.compact
    def __call__(self, x):
        freqs = jnp.arange(self.num_channels//2)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x[:, None] * freqs
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)

        x = nn.Dense(self.num_channels, dtype=self.dtype)(x)
        x = jax.nn.silu(x)
        x = nn.Dense(self.num_channels, dtype=self.dtype)(x)

        return x


def _custom_initializer(label_dim: int)->nn.initializers.Initializer:
    """Custom initializer for the label embedding layer."""
    he = nn.initializers.kaiming_normal()

    def init(key, shape, dtype=jnp.float32):
        return he(key, shape, dtype) * jnp.sqrt(label_dim)

    return init

class DhariwalUNet(nn.Module):
    """ Reimplementation of the DM architecture from the paper
    'Diffusion Models Beat GANS on Image Synthesis'. Equivalent to the
    original implementation by Dhariwal and Nichol, available at
    https://github.com/openai/guided-diffusion

    Parameters:
        out_channels: Number of color channels at output.
        model_channels: Base multiplier for the number of channels.
        channel_mult: Per-resolution multipliers for the number of channels.
        channel_mult_emb: Multiplier for the dimensionality of the embedding vector.
        num_blocks: Number of residual blocks per resolution.
        attn_resolutions: List of resolutions with self-attention.
        dropout: Dropout probability for the residual blocks.
        label_dropout: Dropout probability of class labels for classifier-free guidance.
    """
    out_channels: int = 1
    model_channels: int = 192
    channel_mult: tuple[int] = (1,2,3,4)
    channel_mult_emb: int = 4
    num_blocks: int = 3
    attn_levels: tuple[int] = (1,2,3)
    dropout: float = 0.10
    label_dropout: float = 0

    deterministic: bool = True
    dtype: type|None = None

    @nn.compact
    def __call__(self, x, noise_labels, class_labels=None, *, deterministic=None):
        if deterministic is None:
            deterministic = not self.deterministic
        
        B, H, W, C = x.shape

        emb_channels = self.model_channels * self.channel_mult_emb
        emb = PositionalEmbedding(self.model_channels, dtype=self.dtype)(noise_labels)
        if class_labels:
            label_emb = nn.Dense(
                emb_channels,
                use_bias=False,
                kernel_init=_custom_initializer(class_labels.shape[-1]),
                dtype=self.dtype
            )(class_labels)
            emb += nn.Dropout(self.label_dropout, broadcast_dims=[-1])(label_emb, deterministic=deterministic)
        emb = jax.nn.silu(emb)

        # Encoder.
        skips = []
        for level, mult in enumerate(self.channel_mult):
            features = self.model_channels * mult

            if level == 0:
                x = nn.Conv(features, 3, dtype=self.dtype)(x)
            else:
                x = UNetBlock(
                    features, 
                    resize="down",
                    dropout=self.dropout,
                    dtype=self.dtype
                )(x, emb, deterministic=deterministic)
            skips.append(x)

            for idx in range(self.num_blocks):
                x = UNetBlock(
                    features, 
                    attention=level in self.attn_levels,
                    num_heads=features // 64, 
                    dropout=self.dropout,
                    dtype=self.dtype
                )(x, emb, deterministic=deterministic)
                skips.append(x)

        # Decoder.
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            features = self.model_channels * mult

            if level == len(self.channel_mult) - 1:
                x = UNetBlock(
                    features, 
                    attention=True,
                    num_heads=features // 64, 
                    dropout=self.dropout,
                    dtype=self.dtype
                )(x, emb, deterministic=deterministic)
                x = UNetBlock(
                    features, 
                    attention=False,
                    num_heads=features // 64, 
                    dropout=self.dropout,
                    dtype=self.dtype
                )(x, emb, deterministic=deterministic)
            else:
                x = UNetBlock(
                    features, 
                    resize="up",
                    dropout=self.dropout,
                    dtype=self.dtype
                )(x, emb, deterministic=deterministic)
            
            for idx in range(self.num_blocks + 1):
                x = jnp.concatenate([x, skips.pop()], axis=-1)
                x = UNetBlock(
                    features, 
                    attention=(level in self.attn_levels),
                    num_heads=features // 64, 
                    dropout=self.dropout,
                    dtype=self.dtype
                )(x, emb, deterministic=deterministic)

        x = jax.nn.silu(nn.GroupNorm(32, epsilon=1e-5, dtype=self.dtype)(x))
        x = nn.Conv(self.out_channels, 3, dtype=self.dtype)(x)

        return x

