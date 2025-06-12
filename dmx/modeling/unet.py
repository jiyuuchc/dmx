import flax.linen as nn
import jax
import jax.numpy as jnp

from .common import UNetBlock, LabelEmbedding


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

    dtype: type|None = None

    @nn.compact
    def __call__(self, x, noise_labels, class_labels=None, *, deterministic=True):
        B, H, W, C = x.shape

        emb_channels = self.model_channels * self.channel_mult_emb
        emb = LabelEmbedding(
            emb_channels=emb_channels,
            label_dropout=self.label_dropout,
            dtype=self.dtype,
        )(noise_labels, class_labels=class_labels, deterministic=deterministic)

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
