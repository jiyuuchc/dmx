from dataclasses import dataclass, InitVar

import jax
import jax.numpy as jnp
from flax import nnx

from .common import UNetBlock, LabelEmbedding

@dataclass(repr=False)
class _Encoder(nnx.Module):
    """Encoder block for the Dhariwal UNet."""
    in_channels: int
    emb_channels: int
    dropout: float = 0.0
    model_channels: int = 192
    channel_mult: tuple[int] = (1,2,3,4)
    attn_levels: tuple[int] = (1,2,3)
    num_blocks: int = 3

    dtype: type|None = None
    rngs: InitVar[nnx.Rngs|None] = None

    def __post_init__(self, rngs):
        kwargs = dict(dtype=self.dtype, rngs=rngs)

        encoder = []
        for level, mult in enumerate(self.channel_mult):
            features = self.model_channels * mult

            if level == 0:
                encoder.append(nnx.Conv(self.in_channels, features, 3, **kwargs))
            else:
                encoder.append(UNetBlock(
                    prev_features, features, self.emb_channels,
                    resize="down",
                    dropout=self.dropout,
                    **kwargs
                ))

            for idx in range(self.num_blocks):
                encoder.append(UNetBlock(
                    features, features, self.emb_channels,
                    attention=level in self.attn_levels,
                    num_heads=features // 64, 
                    dropout=self.dropout,
                    **kwargs
                ))

            prev_features = features
        
        self._encoder = encoder

    def __call__(self, x, emb, *, deterministic=True, rngs=None):
        skips = []
        for layer in self._encoder:
            if isinstance(layer, nnx.Conv):
                x = layer(x)
            else:
                x = layer(x, emb, deterministic=deterministic, rngs=rngs)

            skips.append(x)

        return skips

@dataclass(repr=False)
class _Decoder(nnx.Module):
    emb_channels: int
    dropout: float = 0.0
    model_channels: int = 192
    channel_mult: tuple[int] = (1,2,3,4)
    attn_levels: tuple[int] = (1,2,3)
    num_blocks: int = 3

    dtype: type|None = None
    rngs: InitVar[nnx.Rngs|None] = None

    def __post_init__(self, rngs):
        kwargs = dict(dtype=self.dtype, rngs=rngs)

        decoder = []
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            features = self.model_channels * mult

            decoder_block = []

            if level == len(self.channel_mult) - 1:
                pre = UNetBlock(
                    features, features, self.emb_channels,
                    attention=True,
                    num_heads=features // 64, 
                    dropout=self.dropout,
                    **kwargs
                )
            else:
                in_features = self.model_channels * self.channel_mult[level + 1]
                pre = UNetBlock(
                    in_features, features, self.emb_channels,
                    resize="up",
                    dropout=self.dropout,
                    **kwargs,
                )

            for idx in range(self.num_blocks + 1):
                decoder_block.append(UNetBlock(
                    features * 2, features, self.emb_channels,
                    attention=(level in self.attn_levels),
                    num_heads=features // 64, 
                    dropout=self.dropout,
                    **kwargs
                ))

            decoder.append((pre, decoder_block))

        self._decoder = decoder

    def __call__(self, skips, emb, *, deterministic=True, rngs=None):
        x = skips[-1]
        for pre, post in self._decoder:
            x = pre(x, emb, deterministic=deterministic, rngs=rngs)

            for layer in post:
                x = jnp.concatenate([x, skips.pop()], axis=-1)
                x = layer(x, emb, deterministic=deterministic, rngs=rngs)

        return x


@dataclass(repr=False)
class DhariwalUNet(nnx.Module):
    """ Reimplementation of the DM architecture from the paper
    'Diffusion Models Beat GANS on Image Synthesis'. Equivalent to the
    original implementation by Dhariwal and Nichol, available at
    https://github.com/openai/guided-diffusion

    Parameters:
        in_channels: Number of color channels at input.
        out_channels: Number of color channels at output.
        model_channels: Base multiplier for the number of channels.
        channel_mult: Per-resolution multipliers for the number of channels.
        channel_mult_emb: Multiplier for the dimensionality of the embedding vector.
        num_blocks: Number of residual blocks per resolution.
        attn_resolutions: List of resolutions with self-attention.
        dropout: Dropout probability for the residual blocks.
        label_dim: Dimensionality of the class labels for classifier-free guidance. 0 means no labels.
        label_dropout: Dropout probability of class labels for classifier-free guidance.
    """
    in_channels: int = 1
    out_channels: int = 1
    model_channels: int = 192
    channel_mult: tuple[int] = (1,2,3,4)
    channel_mult_emb: int = 4
    num_blocks: int = 3
    attn_levels: tuple[int] = (1,2,3)
    dropout: float = 0.10
    label_dim: int = 0
    label_dropout: float = 0

    dtype: type|None = None

    rngs: InitVar[nnx.Rngs|None] = None
    
    def __post_init__(self, rngs):
        kwargs = dict(dtype=self.dtype, rngs=rngs)

        # embedding
        emb_channels=self.model_channels * self.channel_mult_emb
        self._embedding =  LabelEmbedding(
            emb_channels=emb_channels,
            label_dim=self.label_dim,
            label_dropout=self.label_dropout,
            **kwargs,
        )

        # encoder
        self._encoder = _Encoder(
            in_channels=self.in_channels,
            emb_channels=emb_channels,
            dropout=self.dropout,
            model_channels=self.model_channels,
            channel_mult=self.channel_mult,
            attn_levels=self.attn_levels,
            num_blocks=self.num_blocks,
            **kwargs,
        )

        # decoder
        self._decoder = _Decoder(
            emb_channels=emb_channels,
            dropout=self.dropout,
            model_channels=self.model_channels,
            channel_mult=self.channel_mult,
            attn_levels=self.attn_levels,
            num_blocks=self.num_blocks,
            **kwargs,
        )

        # Final
        features = self.model_channels * self.channel_mult[0]
        self._norm = nnx.GroupNorm(features, num_groups=32, epsilon=1e-5, **kwargs)
        self._conv = nnx.Conv(
            features, self.out_channels, (3,3), **kwargs
        )


    def __call__(self, x, noise_labels, class_labels=None, *, deterministic=True, rngs=None):
        B, H, W, C = x.shape

        assert C == self.in_channels, f"Input channels {C} do not match expected {self.in_channels}."

        emb = self._embedding(noise_labels, class_labels=class_labels, deterministic=deterministic)

        x = self._encoder(x, emb, deterministic=deterministic, rngs=rngs)
        x = self._decoder(x, emb, deterministic=deterministic, rngs=rngs)

        # Final layers
        x = jax.nn.silu(self._norm(x))
        x = self._conv(x)

        return x
