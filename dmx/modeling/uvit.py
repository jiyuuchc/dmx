from dataclasses import dataclass, InitVar

import jax
import jax.numpy as jnp
import jax_wavelets as jw

from flax import nnx

from .common import UNetBlock, LabelEmbedding

@dataclass(repr=False)
class MLPBlock(nnx.Module):
    """MLP block with normalization and dropout."""
    in_channels: int
    emb_channels: int
    expansion_factor: int = 4
    adaptive_scale: bool = True
    dropout: float = 0.0
    
    dtype: type|None = None
    
    rngs: InitVar[nnx.Rngs|None] = None
    
    def __post_init__(self, rngs):
        kwargs = dict(dtype=self.dtype, rngs=rngs)

        self._norm = nnx.LayerNorm(self.in_channels, **kwargs)

        self._proj_in = nnx.Linear(
            self.in_channels, self.in_channels * self.expansion_factor, **kwargs,
        )

        self._proj_out = nnx.Linear(
            self.in_channels * self.expansion_factor, self.in_channels, 
            kernel_init=nnx.initializers.zeros,
            **kwargs,
        )

        if self.adaptive_scale:
            self._scale = nnx.Linear(
                self.emb_channels, self.in_channels * self.expansion_factor, **kwargs
            )

        self._shift = nnx.Linear(
            self.emb_channels, self.in_channels * self.expansion_factor, **kwargs
        )

        self._dropout = nnx.Dropout(self.dropout, rngs=rngs)


    def __call__(self, x, emb=None, *, deterministic=True, rngs=None):
        B, HW, C = x.shape
        assert C == self.in_channels, f"Input channels must match {self.in_channels}, got {C}"

        x = self._norm(x)
        x = jax.nn.silu(self._proj_in(x))
        
        if emb is not None:
            if self.adaptive_scale:
                x = x * self._scale(emb)[:, None, :] + self._shift(emb)[:, None, :]
            else:
                x = x + self._shift(emb)[:, None, :]

        if self.dropout > 0.:
            x = self._dropout(x, deterministic=deterministic, rngs=rngs)
        
        out = self._proj_out(x)
        
        return out


@dataclass(repr=False)
class UVit(nnx.Module):
    """ U-ViT model according to https://proceedings.mlr.press/v202/hoogeboom23a/hoogeboom23a.pdf
    """
    input_shape: tuple[int] = (512, 512, 1)
    out_channels: int = 1
    label_dim: int = 0
    model_channels: int = 128
    channel_mult: tuple[int] = (1,2,4,16)
    channel_mult_emb: int = 8
    num_blocks: tuple[int] = (2, 2, 2)
    num_transformer_blocks: int = 36
    num_heads:int = 4
    dropout: float = 0.10
    transformer_dropout = 0.2
    label_dropout: float = 0
    conv: str = "standard"  # standard|eff
    adaptive_scale: bool = True

    dtype: type|None = None

    rngs: InitVar[nnx.Rngs|None] = None
    
    def __post_init__(self, rngs):
        kwargs = dict(dtype=self.dtype, rngs=rngs)

        # embedding
        emb_channels=self.model_channels * self.channel_mult_emb
        self._embedding = LabelEmbedding(
            emb_channels=emb_channels,
            label_dim=self.label_dim,
            label_dropout=self.label_dropout,
            **kwargs,
        )
        self._conv_in = nnx.Conv(
            self.input_shape[-1] * 16, self.model_channels * self.channel_mult[0], 
            (3, 3),
            **kwargs
        )

        # Encoder
        encoder = []
        prev_features = self.input_shape[-1] * 16
        for level, n_blocks in enumerate(self.num_blocks):
            features = self.model_channels  * self.channel_mult[level]
            block = []
            for _ in range(n_blocks):
                block.append(UNetBlock(
                    features, features, emb_channels,
                    dropout=self.dropout,
                    residual=True,
                    adaptive_scale=self.adaptive_scale,
                    attention=False,
                    **kwargs,
                ))

            down_conv = nnx.Conv(
                features,
                self.model_channels * self.channel_mult[level+1],
                (3, 3), 
                strides=2, 
                **kwargs
            )

            encoder.append((block, down_conv))

        self._encoder = encoder

        # Transformer
        features = self.model_channels * self.channel_mult[-1]
        num_tokens = self.input_shape[1] // 32 * self.input_shape[1] // 32
        self.pos_emb = nnx.Param(
            jax.random.normal(rngs.params(), (num_tokens, features)) * 0.01
        )

        transformer = []
        for _ in range (self.num_transformer_blocks):
            transformer.append((
                nnx.MultiHeadAttention(
                    num_heads=self.num_heads,
                    in_features=features,
                    decode=False,
                    # force_fp32_for_softmax=True,
                    **kwargs,
                ),
                nnx.LayerNorm(features, **kwargs),
                MLPBlock(
                    in_channels=features,
                    emb_channels=emb_channels,
                    dropout=self.transformer_dropout,
                    adaptive_scale=self.adaptive_scale,
                    **kwargs,
                ),
                nnx.LayerNorm(features, **kwargs),
            ))
        self._transformer = transformer

        # Decoder
        decoder = []
        for level, n_blocks in reversed(list(enumerate(self.num_blocks))):
            features = self.model_channels * self.channel_mult[level]
            prev_features = self.model_channels * self.channel_mult[level + 1]
            up_conv = nnx.ConvTranspose(
                prev_features, features, (3, 3), strides=2, **kwargs
            )
            block = []
            for _ in range(n_blocks):
                block.append(UNetBlock(
                    features, features, emb_channels,
                    dropout=self.dropout,
                    residual=True,
                    adaptive_scale=self.adaptive_scale,
                    attention=False,
                    **kwargs,
                ))
            decoder.append((up_conv, block))
        self._decoder = decoder

        # out
        self._conv_out = nnx.Conv(
            self.model_channels * self.channel_mult[0],
            self.out_channels * 16, 
            (3, 3),
            **kwargs
        )


    def __call__ (self, x, noise_labels, class_labels=None, *, deterministic=True, rngs=None):
        assert x.ndim == 4, "Input must be a 4D tensor (batch, height, width, channels)"
        assert x.shape[-3:] == self.input_shape, f"Input shape must match {self.input_shape}, got {x.shape[-3:]}"

        emb = self._embedding(noise_labels, class_labels=class_labels, deterministic=deterministic)

        # patcherize input
        filt = jw.get_filter_bank("bior2.2")
        kernel_dec, _ = jw.make_kernels(filt, x.shape[-1])
        x = jw.wavelet_dec(x, kernel_dec.astype(x.dtype), levels=2)

        x = self._conv_in(x)

        # Encoder
        skips = []
        for block, down_conv in self._encoder:
            for layer in block:
                x = layer(x, emb, deterministic=deterministic, rngs=rngs)
                skips.append(x)
            # Downsample
            x = down_conv(x)

        # Transformer.
        B, H, W, C = x.shape
        x = x.reshape(B, H * W , -1)
        x += self.pos_emb.astype(x.dtype)

        for att, norm_1, mlp, norm_2 in self._transformer:
            x += att(x, deterministic=deterministic, rngs=rngs)
            x = norm_1(x)
            x += mlp(x, emb, deterministic=deterministic, rngs=rngs)
            x = norm_2(x)

        x = x.reshape(B , H , W , C)

        # Decoder
        for up_conv, block in self._decoder:
            x = up_conv(x)

            for layer in block:
                x = layer(x + skips.pop(), emb, deterministic=deterministic, rngs=rngs)

        # Project to output channels
        out = self._conv_out(x)

        _, kernel_rec = jw.make_kernels(filt, self.out_channels)
        out = jw.wavelet_rec(out, kernel_rec.astype(out.dtype), levels=2)

        return out
