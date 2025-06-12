import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_wavelets as jw

from .common import UNetBlock, LabelEmbedding

class MLPBlock(nn.Module):
    """MLP block with normalization and dropout."""
    expansion_factor: int = 4
    dropout: float = 0.0
    dtype: type|None = None

    @nn.compact
    def __call__(self, x, emb=None, *, deterministic=True):
        B, HW, C = x.shape

        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.Dense(self.expansion_factor * C)(x)
        x = jax.nn.silu(x)
        
        if emb is not None:
            scale = nn.Dense(x.shape[-1], dtype=self.dtype)(emb)
            shift = nn.Dense(x.shape[-1], dtype=self.dtype)(emb)
            x = x * scale[:, None, :] + shift[:, None, :]

        if self.dropout > 0.:
            x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        
        out = nn.Dense(C, kernel_init=nn.initializers.zeros)(x)
        
        return out


class UVit(nn.Module):
    """ U-ViT model according to https://proceedings.mlr.press/v202/hoogeboom23a/hoogeboom23a.pdf
    """
    out_channels: int = 1
    model_channels: int = 128
    channel_mult: tuple[int] = (1,2,4,16)
    channel_mult_emb: int = 8
    num_blocks: tuple[int] = (2, 2, 2)
    num_transformer_blocks: int = 36
    num_heads:int = 4
    dropout: float = 0.10
    transformer_dropout = 0.2
    label_dropout: float = 0

    dtype: type|None = None

    @nn.compact
    def __call__ (self, x, noise_labels, class_labels=None, *, deterministic=True):
        emb = LabelEmbedding(
            self.model_channels * self.channel_mult_emb,
            label_dropout=self.label_dropout, 
            dtype=self.dtype,
        )(noise_labels, class_labels=class_labels, deterministic=deterministic)

        # patcherize input
        filt = jw.get_filter_bank("bior2.2")
        kernel_dec, kernel_rec = jw.make_kernels(filt, x.shape[-1])
        x = jw.wavelet_dec(x, kernel_dec.astype(x.dtype), levels=2)

        x = nn.Conv(
            self.channel_mult[0] * self.model_channels, 
            (3, 3), 
            dtype=self.dtype,
        )(x)

        # Encoder
        skips = []
        for level, n_blocks in enumerate(self.num_blocks):
            mult = self.channel_mult[level]

            for _ in range(n_blocks):
                x = UNetBlock(
                    self.model_channels * mult,
                    dropout=self.dropout,
                    residual=True,
                    adaptive_scale=True,
                    attention=False,
                    dtype=self.dtype,
                )(x, emb, deterministic=deterministic)
                skips.append(x)

            # Downsample          
            mult = self.channel_mult[level + 1]
            x = nn.Conv(
                self.model_channels * mult, 
                (3, 3),
                strides=2, 
                dtype=self.dtype
            )(x)

        # Transformer.
        B, H, W, C = x.shape
        x = x.reshape(B, H * W , -1)
        pos_emb = self.param("pos_emb", nn.initializers.normal(0.01), x.shape[-2:])
        x += pos_emb.astype(x.dtype)

        for _ in range (self.num_transformer_blocks):
            x += nn.MultiHeadAttention(
                num_heads=self.num_heads,
                force_fp32_for_softmax=True,
                dtype=self.dtype,
            )(x, deterministic=deterministic)
            x = nn.LayerNorm(dtype=self.dtype)(x)
            x += MLPBlock(
                expansion_factor=4,
                dropout=self.transformer_dropout,
                dtype=self.dtype
            )(x, emb, deterministic=deterministic)
            x = nn.LayerNorm(dtype=self.dtype)(x)
        x = x.reshape(B , H , W , C)

        # Up path
        for level, n_blocks in reversed(list(enumerate(self.num_blocks))):
            mult = self.channel_mult[level]
            x = nn.ConvTranspose(
                self.model_channels * mult, 
                (3, 3), 
                strides=2, 
                dtype=self.dtype
            )(x)

            for _ in range(n_blocks):
                x = UNetBlock(
                    self.model_channels * mult,
                    dropout=self.dropout,
                    residual=True,
                    attention=False,
                    dtype=self.dtype,
                )(x + skips.pop(), emb, deterministic=deterministic)

        # Project to output channels
        out = nn.Dense(self.out_channels * 16, dtype=self.dtype)(x)
        out = jw.wavelet_rec(out, kernel_rec.astype(out.dtype), levels=2)

        return out
