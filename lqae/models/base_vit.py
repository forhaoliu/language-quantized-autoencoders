import enum
import math
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.attention import dot_product_attention_weights
from flax.linen.initializers import variance_scaling
from ml_collections import ConfigDict

from .model_utils import ACT2FN


# copied from https://github.com/deepmind/dm-haiku/blob/3f31e279d4ce613ae3e47b97031f8b2d732071b7/haiku/_src/spectral_norm.py#L46
def l2_normalize(x, axis=None, eps=1e-12):
    """Normalizes along dimension `axis` using an L2 norm.
    This specialized function exists for numerical stability reasons.
    Args:
      x: An input ndarray.
      axis: Dimension along which to normalize, e.g. `1` to separately normalize
        vectors in a batch. Passing `None` views `t` as a flattened vector when
        calculating the norm (equivalent to Frobenius norm).
      eps: Epsilon to avoid dividing by zero.
    Returns:
      An array of the same shape as 'x' L2-normalized along 'axis'.
    """
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


# Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_flax_gptj.py
def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype(
        "float32"
    )
    sin, cos = np.sin(sinusoid_inp), np.cos(sinusoid_inp)
    sentinel = dim // 2 + dim % 2
    out = np.zeros((num_pos, dim))
    out[:, 0:sentinel] = sin
    out[:, sentinel:] = cos
    return jnp.array(out)


def patch_unflatten(patch, shape):
    return patch.reshape(patch.shape[0], *shape, patch.shape[-1])


def patch_flatten(patch):
    return patch.reshape(patch.shape[0], -1, patch.shape[-1]), patch.shape[1:-1]


class ConvPatches(nn.Module):
    patch_size: Tuple[int]
    hidden_size: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, pixel_values):
        patch_embeds = nn.Conv(
            self.hidden_size,
            kernel_size=[self.patch_size, self.patch_size],
            strides=[self.patch_size, self.patch_size],
            padding="VALID",
            use_bias=False,
            dtype=self.dtype,
            name="patch_embeds",
            kernel_init=jax.nn.initializers.normal(0.02),
        )(pixel_values)
        return patch_embeds


class GLU(nn.Module):
    dim1: int
    dim2: int
    activation: str
    dropout: float
    space_only_conv: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, deterministic: bool = True):
        Dense = partial(
            nn.Dense,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
        )

        hidden_states = nn.LayerNorm(
            epsilon=1e-5, dtype=self.dtype, name="layernorm_0"
        )(hidden_states)

        hidden_gelu = Dense(features=self.dim1, name="fc1")(hidden_states)
        hidden_gelu = ACT2FN[self.activation](hidden_gelu)

        hidden_linear = Dense(features=self.dim1, name="fc2")(hidden_states)

        hidden_states = hidden_gelu * hidden_linear

        # suggestion from Katherine Crowson
        ndims = len(hidden_states.shape[1:-1])
        assert ndims in [2, 3]
        if self.space_only_conv:
            kernel = (3, 3) if ndims == 2 else (1, 3, 3)
        else:
            kernel = (3,) * ndims
        hidden_states = nn.Conv(
            self.dim1,
            kernel_size=kernel,
            strides=(1,) * ndims,
            padding="SAME",
            feature_group_count=self.dim1,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            name="mid_ffn_conv",
        )(hidden_states)

        hidden_states = nn.LayerNorm(
            epsilon=1e-5, dtype=self.dtype, name="layernorm_1"
        )(hidden_states)

        hidden_states = nn.Dropout(rate=self.dropout)(
            hidden_states, deterministic=deterministic
        )
        hidden_states = Dense(features=self.dim2, name="fc_out")(hidden_states)
        hidden_states = nn.Dropout(rate=self.dropout)(
            hidden_states, deterministic=deterministic
        )
        return hidden_states


class Attention(nn.Module):
    hidden_size: int
    num_heads: int
    dropout: float
    dtype: jnp.dtype = jnp.float32

    def _split_heads(self, hidden_states):
        head_dim = self.hidden_size // self.num_heads
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.num_heads, head_dim)
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    @nn.compact
    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        mask: Optional[jnp.ndarray] = None,
    ):
        Dense = partial(
            nn.Dense,
            self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
        )
        hidden_states, shape = patch_flatten(hidden_states)

        query = Dense()(hidden_states)
        key = Dense()(hidden_states)
        value = Dense()(hidden_states)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=None,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
            mask=mask,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = Dense()(attn_output)
        attn_output = patch_unflatten(attn_output, shape)
        return attn_output


class TransformerBlock(nn.Module):
    hidden_size: int
    intermediate_size: int
    num_heads: int
    dropout: float
    deterministic: bool
    space_only_conv: bool = False
    AttentionModule: Any = Attention
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, mask: Optional[jnp.ndarray] = None):
        deterministic = self.deterministic
        residual = hidden_states

        hidden_states = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(hidden_states)
        hidden_states = self.AttentionModule(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            dtype=self.dtype,
        )(hidden_states=hidden_states, deterministic=deterministic, mask=mask)
        # normformer
        hidden_states = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = GLU(
            dim1=self.intermediate_size,
            dim2=self.hidden_size,
            activation="gelu",
            dropout=self.dropout,
            space_only_conv=self.space_only_conv,
            dtype=self.dtype,
        )(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        return hidden_states


class Transformer(nn.Module):
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    dropout: float
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        mask: Optional[jnp.ndarray] = None,
    ):
        space_only_conv = mask is not None
        layer_outputs = []
        for i in range(self.num_layers):
            AttentionModule = Attention
            assert mask is None
            mask_in = mask

            hidden_states = nn.remat(TransformerBlock)(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_heads=self.num_heads,
                dropout=self.dropout,
                name=str(i),
                deterministic=deterministic,
                space_only_conv=space_only_conv,
                AttentionModule=AttentionModule,
                dtype=self.dtype,
            )(hidden_states, mask_in)
            layer_outputs.append(hidden_states)
        return hidden_states, layer_outputs


class VitEncoder(nn.Module):
    config: ConfigDict
    causal_mask: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        mask: Optional[jnp.ndarray] = None,
    ):
        hidden_states = ConvPatches(
            patch_size=self.config.patch_size,
            hidden_size=self.config.hidden_size,
            dtype=self.dtype,
        )(pixel_values)

        position_embeddings = self.param(
            "pos_embedding",
            jax.nn.initializers.normal(0.02, dtype=jnp.float32),
            (1, *hidden_states.shape[1:-1], self.config.hidden_size),
        )
        hidden_states += position_embeddings
        hidden_states = nn.Dropout(rate=self.config.dropout)(
            hidden_states, deterministic=deterministic
        )

        if self.causal_mask:
            assert len(hidden_states.shape[1:-1]) == 3, hidden_states.shape
            T = hidden_states.shape[1]
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        else:
            mask = None

        hidden_states, layer_outputs = Transformer(
            num_layers=self.config.enc_num_layers,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_heads=self.config.enc_num_heads,
            dropout=self.config.dropout,
            dtype=self.dtype,
        )(hidden_states, deterministic=deterministic, mask=mask)
        return hidden_states, layer_outputs


class VitDecoder(nn.Module):
    config: ConfigDict
    causal_mask: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        mask: Optional[jnp.ndarray] = None,
    ):
        position_embeddings = self.param(
            "pos_embedding",
            jax.nn.initializers.normal(0.02, dtype=jnp.float32),
            (1, *hidden_states.shape[1:-1], self.config.hidden_size),
        )
        hidden_states += position_embeddings

        if self.causal_mask:
            assert len(hidden_states.shape[1:-1]) == 3, hidden_states.shape
            T = hidden_states.shape[1]
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        else:
            mask = None

        hidden_states, layer_outputs = Transformer(
            num_layers=self.config.dec_num_layers,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_heads=self.config.dec_num_heads,
            dropout=self.config.dropout,
            dtype=self.dtype,
        )(hidden_states, deterministic=deterministic, mask=mask)

        images = nn.ConvTranspose(
            3,
            kernel_size=[self.config.patch_size, self.config.patch_size],
            strides=[self.config.patch_size, self.config.patch_size],
            padding="VALID",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
        )(hidden_states)
        return images
