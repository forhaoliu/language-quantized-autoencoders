import functools
import io
import math
import os
from typing import Any, Callable, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import requests
from ml_collections import ConfigDict

from .model_utils import ACT2FN


class ResBlock(nn.Module):
    """Basic Residual Block."""

    filters: int
    norm_fn: Any
    conv_fn: Any
    dtype: int = jnp.float32
    activation_fn: Any = nn.relu
    use_conv_shortcut: bool = False

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        residual = x
        x = self.norm_fn()(x)
        x = self.activation_fn(x)
        x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
        x = self.norm_fn()(x)
        x = self.activation_fn(x)
        x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)

        if input_dim != self.filters:
            if self.use_conv_shortcut:
                residual = self.conv_fn(
                    self.filters, kernel_size=(3, 3), use_bias=False
                )(x)
            else:
                residual = self.conv_fn(
                    self.filters, kernel_size=(1, 1), use_bias=False
                )(x)
        return x + residual


class ResNetEncoder(nn.Module):
    """Encoder Blocks."""

    config: ConfigDict
    dtype: int = jnp.float32

    def setup(self):
        self.filters = self.config.filters
        self.num_res_blocks = self.config.num_res_blocks
        self.channel_multipliers = self.config.channel_multipliers
        self.hidden_size = self.config.hidden_size
        self.conv_downsample = self.config.conv_downsample
        self.norm_type = "GN"
        self.activation_fn = ACT2FN["swish"]

    @nn.compact
    def __call__(self, x, train):
        conv_fn = nn.Conv
        norm_fn = get_norm_layer(
            train=train, dtype=self.dtype, norm_type=self.norm_type
        )
        block_args = dict(
            norm_fn=norm_fn,
            conv_fn=conv_fn,
            dtype=self.dtype,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
        )
        x = conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
        num_blocks = len(self.channel_multipliers)
        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            for _ in range(self.num_res_blocks):
                x = ResBlock(filters, **block_args)(x)
            if i < num_blocks - 1:
                if self.conv_downsample:
                    x = conv_fn(filters, kernel_size=(4, 4), strides=(2, 2))(x)
                else:
                    x = dsample(x)
        for _ in range(self.num_res_blocks):
            x = ResBlock(filters, **block_args)(x)
        x = norm_fn()(x)
        x = self.activation_fn(x)
        x = conv_fn(self.hidden_size, kernel_size=(1, 1))(x)
        return x, None


class ResNetDecoder(nn.Module):
    """Decoder Blocks."""

    config: ConfigDict
    output_dim: int = 3
    dtype: Any = jnp.float32

    def setup(self):
        self.filters = self.config.filters
        self.num_res_blocks = self.config.num_res_blocks
        self.channel_multipliers = self.config.channel_multipliers
        self.norm_type = "GN"
        self.activation_fn = ACT2FN["swish"]

    @nn.compact
    def __call__(self, x, train):
        conv_fn = nn.Conv
        norm_fn = get_norm_layer(
            train=train, dtype=self.dtype, norm_type=self.norm_type
        )
        block_args = dict(
            norm_fn=norm_fn,
            conv_fn=conv_fn,
            dtype=self.dtype,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
        )
        num_blocks = len(self.channel_multipliers)
        filters = self.filters * self.channel_multipliers[-1]
        x = conv_fn(filters, kernel_size=(3, 3), use_bias=True)(x)
        for _ in range(self.num_res_blocks):
            x = ResBlock(filters, **block_args)(x)
        for i in reversed(range(num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            for _ in range(self.num_res_blocks):
                x = ResBlock(filters, **block_args)(x)
            if i > 0:
                x = upsample(x, 2)
                x = conv_fn(filters, kernel_size=(3, 3))(x)
        x = norm_fn()(x)
        x = self.activation_fn(x)
        x = conv_fn(self.output_dim, kernel_size=(3, 3))(x)
        return x


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


def tensorflow_style_avg_pooling(x, window_shape, strides, padding: str):
    """Avg pooling as done by TF (Flax layer gives different results).
    To be specific, Flax includes padding cells when taking the average,
    while TF does not.
    Args:
      x: Input tensor
      window_shape: Shape of pooling window; if 1-dim tuple is just 1d pooling, if
        2-dim tuple one gets 2d pooling.
      strides: Must have the same dimension as the window_shape.
      padding: Either 'SAME' or 'VALID' to indicate pooling method.
    Returns:
      pooled: Tensor after applying pooling.
    """
    pool_sum = jax.lax.reduce_window(
        x, 0.0, jax.lax.add, (1,) + window_shape + (1,), (1,) + strides + (1,), padding
    )
    pool_denom = jax.lax.reduce_window(
        jnp.ones_like(x),
        0.0,
        jax.lax.add,
        (1,) + window_shape + (1,),
        (1,) + strides + (1,),
        padding,
    )
    return pool_sum / pool_denom


def upsample(x, factor=2):
    n, h, w, c = x.shape
    x = jax.image.resize(x, (n, h * factor, w * factor, c), method="nearest")
    return x


def dsample(x):
    return tensorflow_style_avg_pooling(x, (2, 2), strides=(2, 2), padding="same")


def get_norm_layer(train, dtype, norm_type="BN"):
    """Normalization layer."""
    if norm_type == "BN":
        norm_fn = functools.partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            axis_name=None,
            axis_index_groups=None,
            dtype=jnp.float32,
        )
    elif norm_type == "LN":
        norm_fn = functools.partial(nn.LayerNorm, dtype=dtype)
    elif norm_type == "GN":
        norm_fn = functools.partial(nn.GroupNorm, dtype=dtype)
    else:
        raise NotImplementedError
    return norm_fn
