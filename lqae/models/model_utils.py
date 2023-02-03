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
from functools import partial




def normalize_func(x, axis=None, eps=1e-12, use_l2_normalize=True):
    if use_l2_normalize:
        return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)
    else:
        return x


def squared_euclidean_distance(
    a: jnp.ndarray, b: jnp.ndarray, b2: jnp.ndarray = None, precision: Any = None,
    dot_product: bool = False,
) -> jnp.ndarray:
    """Computes the pairwise squared Euclidean distance.
    Args:
      a: float32: (n, d): An array of points.
      b: float32: (m, d): An array of points.
      b2: float32: (d, m): b square transpose.
      precision: use DEFAULT precision by default
    Returns:
      d: float32: (n, m): Where d[i, j] is the squared Euclidean distance between
      a[i] and b[j].
    """
    if dot_product:
        return jnp.matmul(a, b.T, precision=precision)
    if b2 is None:
        b2 = jnp.sum(b.T**2, axis=0, keepdims=True)
    a2 = jnp.sum(a**2, axis=1, keepdims=True)
    ab = jnp.matmul(a, b.T, precision=precision)
    d = a2 - 2 * ab + b2
    return d


def entropy_loss_fn(affinity, loss_type="softmax", temperature=1.0):
    """Calculates the entropy loss."""
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = jax.nn.softmax(flat_affinity, axis=-1)
    log_probs = jax.nn.log_softmax(flat_affinity + 1e-5, axis=-1)
    if loss_type == "softmax":
        target_probs = probs
    elif loss_type == "argmax":
        codes = jnp.argmax(flat_affinity, axis=-1)
        onehots = jax.nn.one_hot(
            codes, flat_affinity.shape[-1], dtype=flat_affinity.dtype
        )
        onehots = probs - jax.lax.stop_gradient(probs - onehots)
        target_probs = onehots
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = jnp.mean(target_probs, axis=0)
    avg_entropy = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-5))
    sample_entropy = -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))
    loss = sample_entropy - avg_entropy
    return loss


class LinearCLS(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        norm = functools.partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            use_scale=False,
            use_bias=False,
        )
        x = norm(name="bn")(x)
        logits = nn.Dense(self.num_classes)(x)
        return logits


def update_vit_config(model_type, config):
    def get_config(model_type):
        if model_type == "small":
            return 12, 6
        elif model_type == "base":
            return 12, 12
        elif model_type == "large":
            return 24, 16
        elif model_type == "huge":
            return 32, 16
        elif model_type == "decoder_small":
            return 8, 16
        elif model_type == "decoder_base":
            return 12, 16
        elif model_type == "decoder_large":
            return 16, 16
        else:
            return tuple(int(x) for x in model_type.split("/"))

    if model_type == "debug":
        config.enc_num_layers = 2
        config.enc_num_heads = 4
        config.dec_num_layers = 2
        config.dec_num_heads = 4
    elif ":" not in model_type:
        config.enc_num_layers, config.enc_num_heads = get_config(model_type)
        config.dec_num_layers, config.dec_num_heads = get_config(model_type)
    else:
        encoder_type, decoder_type = model_type.split(":")
        config.enc_num_layers, config.enc_num_heads = get_config(encoder_type)
        config.dec_num_layers, config.dec_num_heads = get_config(decoder_type)

    assert_hidden_size(config)


def assert_avg_rnd(config):
    if config.top_k_avg and config.top_k_rnd:
        raise ValueError("top_k_avg and top_k_rnd are mutually exclusive")
    if config.top_k_value > 1:
        assert (
            config.top_k_avg or config.top_k_rnd
        ), "top_k_avg or top_k_rnd must be True when top_k_value > 1"
    elif config.top_k_value == 1:
        assert (
            not config.top_k_avg and not config.top_k_rnd
        ), "top_k_avg and top_k_rnd must be False when top_k_value == 1"


def assert_hidden_size(config):
    assert config.hidden_size % config.enc_num_heads == 0
    assert config.hidden_size % config.dec_num_heads == 0


ACT2FN = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
    "tanh": nn.tanh,
}
