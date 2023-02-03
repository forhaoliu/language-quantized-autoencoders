import os
import time
from functools import partial
from typing import Any, Mapping, Text, Tuple, Union

import dill
import flax
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from flax import jax_utils


class JaxRNG(object):
    """A convenient stateful Jax RNG wrapper. Can be used to wrap RNG inside pure function."""

    @classmethod
    def from_seed(cls, seed):
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


def wrap_function_with_rng(rng):
    """To be used as decorator, automatically bookkeep a RNG for the wrapped function."""

    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, split_rng = jax.random.split(rng)
            return function(split_rng, *args, **kwargs)

        return wrapped

    return wrap_function


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)


def get_metrics(metrics, unreplicate=False, stack=False):
    if unreplicate:
        metrics = flax.jax_utils.unreplicate(metrics)
    metrics = jax.device_get(metrics)
    if stack:
        return jax.tree_map(lambda *args: np.stack(args), *metrics)
    else:
        return {key: float(val) for key, val in metrics.items()}


def get_onehot(labels, num_classes, on_value=1.0, off_value=0.0):
    x = labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,))
    x = jax.lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return x.astype(jnp.float32)


def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def mse_loss(val, target):
    return jnp.mean(jnp.square(val - target))


def cross_entropy_loss(logits, labels, smoothing_factor=0.0):
    num_classes = logits.shape[-1]
    if labels.dtype == jnp.int32 or labels.dtype == jnp.int64:
        labels = jax.nn.one_hot(labels, num_classes)
    if smoothing_factor > 0.0:
        labels = labels * (1.0 - smoothing_factor) + smoothing_factor / num_classes
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(logp * labels, axis=-1))


def accumulated_gradient(
    state, accumulated_grads, accumulated_steps, grads, apply_steps, apply_fn=None
):
    accumulated_grads = jax.tree_map(
        lambda x, y: x + y,
        accumulated_grads,
        jax.tree_map(lambda x: x / float(apply_steps), grads),
    )
    accumulated_steps = (accumulated_steps + 1) % apply_steps
    if apply_fn is None:
        apply_fn = lambda s, g: s.apply_gradients(grads=g)
    state = jax.lax.cond(
        accumulated_steps == 0,
        lambda: apply_fn(state, accumulated_grads),
        lambda: state,
    )
    accumulated_grads = jax.lax.cond(
        accumulated_steps == 0,
        lambda: jax.tree_map(jnp.zeros_like, accumulated_grads),
        lambda: accumulated_grads,
    )
    return state, accumulated_grads, accumulated_steps


@partial(jax.pmap, axis_name="pmap", donate_argnums=0)
def sync_state_across_devices(state):
    i = jax.lax.axis_index("pmap")

    def select(x):
        return jax.lax.psum(jnp.where(i == 0, x, jnp.zeros_like(x)), "pmap")

    return jax.tree_map(select, state)


def get_random_bounding_box(
    image_shape: Tuple[int, int], lambda_cutmix: float, margin: float = 0.0
) -> Tuple[int, int, int, int]:
    ratio = np.sqrt(1 - lambda_cutmix)
    img_h, img_w = image_shape
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y)
    cx = np.random.randint(0 + margin_x, img_w - margin_x)
    y_min = np.clip(cy - cut_h // 2, 0, img_h)
    y_max = np.clip(cy + cut_h // 2, 0, img_h)
    x_min = np.clip(cx - cut_w // 2, 0, img_w)
    x_max = np.clip(cx + cut_w // 2, 0, img_w)
    return y_min, y_max, x_min, x_max


def label_smoothing_fn(labels, smoothing_factor):
    num_classes = labels.shape[-1]
    labels = labels * (1.0 - smoothing_factor) + smoothing_factor / num_classes
    return labels


def mixup_cutmix(
    images: jnp.ndarray,
    labels: jnp.ndarray,
    rng: Any,
    num_classes: int,
    mixup_alpha: float = 1.0,
    cutmix_alpha: float = 0.0,
    switch_prob: float = 0.5,
    label_smoothing: float = 0.0,
    image_format: str = "NHWC"
    ):
    if len(labels.shape) == 1:
        labels = jax.nn.one_hot(labels, num_classes)

    if "N" not in image_format:
        raise ValueError('Mixup requires "N" to be in "image_format".')

    batch_size = labels.shape[0]

    if cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < switch_prob):
        do_mixup = False
        do_cutmix = True
    elif mixup_alpha > 0:
        do_mixup = True
        do_cutmix = False
    else:
        return images, labels

    if do_mixup:
        weight = jax.random.beta(rng, mixup_alpha, mixup_alpha)
        weight *= jnp.ones((batch_size, 1))

        # Mixup inputs.
        # Shape calculations use np to avoid device memory fragmentation:
        image_weight_shape = np.ones((images.ndim))
        image_weight_shape[image_format.index("N")] = batch_size
        image_weight = jnp.reshape(
            weight, image_weight_shape.astype(jnp.int32)
        )
        reverse = tuple(
            slice(images.shape[i]) if d != "N" else slice(-1, None, -1)
            for i, d in enumerate(image_format)
        )
        mixed_images = image_weight * images + (1.0 - image_weight) * images[reverse]
        label_weight = weight

    elif do_cutmix:
        if image_format not in {"NHWC", "NTHWC"}:
            raise ValueError(
                "Cutmix is only supported for inputs in format"
                f" NHWC or NTHWC. Got {image_format}."
            )
        cutmix_lambda = np.random.beta(cutmix_alpha, cutmix_alpha)

        y_min, y_max, x_min, x_max = get_random_bounding_box(
            images.shape[-3:-1], cutmix_lambda
        )
        image_mask = np.ones(images.shape)
        if image_format == "NHWC":
            image_mask[:, y_min:y_max, x_min:x_max, :] = 0.0
        else:
            image_mask[:, :, y_min:y_max, x_min:x_max, :] = 0.0
        height, width = images.shape[-3], images.shape[-2]

        mixed_images = images * image_mask + jnp.flip(images, axis=0) * (
            1.0 - image_mask
        )
        box_area = (y_max - y_min) * (x_max - x_min)
        label_weight = 1.0 - box_area / float(height * width)

    # Mixup label
    if label_smoothing > 0:
        labels = label_smoothing_fn(labels, label_smoothing)

    mixed_labels = label_weight * labels + (1.0 - label_weight) * labels[::-1]

    return mixed_images, mixed_labels
