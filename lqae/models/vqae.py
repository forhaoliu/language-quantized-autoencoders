import functools
import io
import math
import os
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import requests
from flax.linen.initializers import variance_scaling
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from PIL import Image, ImageFilter

from .base_resnet import ResNetDecoder, ResNetEncoder
from .base_vit import VitDecoder, VitEncoder
from .model_utils import (
    assert_avg_rnd,
    update_vit_config,
    entropy_loss_fn,
    normalize_func,
    squared_euclidean_distance,
)


class VectorQuantizer(nn.Module):
    """Basic vector quantizer."""

    config: ConfigDict
    dtype: int = jnp.float32

    def setup(self):
        if self.config.quantizer_latent_dim > 0:
            self.input_to_latent = nn.Dense(
                self.config.quantizer_latent_dim, dtype=self.dtype
            )
            self.code_to_latent = nn.Dense(
                self.config.quantizer_latent_dim, dtype=self.dtype
            )
        else:
            self.input_to_latent = self.code_to_latent = lambda x: x

    @nn.compact
    def __call__(self, x, train, rng):
        l2_normalize = lambda x, axis=1: normalize_func(
            x, axis=axis, use_l2_normalize=self.config.l2_normalize
        )
        codebook_size = self.config.codebook_size
        embed_init = variance_scaling(1.0, "fan_in", "normal", out_axis=0)
        codebook = self.param(
            "codebook",
            embed_init,
            (codebook_size, x.shape[-1]),
        )
        if self.config.strawman_codebook:
            strawman_codebook = self.param(
                "strawman_codebook",
                jax.nn.initializers.normal(0.02, dtype=jnp.float32),
                (codebook_size, self.config.quantizer_latent_dim),
            )
            strawman_codebook = jnp.asarray(strawman_codebook, dtype=self.dtype)
            latent_input = self.input_to_latent(jnp.reshape(x, (-1, x.shape[-1])))
            latent_input = l2_normalize(latent_input, axis=1)
            sg_strawman_codebook = jax.lax.stop_gradient(
                l2_normalize(strawman_codebook, axis=1)
            )
            distances = jnp.reshape(
                squared_euclidean_distance(latent_input, sg_strawman_codebook),
                x.shape[:-1] + (codebook_size,),
            )
        else:
            codebook = jnp.asarray(codebook, dtype=self.dtype)
            latent_input = self.input_to_latent(jnp.reshape(x, (-1, x.shape[-1])))
            latent_input = l2_normalize(latent_input, axis=1)
            latent_codebook = self.code_to_latent(codebook)
            latent_codebook = l2_normalize(latent_codebook, axis=1)
            sg_latent_codebook = jax.lax.stop_gradient(
                l2_normalize(latent_codebook, axis=1)
            )
            distances = jnp.reshape(
                squared_euclidean_distance(latent_input, sg_latent_codebook),
                x.shape[:-1] + (codebook_size,),
            )

        encoding_indices = jax.lax.approx_min_k(
            distances,
            k=self.config.top_k_value,
            reduction_dimension=-1,
            aggregate_to_topk=True,
        )[1]

        encoding_indices, encodings, quantized = self.get_encoding_quantized(
            encoding_indices, train, rng, codebook_size
        )

        codebook_usage = jnp.sum(encodings, axis=(0, 1)) > 0
        codebook_usage = jnp.sum(codebook_usage) / codebook_size
        if self.config.top_k_avg:
            codebook_usage = codebook_usage / self.config.top_k_value
        result_dict = dict()
        if train:
            result_dict = self.get_train_loss(quantized, x, distances)

            if self.config.strawman_codebook:
                strawman_quantized = self.quantize_strawman(encodings)
                strawman_result_dict = self.get_train_loss(
                    strawman_quantized, self.input_to_latent(x), distances
                )
                for k, v in result_dict.items():
                    result_dict[k] = v + strawman_result_dict[k]
            else:
                latent_quantized = self.code_to_latent(quantized)
                latent_result_dict = self.get_train_loss(
                    latent_quantized, self.input_to_latent(x), distances
                )
                for k, v in result_dict.items():
                    result_dict[k] = v + latent_result_dict[k]

            quantized = x + jax.lax.stop_gradient(quantized - x)

        avg_probs = jnp.mean(encodings.reshape(-1, encodings.shape[-1]), axis=0)
        log_perplexity = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10))
        perplexity = jnp.exp(log_perplexity)

        if "quantizer_loss" in result_dict:
            result_dict["quantizer_loss"] = (
                result_dict["quantizer_loss"]
                + self.config.quantizer_loss_perplexity * log_perplexity
            )
        result_dict.update(
            {
                "encodings": encodings,
                "encoding_indices": encoding_indices,
                "raw": x,
                "perplexity": perplexity,
                "codebook_usage": codebook_usage,
            }
        )
        return quantized, result_dict

    def quantize(self, z: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(z, self.variables["params"]["codebook"])

    def get_codebook(self) -> jnp.ndarray:
        return self.variables["params"]["codebook"]

    def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(self.variables["params"]["codebook"], ids, axis=0)

    def quantize_strawman(self, z: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(z, self.variables["params"]["strawman_codebook"])

    def get_train_loss(self, quantized, x, distances):
        e_latent_loss = (
            jnp.mean((jax.lax.stop_gradient(quantized) - x) ** 2)
            * self.config.quantizer_loss_commitment
        )
        q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(x)) ** 2)
        entropy_loss = 0.0
        if self.config.quantizer_loss_entropy != 0:
            entropy_loss = (
                entropy_loss_fn(
                    -distances,
                    loss_type=self.config.entropy_loss_type,
                    temperature=self.config.entropy_temperature,
                )
                * self.config.quantizer_loss_entropy
            )
        e_latent_loss = jnp.asarray(e_latent_loss, jnp.float32)
        q_latent_loss = jnp.asarray(q_latent_loss, jnp.float32)
        entropy_loss = jnp.asarray(entropy_loss, jnp.float32)
        loss = e_latent_loss + q_latent_loss + entropy_loss

        result_dict = dict(
            quantizer_loss=loss,
            e_latent_loss=e_latent_loss,
            q_latent_loss=q_latent_loss,
            entropy_loss=entropy_loss,
        )
        return result_dict

    def get_encoding_quantized(self, encoding_indices, train, rng, codebook_size):
        if self.config.top_k_rnd:
            if train:
                encoding_indices = jax.random.choice(rng, encoding_indices, axis=-1)
            else:
                encoding_indices = encoding_indices[..., 0]
            encodings = jax.nn.one_hot(
                encoding_indices, codebook_size, dtype=self.dtype
            )
            quantized = self.quantize(encodings)
        elif self.config.top_k_avg:
            encodings = jax.nn.one_hot(
                encoding_indices, codebook_size, dtype=self.dtype
            )
            quantized = self.quantize(encodings)
            quantized = jnp.mean(quantized, axis=-2)
            encoding_indices = encoding_indices[..., 0]
        else:
            encoding_indices = encoding_indices[..., 0]
            encodings = jax.nn.one_hot(
                encoding_indices, codebook_size, dtype=self.dtype
            )
            quantized = self.quantize(encodings)
        return encoding_indices, encodings, quantized


class VQAE(nn.Module):
    config_updates: ... = None
    dtype: int = jnp.float32

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = ConfigDict()

        # Quantizer config
        config.quantizer_loss_entropy = 0.0
        config.entropy_temperature = 0.01
        config.entropy_loss_type = "softmax"
        config.quantizer_loss_commitment = 0.25
        config.l2_normalize = False
        config.top_k_value = 1
        config.top_k_avg = False
        config.top_k_rnd = False
        config.quantizer_latent_dim = 0
        config.strawman_codebook = False
        config.quantizer_loss_perplexity = 0.0
        # VQ quantizer config
        config.codebook_size = 1024

        # ResNet config
        config.filters = 128
        config.num_res_blocks = 2
        config.channel_multipliers = [1, 1, 2, 2, 4]
        config.hidden_size = 768
        config.conv_downsample = False

        # VIT config
        config.vit_encoder_decoder = False
        config.vit_model_type = config_dict.placeholder(str)
        config.patch_size = 16
        config.dropout = 0.0
        config.hidden_size = 768
        config.mlp_ratio = 4
        config.intermediate_size = config.hidden_size * config.mlp_ratio

        # Bert config
        config.bert = "roberta-base"
        config.bert_min_ratio = 0.15
        config.bert_max_ratio = 0.15
        config.use_bert_codebook = True
        config.bert_loss_mask_only = True
        config.bert_mask_loss_weight = 0.0
        config.bert_channel_image_loss_weight = 0.0
        config.nochannel_image_loss_weight = 0.0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        if config.vit_model_type is not None:
            update_vit_config(config.vit_model_type, config)
        assert_avg_rnd(config)

        return config

    @nn.nowrap
    def get_config(self):
        return self.get_default_config(self.config_updates)

    @nn.nowrap
    def rng_keys(self):
        return ("params", "dropout", "drop_path", "quantizer")

    @nn.nowrap
    def no_decay_list(self):
        no_decay = ["bias", "embedding"]
        return no_decay

    def setup(self):
        self.config = self.get_default_config(self.config_updates)
        self.quantizer = VectorQuantizer(config=self.config, dtype=self.dtype)
        if self.config.vit_encoder_decoder:
            self.encoder = VitEncoder(config=self.config, dtype=self.dtype)
            self.decoder = VitDecoder(config=self.config, dtype=self.dtype)
        else:
            self.encoder = ResNetEncoder(config=self.config, dtype=self.dtype)
            self.decoder = ResNetDecoder(config=self.config, dtype=self.dtype)

    def encode(self, image, train):
        encoded_feature, _ = self.encoder(image, train)
        quantized, result_dict = self.quantizer(
            encoded_feature, train, self.make_rng("quantizer")
        )
        return quantized, result_dict

    def forward_image_representation(self, image, train):
        output = {}
        encoded_feature, encoder_embedding = self.encoder(image, train)
        if encoder_embedding is not None:
            encoder_embedding = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (x.shape[0], -1, x.shape[-1])),
                encoder_embedding,
            )
            output["encoder_embedding"] = encoder_embedding
        else:
            encoded_feature = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (x.shape[0], -1, x.shape[-1])), encoded_feature
            )
            output["encoder_embedding"] = [encoded_feature]
        all_embedding = [x for x in output["encoder_embedding"]]
        output["all_embedding"] = all_embedding
        return output

    def decode(self, x: jnp.ndarray, train) -> jnp.ndarray:
        reconstructed = self.decoder(x, train)
        return reconstructed

    def get_codebook_funct(self):
        return self.quantizer.get_codebook()

    def decode_from_indices(self, inputs, train):
        if isinstance(inputs, dict):
            ids = inputs["encoding_indices"]
        else:
            ids = inputs
        features = self.quantizer.decode_ids(ids)
        reconstructed_image = self.decode(features, train)
        return reconstructed_image

    def encode_to_indices(self, inputs, train):
        if isinstance(inputs, dict):
            image = inputs["image"]
        else:
            image = inputs
        encoded_feature, _ = self.encoder(image, train)
        _, result_dict = self.quantizer(
            encoded_feature, train, self.make_rng("quantizer")
        )
        ids = result_dict["encoding_indices"]
        return ids

    def __call__(self, image, train, ratio=None):
        del ratio
        quantized, result_dict = self.encode(image, train)
        image_output = self.decode(quantized, train)
        return {"image_output": image_output}, result_dict
