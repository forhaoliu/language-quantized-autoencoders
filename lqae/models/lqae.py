import functools
import io
import math
import os
from typing import Any, Callable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import requests
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from PIL import Image, ImageFilter
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_flax_roberta import (
    FlaxBaseModelOutputWithPoolingAndCrossAttentions,
    FlaxMaskedLMOutput,
    FlaxRobertaEncoder,
    FlaxRobertaForMaskedLM,
    FlaxRobertaLMHead,
    FlaxRobertaPooler,
    RobertaConfig,
    create_position_ids_from_input_ids,
)

from ..jax_utils import JaxRNG, get_onehot, next_rng
from .base_resnet import ResNetDecoder, ResNetEncoder
from .base_vit import VitDecoder, VitEncoder
from .model_utils import (
    assert_avg_rnd,
    update_vit_config,
    entropy_loss_fn,
    normalize_func,
    squared_euclidean_distance,
)


class LanguageQuantizer(nn.Module):
    """Language quantizer."""

    config: ConfigDict
    codebook: jnp.array
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
        codebook_size = self.codebook.shape[0]
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
                squared_euclidean_distance(latent_input, sg_strawman_codebook,
                dot_product=self.config.dot_product),
                x.shape[:-1] + (codebook_size,),
            )
        else:
            codebook = jnp.asarray(self.codebook, dtype=self.dtype)
            latent_input = self.input_to_latent(jnp.reshape(x, (-1, x.shape[-1])))
            latent_input = l2_normalize(latent_input, axis=1)
            latent_codebook = self.code_to_latent(codebook)
            latent_codebook = l2_normalize(latent_codebook, axis=1)
            sg_latent_codebook = jax.lax.stop_gradient(
                l2_normalize(latent_codebook, axis=1)
            )
            distances = jnp.reshape(
                squared_euclidean_distance(latent_input, sg_latent_codebook,
                dot_product=self.config.dot_product),
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
            # disable gradient for LQAE
            quantized = jax.lax.stop_gradient(quantized)
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
        return jnp.dot(z, self.codebook)

    def get_codebook(self) -> jnp.ndarray:
        return self.codebook

    def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(self.codebook, ids, axis=0)

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


# Removed word embeddings. Copied from
# https://github.com/huggingface/transformers/blob/12ce2941c7b67c0dedac0f0468b3ed854fa940ab/src/transformers/models/roberta/modeling_flax_roberta.py#L139-L176
class Add_Pos(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
        )
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
        )
        self.LayerNorm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(
        self,
        inputs_embeds,
        token_type_ids,
        position_ids,
        attention_mask,
        deterministic: bool = True,
    ):
        # Embed
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # Sum all embeddings
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


@flax.struct.dataclass
class NewFlaxMaskedLMOutput(FlaxMaskedLMOutput):
    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    last_hidden_states: Optional[Tuple[jnp.ndarray]] = None


# Removed word embeddings. Copied from
# https://github.com/huggingface/transformers/blob/12ce2941c7b67c0dedac0f0468b3ed854fa940ab/src/transformers/models/roberta/modeling_flax_roberta.py#L914-L982
# and
# https://github.com/huggingface/transformers/blob/12ce2941c7b67c0dedac0f0468b3ed854fa940ab/src/transformers/models/roberta/modeling_flax_roberta.py#L998-L1053
class Language_Model(nn.Module):
    config: RobertaConfig
    bert: str = "roberta-base"
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = False
    gradient_checkpointing: bool = False

    def setup(self):
        self.embeddings = Add_Pos(self.config)
        self.encoder = FlaxRobertaEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.pooler = FlaxRobertaPooler(self.config, dtype=self.dtype)
        self.lm_head = FlaxRobertaLMHead(config=self.config, dtype=self.dtype)

        pretrained_model = FlaxRobertaForMaskedLM.from_pretrained(self.bert)
        self.word_embeddings = pretrained_model.params["roberta"]["embeddings"][
            "word_embeddings"
        ]["embedding"]

    @nn.nowrap
    def rng_keys(self):
        return ("params", "dropout")

    def __call__(
        self,
        hidden_states,
        input_ids,
        attention_mask,
        token_type_ids: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):

        # make sure `token_type_ids` is correctly initialized when not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # make sure `position_ids` is correctly initialized when not passed
        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(
                input_ids, self.config.pad_token_id
            )

        hidden_states = self.embeddings(
            hidden_states,
            token_type_ids,
            position_ids,
            attention_mask,
            deterministic=deterministic,
        )

        outputs = self.encoder(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                outputs = (hidden_states,) + outputs[1:]
            else:
                outputs = (hidden_states, pooled) + outputs[1:]
        else:
            outputs = FlaxBaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=hidden_states,
                pooler_output=pooled,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions,
            )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.word_embeddings
        else:
            shared_embedding = None

        # Compute the prediction scores
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return NewFlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            last_hidden_states=hidden_states,
        )


class LQAE(nn.Module):
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
        config.strawman_codebook_init = "normal:1.0"
        config.quantizer_loss_perplexity = 0.0
        config.dot_product = False

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
        config.use_bert_ste = True

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
    def get_bert_config(self):
        config = self.get_default_config(self.config_updates)
        return RobertaConfig.from_pretrained(config.bert)

    @nn.nowrap
    def get_num_bert_layers(self):
        return self.get_bert_config().num_hidden_layers + 1

    @nn.nowrap
    def get_num_encoder_layers(self):
        return self.get_config().enc_num_layers

    @nn.nowrap
    def get_hidden_size(self):
        return self.get_config().hidden_size

    @nn.nowrap
    def rng_keys(self):
        return ("params", "dropout", "drop_path", "shuffle", "mask", "quantizer")

    @nn.nowrap
    def no_decay_list(self):
        no_decay = ["bias", "embedding"]
        return no_decay

    def setup(self):
        self.config = self.get_default_config(self.config_updates)
        (
            self.lang_model,
            self.codebook,
            self.mask_code,
            self.tokenizer,
        ) = self.config_language_model()
        self.quantizer = LanguageQuantizer(
            config=self.config,
            codebook=self.codebook,
            dtype=self.dtype,
        )
        if self.config.vit_encoder_decoder:
            self.encoder = VitEncoder(config=self.config, dtype=self.dtype)
            self.decoder = VitDecoder(config=self.config, dtype=self.dtype)
        else:
            self.encoder = ResNetEncoder(config=self.config, dtype=self.dtype)
            self.decoder = ResNetDecoder(config=self.config, dtype=self.dtype)

    def config_language_model(self):
        pretrained_bert = FlaxRobertaForMaskedLM.from_pretrained(self.config.bert)
        language_model = Language_Model(pretrained_bert.config, self.config.bert)
        codebook = pretrained_bert.params["roberta"]["embeddings"]["word_embeddings"][
            "embedding"
        ]

        tokenizer = RobertaTokenizer.from_pretrained(self.config.bert)
        mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        mask_code = codebook[mask_token_id]

        return language_model, codebook, mask_code, tokenizer

    @nn.nowrap
    def load_bert_params(self, params, pretrain=True):
        config = self.get_default_config(self.config_updates)
        pretrained_bert = FlaxRobertaForMaskedLM.from_pretrained(config.bert)
        pretrained_params = flax.core.unfreeze(pretrained_bert.params)

        if pretrain:
            lang_params = flax.core.unfreeze(params)["params"]["lang_model"]
        else:
            lang_params = flax.core.unfreeze(params)["params"]['backbone']["lang_model"]

        for key in lang_params.keys():
            if key == "embeddings":
                for k in lang_params["embeddings"].keys():
                    assert (
                        k in pretrained_params["roberta"]["embeddings"].keys()
                    ), f"pretrained model miss key={key}"
                    lang_params["embeddings"][k] = pretrained_params["roberta"][
                        "embeddings"
                    ][k]
            elif key == "lm_head":
                assert (
                    key in pretrained_params.keys()
                ), f"pretrained model miss key={key}"
                lang_params[key] = pretrained_params[key]
            else:
                assert (
                    key in pretrained_params["roberta"].keys()
                ), f"pretrained model miss key={key}"
                lang_params[key] = pretrained_params["roberta"][key]

        params = flax.core.unfreeze(params)
        if pretrain:
            params["params"].update({"lang_model": lang_params})
        else:
            params["params"]["backbone"].update({"lang_model": lang_params})
        params = flax.core.freeze(params)
        return params

    def languge_model_encode_decode(
        self,
        input_code,
        input_ids,
        ratio={},
        output_hidden_states=False,
    ):
        input_shape = input_code.shape
        if len(input_code.shape) == 4:
            input_code = jnp.reshape(
                input_code, (input_code.shape[0], -1, input_code.shape[-1])
            )
            input_ids = jnp.reshape(input_ids, (input_ids.shape[0], -1))

        min_ratio = ratio.get("min_ratio", self.config.bert_min_ratio)
        max_ratio = ratio.get("max_ratio", self.config.bert_max_ratio)
        assert min_ratio <= max_ratio, "min_ratio must be less than max_ratio"
        use_mask = random_ratio_mask(
            jnp.zeros((input_code.shape[0], input_code.shape[1])),
            min_ratio,
            max_ratio,
            self.make_rng("mask"),
        ).astype(bool)
        input_code = jnp.where(
            use_mask[..., None], self.mask_code[None, None, ...], input_code
        )

        attention_mask = jnp.ones(
            (input_code.shape[0], input_code.shape[1]), dtype=jnp.uint8
        )
        bert_output = self.lang_model(
            input_code,
            input_ids,
            attention_mask,
            output_hidden_states=output_hidden_states,
            deterministic=True,
        )
        if self.config.use_bert_ste:
            logits = bert_output.logits
            decoding_indices = jnp.argmax(logits, axis=-1)
            codebook_size = self.codebook.shape[0]
            encodings = jax.nn.one_hot(decoding_indices, codebook_size, dtype=self.dtype)
            argmax_code = jnp.dot(encodings, self.codebook)
            softmax_code = jnp.dot(jax.nn.softmax(logits, axis=-1), self.codebook)
            output = softmax_code + jax.lax.stop_gradient(argmax_code - softmax_code)
            output = jnp.reshape(output, input_shape)
        else:
            output = bert_output.last_hidden_states
            output = jnp.reshape(output, input_shape)

        logits = bert_output.logits
        bert_loss = optax.softmax_cross_entropy(
            logits, get_onehot(input_ids, logits.shape[-1])
        )
        if self.config.bert_loss_mask_only:
            bert_loss = bert_loss * use_mask
            bert_loss = jnp.sum(bert_loss, axis=1) / jnp.sum(use_mask, axis=1)

        bert_loss = jnp.mean(bert_loss) * self.config.bert_mask_loss_weight
        language_model_output = {
            "bert_logits": bert_output.logits,
            "bert_hidden_states": bert_output.hidden_states,
            "bert_loss": bert_loss,
        }
        return output, language_model_output

    def encode(self, image, train: bool):
        encoded_feature, _ = self.encoder(image, train)
        quantized, result_dict = self.quantizer(
            encoded_feature, train, rng=self.make_rng("quantizer")
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
        quantized, result_dict = self.quantizer(
            encoded_feature, train, rng=self.make_rng("quantizer")
        )
        _, language_model_output = self.languge_model_encode_decode(
            quantized,
            result_dict["encoding_indices"],
            ratio={"min_ratio": 0, "max_ratio": 0},
            output_hidden_states=True,
        )
        bert_embedding = language_model_output["bert_hidden_states"]
        output["bert_embedding"] = bert_embedding
        return output

    def decode(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
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
            encoded_feature, train, rng=self.make_rng("quantizer")
        )
        ids = result_dict["encoding_indices"]
        return ids

    def __call__(self, image, train, ratio={}):
        quantized, result_dict = self.encode(image, train)
        bert_quantized, language_model_output = self.languge_model_encode_decode(
            quantized, result_dict["encoding_indices"], ratio
        )
        result_dict = {**result_dict, **language_model_output}
        image_output = self.decoder(quantized, train)
        bert_channel_image_output = self.decoder(bert_quantized, train)
        output = {
            "image_output": image_output,
            "bert_channel_image_output": bert_channel_image_output,
        }
        return output, result_dict


def random_mask(x, ratio, rng):
    return (jax.random.uniform(rng, shape=x.shape[:2]) < ratio).astype(jnp.float32)


def random_ratio_mask(x, min_ratio, max_ratio, rng):
    rng_generator = JaxRNG(rng)
    ratio = jax.random.uniform(
        rng_generator(), shape=x.shape[:2], minval=min_ratio, maxval=max_ratio
    )
    return random_mask(x, ratio, rng_generator())
