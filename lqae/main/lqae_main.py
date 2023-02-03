import dataclasses
import pprint
from copy import copy, deepcopy
from functools import partial

import absl.app
import absl.flags
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from absl import logging
from flax.jax_utils import prefetch_to_device
from flax.training.train_state import TrainState
from tqdm.auto import tqdm, trange

import wandb

from ..data import ImageNetDataset, ImageTextDataset
from ..jax_utils import (
    JaxRNG,
    accumulated_gradient,
    get_metrics,
    next_rng,
    sync_state_across_devices,
)
from ..models import LQAE, VQAE
from ..utils import (
    WandBLogger,
    create_log_images,
    define_flags_with_default,
    get_user_flags,
    image_float2int,
    load_pickle,
    set_random_seed,
)

FLAGS_DEF = define_flags_with_default(
    seed=42,
    epochs=200,
    batch_size=0,
    accumulate_grad_steps=1,
    dataloader_n_workers=0,
    dataloader_shuffle=False,
    log_freq=50,
    plot_freq=1000,
    save_model_freq=0,
    clip_gradient=1e9,
    lr_init_value=0.0,
    lr_end_value=0.0,
    lr_peak_value=1.0e-4,
    lr_warmup_epochs=0,
    weight_decay=0.0001,
    load_checkpoint="",
    load_pretrained="",
    dataset="imagenet",
    cc12m_data=ImageTextDataset.get_default_config(),
    imagenet_data=ImageNetDataset.get_default_config(),
    # lqae: encoder-decoder with frozen BERT
    # vqae: encoder-decoder without BERT
    # bert: trainable BERT with frozen encoder-decoder
    model_type="lqae",
    lqae=LQAE.get_default_config(),
    vqae=VQAE.get_default_config(),
    logging=WandBLogger.get_default_config(),
    log_all_worker=False,
)
FLAGS = absl.flags.FLAGS


def main(argv):
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    assert FLAGS.model_type in [
        "lqae",
        "vqae",
        "bert",
    ], "model_type must be one of lqae, vqae, bert"

    variant["jax_process_index"] = jax_process_index = jax.process_index()
    variant["jax_process_count"] = jax_process_count = jax.process_count()
    assert FLAGS.batch_size % jax_process_count == 0
    variant["process_batch_size"] = process_batch_size = (
        FLAGS.batch_size // jax_process_count
    )
    variant["device_batch_size"] = process_batch_size // jax.local_device_count()
    lr_scale = FLAGS.batch_size / 256
    variant["effective_lr"] = FLAGS.lr_peak_value * lr_scale
    jax_devices = jax.local_devices()
    n_devices = len(jax_devices)
    assert process_batch_size % n_devices == 0

    logger = WandBLogger(
        config=FLAGS.logging,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax_process_index == 0),
    )
    set_random_seed(FLAGS.seed * (jax_process_index + 1))

    if FLAGS.dataset == "cc12m":
        FLAGS.cc12m_data.image_only = True
        dataset = ImageTextDataset(
            FLAGS.cc12m_data, jax_process_index / jax_process_count
        )
    elif FLAGS.dataset == "imagenet":
        FLAGS.imagenet_data.image_only = True
        dataset = ImageNetDataset(
            FLAGS.imagenet_data, jax_process_index / jax_process_count
        )
    else:
        raise ValueError("Unsupported dataset!")

    val_flags = deepcopy(FLAGS.imagenet_data)
    val_flags.partition = "val"
    val_flags.transform_type = "test"
    val_dataset = ImageNetDataset(val_flags, jax_process_index / jax_process_count)

    steps_per_epoch = int(len(dataset) / FLAGS.batch_size)
    total_steps = steps_per_epoch * FLAGS.epochs
    val_steps = int(len(val_dataset) / FLAGS.batch_size)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=process_batch_size,
        shuffle=FLAGS.dataloader_shuffle,
        drop_last=True,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=FLAGS.dataloader_n_workers > 0,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=process_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=FLAGS.dataloader_n_workers > 0,
    )

    if FLAGS.model_type == "lqae" or FLAGS.model_type == "bert":
        logging.info(f"Using LQAE model for {FLAGS.model_type}")
        model = LQAE(FLAGS.lqae)
    elif FLAGS.model_type == "vqae":
        logging.info(f"Using LQAE model for {FLAGS.model_type}")
        model = VQAE(FLAGS.vqae)

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=FLAGS.lr_init_value * lr_scale,
        peak_value=FLAGS.lr_peak_value * lr_scale,
        warmup_steps=FLAGS.lr_warmup_epochs
        * steps_per_epoch
        // FLAGS.accumulate_grad_steps,
        decay_steps=total_steps // FLAGS.accumulate_grad_steps,
        end_value=FLAGS.lr_end_value * lr_scale,
    )

    def get_loss(output, result_dict, image, train=True):
        if "bert_loss" in result_dict:
            bert_loss = result_dict["bert_loss"]
        else:
            bert_loss = 0.0

        if FLAGS.model_type == "lqae" or FLAGS.model_type == "bert":
            recon_loss = FLAGS.lqae.bert_channel_image_loss_weight * jnp.mean(
                (image - output["bert_channel_image_output"]) ** 2
            ) + FLAGS.lqae.nochannel_image_loss_weight * jnp.mean(
                (image - output["image_output"]) ** 2
            )
        elif FLAGS.model_type == "vqae":
            recon_loss = jnp.mean((image - output["image_output"]) ** 2)

        if train:
            quantizer_loss = result_dict["quantizer_loss"]
            return quantizer_loss, bert_loss, recon_loss
        else:
            return bert_loss, recon_loss

    @partial(jax.pmap, axis_name="pmap", donate_argnums=0)
    def train_step_fn(state, rng, accumulated_grads, accumulated_steps, image):
        rng_generator = JaxRNG(rng)

        def loss_fn(params):
            output, result_dict = model.apply(
                params,
                image,
                train=True,
                rngs=rng_generator(keys=model.rng_keys()),
            )

            quantizer_loss, bert_loss, recon_loss = get_loss(
                output, result_dict, image, train=True
            )

            total_loss = recon_loss + quantizer_loss + bert_loss

            aux = dict(
                recon_loss=recon_loss,
                quantizer_loss=quantizer_loss,
                bert_loss=bert_loss,
                total_loss=total_loss,
                e_latent_loss=result_dict["e_latent_loss"],
                q_latent_loss=result_dict["q_latent_loss"],
                entropy_loss=result_dict["entropy_loss"],
                perplexity=result_dict["perplexity"],
                codebook_usage=result_dict["codebook_usage"],
            )
            encoding_indices = result_dict["encoding_indices"]

            return total_loss, (aux, encoding_indices)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (aux, encoding_indices)), grads = grad_fn(state.params)
        encoding_indices = jax.lax.all_gather(encoding_indices, axis_name="pmap")
        loss, aux = jax.lax.pmean((loss, aux), axis_name="pmap")
        aux["train_state_step"] = state.step
        aux["learning_rate"] = learning_rate(state.step)

        def global_norm(tree):
            """ Return the global L2 norm of a pytree. """
            squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
            flattened, _ = jax.flatten_util.ravel_pytree(squared)
            return jnp.sqrt(jnp.sum(flattened))

        grad_norm = global_norm(grads)
        aux["grad_norm"] = grad_norm

        if FLAGS.accumulate_grad_steps > 1:
            state, accumulated_grads, accumulated_steps = accumulated_gradient(
                state,
                accumulated_grads,
                accumulated_steps,
                grads,
                FLAGS.accumulate_grad_steps,
                lambda s, g: s.apply_gradients(
                    grads=jax.lax.pmean(g, axis_name="pmap")
                ),
            )
        else:
            state = state.apply_gradients(grads=jax.lax.pmean(grads, axis_name="pmap"))
        return (
            state,
            aux,
            rng_generator(),
            accumulated_grads,
            accumulated_steps,
            encoding_indices,
        )

    @partial(jax.pmap, axis_name="pmap")
    def val_step_fn(state, rng, image):
        rng_generator = JaxRNG(rng)

        output, result_dict = model.apply(
            state.params,
            image,
            train=False,
            ratio={"min_ratio": 0, "max_ratio": 0},
            rngs=rng_generator(keys=model.rng_keys()),
        )

        bert_loss, recon_loss = get_loss(output, result_dict, image, train=False)

        aux = dict(
            recon_loss=recon_loss,
            bert_loss=bert_loss,
            perplexity=result_dict["perplexity"],
            codebook_usage=result_dict["codebook_usage"],
        )

        encoding_indices = result_dict["encoding_indices"]
        encoding_indices = jax.lax.all_gather(encoding_indices, axis_name="pmap")

        aux = jax.lax.pmean(aux, axis_name="pmap")
        return aux, rng_generator(), encoding_indices

    @partial(jax.pmap, axis_name="pmap")
    def reconstruction_fn(state, rng, image):
        rng_generator = JaxRNG(rng)

        output, _ = model.apply(
            state.params,
            image,
            train=False,
            ratio={"min_ratio": 0, "max_ratio": 0},
            rngs=rng_generator(keys=model.rng_keys()),
        )
        if FLAGS.model_type == "lqae" or FLAGS.model_type == "bert":
            image_output = output["image_output"]
            image_output = jnp.clip(image_output, 0, 1)
            bert_channel_image_output = output["bert_channel_image_output"]
            bert_channel_image_output = jnp.clip(bert_channel_image_output, 0, 1)
            return image, image_output, bert_channel_image_output
        elif FLAGS.model_type == "vqae":
            image_output = output["image_output"]
            image_output = jnp.clip(image_output, 0, 1)
            return image, image_output

    def get_weight_decay_mask(params):
        flattened_params = flax.traverse_util.flatten_dict(flax.core.unfreeze(params))

        def decay(key):
            return all([k not in model.no_decay_list() for k in key])

        return flax.traverse_util.unflatten_dict(
            {key: decay(key) for key in flattened_params.keys()}
        )

    def get_no_gradient_update_fn(model_type):
        if model_type == "lqae":

            def func(key):
                return "lang_model" in key

        elif model_type == "bert":

            def func(key):
                return "encoder" in key or "decoder" in key

        elif model_type == "vqae":

            def func(key):
                return False

        return func

    no_gradient_update = get_no_gradient_update_fn(FLAGS.model_type)

    if FLAGS.load_checkpoint != "":
        checkpoint_data = load_pickle(FLAGS.load_checkpoint)
        state = flax.jax_utils.replicate(checkpoint_data["state"], jax_devices)
        start_step = checkpoint_data["step"]
    else:
        image = jnp.zeros((6, 256, 256, 3), dtype=jnp.float32)
        rngs = next_rng(keys=model.rng_keys())
        params = model.init(rngs, image, train=True)

        if FLAGS.model_type == "bert":
            if FLAGS.load_pretrained != "":
                checkpoint_data = load_pickle(FLAGS.load_pretrained)
                checkpoint_params = checkpoint_data["state"].params["params"]
                checkpoint_params = flax.core.unfreeze(checkpoint_params)
                params = flax.core.unfreeze(params["params"])
                for key in params.keys():
                    if key not in checkpoint_params.keys():
                        if key in ["lang_model"]:
                            continue
                        else:
                            raise ValueError(f"pretrained model miss key={key}")
                params = flax.core.freeze({"params": params})
            params = model.load_bert_params(params)
        elif FLAGS.model_type == "lqae":
            if FLAGS.lqae.use_bert_codebook:
                params = model.load_bert_params(params)

        transform_fn = {
            True: optax.set_to_zero(),
            False: optax.chain(
                optax.clip_by_global_norm(FLAGS.clip_gradient),
                optax.adamw(
                    learning_rate=learning_rate,
                    b1=0.9,
                    b2=0.95,
                    weight_decay=FLAGS.weight_decay,
                ),
            ),
        }

        def label_fn(params):
            flattened_params = flax.traverse_util.flatten_dict(params)
            return flax.traverse_util.unflatten_dict(
                {
                    key: no_gradient_update(key)
                    for key, value in flattened_params.items()
                }
            )

        def count_params(params):
            flattened_params = flax.traverse_util.flatten_dict(params)
            tree = flax.traverse_util.unflatten_dict(
                {
                    key: value
                    for key, value in flattened_params.items()
                    if no_gradient_update(key)
                }
            )
            num_params = sum(p.size for p in jax.tree_leaves(tree))
            return num_params

        num_params = count_params(params)
        logger.log({"num_learnable_params": num_params})

        opt = optax.multi_transform(transform_fn, label_fn)
        state = flax.jax_utils.replicate(
            TrainState.create(
                params=flax.core.frozen_dict.unfreeze(params),
                apply_fn=None,
                tx=opt,
            ),
            jax_devices,
        )
        start_step = 0

        del params

    def generate_batch(iterator):
        while True:
            for images in iterator:
                yield images.numpy().reshape(n_devices, -1, *images.shape[1:])

    state = sync_state_across_devices(state)
    sharded_rng = jax.device_put_sharded(next_rng(n_devices), jax_devices)

    if FLAGS.accumulate_grad_steps > 1:
        accumulated_grads = flax.jax_utils.replicate(
            jax.tree_map(jnp.zeros_like, flax.jax_utils.unreplicate(state).params),
            jax_devices,
        )
        accumulated_steps = flax.jax_utils.replicate(
            jnp.array(0, jnp.int32), jax_devices
        )
    else:
        accumulated_grads = flax.jax_utils.replicate(
            jnp.array(0, jnp.int32), jax_devices
        )
        accumulated_steps = flax.jax_utils.replicate(
            jnp.array(0, jnp.int32), jax_devices
        )

    data_iterator = prefetch_to_device(generate_batch(dataloader), 2, jax_devices)
    val_data_iterator = prefetch_to_device(
        generate_batch(val_dataloader), 2, jax_devices
    )
    step_counter = trange(start_step, total_steps, ncols=0, desc="Train")
    if FLAGS.model_type == "lqae" or FLAGS.model_type == "bert":
        codebook_size = 50265
    elif FLAGS.model_type == "vqae":
        codebook_size = FLAGS.vqae.codebook_size

    step = 0
    for step, image in zip(step_counter, data_iterator):
        epoch = int(step * jax_process_count / len(dataloader))
        image = image.astype(jnp.float32)
        (
            state,
            metrics,
            sharded_rng,
            accumulated_grads,
            accumulated_steps,
            encoding_indices,
        ) = train_step_fn(
            state, sharded_rng, accumulated_grads, accumulated_steps, image
        )
        if step % FLAGS.log_freq == 0:
            log_metrics = {"step": step, "epoch": epoch}
            log_metrics.update(get_metrics(metrics, unreplicate=True))
            encoding_indices = jax.device_get(
                flax.jax_utils.unreplicate(encoding_indices)
            )
            indices_histogram = jnp.histogram(
                encoding_indices, bins=512, range=(0, codebook_size - 1)
            )
            log_metrics.update(
                {
                    "indices_histogram": wandb.Histogram(
                        np_histogram=indices_histogram
                    ),
                    "encoding_indices": wandb.Histogram(encoding_indices),
                }
            )
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

        if FLAGS.plot_freq > 0 and step % FLAGS.plot_freq == 0:
            log_image = create_log_images(
                jax.device_get(reconstruction_fn(state, sharded_rng, image)),
                mean=dataset.image_mean,
                std=dataset.image_std,
            )
            if jax_process_index == 0:
                logger.log({"image_prediction": wandb.Image(log_image)})

        if step % steps_per_epoch == 0:
            val_metrics = []
            val_encoding_indices = []
            for _, val_image in zip(
                trange(val_steps, ncols=0, desc="val"), val_data_iterator
            ):
                val_image = val_image.astype(jnp.float32)
                metrics, sharded_rng, encoding_indices = val_step_fn(
                    state, sharded_rng, val_image
                )
                val_metrics.append(metrics)
                val_encoding_indices.append(encoding_indices)
            log_metrics = get_metrics(val_metrics, unreplicate=True, stack=True)
            val_encoding_indices = jax.tree_map(
                lambda x: jax.device_get(flax.jax_utils.unreplicate(x)),
                val_encoding_indices,
            )
            val_encoding_indices = jnp.concatenate(val_encoding_indices, axis=0)
            log_metrics = {
                f"val_{k}": v
                for k, v in jax.tree_map(lambda x: x.mean(), log_metrics).items()
            }
            val_indices_histogram = jnp.histogram(
                val_encoding_indices, bins=512, range=(0, codebook_size - 1)
            )
            log_metrics.update(
                {
                    "val_indices_histogram": wandb.Histogram(
                        np_histogram=val_indices_histogram
                    ),
                    "val_encoding_indices": wandb.Histogram(val_encoding_indices),
                }
            )
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

        if FLAGS.save_model_freq > 0 and step % FLAGS.save_model_freq == 0:
            save_data = {
                "step": step,
                "epoch": epoch,
                "variant": variant,
                "state": jax.device_get(flax.jax_utils.unreplicate(state)),
            }
            if jax_process_index == 0:
                logger.save_pickle(save_data, "model.pkl")

    if FLAGS.save_model_freq > 0:
        save_data = {
            "step": step,
            "epoch": epoch,
            "variant": variant,
            "state": jax.device_get(flax.jax_utils.unreplicate(state)),
        }
        if jax_process_index == 0:
            logger.save_pickle(save_data, "model.pkl")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    absl.app.run(main)
