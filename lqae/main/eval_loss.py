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
    # eval loss ratio
    min_ratio=0.15,
    max_ratio=0.15,
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
    val_steps = int(len(val_dataset) / FLAGS.batch_size)

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

    @partial(jax.pmap, axis_name="pmap")
    def val_step_fn(state, rng, image):
        rng_generator = JaxRNG(rng)

        output, result_dict = model.apply(
            state.params,
            image,
            train=False,
            ratio={"min_ratio": FLAGS.min_ratio, "max_ratio": FLAGS.max_ratio},
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

    if FLAGS.load_checkpoint != "":
        checkpoint_data = load_pickle(FLAGS.load_checkpoint)
        state = flax.jax_utils.replicate(checkpoint_data["state"], jax_devices)
    else:
        image = jnp.zeros((6, 256, 256, 3), dtype=jnp.float32)
        rngs = next_rng(keys=model.rng_keys())
        params = model.init(rngs, image, train=True)

        assert FLAGS.model_type == "lqae"
        if FLAGS.lqae.use_bert_codebook:
            params = model.load_bert_params(params)

        state = flax.jax_utils.replicate(
            TrainState.create(
                params=flax.core.frozen_dict.unfreeze(params),
                apply_fn=None,
                tx=optax.lars(
                    learning_rate=0,
                    weight_decay=0,
                    momentum=0,
                ),
            ),
            jax_devices,
        )

        del params

    def generate_batch(iterator):
        while True:
            for images in iterator:
                yield images.numpy().reshape(n_devices, -1, *images.shape[1:])

    state = sync_state_across_devices(state)
    sharded_rng = jax.device_put_sharded(next_rng(n_devices), jax_devices)

    val_data_iterator = prefetch_to_device(
        generate_batch(val_dataloader), 2, jax_devices
    )
    if FLAGS.model_type == "lqae" or FLAGS.model_type == "bert":
        codebook_size = 50265
    elif FLAGS.model_type == "vqae":
        codebook_size = FLAGS.vqae.codebook_size

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


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    absl.app.run(main)
