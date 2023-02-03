import dataclasses
import os
import pprint
from copy import copy, deepcopy
from functools import partial
from typing import Any, Callable, Optional

import absl.app
import absl.flags
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training import train_state
from tqdm.auto import tqdm, trange

from ..data import ImageNetDataset
from ..jax_utils import (
    JaxRNG,
    cross_entropy_loss,
    get_metrics,
    next_rng,
    sync_state_across_devices,
    mixup_cutmix,
)
from ..models import LQAE
from ..utils import (
    WandBLogger,
    define_flags_with_default,
    get_user_flags,
    load_checkpoint,
    load_pickle,
    set_random_seed,
)


FLAGS_DEF = define_flags_with_default(
    seed=42,
    epochs=200,
    batch_size=0,
    dataloader_n_workers=0,
    dataloader_shuffle=False,
    log_freq=50,
    save_model_freq=0,
    lr_init_value=0.0,
    lr_end_value=0.0,
    lr_peak_value=1.0e-4,
    lr_warmup_epochs=0,
    momentum=0.9,
    weight_decay=0.0001,
    clip_gradient=1e9,
    load_pretrained="",
    load_checkpoint="",
    last_embedding_layers="all",
    imagenet_data=ImageNetDataset.get_default_config(),
    lqae=LQAE.get_default_config(),
    logging=WandBLogger.get_default_config(),
    log_all_worker=False,
    load_bert_params=True,
)
FLAGS = absl.flags.FLAGS


def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

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

    train_dataset = ImageNetDataset(
        FLAGS.imagenet_data, jax_process_index / jax_process_count
    )
    val_flags = deepcopy(FLAGS.imagenet_data)
    val_flags.partition = "val"
    val_flags.transform_type = "test"
    val_dataset = ImageNetDataset(val_flags, jax_process_index / jax_process_count)

    steps_per_epoch = int(len(train_dataset) / FLAGS.batch_size)
    total_steps = steps_per_epoch * FLAGS.epochs
    val_steps = int(len(val_dataset) / FLAGS.batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=process_batch_size,
        shuffle=FLAGS.dataloader_shuffle,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=FLAGS.dataloader_n_workers > 0,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=process_batch_size,
        shuffle=False,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=FLAGS.dataloader_n_workers > 0,
        drop_last=True,
    )

    def combine_representation(output):
        encoder_embedding = [x for x in output["encoder_embedding"]]
        # Stop gradient on encoder_embedding in Linear
        encoder_embedding = jax.lax.stop_gradient(encoder_embedding)
        bert_embedding = [x for x in output["bert_embedding"]]
        # Stop gradient on bert_embedding in Linear
        bert_embedding = jax.lax.stop_gradient(bert_embedding)
        all_embedding = encoder_embedding + bert_embedding
        if FLAGS.last_embedding_layers == "all":
            representation = all_embedding
        elif FLAGS.last_embedding_layers == "all_bert":
            representation = bert_embedding
        elif FLAGS.last_embedding_layers == "all_encoder":
            representation = encoder_embedding
        else:
            representation = [
                all_embedding[-int(x)] for x in FLAGS.last_embedding_layers.split(",")
            ]
        representation = jax.tree_util.tree_map(
            lambda x: jnp.mean(x[:, 1:, :], axis=1), representation
        )
        representation = jnp.concatenate(representation, axis=-1)
        return representation

    class FinetuneCLS(nn.Module):
        backbone: nn.Module
        num_classes: int

        @nn.nowrap
        def rng_keys(self):
            return ('params', 'noise', 'drop_path')

        @nn.compact
        def __call__(self, x, deterministic=False):
            output = self.backbone.forward_image_representation(x, deterministic)
            x = combine_representation(output)
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.num_classes)(x)
            logits = x
            return logits

    backbone = LQAE(FLAGS.lqae)
    model = FinetuneCLS(
        backbone=backbone,
        num_classes=train_dataset.num_classes(),
    )

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=FLAGS.lr_init_value * lr_scale,
        peak_value=FLAGS.lr_peak_value * lr_scale,
        warmup_steps=FLAGS.lr_warmup_epochs * steps_per_epoch,
        decay_steps=total_steps,
        end_value=FLAGS.lr_end_value * lr_scale,
    )

    @partial(jax.pmap, axis_name="pmap", donate_argnums=[0])
    def train_step_fn(state, rng, image, label):
        rng_generator = JaxRNG(rng)
        def loss_fn(params):
            logits = model.apply(
                params,
                image,
                deterministic=False,
                rngs=rng_generator(keys=backbone.rng_keys()),
            )
            loss = cross_entropy_loss(logits, label)
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(label, axis=-1))
            aux = dict(loss=loss, accuracy=accuracy)
            return loss, aux
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = jax.lax.pmean(
            grad_fn(state.params),
            axis_name="pmap",
        )
        aux["learning_rate"] = learning_rate(state.step)
        state = state.apply_gradients(grads=grads)
        return state, aux, rng_generator()

    @partial(jax.pmap, axis_name="pmap")
    def eval_step_fn(state, rng, image, label):
        rng_generator = JaxRNG(rng)
        logits = model.apply(
            state.params,
            image,
            deterministic=True,
            rngs=rng_generator(keys=backbone.rng_keys()),
        )
        accuracy = jax.lax.pmean(
            jnp.mean(jnp.argmax(logits, axis=-1) == label), axis_name="pmap"
        )
        aux = dict(accuracy=accuracy)
        aux = jax.lax.pmean(aux, axis_name="pmap")
        return aux, rng_generator()

    if FLAGS.load_checkpoint != "":
        checkpoint_data = load_pickle(FLAGS.load_checkpoint)
        state = flax.jax_utils.replicate(checkpoint_data, jax_devices)
        start_step = checkpoint_data["step"]
    else:
        image = jnp.zeros((2, 256, 256, 3), dtype=jnp.float32)
        rngs = next_rng(keys=backbone.rng_keys())
        params = model.init(rngs, image)

        if FLAGS.load_pretrained != "":
            checkpoint_data = load_checkpoint(FLAGS.load_pretrained)
            checkpoint_params = checkpoint_data["state"].params["params"]
            checkpoint_params = flax.core.unfreeze(checkpoint_params)
            backbone_params = flax.core.unfreeze(params['params']["backbone"])
            for key in backbone_params.keys():
                if key in ["decoder", "lang_model"]:
                    continue
                else:
                    assert (
                        key in checkpoint_params.keys()
                    ), f"pretrained model miss key={key}"
                    backbone_params[key] = checkpoint_params[key]
            params = flax.core.unfreeze(params)
            params["params"].update({"backbone": backbone_params})
            params = flax.core.freeze(params)

        # make sure BERT pretrained code is loaded
        if FLAGS.load_bert_params:
            params = backbone.load_bert_params(params, False)

        state = train_state.TrainState.create(
            params=flax.core.unfreeze(params),
            apply_fn=None,
            tx=optax.lars(
                learning_rate=learning_rate,
                weight_decay=FLAGS.weight_decay,
                momentum=FLAGS.momentum,
            ),
        )
        state = flax.jax_utils.replicate(state, jax_devices)
        start_step = 0

        del params

    state = sync_state_across_devices(state)
    sharded_rng = jax.device_put_sharded(next_rng(n_devices), jax_devices)

    def generate_batch(iterator):
        while True:
            for batch in iterator:
                imgs = batch[0].numpy()
                imgs = imgs.reshape(n_devices, -1, *imgs.shape[1:])
                labels = batch[1].numpy()
                labels = labels.reshape(n_devices, -1, *labels.shape[1:])
                yield tuple([imgs, labels])

    train_iterator = prefetch_to_device(generate_batch(train_loader), 2, jax_devices)
    val_iterator = prefetch_to_device(generate_batch(val_loader), 2, jax_devices)

    best_val_acc = 0.0
    step_counter = trange(start_step, total_steps, desc="train", ncols=0)

    for step, (image, label) in zip(step_counter, train_iterator):
        epoch = step // steps_per_epoch
        if step % steps_per_epoch == 0:
            train_metrics = []

        image = image.astype(jnp.float32)
        label = label.astype(jnp.int32)

        state, metrics, sharded_rng = train_step_fn(state, sharded_rng, image, label)
        train_metrics.append(metrics)

        if step % FLAGS.log_freq == 0:
            log_metrics = get_metrics(train_metrics, unreplicate=True, stack=True)
            log_metrics = {
                f"train_{k}": v
                for k, v in jax.tree_map(lambda x: x.mean(), log_metrics).items()
            }
            log_metrics.update({"step": step, "epoch": epoch})
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

        if FLAGS.save_model_freq > 0 and step % FLAGS.save_model_freq == 0:
            save_data = {
                "step": step,
                "epoch": epoch,
                "variant": variant,
                "state": jax.device_get(flax.jax_utils.unreplicate(state)),
                "best_val_acc": best_val_acc,
            }
            if jax_process_index == 0:
                logger.save_pickle(save_data, "model.pkl")

        if step % steps_per_epoch == 0:
            val_metrics = []
            for _, (image, label) in zip(
                trange(val_steps, desc="val", ncols=0), val_iterator
            ):
                image = image.astype(jnp.float32)
                label = label.astype(jnp.int32)

                metrics, sharded_rng = eval_step_fn(state, sharded_rng, image, label)
                val_metrics.append(metrics)

            log_metrics = get_metrics(val_metrics, unreplicate=True, stack=True)
            accuracy = log_metrics["accuracy"].mean()
            log_metrics = {
                f"val_{k}": v
                for k, v in jax.tree_map(lambda x: x.mean(), log_metrics).items()
            }
            log_metrics.update({"step": step, "epoch": epoch})
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if accuracy > best_val_acc:
                best_val_acc = accuracy

                if FLAGS.save_model_freq > 0:
                    save_data = {
                        "epoch": epoch,
                        "step": step,
                        "variant": variant,
                        "state": jax.device_get(flax.jax_utils.unreplicate(state)),
                        "best_val_acc": best_val_acc,
                    }
                    if jax_process_index == 0:
                        logger.save_pickle(save_data, "best_model.pkl")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    absl.app.run(main)
