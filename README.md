# Language Quantized AutoEncoders

This is a Jax implementation of our work [Language Quantized AutoEncoders](https://arxiv.org/abs/2302.00902).

It contains training and evalutation code.

This implementation has been tested on multi-GPU and Google Cloud TPU and supports both multi-host training with TPU Pods and multi-GPU training.

## Usage
Experiments can be launched via the following commands.

An example script of launching a LQAE training job is:
```
export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

echo $PYTHONPATH
export WANDB_API_KEY=''

export experiment_name='lqae'
export project_id='lqae'
export wu='5'
export ep='100'
export model='lqae'
export experiment_note=""
export experiment_id="lqae-base"

python3 -m lqae.main.lqae_main \
    --model_type="$model" \
    --lqae.bert_min_ratio=0.5 \
    --lqae.bert_max_ratio=0.5 \
    --lqae.quantizer_loss_commitment=0.005 \
    --lqae.quantizer_loss_entropy=0.0 \
    --lqae.quantizer_loss_perplexity=0.0 \
    --lqae.l2_normalize=True \
    --lqae.top_k_value=1 \
    --lqae.top_k_avg=False \
    --lqae.top_k_rnd=False \
    --lqae.vit_encoder_decoder=True \
    --lqae.vit_model_type='base' \
    --lqae.patch_size=16 \
    --lqae.use_bert_codebook=True \
    --lqae.bert_mask_loss_weight=0.0001 \
    --lqae.bert_channel_image_loss_weight=1.0 \
    --lqae.nochannel_image_loss_weight=0.0 \
    --lqae.quantizer_latent_dim=0 \
    --lqae.strawman_codebook=False \
    --lqae.use_bert_ste=False \
    --seed=42 \
    --epochs="$ep" \
    --lr_warmup_epochs="$wu" \
    --batch_size=512 \
    --dataloader_n_workers=16 \
    --log_freq=500 \
    --plot_freq=2000 \
    --save_model_freq=10000 \
    --lr_peak_value=1.5e-4 \
    --weight_decay=0.0005 \
    --load_checkpoint='' \
    --dataset='imagenet' \
    --imagenet_data.path="YOUR IMAGENET FILE in HDF5" \
    --imagenet_data.random_start=True \
    --log_all_worker=False \
    --logging.online=True \
    --logging.project_id="$project_id" \
    --logging.experiment_id="$experiment_id" \
    --logging.experiment_note="$experiment_note" \
    --logging.output_dir="$HOME/experiment_output/$project_id"

```

Example of running LLM based evaluation using LQAE pretrained model is at this [colab](https://colab.research.google.com/drive/1_nzC8W6yO9fYP8GLfUmY11hoVQUW9e6Q?usp=sharing).

To run experiments more conveniently on TPUs, you may want to use the script in jobs folder to manage TPUs jobs.
