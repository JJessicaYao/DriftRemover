"""
Adapted fine-tuning script for Stable Diffusion 2 inpainting with LoRA.

Training code adapted from Diffusers (http://www.apache.org/licenses/LICENSE-2.0)
 * https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py
 * https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

Masking code adapted from github.com/justinpinkney/stable-diffusion (MIT)
 * https://github.com/justinpinkney/stable-diffusion/blob/main/ldm/data/inpainting/synthetic_mask.py

Example Usage (9GB VRAM, tested 3090 Ti):
$ accelerate launch train_text_to_image_lora_sd2_inpaint.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" \
  --dataset_name="sshh12/sentinel-2-rgb-captioned" \
  --caption_column="text" \
  --mask_mode="512train-very-large" \
  --mixed_precision="fp16" \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=600 \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --seed=0 \
  --validation_epochs=1 \
  --validation_file="validation.jsonl"\
  --output_dir="output" \
  --enable_xformers_memory_efficient_attention \
  --report_to="wandb"

Example `validation.jsonl`
    {"file_name": "img1.png", "mask_file_name": "vertical-bar-mask.png", "text": "a satellite image"}
    {"file_name": "img1.png", "mask_file_name": "vertical-bar-mask.png", "text": "a satellite image of a beach"}
    ...

Usage:
    pipe = StableDiffusionInpaintPipeline(...)
    pipe.unet.load_attn_procs('.../pytorch_model.bin', use_safetensors=False)
"""

import argparse
import logging
import math
import os
import random
import shutil
import json
from pathlib import Path
from PIL import Image, ImageDraw

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset
from torchvision.transforms import functional

import diffuser
from diffuser import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
import safetensors
from diffuser.loaders import AttnProcsLayers
from diffuser.models.attention_processor import LoRAAttnProcessor
from diffuser.optimization import get_scheduler
from diffuser.utils import is_wandb_available
from diffuser.utils.import_utils import is_xformers_available

import warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__, log_level="INFO")

class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A file for images to test for validation.",
    )
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="512train-large",
        help="Mask mode to use for training.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--random_hflip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--random_vflip",
        action="store_true",
        help="whether to randomly flip images vertically",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0, help="The scale of noise offset."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    parser.add_argument(
        "--type",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


DATASET_NAME_MAPPING = {}

def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "safetensors"})
    else:
        torch.save(learned_embeds_dict, save_path)


def main(type,placeholder):
    args = parse_args()
    args.type = type
    args.placeholder_token = type + '_' + placeholder
    args.initializer_token = placeholder.split('_')[0]

    logging_dir = Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffuser.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffuser.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        # torch_dtype=torch.float32,
        safety_checker = None,
        requires_safety_checker = False
    )

    # Load scheduler, tokenizer and models.
    noise_scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path+'/unet/'
    )
    pipe.unet=unet
    
    # Add the placeholder token in tokenizer
    placeholder_tokens = [args.placeholder_token]

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    # if len(token_ids) > 1:
    #     raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    # text_encoder.requires_grad_(True)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
        # optimizer_cls = torch.optim.SGD

    optimizer = optimizer_cls(
        # text_encoder.parameters(),
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # optimizer = optimizer_cls(
    #     text_encoder.parameters(),
    #     lr=args.learning_rate,
    #     weight_decay=0.1,
    # )
    

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir+args.type,
        )
    else:
        data_files = {}
        if args.train_data_dir+args.type is not None:
            data_files["train"] = os.path.join(args.train_data_dir,args.type, "**")
        dataset = load_dataset(
            # "imagefolder",
            args.train_data_dir+args.type+'/script/{}_train_script_{}.py'.format(args.type,placeholder),
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    
    args.fault_column = 'fault'
    if args.fault_column is None:
        fault_column = (
            dataset_columns[1] if dataset_columns is not None else column_names[1]
        )
    else:
        fault_column = args.fault_column
        if fault_column not in column_names:
            raise ValueError(
                f"--fault_column' value '{args.fault_column}' needs to be one of: {', '.join(column_names)}"
            )

    args.mask_column = 'mask'
    if args.mask_column is None:
        mask_column = (
            dataset_columns[2] if dataset_columns is not None else column_names[2]
        )
    else:
        mask_column = args.mask_column
        if mask_column not in column_names:
            raise ValueError(
                f"--mask_column' value '{args.mask_column}' needs to be one of: {', '.join(column_names)}"
            )
        
    if args.caption_column is None:
        caption_column = (
            dataset_columns[-1] if dataset_columns is not None else column_names[-1]
        )
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    imagenet_templates_small = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a dark photo of the {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a rendition of a {}",
    ]

    class RandomHorizontalFlip(object):
        def __init__(self, flip_prob):
            self.flip_prob = flip_prob

        def __call__(self, image, target):
            # for i in range(image.size(0)):
            if random.random() < self.flip_prob:
                image = functional.hflip(image)
                if target is not None:
                    target = functional.hflip(target)
            return image, target
        
    class RandomVerticalFlip(object):
        def __init__(self, flip_prob):
            self.flip_prob = flip_prob

        def __call__(self, image, target):
            # for i in range(image.size(0)):
            if random.random() < self.flip_prob:
                image = functional.vflip(image)
                if target is not None:
                    target = functional.vflip(target)
            return image, target
    
    class RandomRotation(object):
        def __init__(self, flip_prob):
            self.flip_prob = flip_prob

        def __call__(self, image, target):
            # for i in range(image.size(0)):
            random_angle = random.randint(0, self.flip_prob)
            image = functional.rotate(image,random_angle)
            if target is not None:
                target = functional.rotate(target,random_angle)
            return image, target

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                caption = random.choice(imagenet_templates_small).format(args.type + '_' + caption.split('; ')[-1]) #+ ';' + caption.split('; ')[0]
                # caption = args.type + '_' + caption.split('; ')[-1]
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return inputs.input_ids

    # Preprocessing the datasets.
    all_transforms = [
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
            RandomRotation(20),
        ]
    

    # train_transforms = transforms.Compose(
    #     [
    #         transforms.Resize(
    #             args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
    #         ),
    #         transforms.andomHorizontalFlip()
    #         if args.random_hflip
    #         else transforms.Lambda(lambda x: x),
    #         transforms.RandomVerticalFlip()
    #         if args.random_vflip
    #         else transforms.Lambda(lambda x: x),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5]),
    #         RandomRotation(20),
    #     ]
    # )

    # mask_transforms = transforms.Compose(
    #     [
    #         transforms.Resize(
    #             args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
    #         ),
    #         transforms.RandomHorizontalFlip()
    #         if args.random_hflip
    #         else transforms.Lambda(lambda x: x),
    #         transforms.RandomVerticalFlip()
    #         if args.random_vflip
    #         else transforms.Lambda(lambda x: x),
    #         transforms.ToTensor(),
    #         RandomRotation(20),
    #     ]
    # )

    def preprocess_train(examples):
        # images = [image.convert("RGB") for image in examples[image_column]]
        # examples["good_image"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)

        faults = [fault.convert("RGB") for fault in examples[fault_column]]
        # examples["fault_image"] = [train_transforms(fault) for fault in faults]

        masks = []
        masked_image = []
        images = []
        count=0

        resize_512=transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        to_tensor = transforms.ToTensor()
        
        # for mask in examples[mask_column]:
        for count in range(len(examples[mask_column])):
            mask = examples[mask_column][count].convert("L")
            mask =resize_512(mask)
            fault_img = resize_512(faults[count])
            for block in all_transforms:
                fault_img, mask = block(fault_img, mask)
            fault_img = to_tensor(fault_img)
            mask = to_tensor(mask)
            # mask = mask_transforms(mask.convert("L"))
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            images.append(fault_img)
            masks.append(mask)
            masked_image.append(fault_img * (mask < 0.5))
            # count += 1
        assert len(masks) == len(images) == len(masked_image)
        examples["fault_image"] = images 
        examples["masks"] = masks
        examples["masked_images"] = masked_image
        # examples["masks"] = []
        # examples["masked_images"] = []
        # for pixel_values in examples["pixel_values"]:
        #     mask, masked_image = generate_mask(pixel_values, args.mask_mode)
        #     examples["masks"].append(mask)
        #     examples["masked_images"].append(masked_image)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=args.seed)
                .select(range(args.max_train_samples))
            )
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        def _collate_imgs(vals):
            vals = torch.stack(vals)
            vals = vals.to(memory_format=torch.contiguous_format).float()
            return vals

        # good_image = _collate_imgs([example["good_image"] for example in examples])
        fault_image = _collate_imgs([example["fault_image"] for example in examples])
        masks = _collate_imgs([example["masks"] for example in examples])
        masked_images = _collate_imgs(
            [example["masked_images"] for example in examples]
        )
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {
            # "good_image": good_image,
            "fault_image": fault_image,
            "input_ids": input_ids,
            "masks": masks,
            "masked_images": masked_images,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        # Repeat(train_dataset,5*len(train_dataset)),
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    text_encoder.train()

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    fold_name="text_inversion/bs4-lr10-4-mse-only%s_init_emb_%s"%(args.type,args.placeholder_token)
    if accelerator.is_main_process:
        accelerator.init_trackers(fold_name, config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Place Holder = {args.placeholder_token}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 1

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    for epoch in range(first_epoch, args.num_train_epochs+1):
        train_loss = 0.0
        text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            # Image.fromarray((batch['fault_image'][0].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)).save('fault_image.png')
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(
                    batch["fault_image"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # go_latents = vae.encode(
                #     batch["good_image"].to(dtype=weight_dtype)
                # ).latent_dist.sample()
                # go_latents = go_latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  ##给target image加噪
                
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(
                        prediction_type=args.prediction_type
                    )

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    target = latents
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                masked_image_latents = vae.encode(
                    batch["masked_images"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae.config.scaling_factor

                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                mask = torch.nn.functional.interpolate(
                    batch["masks"],
                    size=(
                        args.resolution // vae_scale_factor,
                        args.resolution // vae_scale_factor,
                    ),
                )
                mask = mask.to(device=latents.device, dtype=weight_dtype)

                num_channels_latents = vae.config.latent_channels
                num_channels_unet = unet.config.in_channels
                num_channels_mask = mask.shape[1]
                num_channels_masked_image = masked_image_latents.shape[1]
                if (
                    num_channels_latents + num_channels_mask + num_channels_masked_image
                    != num_channels_unet
                ):
                    raise ValueError(
                        f"Incorrect configuration settings! The config of `pipeline.unet`: {unet.config} expects"
                        f" {num_channels_unet} but received `num_channels_latents`: {num_channels_latents} +"
                        f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                        f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                        " `pipeline.unet` or your `mask_image` or `image` input."
                    )

                latent_model_input = torch.cat(
                    [noisy_latents, mask, masked_image_latents], dim=1
                )
                # latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                
                # Predict the noise residual and compute loss
                # print(mask.shape)
                model_pred = unet(
                    latent_model_input, timesteps, encoder_hidden_states, #attention_mask=mask,
                ).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     params_to_clip = text_encoder.parameters()
                #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]
                

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir,fold_name, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # text_encoder.to(torch.float32)
                        # text_encoder.module.save_pretrained(os.path.join(args.output_dir,fold_name,f"checkpoint-{global_step}"))
                        # logger.info(f"Saved text_encoder model to {save_path}")
                        save_progress(
                            text_encoder,
                            placeholder_token_ids,
                            accelerator,
                            args,
                            save_path+'/model.safetensors'
                        )   



            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        
        if accelerator.is_main_process:
            if args.validation_file is not None and epoch % args.validation_epochs == 0:
                
                val_examples = []
                with open(os.path.join(args.validation_file,args.type,args.type+'_no_expand.jsonl'), "r") as val_index_fp:
                    for line in val_index_fp:
                        val_examples.append(json.loads(line))

                logger.info(
                    f"Running validation... \n Generating {len(val_examples)} images"
                )
                # create pipeline
                pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    revision=args.revision,
                    torch_dtype=weight_dtype,
                    safety_checker = None,
                    requires_safety_checker = False
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)
                
                # run inference
                generator = torch.Generator(device=accelerator.device)
                if args.seed is not None:
                    generator = generator.manual_seed(args.seed)
                images = []
                
                for val_example in val_examples[:30:6]:
                    val_image = Image.open(
                        os.path.join(val_example["good"])
                    ).resize((args.resolution, args.resolution))
                    mask_image = (
                        Image.open(os.path.join(val_example["mask"]))
                        .convert("L")
                        .resize((args.resolution, args.resolution))
                    )
                    images.append(
                        pipeline(
                            prompt=args.type+'_'+val_example["text"].split('; ')[-1],
                            image=val_image,
                            mask_image=mask_image,
                            num_inference_steps=30,
                            # strength=1.0,
                            generator=generator,
                        ).images[0]
                    )
                assert all(np.all(item == 0) for item in images) == False, 'all generating image is black'
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images(
                            "validation", np_images, epoch, dataformats="NHWC"
                        )
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(
                                        image, caption=f"{i}: {val_example['text']}"
                                    )
                                    for i, (val_example, image) in enumerate(
                                        zip(val_examples, images)
                                    )
                                ]
                            }
                        )

                del pipeline
                torch.cuda.empty_cache()

    
    # Save the lora layers
    accelerator.wait_for_everyone()
    # breakpoint()
    if accelerator.is_main_process:
        text_encoder = text_encoder.to(torch.float32)
        # unet.save_attn_procs(os.path.join(args.output_dir,fold_name))
        text_encoder.module.save_pretrained(os.path.join(args.output_dir,fold_name))
        save_progress(
            text_encoder,
            placeholder_token_ids,
            accelerator,
            args,
            os.path.join(args.output_dir,fold_name)+'/model.safetensors'
        )   


    # Final inference
    # # Load previous pipeline
    # pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    #     args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype, 
    #     safety_checker = None, requires_safety_checker = False
    # )
    # pipeline = pipeline.to(accelerator.device)
    
    # # breakpoint()
    # # load attention processors
    # # pipeline.unet.load_attn_procs(os.path.join(args.output_dir,fold_name))
    # # text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.output_dir,fold_name),use_safetensors=False).to(accelerator.device)
    # # pipeline.text_encoder = text_encoder
    # pipeline.load_textual_inversion(os.path.join(args.output_dir,fold_name), token=args.type + '_' + placeholder,
    #                                 use_safetensors=True,
    #                                 text_encoder=pipeline.text_encoder,
    #                                 tokenizer=pipeline.tokenizer).to(accelerator.device)


    # # run inference
    # generator = torch.Generator(device=accelerator.device)
    # if args.seed is not None:
    #     generator = generator.manual_seed(args.seed)
    
    # val_examples = []
    # with open(os.path.join(args.validation_file,args.type,args.type+'_no_expand.jsonl'), "r") as val_index_fp:
    #     for line in val_index_fp:
    #         val_examples.append(json.loads(line))
    # images=[]
    # for val_example in val_examples[::30]:
    #     val_image = Image.open(
    #         os.path.join('../stable-diffusion-main/',val_example["good"])
    #     ).resize((args.resolution, args.resolution))
    #     mask_image = (
    #         Image.open(os.path.join('../stable-diffusion-main/',val_example["mask"]))
    #         .convert("L")
    #         .resize((args.resolution, args.resolution))
    #     )
    #     images.append(
    #         pipeline(
    #             prompt=args.type+'_'+val_example["text"].split('; ')[-1],
    #             image=val_image,
    #             mask_image=mask_image,
    #             num_inference_steps=50,
    #             strength=1.0,
    #             generator=generator,
    #         ).images[0]
    #     )
    # # for _ in range(args.num_validation_images):
    # #     images.append(pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0])
    
    # if accelerator.is_main_process:
    #     for tracker in accelerator.trackers:
    #         if len(images) != 0:
    #             print('generating the final image')
    #             if tracker.name == "tensorboard":
    #                 np_images = np.stack([np.asarray(img) for img in images])
    #                 tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
    #             if tracker.name == "wandb":
    #                 tracker.log(
    #                     {
    #                         "test": [
    #                             wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
    #                             for i, image in enumerate(images)
    #                         ]
    #                     }
    #                 )

    accelerator.end_training()


if __name__ == "__main__":
    types = [
        'bottle',
        'cable',
        'capsule',
        'carpet',
        'grid',
        'hazelnut',
        'leather',
        'metal_nut',
        'pill',
        'screw',
        'tile',
        'toothbrush',
        'transistor',
        'wood',
        'zipper'
    ]
    
    for type in types:
        bks = os.listdir('dataset/%s/ground_truth/'%type)
        for bk in bks:
            print(type+'_'+bk)
            main(type,bk)
