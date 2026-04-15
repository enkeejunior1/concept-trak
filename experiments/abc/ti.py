import argparse
import json
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
# from diffusers.utils.hub_utils import populate_model_card # load_or_create_model_card
from diffusers.utils.import_utils import is_xformers_available


if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.33.0.dev0")
# logger = get_logger(__name__)

def save_model_card(repo_id: str, images: list = None, base_model: str = None, repo_folder: str = None):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"
    model_description = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "textual_inversion",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=False):
    print("Saving embeddings")
    learned_embeds = (
        text_encoder
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def parse_args():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # related to AbC
    # parser.add_argument( "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data.")
    parser.add_argument( "--task_idx", type=int, default=None, required=True, help="A folder containing the training data.")
    parser.add_argument( "--output_dir", type=str, default=str(experiment_dir / "results" / "ti"), help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument( "--sd_model_path", type=str, default="/home/yonghyun.park/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b")
    parser.add_argument( "--data_dir", type=str, default=str(experiment_dir / "data"))
    parser.add_argument( "--task_json", type=str, default=str(experiment_dir / "configs" / "all_tasks.json"))
    parser.add_argument( "--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument( "--validation_prompt", type=str, default=None, help="A prompt that is used during validation to verify that the model is learning.")
    parser.add_argument( "--max_train_steps", type=int, default=5000, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")

    # default values
    parser.add_argument( "--num_train_epochs", type=int, default=100)
    parser.add_argument( "--save_steps", type=int, default=500, help="Save learned_embeds.bin every X updates steps.")
    parser.add_argument( "--save_as_full_pipeline", action="store_true", help="Save the complete stable diffusion pipeline.")
    parser.add_argument( "--num_vectors", type=int, default=1, help="How many textual inversion vectors shall be used to learn the concept.")
    # parser.add_argument( "--pretrained_model_name_or_path", type=str, default=None, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument( "--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument( "--variant", type=str, default=None, help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")
    parser.add_argument( "--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    # parser.add_argument( "--placeholder_token", type=str, default=None, required=True, help="A token to use as a placeholder for the concept.")
    # parser.add_argument( "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word.")
    parser.add_argument( "--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument( "--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument( "--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument( "--resolution", type=int, default=512, help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
    parser.add_argument( "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution.")
    parser.add_argument( "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument( "--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument( "--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument( "--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument( "--lr_scheduler", type=str, default="constant", help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument( "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument( "--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument( "--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument( "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument( "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument( "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument( "--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument( "--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument( "--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument( "--hub_model_id", type=str, default=None, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument( "--logging_dir", type=str, default="logs", help="TensorBoard log directory.")
    parser.add_argument( "--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.")
    parser.add_argument( "--allow_tf32", action="store_true", help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    parser.add_argument( "--report_to", type=str, default="tensorboard", help="The integration to report the results and logs to. Supported platforms are `\"tensorboard\"` (default), `\"wandb\"` and `\"comet_ml\"`. Use `\"all\"` to report to all integrations.")
    parser.add_argument( "--num_validation_images", type=int, default=4, help="Number of images that should be generated during validation with `validation_prompt`.")
    parser.add_argument( "--validation_steps", type=int, default=1000, help="Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images` and logging the images.")
    parser.add_argument( "--validation_epochs", type=int, default=None, help="Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images` and logging the images.")
    parser.add_argument( "--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument( "--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming using `--resume_from_checkpoint`.")
    parser.add_argument( "--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store.")
    parser.add_argument( "--resume_from_checkpoint", type=str, default=None, help="Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `\"latest\"` to automatically select the last available checkpoint.")
    parser.add_argument( "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument( "--no_safe_serialization", action="store_true", help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # if args.train_data_dir is None:
    #     raise ValueError("You must specify a train data directory.")

    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        custom_prompt_templates=None,
        image_paths=None,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        
        if image_paths is None:
            self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        else:
            self.image_paths = image_paths

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        if custom_prompt_templates is not None:
            self.templates = custom_prompt_templates
        else:
            self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def main():
    args = parse_args()

    # ------------------------------------------------------------
    # AbC benchmark 
    # ------------------------------------------------------------
    # load task
    sd_version = args.sd_model_path
    args.pretrained_model_name_or_path = sd_version
    data_path = args.data_dir
    task_json = args.task_json
    task_idx = args.task_idx

    # load task json 
    with open(task_json, 'r') as f:
        tasks = json.load(f)
    task = tasks[task_idx]
    task['model_path'] = task['model_path'].replace('models', 'models-ti')
    task['synth_image_path'] = task['synth_image_path'].replace('synth', 'synth-ti')

    # prompt 
    prompt = '<new2><new2><new2><new2> ' + task['prompt']
    args.initializer_token = task['model_path'].split('/')[-1]
    args.placeholder_token = '<new2>'
    assert '<new1>' in prompt and '<new2>' in prompt

    # path 
    emb_save_path = os.path.join(args.output_dir, str(task_idx))
    os.makedirs(emb_save_path, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset 
    from utils import get_synth_latent_text_embed
    x0, _, _, _ = get_synth_latent_text_embed(
        sd_version,
        f'{data_path}/{task["model_path"]}',
        f'{data_path}/{task["synth_image_path"]}',
        prompt,
        prompt,
        batch_size=1,
        device='cuda',
        dtype=torch.float32,
    )

    # ------------------------------------------------------------
    if os.path.exists(os.path.join(emb_save_path, "new2.bin")):
        print(f'Already got textual inversion weights: {emb_save_path}')
    else:
        # load models
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(sd_version).to(device)
        model_path = f'{data_path}/{task["model_path"]}'
        pipe.load_textual_inversion(model_path, weight_name="new1.bin")
        del pipe.vae
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        unet = pipe.unet
        noise_scheduler = pipe.scheduler
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # add placeholder token in tokenizer
        placeholder_tokens = ['<new2>']
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)

        # convert initializer_token, placeholder_token to ids
        initializer_token_id  = tokenizer.encode(args.initializer_token, add_special_tokens=False)[0] # '<new1>'
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

        # resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id in placeholder_token_ids:
                token_embeds[token_id] = token_embeds[initializer_token_id].clone()

        # freeze unet and text encoder
        unet.requires_grad_(False)
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Scheduler and math around the number of training steps.
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
            num_cycles=args.lr_num_cycles,
        )

        # Set up mixed precision training
        weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
            # Use autocast for mixed precision
            scaler = torch.cuda.amp.GradScaler()
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        # Move vae and unet to device and cast to weight_dtype
        unet.to(device, dtype=weight_dtype)
        
        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
        )

        # keep original embeddings as reference
        orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()
        bsz = args.train_batch_size

        for step in range(args.max_train_steps):
            text_encoder.train()

            # reset gradients only at the beginning of accumulation cycle
            if step % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()
            
            # move batch to device
            x0 = x0.to(device)
            latents = x0.repeat(bsz, 1, 1, 1)
            noise = torch.randn_like(latents)

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict the noise residual
            tokens = tokenizer([prompt], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")['input_ids'].to(device)
            encoder_hidden_states = text_encoder(tokens)[0].repeat(bsz, 1, 1)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            assert noise_scheduler.config.prediction_type == "epsilon"
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # Scale loss by gradient accumulation steps
            loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights and scheduler only at the end of accumulation cycle
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                
                # make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

                with torch.no_grad():
                    text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]

                global_step += 1

            # update progress
            progress_bar.update(1)
            
            # log metrics
            logs = {"loss": loss.item() * args.gradient_accumulation_steps, "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
        # create the pipeline using the trained modules and save it.
        # Save the newly trained embeddings
        weight_name = "new2.bin" # if args.no_safe_serialization else "<new2>.safetensors"
        save_path = os.path.join(emb_save_path, weight_name)
        save_progress(
            text_encoder,
            placeholder_token_ids,
            None,  # no accelerator
            args,
            save_path,
            safe_serialization=False,
        )
        
        # delete all models to flush the GPU memory
        del text_encoder
        del unet
        del tokenizer
        del noise_scheduler
        
        # force CUDA to release memory
        torch.cuda.empty_cache()
        
        # print memory status after cleanup
        if torch.cuda.is_available():
            print(f"GPU memory allocated after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU memory reserved after cleanup: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # ------------------------------------------------------------
    # Abc
    # ------------------------------------------------------------
    # load model 
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path).to(device) 
    pipe.load_textual_inversion(f'{data_path}/{task["model_path"]}', weight_name="new1.bin")
    pipe.load_textual_inversion(emb_save_path, weight_name="new2.bin")
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.to(device)
    assert '<new1>' in task['prompt']

    test_image_save_path = f'{data_path}/{task["synth_image_path"]}'
    image = Image.open(test_image_save_path)
    image.save(f'{emb_save_path}/original.png')
    batch_size = args.train_batch_size * args.gradient_accumulation_steps
    seed_image = int(test_image_save_path.split('/')[-1].split('.')[0])

    image = pipe(task['prompt'], generator=torch.Generator(device=device).manual_seed(seed_image)).images[0]
    image.save(f'{emb_save_path}/new1-{args.max_train_steps}-{batch_size}.png')

    image = pipe('<new2> ' + task['prompt'], generator=torch.Generator(device=device).manual_seed(seed_image)).images[0]
    image.save(f'{emb_save_path}/new2_new1-{args.max_train_steps}-{batch_size}.png')

    image = pipe('<new1> ' + task['prompt'], generator=torch.Generator(device=device).manual_seed(seed_image)).images[0]
    image.save(f'{emb_save_path}/new1_new1-{args.max_train_steps}-{batch_size}.png')

if __name__ == "__main__":
    main()