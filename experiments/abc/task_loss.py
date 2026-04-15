# TRAK-based + LoGRA concept attribution
import gc
import random
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from einops import einsum

import numpy as np
import torch
import torch.nn.functional as F
from utils import ExemplarDataset, seed_everything, get_dps_guidance

def flush():
    torch.cuda.empty_cache()
    gc.collect()

def vectorize(g):
    return torch.cat([x.flatten(start_dim=1) for x in g], dim=-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def arg_parser():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_idx', type=int, required=True)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--ddim_inversion', action='store_true')
    parser.add_argument('--NFE', type=int, default=10)
    parser.add_argument('--dtype', type=str, default='fp32')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--output_dir', type=str, default=str(experiment_dir / 'results' / 'grads'))
    parser.add_argument('--sd_model_path', type=str, default='/home/yonghyun.park/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b')
    parser.add_argument('--data_dir', type=str, default=str(experiment_dir / 'data'))
    parser.add_argument('--task_json', type=str, default=str(experiment_dir / 'configs' / 'all_tasks.json'))
    return parser.parse_args()

seed_everything(42)
if __name__ == "__main__":
    # ------------------------------------------------------------
    # Args
    # ------------------------------------------------------------
    args = arg_parser()
    sd_version = args.sd_model_path
    data_path = args.data_dir
    task_json = args.task_json
    task_idx = args.task_idx
    
    # Args (global)
    dtype = torch.float32
    device = 'cuda'
    loss_rescale = 1e4

    # ------------------------------------------------------------
    # setting 
    # ------------------------------------------------------------
    save_path = f'{args.output_dir}/loss-NFE{args.NFE}'
    if args.normalize:
        save_path += '-norm'
    if args.ddim_inversion:
        save_path += f'-ddim-gs_{args.guidance_scale}'
    save_path = Path(save_path)
    feature_path = f'{save_path}/task_loss-{args.task_idx}.npy'
    os.makedirs(save_path, exist_ok=True)

    # load task json 
    with open(task_json, 'r') as f:
        tasks = json.load(f)
    task = tasks[task_idx]
    task['model_path'] = task['model_path'].replace('models', 'models-ti')
    task['synth_image_path'] = task['synth_image_path'].replace('synth', 'synth-ti')
    print('current task:', task)

    # ------------------------------------------------------------
    # load models 
    # ------------------------------------------------------------
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(sd_version, torch_dtype=dtype).to(device)
    unet = pipe.unet.to(device, dtype=dtype)
    unet.requires_grad_(False)
    unet.eval()

    # load customized model weights (w/o KV finetuning)
    model_path = f'{data_path}/{task["model_path"]}'
    pipe.load_textual_inversion(model_path, weight_name="new1.bin")

    if args.ddim_inversion:
        with torch.no_grad():
            tokens = pipe.tokenizer(
                [''],
                max_length=pipe.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )['input_ids'].to(device)
            p_base = pipe.text_encoder(tokens)[0].to(device, dtype=dtype)

    del pipe.vae
    del pipe.text_encoder
    flush()

    # ------------------------------------------------------------
    # load GT dataset
    # ------------------------------------------------------------
    from utils import ExemplarDataset
    train_ds = ExemplarDataset(
        sd_version,
        f'{data_path}/{task["model_path"]}',
        task['test_case'],
        task['test_case_ind'],
        task['train_prompt'],
        vae_batch_size=20,
        vae_device='cuda',
        dataroot=data_path,
        mode='no_flip' # TODO: change to no_flip_and_flip
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # ------------------------------------------------------------
    # compute train loss (DSM Loss)
    # ------------------------------------------------------------
    dstore_keys = np.memmap(feature_path, dtype=np.float32, mode='w+', shape=(len(train_ds)))

    # precompute timesteps needed for forward pass
    selected_timesteps = torch.arange(1000 // args.NFE, 1000, 1000 // args.NFE, device=device)
    generator = torch.Generator(device).manual_seed(42)
    selected_noises = torch.randn(args.NFE, 4, 64, 64, device=device, dtype=torch.float16, generator=generator).to(device, dtype=dtype).split(1, dim=0)

    sample_idx = 0
    for x0, p_emb in tqdm(train_dl, desc='Compute task loss: Iterate over task sample...'):
        num_samples = x0.shape[0]
        x0, p_emb = x0.to(device, dtype=dtype), p_emb.to(device, dtype=dtype)

        # Sample xt for loss
        if args.ddim_inversion:
            from utils import CustomScheduler, invert
            scheduler = CustomScheduler(alphas_cumprod=pipe.scheduler.alphas_cumprod, device=device, dtype=dtype)
            scheduler.set_timesteps(51, device=device, is_inversion=True)
            traj_dict = invert(
                x0, 51, unet, p_emb, p_base.repeat(num_samples, 1, 1), 
                scheduler=scheduler, guidance_scale=args.guidance_scale,
                device=device, dtype=dtype
            )[1]

            noise_list = traj_dict['noise'][4::5]
            xt_list = traj_dict['xt'][4::5]
            t_list = traj_dict['t'][4::5]
            t_list = [torch.tensor([t]*num_samples, device=device).long() for t in t_list]

        else:
            noise_list = []
            xt_list = []
            t_list = []
            for t, noise in zip(selected_timesteps, selected_noises):
                timesteps = torch.tensor([t]*num_samples, device=device).long()
                vec_noise = noise.repeat(num_samples, 1, 1, 1).to(device, dtype=dtype)
                xt = pipe.scheduler.add_noise(x0, vec_noise, timesteps).to(device, dtype=dtype)
                noise_list.append(vec_noise)
                xt_list.append(xt)
                t_list.append(timesteps)

        # Compute loss
        for index_t, (t, noise, xt) in enumerate(zip(t_list, noise_list, xt_list)):
            with torch.no_grad():
                et = unet(xt, t, p_emb).sample
                loss_t = F.mse_loss(noise, et, reduction="none").mean(dim=(1,2,3))
            
            if index_t==0:
                loss = loss_t
            else:
                loss += loss_t

        if sample_idx==0:
            print(f'loss[0], {loss[0]} | loss.shape, {loss.shape}')
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
        if loss.isnan().any():
            print(f"Warning: loss is nan at sample {sample_idx}")
        
        dstore_keys[sample_idx:sample_idx+num_samples] = loss.cpu().numpy()
        sample_idx += num_samples

        del loss, loss_t
        flush()