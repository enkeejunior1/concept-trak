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
from utils import seed_everything, get_dps_guidance, make_random_project_func, normalize_objective_name

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
    parser.add_argument('--split_idx', type=int, required=True)
    parser.add_argument('--num_split', type=int, default=8)
    parser.add_argument('--layer', type=str, required=True)
    parser.add_argument('--f', type=str, required=True)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--ddim_inversion', action='store_true')
    parser.add_argument('--NFE', type=int, default=10)
    parser.add_argument('--dtype', type=str, default='fp32')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--proj_type', type=str, default='random_mask')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--output_dir', type=str, default=str(experiment_dir / 'results' / 'grads'))
    parser.add_argument(
        '--sd_model_path',
        type=str,
        default='CompVis/stable-diffusion-v1-4',
        help='HF repo id (e.g. CompVis/stable-diffusion-v1-4) or local snapshot directory.',
    )
    parser.add_argument('--data_dir', type=str, default=str(experiment_dir / 'data'))
    return parser.parse_args()

seed_everything(42)
if __name__ == "__main__":
    # ------------------------------------------------------------
    # Args
    # ------------------------------------------------------------
    args = arg_parser()
    objective = normalize_objective_name(args.f)
    sd_version = args.sd_model_path
    data_path = args.data_dir

    # Args (global)
    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
    device = 'cuda'
    loss_rescale = 1e4

    # ------------------------------------------------------------
    # setting 
    # ------------------------------------------------------------
    save_path = f'{args.output_dir}/{args.layer}-{objective}-NFE{args.NFE}'
    if args.normalize:
        save_path += '-norm'
    if args.ddim_inversion:
        save_path += f'-ddim-gs_{args.guidance_scale}'
    save_path = Path(save_path)
    feature_path = f'{save_path}/train_grad-{args.split_idx}.npy'
    os.makedirs(save_path, exist_ok=True)
    save_loss = objective == 'das'
    if save_loss:
        loss_path = f'{save_path}/train_loss-{args.split_idx}.npy'

    # ------------------------------------------------------------
    # load models 
    # ------------------------------------------------------------
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(sd_version, torch_dtype=dtype).to(device)
    unet = pipe.unet.to(device, dtype=dtype)
    unet.requires_grad_(False)
    unet.eval()
    
    for name, param in unet.named_parameters():
        if args.layer in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    proj_dim = 2**15

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
    # setting: TRAK
    # ------------------------------------------------------------
    project_func = make_random_project_func(
        count_parameters(unet),
        proj_dim=proj_dim,
        proj_max_batch_size=16,
        proj_type=args.proj_type,
        device=device,
    )
    
    # ------------------------------------------------------------
    # setting: vmap
    # ------------------------------------------------------------
    from torch.func import functional_call, vmap, grad 
    params = {k: v.detach() for k, v in unet.named_parameters() if v.requires_grad==True}
    buffers = {k: v.detach() for k, v in unet.named_buffers() if v.requires_grad==True}

    if objective == 'dtrak':
        def compute_f(params, buffers, xt, t, noise, p_emb_):
            xt, t, noise, p_emb_ = xt.unsqueeze(0), t.unsqueeze(0), noise.unsqueeze(0), p_emb_.unsqueeze(0)
            et = functional_call(unet, (params, buffers), args=xt, kwargs={'timestep': t, 'encoder_hidden_states': p_emb_})
            et = et.sample
            f = F.mse_loss(torch.zeros_like(noise), et, reduction="none").mean(dim=(1,2,3)).sum()
            return loss_rescale * f
    elif objective == 'dsm':
        def compute_f(params, buffers, xt, t, noise, p_emb_):
            xt, t, noise, p_emb_ = xt.unsqueeze(0), t.unsqueeze(0), noise.unsqueeze(0), p_emb_.unsqueeze(0)
            et = functional_call(unet, (params, buffers), args=xt, kwargs={'timestep': t, 'encoder_hidden_states': p_emb_})
            et = et.sample
            f = F.mse_loss(noise, et, reduction="none").mean(dim=(1,2,3)).sum()
            return loss_rescale * f
    elif objective == 'dps':
        def compute_f(params, buffers, xt, t, guidance, p_emb_):
            xt, t, guidance, p_emb_ = xt.unsqueeze(0), t.unsqueeze(0), guidance.unsqueeze(0), p_emb_.unsqueeze(0)
            et = functional_call(unet, (params, buffers), args=xt, kwargs={'timestep': t, 'encoder_hidden_states': p_emb_})
            et = et.sample
            f = F.mse_loss((et - guidance).detach(), et, reduction="none").mean(dim=(1,2,3)).sum() 
            return loss_rescale * f
    elif objective == 'das':
        def compute_f(params, buffers, xt, t, noise, p_emb_):
            xt, t, noise, p_emb_ = xt.unsqueeze(0), t.unsqueeze(0), noise.unsqueeze(0), p_emb_.unsqueeze(0)
            et = functional_call(unet, (params, buffers), args=xt, kwargs={'timestep': t, 'encoder_hidden_states': p_emb_})
            et = et.sample
            f = F.l1_loss(torch.zeros_like(noise), et, reduction="none").mean(dim=(1,2,3)).sum()
            return loss_rescale * f
    else:
        raise ValueError(f'Invalid f: {objective}')

    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0, 0, 0))

    # ------------------------------------------------------------
    # load GT dataset
    # ------------------------------------------------------------
    from utils import LaionDataset
    split_size = 100_000 // args.num_split
    train_ds = LaionDataset(
        data_path, subset_size=100_000, mode='no_flip'
    )
    train_ds = torch.utils.data.Subset(
        train_ds, range(args.split_idx * split_size, min((args.split_idx + 1) * split_size, len(train_ds)))
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, prefetch_factor=4
    )

    # ------------------------------------------------------------
    # compute train grad (DSM Loss)
    # ------------------------------------------------------------
    dstore_keys = np.memmap(feature_path, dtype=np.float32, mode='w+', shape=(len(train_ds), proj_dim))
    if save_loss:
        dstore_loss = np.memmap(loss_path, dtype=np.float32, mode='w+', shape=(len(train_ds)))

    # precompute timesteps needed for forward pass
    selected_timesteps = torch.arange(1000 // args.NFE, 1000, 1000 // args.NFE, device=device)
    generator = torch.Generator(device).manual_seed(42)
    selected_noises = torch.randn(args.NFE, 4, 64, 64, device=device, dtype=torch.float16, generator=generator).to(device, dtype=dtype).split(1, dim=0)

    sample_idx = 0
    for x0, p_emb in tqdm(train_dl, desc=f'Compute train grad: Iterate over train sample...'):
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

        # Compute gradients
        for index_t, (t, noise, xt) in enumerate(zip(t_list, noise_list, xt_list)):
            if objective == 'dps':
                alpha_prod_t = pipe.scheduler.alphas_cumprod[t[0].item()].item()
                guidance = get_dps_guidance(unet, xt, p_emb, t, x0, alpha_prod_t).detach()
                f_args = (params, buffers, xt, t, guidance, p_emb)
            else:
                f_args = (params, buffers, xt, t, noise, p_emb)
                
            # with torch.cuda.amp.autocast(enabled=(args.dtype == 'fp16')):
            ft_per_sample_grads = ft_compute_sample_grad(*f_args)
            ft_per_sample_grads = vectorize(list(ft_per_sample_grads.values()))

            # normalize: skip normalization if the row is all zeros
            if args.normalize:
                non_zero_mask = torch.any(ft_per_sample_grads != 0, dim=1)
                ft_per_sample_grads_norm = ft_per_sample_grads.clone()
                ft_per_sample_grads_norm[non_zero_mask] = ft_per_sample_grads[non_zero_mask] / ft_per_sample_grads[non_zero_mask].norm(dim=-1, keepdim=True)
                ft_per_sample_grads = ft_per_sample_grads_norm
            
            if index_t==0:
                emb = ft_per_sample_grads
            else:
                emb += ft_per_sample_grads

            if save_loss:
                with torch.no_grad():
                    et = unet(xt, t, p_emb).sample
                    loss_t = F.mse_loss(noise, et, reduction="none").mean(dim=(1,2,3))
                if index_t == 0:
                    sample_loss = loss_t
                else:
                    sample_loss += loss_t

        if sample_idx==0:
            print(emb[0])
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
        if emb.isnan().any():
            print(f"Warning: emb is nan at sample {sample_idx}")
        
        emb = project_func(emb.float())
        dstore_keys[sample_idx:sample_idx+num_samples] = emb.cpu().numpy()
        if save_loss:
            dstore_loss[sample_idx:sample_idx+num_samples] = sample_loss.cpu().numpy()
        sample_idx += num_samples
        
        del emb
        if save_loss:
            del sample_loss
        flush()
