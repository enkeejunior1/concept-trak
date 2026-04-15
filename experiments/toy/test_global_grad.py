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

from utils import (
    SyntheticClassDataset,
    create_model,
    seed_everything,
    num_conds,
    num_class,
)

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
    parser.add_argument('--shape_idx', type=int, default=-1)
    parser.add_argument('--color_idx', type=int, default=-1)
    parser.add_argument('--target_concept_dim', type=int, default=0)
    parser.add_argument('--target_concept_idx', type=int, default=0)
    parser.add_argument('--model_path', type=str, default=str(experiment_dir / 'weights' / 'model.bin'))
    parser.add_argument('--f', type=str, default='slider')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--NFE', type=int, default=10)
    parser.add_argument('--dtype', type=str, default='fp32')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=256)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--random_neg', action='store_true')
    parser.add_argument('--output_dir', type=str, default=str(experiment_dir / 'results' / 'grads'))
    return parser.parse_args()

@torch.no_grad()
def reverse_process(
    unet,
    scheduler,
    latents,
    c_pos,
    c_neg,
    total_timesteps=1000,
    start_timesteps=0,
    guidance_scale=1.0,
):
    bsz = latents.shape[0]
    for timestep in scheduler.timesteps[start_timesteps:total_timesteps]:
        if guidance_scale != 0.0 and guidance_scale != 1.0:
            latent_model_input = torch.cat([latents] * 2)
            all_conditions = torch.cat([c_pos, c_neg], dim=0)
            noise_pred = unet(latent_model_input, timestep.repeat(2*bsz), y=all_conditions)
            
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            all_conditions = c_neg if guidance_scale == 0 else c_pos
            latent_model_input = latents
            noise_pred = unet(latent_model_input, timestep.repeat(bsz), y=all_conditions)

        latents = scheduler.step(noise_pred, timestep, latents).prev_sample
    return latents

seed_everything(42)
if __name__ == "__main__":
    # ------------------------------------------------------------
    # Args
    # ------------------------------------------------------------
    args = arg_parser()

    # Args (global)
    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
    device = 'cuda'
    loss_rescale = 1e4

    # ------------------------------------------------------------
    # setting 
    # ------------------------------------------------------------
    save_path = f'{args.output_dir}/slider-NFE{args.NFE}'
    if args.normalize:
        save_path += '-norm'
    save_path += f'-ddim-gs_{args.guidance_scale}'
    save_path = Path(save_path)
    feature_path = f'{save_path}/concept_grad-target_dim_{args.target_concept_dim}-target_idx_{args.target_concept_idx}-shape_{args.shape_idx}-color_{args.color_idx}.npy'
    os.makedirs(save_path, exist_ok=True)

    # ------------------------------------------------------------
    # load models 
    # ------------------------------------------------------------
    unet = create_model(device, cond_dim=num_conds*num_class)
    unet.load_state_dict(torch.load(args.model_path, map_location=device))
    unet.eval()
    unet = unet.to(device, dtype=dtype)

    for param in unet.parameters():
        param.requires_grad = True
    proj_dim = 2**15

    from diffusers import DDPMScheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # ------------------------------------------------------------
    # setting: TRAK
    # ------------------------------------------------------------
    from dattri.func.projection import random_project
    project_func = random_project(
        torch.randn(count_parameters(unet), device=device), 
        count_parameters(unet), 
        proj_max_batch_size=16, proj_dim=proj_dim, device=device
    )
    
    # ------------------------------------------------------------
    # setting: vmap
    # ------------------------------------------------------------
    from torch.func import functional_call, vmap, grad 
    params = {k: v.detach() for k, v in unet.named_parameters() if v.requires_grad==True}
    buffers = {k: v.detach() for k, v in unet.named_buffers() if v.requires_grad==True}

    # slider loss function
    def compute_f(params, buffers, xt, t, c_pos, c_neg):
        xt, t, c_pos, c_neg = xt.unsqueeze(0), t.unsqueeze(0), c_pos.unsqueeze(0), c_neg.unsqueeze(0)
        
        # positive and negative conditions
        et_pos = functional_call(unet, (params, buffers), args=xt, kwargs={'t': t, 'y': c_pos})
        et_neg = functional_call(unet, (params, buffers), args=xt, kwargs={'t': t, 'y': c_neg})
        
        # guidance calculation (slider loss)
        guidance = args.guidance_scale * (et_pos - et_neg)
        f = F.mse_loss((et_pos + guidance).detach(), et_pos, reduction='none').mean(dim=(1,2,3)).sum()
        return loss_rescale * f

    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0, 0, 0))

    # ------------------------------------------------------------
    # compute concept grad (slider loss)
    # ------------------------------------------------------------
    # if os.path.exists(feature_path):
    #     existing_data = np.memmap(feature_path, dtype=np.float32, mode='r', shape=(1, proj_dim))
    #     if np.all(existing_data == 0):
    #         print(f"Feature file {feature_path} exists but is all zeros. Will overwrite.")
    #     else:
    #         print(f"Feature file {feature_path} exists and contains non-zero data. Will skip.")
    #         exit(0)
    # else:
    #     print(f"Feature file {feature_path} does not exist. Will create new file.")
    dstore_keys = np.memmap(feature_path, dtype=np.float32, mode='w+', shape=(1, proj_dim))
    DDIM_NFE = 50

    c_idx = args.target_concept_idx
    c_dim = args.target_concept_dim
    conds = [args.shape_idx, args.color_idx]
    
    print(f"Using {c_dim} concept with index {c_idx}")
    for i in tqdm(range(args.epochs), desc='concept grad slider...'):
        c_pos = torch.tensor(conds, device=device).repeat(args.batch_size, 1)
        c_pos[:, c_dim] = c_idx  # Set the concept of interest to class 1
        c_pos = torch.nn.functional.one_hot(c_pos.long(), num_classes=num_class).float().flatten(start_dim=1)
            
        c_neg = torch.tensor(conds, device=device).repeat(args.batch_size, 1)
        random_idx_for_c_dim = [idx for idx in range(num_class) if idx != c_idx]
        c_neg[:, c_dim] = torch.tensor([random_idx_for_c_dim[i] for i in torch.randint(0, len(random_idx_for_c_dim), (args.batch_size,), device=device)], device=device)
        c_neg = torch.nn.functional.one_hot(c_neg.long(), num_classes=num_class).float().flatten(start_dim=1)
        assert (c_neg != c_pos).any(dim=1).all(), 'c_neg == c_pos'
            
        # xT -> xt (DDIM inversion, from T to tgt_t)
        with torch.no_grad():
            noise_scheduler.set_timesteps(DDIM_NFE, device=device)

            # sample target timestep
            tgt_t = (DDIM_NFE // args.NFE) * torch.randint(1, args.NFE, (1,)).item()
            xT = torch.randn(args.batch_size, 3, 64, 64, device=device, dtype=dtype)
            xT = xT * noise_scheduler.init_noise_sigma

            # xT -> xt
            xt = reverse_process(
                unet, noise_scheduler, xT, c_pos, c_neg,
                start_timesteps=0, total_timesteps=tgt_t, guidance_scale=args.guidance_scale,
            )
            
            noise_scheduler.set_timesteps(1000)
            t = noise_scheduler.timesteps[int(tgt_t * 1000 / DDIM_NFE)]

        vec_t = torch.tensor([t] * args.batch_size, device=device).long()
        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, xt, vec_t, c_pos, c_neg)
        ft_per_sample_grads = vectorize(list(ft_per_sample_grads.values()))
        
        if args.normalize:
            non_zero_mask = torch.any(ft_per_sample_grads != 0, dim=1)
            ft_per_sample_grads_norm = ft_per_sample_grads.clone()
            ft_per_sample_grads_norm[non_zero_mask] = ft_per_sample_grads[non_zero_mask] / ft_per_sample_grads[non_zero_mask].norm(dim=-1, keepdim=True)
            ft_per_sample_grads = ft_per_sample_grads_norm
        
        if i==0:
            emb = ft_per_sample_grads.sum(dim=0, keepdim=True)
            print(emb[0])
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        else:
            emb += ft_per_sample_grads.sum(dim=0, keepdim=True)

    emb = project_func(emb.float()) / args.epochs
    dstore_keys[0] = emb.cpu().numpy()