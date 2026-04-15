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

# torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.disable = True

from utils import (
    SyntheticClassDataset,
    create_model,
    check_gpu_health_and_set_device,
    load_pipeline,
    seed_everything,
    get_dps_guidance,
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
    parser.add_argument('--split_idx', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=str(experiment_dir / 'weights' / 'model.bin'))
    parser.add_argument('--num_split', type=int, default=8)
    parser.add_argument('--f', type=str, required=True)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--NFE', type=int, default=10)
    parser.add_argument('--dtype', type=str, default='fp32')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ddim_inversion', action='store_true')
    parser.add_argument('--guidance_scale', type=float, default=0.0)
    parser.add_argument('--base_dir', type=str, default=str(experiment_dir), help='toy experiment directory')
    parser.add_argument('--output_dir', type=str, default=str(experiment_dir / 'results' / 'grads'), help='output directory')
    return parser.parse_args()

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
    save_path = f'{args.output_dir}/{args.f}-NFE{args.NFE}'
    if args.normalize:
        save_path += '-norm'
    if args.ddim_inversion:
        save_path += f'-ddim-gs_{args.guidance_scale}'
    save_path = Path(save_path)
    feature_path = f'{save_path}/train_grad-{args.split_idx}.npy'
    os.makedirs(save_path, exist_ok=True)

    # ------------------------------------------------------------
    # load GT dataset
    # ------------------------------------------------------------
    import torchvision.transforms as transforms
    split_size = 10000 // args.num_split
    imgs_path = f"{args.base_dir}/data/images"
    attr_path = f"{args.base_dir}/data/labels/metadata.npy"
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_ds = SyntheticClassDataset(
        imgs_path=imgs_path, attr_path=attr_path, num_conds=num_conds, num_class=num_class, transform=transform, 
    )
    num_conds = train_ds.num_conds
    num_class = train_ds.num_class
    train_ds = torch.utils.data.Subset(
        train_ds, range(args.split_idx * split_size, min((args.split_idx + 1) * split_size, len(train_ds)))
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, prefetch_factor=4
    )

    # ------------------------------------------------------------
    # load models 
    # ------------------------------------------------------------
    unet = create_model(device, cond_dim=num_conds*num_class)
    unet.load_state_dict(torch.load(args.model_path, map_location=device))
    unet.eval()
    unet = unet.to(device, dtype=dtype)

    for p in unet.parameters():
        p.requires_grad = True
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

    if args.f == 'dtrakv1':
        def compute_f(params, buffers, xt, t, noise, c):
            xt, t, noise, c = xt.unsqueeze(0), t.unsqueeze(0), noise.unsqueeze(0), c.unsqueeze(0)
            et = functional_call(unet, (params, buffers), args=xt, kwargs={'t': t, 'y': c})
            f = F.mse_loss(torch.zeros_like(noise), et, reduction="none").mean(dim=(1,2,3)).sum()
            return loss_rescale * f
    elif args.f == 'ttrakv1': # exactly same as L_dsm
        def compute_f(params, buffers, xt, t, noise, c):
            xt, t, noise, c = xt.unsqueeze(0), t.unsqueeze(0), noise.unsqueeze(0), c.unsqueeze(0)
            et = functional_call(unet, (params, buffers), args=xt, kwargs={'t': t, 'y': c})
            f = F.mse_loss(noise, et, reduction="none").mean(dim=(1,2,3)).sum()
            return loss_rescale * f
    elif args.f == 'dasv1': 
        def compute_f(params, buffers, xt, t, noise, c):
            xt, t, noise, c = xt.unsqueeze(0), t.unsqueeze(0), noise.unsqueeze(0), c.unsqueeze(0)
            et = functional_call(unet, (params, buffers), args=xt, kwargs={'t': t, 'y': c})
            f = F.l1_loss(torch.zeros_like(noise), et, reduction="none").mean(dim=(1,2,3)).sum()
            return loss_rescale * f
    elif args.f == 'dpsv1':
        def compute_f(params, buffers, xt, t, guidance, c):
            xt, t, guidance, c = xt.unsqueeze(0), t.unsqueeze(0), guidance.unsqueeze(0), c.unsqueeze(0)
            et = functional_call(unet, (params, buffers), args=xt, kwargs={'t': t, 'y': c})
            f = F.mse_loss((et - guidance).detach(), et, reduction="none").mean(dim=(1,2,3)).sum() 
            return loss_rescale * f
    else:
        raise ValueError(f'Invalid f: {args.f}')

    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0, 0, 0))

    # ------------------------------------------------------------
    # compute train grad (DSM Loss)
    # ------------------------------------------------------------
    # Delete existing file if it exists
    dstore_keys = np.memmap(feature_path, dtype=np.float32, mode='w+', shape=(len(train_ds), proj_dim))

    # precompute timesteps needed for forward pass
    generator = torch.Generator(device).manual_seed(42)
    selected_noises = torch.randn(args.NFE, 3, 64, 64, device=device, dtype=torch.float16, generator=generator).to(device, dtype=dtype).split(1, dim=0)
    selected_timesteps = torch.arange(1000 // args.NFE, 1000, 1000 // args.NFE, device=device)

    sample_idx = 0
    for x0, c in tqdm(train_dl, desc=f'Compute train grad: Iterate over train sample...'):
        num_samples = x0.shape[0]
        x0, c = x0.to(device, dtype=dtype), c.to(device, dtype=dtype)
        c = torch.nn.functional.one_hot(c.long(), num_classes=num_class).float().flatten(start_dim=1)
        c_neg = torch.zeros_like(c, dtype=torch.float32).flatten(start_dim=1)

        def c_neg_gen(c):
            while True:
                c_neg = torch.randint(0, num_class, (c.shape[0], num_conds), device=device, dtype=torch.long)
                c_neg = torch.nn.functional.one_hot(c_neg, num_classes=num_class).float().flatten(start_dim=1)
                if (c_neg != c).any():
                    break
            return c_neg.to(c)

        # Sample xt for loss
        if args.ddim_inversion:
            from utils import CustomScheduler, invert
            scheduler = CustomScheduler(alphas_cumprod=noise_scheduler.alphas_cumprod, device=device, dtype=dtype)
            scheduler.set_timesteps(51, device=device, is_inversion=True)
            traj_dict = invert(
                x0, 51, unet, c, p_neg_gen=c_neg_gen, scheduler=scheduler, 
                device=device, dtype=dtype, guidance_scale=args.guidance_scale, 
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
                xt = noise_scheduler.add_noise(x0, vec_noise, timesteps).to(device, dtype=dtype)
                noise_list.append(vec_noise)
                xt_list.append(xt)
                t_list.append(timesteps)

        # Compute gradients
        for index_t, (t, noise, xt) in enumerate(zip(t_list, noise_list, xt_list)):
            if 'dps' in args.f:
                alpha_prod_t = noise_scheduler.alphas_cumprod[t[0].item()].item()
                guidance = get_dps_guidance(unet, xt, c, t, x0, alpha_prod_t).detach()
                f_args = (params, buffers, xt, t, guidance, c)
            else:
                f_args = (params, buffers, xt, t, noise, c)
                
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

        if sample_idx==0:
            print(emb[0], flush=True)
            print(emb.norm(dim=-1, keepdim=True), flush=True)
            print(ft_per_sample_grads.norm(dim=-1, keepdim=True), flush=True)
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB", flush=True)
            
        
        if emb.isnan().any():
            print(f"Warning: emb is nan at sample {sample_idx}")
        
        emb = project_func(emb.float())
        dstore_keys[sample_idx:sample_idx+num_samples] = emb.cpu().numpy()
        sample_idx += num_samples
        
        del emb
        flush()