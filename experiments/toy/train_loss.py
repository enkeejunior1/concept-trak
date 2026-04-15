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
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--NFE', type=int, default=10)
    parser.add_argument('--dtype', type=str, default='fp32')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ddim_inversion', action='store_true')
    parser.add_argument('--guidance_scale', type=float, default=0.0)
    parser.add_argument('--random_neg', action='store_true', help='use random negative samples')
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
    save_path = f'{args.output_dir}/loss-NFE{args.NFE}'
    if args.normalize:
        save_path += '-norm'
    if args.ddim_inversion:
        save_path += f'-ddim-gs_{args.guidance_scale}'
    save_path = Path(save_path)
    feature_path = f'{save_path}/train_loss-{args.split_idx}.npy'
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
    unet.requires_grad_(False)

    from diffusers import DDPMScheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    if args.ddim_inversion:
        c_base = torch.zeros(1, num_conds*num_class, device=device, dtype=dtype)

    # ------------------------------------------------------------
    # compute train loss (DSM Loss)
    # ------------------------------------------------------------
    dstore_keys = np.memmap(feature_path, dtype=np.float32, mode='w+', shape=(len(train_ds)))

    # precompute timesteps needed for forward pass
    selected_timesteps = torch.arange(1000 // args.NFE, 1000, 1000 // args.NFE, device=device)
    generator = torch.Generator(device).manual_seed(42)
    selected_noises = torch.randn(args.NFE, 3, 64, 64, device=device, dtype=torch.float16, generator=generator).to(device, dtype=dtype).split(1, dim=0)

    sample_idx = 0
    for x0, c in tqdm(train_dl, desc=f'Compute train loss: Iterate over train sample...'):
        num_samples = x0.shape[0]
        x0, c = x0.to(device, dtype=dtype), c.to(device, dtype=dtype)
        c = torch.nn.functional.one_hot(c.long(), num_classes=num_class).float().flatten(start_dim=1)
        
        if args.random_neg:
            c_neg_gen = lambda: torch.randint(0, num_class, (num_samples,), device=device, dtype=torch.long)
            c_neg = None
        else:
            c_neg_gen = None
            c_neg = torch.zeros_like(c, dtype=torch.float32).flatten(start_dim=1)

        # Sample xt for loss
        if args.ddim_inversion:
            from utils import CustomScheduler, invert
            scheduler = CustomScheduler(alphas_cumprod=noise_scheduler.alphas_cumprod, device=device, dtype=dtype)
            scheduler.set_timesteps(51, device=device, is_inversion=True)
            traj_dict = invert(
                x0, 51, unet, c, c_base.repeat(num_samples, 1), 
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
                xt = noise_scheduler.add_noise(x0, vec_noise, timesteps).to(device, dtype=dtype)
                noise_list.append(vec_noise)
                xt_list.append(xt)
                t_list.append(timesteps)

        # Compute loss
        for index_t, (t, noise, xt) in enumerate(zip(t_list, noise_list, xt_list)):
            with torch.no_grad():
                et = unet(xt, t=t, y=c)
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