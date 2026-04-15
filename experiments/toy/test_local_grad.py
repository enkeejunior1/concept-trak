import gc
import random
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from einops import einsum
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.disable = True

from utils import (
    SyntheticClassDataset,
    create_model,
    num_conds,
    num_class,
    check_gpu_health_and_set_device,
    load_pipeline,
    seed_everything,
    get_dps_guidance,
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
    parser.add_argument('--sample_idx', type=int, required=True)
    parser.add_argument('--shape_idx', type=int, required=True)
    parser.add_argument('--color_idx', type=int, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=str(experiment_dir / 'weights' / 'model.bin'))
    parser.add_argument('--f', type=str, default='slider_local_1')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--NFE', type=int, default=10)
    parser.add_argument('--dtype', type=str, default='fp32')
    parser.add_argument('--ddim_inversion', action='store_true')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--output_dir', type=str, default=str(experiment_dir / 'results' / 'grads'))
    parser.add_argument('--images_dir', type=str, default=str(experiment_dir / 'results' / 'generated_samples'))
    parser.add_argument('--base_dir', type=str, default=str(experiment_dir))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1024)
    parser.add_argument('--target_concept_dim', type=int, default=1024)
    parser.add_argument('--target_concept_idx', type=int, default=1024)
    parser.add_argument('--eta', type=float, default=0.1)
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
    if args.num_samples == 1:
        feature_path = f'{save_path}/concept_grad-target_dim_{args.target_concept_dim}-target_idx_{args.target_concept_idx}-shape_{args.shape_idx}-color_{args.color_idx}-sample_{args.sample_idx}-eta_{args.eta}.npy'
    else:
        raise ValueError(f'Invalid num_samples: {args.num_samples}')
    os.makedirs(save_path, exist_ok=True)

    # ------------------------------------------------------------
    # load GT dataset
    # ------------------------------------------------------------
    # Define paths for synthetic data (matching 0train_gen.py)
    imgs_path = f"{args.base_dir}/data/images"
    attr_path = f"{args.base_dir}/data/labels/metadata.npy"
    
    # Create transform to match training (from 0train_gen.py)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load generated images instead of training samples
    import json
    from PIL import Image
    if args.num_samples == 1:
        # For single sample, use the specific sample_idx
        cond_label = "_".join([str(c) for c in [args.shape_idx, args.color_idx]])
        image_path = os.path.join(
            args.images_dir, f'{args.sample_idx}-{cond_label}.png'
        )
        image = Image.open(image_path)
        image = transform(image)
        
        # Get condition from metadata
        cond = torch.tensor([args.shape_idx, args.color_idx])
        concept_images = image.unsqueeze(0)
        concept_conds = cond.unsqueeze(0)

    # Convert to tensors and process conditions (matching 0train_gen.py format)
    images = concept_images.to(device, dtype=dtype)
    conds = concept_conds.to(device, dtype=dtype)
    assert len(images) == len(conds) == 1

    # ------------------------------------------------------------
    # load models 
    # ------------------------------------------------------------
    # Create model with matching conditioning dimension (from 0train_gen.py)
    unet = create_model(device, cond_dim=num_conds*num_class)
    unet.load_state_dict(torch.load(args.model_path, map_location=device))
    unet.eval()
    unet = unet.to(device, dtype=dtype)

    for p in unet.parameters():
        p.requires_grad = True
    proj_dim = 2**15

    # Use same scheduler as training (from 0train_gen.py)
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

    if args.f in {'slider_local', 'slider_local_1'}:
        def compute_f(params, buffers, xt, t, c_pos, c_neg):
            xt, t, c_pos, c_neg = xt.unsqueeze(0), t.unsqueeze(0), c_pos.unsqueeze(0), c_neg.unsqueeze(0)
            et_pos = functional_call(unet, (params, buffers), args=xt, kwargs={'t': t, 'y': c_pos})
            et_neg = functional_call(unet, (params, buffers), args=xt, kwargs={'t': t, 'y': c_neg})
            guidance = args.guidance_scale * (et_pos - et_neg)
            f = F.mse_loss((et_pos + guidance).detach(), et_pos, reduction='none').mean(dim=(1,2,3)).sum()
            return loss_rescale * f
    else:
        raise ValueError(f'Invalid f: {args.f}')

    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0, 0, 0))

    # ------------------------------------------------------------
    # compute train grad (DSM Loss)
    # ------------------------------------------------------------
    # Check if feature_path exists and is all zeros
    if os.path.exists(feature_path):
        existing_data = np.memmap(feature_path, dtype=np.float32, mode='r', shape=(1, proj_dim))
        if np.all(existing_data == 0):
            print(f"Feature file {feature_path} exists but is all zeros. Will overwrite.")
        else:
            print(f"Feature file {feature_path} exists and contains non-zero data. Will skip.")
            exit(0)
    else:
        print(f"Feature file {feature_path} does not exist. Will create new file.")
    dstore_keys = np.memmap(feature_path, dtype=np.float32, mode='w+', shape=(1, proj_dim))

    # precompute timesteps needed for forward pass
    selected_timesteps = torch.arange(1000 // args.NFE, 1000, 1000 // args.NFE, device=device)

    print(f'len(images): {len(images)}, len(conds): {len(conds)}')
    c_idx = args.target_concept_idx
    c_dim = args.target_concept_dim
    for batch_idx in tqdm(range(args.epochs), desc='concept grad slider...'):
        x0, c_pos = images.repeat(args.batch_size, 1, 1, 1), conds.repeat(args.batch_size, 1)
        num_samples = x0.shape[0]
        x0, c_pos = x0.to(device, dtype=dtype), c_pos.to(device, dtype=dtype)
        c_pos = torch.nn.functional.one_hot(c_pos.long(), num_classes=num_class).float().flatten(start_dim=1)
            
        c_neg = conds.repeat(args.batch_size, 1).to(device, dtype=dtype)
        c_neg[:, c_dim] = 1 if c_idx == 0 else 0
        c_neg = torch.nn.functional.one_hot(c_neg.long(), num_classes=num_class).float().flatten(start_dim=1)
        assert (c_neg != c_pos).any(dim=1).all(), f'c_neg == c_pos, c_idx: {c_idx}, c_dim: {c_dim}, c_neg: {c_neg}, c_pos: {c_pos}'

        # Sample xt for loss
        if args.ddim_inversion:
            from utils import CustomScheduler, invert
            scheduler = CustomScheduler(alphas_cumprod=noise_scheduler.alphas_cumprod, device=device, dtype=dtype)
            scheduler.set_timesteps(51, device=device, is_inversion=True)
            traj_dict = invert(
                x0, 51, unet, c_pos, c_neg, scheduler=scheduler, 
                device=device, dtype=dtype, guidance_scale=args.guidance_scale, eta=args.eta
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
            vec_t = t
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, xt, vec_t, c_pos, c_neg)
            ft_per_sample_grads = vectorize(list(ft_per_sample_grads.values()))
            
            # normalize: skip normalization if the row is all zeros
            if args.normalize:
                non_zero_mask = torch.any(ft_per_sample_grads != 0, dim=1)
                ft_per_sample_grads_norm = ft_per_sample_grads.clone()
                ft_per_sample_grads_norm[non_zero_mask] = ft_per_sample_grads[non_zero_mask] / ft_per_sample_grads[non_zero_mask].norm(dim=-1, keepdim=True)
                ft_per_sample_grads = ft_per_sample_grads_norm
            
            if index_t==0 and batch_idx==0:
                emb = ft_per_sample_grads.sum(dim=0, keepdim=True) / args.batch_size / args.epochs
            else:
                emb += ft_per_sample_grads.sum(dim=0, keepdim=True) / args.batch_size / args.epochs

        if batch_idx==0:
            print(emb[0])
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
        if emb.isnan().any():
            print(f"Warning: emb is nan at sample {batch_idx}")
            raise ValueError(f"emb is nan at sample {batch_idx}")
    
    emb = emb 
    emb = project_func(emb.float())
    dstore_keys[0] = emb.cpu().numpy()
    
    del emb
    flush()