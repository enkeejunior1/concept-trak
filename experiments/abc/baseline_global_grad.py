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
    parser.add_argument('--layer', type=str, required=True)
    parser.add_argument('--f', type=str, required=True)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--NFE', type=int, default=10)
    parser.add_argument('--dtype', type=str, default='fp32')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ddim_inversion', action='store_true')
    parser.add_argument('--num_samples', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default=str(experiment_dir / 'results' / 'grads'))
    parser.add_argument('--sd_model_path', type=str, default='/home/yonghyun.park/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b')
    parser.add_argument('--data_dir', type=str, default=str(experiment_dir / 'data'))
    parser.add_argument('--task_json', type=str, default=str(experiment_dir / 'configs' / 'all_tasks.json'))
    parser.add_argument('--prompt', type=str, default='special')
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
    
    batch_size = args.batch_size
    normalize = args.normalize
    ddim_inversion = args.ddim_inversion
    device = 'cuda'
    dtype = torch.float32 if args.dtype == 'fp32' else torch.float16
    loss_rescale = 1e4 if args.dtype == 'fp16' else 1.0
    NFE = args.NFE
    f = args.f
    
    # ------------------------------------------------------------
    # setting 
    # ------------------------------------------------------------
    save_path = f'{args.output_dir}/{args.layer}-{args.f}-NFE{args.NFE}'
    if normalize:
        save_path += '-norm'
    if ddim_inversion:
        save_path += f'-ddim-gs_{args.guidance_scale}'
    if args.prompt == 'special':
        save_path += '-special'
    save_path = Path(save_path)
    os.makedirs(save_path, exist_ok=True)
    feature_path = f'{save_path}/concept_grad-{args.task_idx}.npy'

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
    pipe = DiffusionPipeline.from_pretrained(sd_version).to(device, dtype=dtype)
    unet = pipe.unet.to(device, dtype=dtype)
    unet.requires_grad_(False)
    unet.eval()

    for name, param in unet.named_parameters():
        if args.layer in name: # TODO: all attn -> self-attention or cross-attention
            param.requires_grad = True
        else:
            param.requires_grad = False
    proj_dim = 2**15

    # load customized model weights 
    model_path = f'{data_path}/{task["model_path"]}'
    pipe.load_textual_inversion(model_path, weight_name="new1.bin")

    # for global concept attribution, we need multiple images for single concept <V>
    pipe.safety_checker = None
    pipe._safety_check = False
    latents = []
    for i in range(args.num_samples // 32):
        latents.append(pipe(
            '<new1>' if args.prompt == 'special' else task['prompt'], 
            num_images_per_prompt=32,
            output_type='latent',
            return_dict=False,
        )[0])
    latents = torch.cat(latents, dim=0).to(device, dtype=dtype)

    del pipe.vae
    del pipe.text_encoder
    flush()

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

    if args.f == 'dtrakv1_global':
        def compute_f(params, buffers, xt, t, noise, p_emb_):
            xt, t, noise, p_emb_ = xt.unsqueeze(0), t.unsqueeze(0), noise.unsqueeze(0), p_emb_.unsqueeze(0)
            et = functional_call(unet, (params, buffers), args=xt, kwargs={'timestep': t, 'encoder_hidden_states': p_emb_})
            et = et.sample
            f = F.mse_loss(torch.zeros_like(noise), et, reduction="none").mean(dim=(1,2,3)).sum()
            return loss_rescale * f
    elif args.f == 'ttrakv1_global': # exactly same as L_dsm
        def compute_f(params, buffers, xt, t, noise, p_emb_):
            xt, t, noise, p_emb_ = xt.unsqueeze(0), t.unsqueeze(0), noise.unsqueeze(0), p_emb_.unsqueeze(0)
            et = functional_call(unet, (params, buffers), args=xt, kwargs={'timestep': t, 'encoder_hidden_states': p_emb_})
            et = et.sample
            f = F.mse_loss(noise, et, reduction="none").mean(dim=(1,2,3)).sum()
            return loss_rescale * f
    elif args.f == 'dasv1_global': 
        def compute_f(params, buffers, xt, t, noise, p_emb_):
            xt, t, noise, p_emb_ = xt.unsqueeze(0), t.unsqueeze(0), noise.unsqueeze(0), p_emb_.unsqueeze(0)
            et = functional_call(unet, (params, buffers), args=xt, kwargs={'timestep': t, 'encoder_hidden_states': p_emb_})
            et = et.sample
            f = F.l1_loss(torch.zeros_like(noise), et, reduction="none").mean(dim=(1,2,3)).sum()
            return loss_rescale * f
    else:
        raise ValueError(f'Invalid f: {args.f}')

    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0, 0, 0))

    # ------------------------------------------------------------
    # load test dataset
    # ------------------------------------------------------------
    from utils import get_synth_latent_text_embed
    x0, p_pos, p_neg, p_base = get_synth_latent_text_embed(
        sd_version,
        f'{data_path}/{task["model_path"]}',
        f'{data_path}/{task["synth_image_path"]}',
        task['prompt'],
        task['prompt'].replace('<new1> ', ''),
        batch_size=1,
        device='cuda',
        dtype=dtype,
    )
    assert batch_size == 1, 'batch_size must be 1'
    x0, p_pos, p_neg, p_base = x0.repeat(batch_size, 1, 1, 1), p_pos.repeat(batch_size, 1, 1), p_neg.repeat(batch_size, 1, 1), p_base.repeat(batch_size, 1, 1)
    x0, p_pos, p_neg, p_base = x0.to(device, dtype=dtype), p_pos.to(device, dtype=dtype), p_neg.to(device, dtype=dtype), p_base.to(device, dtype=dtype)
    print('prompt:', task['prompt'])
    print('prompt_neg:', task['prompt'].replace('<new1> ', ''))

    # ------------------------------------------------------------
    # compute concept grad (concept slider loss)
    # ------------------------------------------------------------
    dstore_keys = np.memmap(feature_path, dtype=np.float32, mode='w+', shape=(1, proj_dim))
    selected_timesteps = torch.arange(1000 // NFE, 1000, 1000 // NFE, device=device)

    # precompute timesteps needed for forward pass
    generator = torch.Generator(device).manual_seed(42)
    selected_noises = torch.randn(NFE, 4, 64, 64, device=device, dtype=dtype, generator=generator).split(batch_size, dim=0)

    for sample_idx, x0 in enumerate(latents.split(1, dim=0)):
        # Sample a random timestep for each image
        if ddim_inversion:
            from utils import CustomScheduler, invert
            scheduler = CustomScheduler(alphas_cumprod=pipe.scheduler.alphas_cumprod, device=device, dtype=dtype)
            scheduler.set_timesteps(51, device=device, is_inversion=True)
            traj_dict = invert(x0, 51, unet, p_pos, p_base, scheduler=scheduler, device=device, dtype=dtype)[1]

            noise_list = traj_dict['noise'][4::5]
            xt_list = traj_dict['xt'][4::5]
            t_list = traj_dict['t'][4::5]
            t_list = [torch.tensor([t], device=device).long() for t in t_list]

        else:
            noise_list = []
            xt_list = []
            t_list = []
            for t, noise in zip(selected_timesteps, selected_noises):
                timesteps = torch.tensor([t], device=device).long()
                vec_noise = noise.repeat(1, 1, 1, 1).to(device, dtype=dtype)
                xt = pipe.scheduler.add_noise(x0, vec_noise, timesteps).to(device, dtype=dtype)
                noise_list.append(vec_noise)
                xt_list.append(xt)
                t_list.append(timesteps)

        # Compute gradients
        for index_t, (t, noise, xt) in enumerate(zip(t_list, noise_list, xt_list)):
            if 'dps' in args.f:
                alpha_prod_t = pipe.scheduler.alphas_cumprod[t[0].item()].item()
                guidance = get_dps_guidance(unet, xt, p_pos, t, x0, alpha_prod_t).detach()
                f_args = (params, buffers, xt, t, guidance, p_pos)
            else:
                f_args = (params, buffers, xt, t, noise, p_pos)

            # with torch.cuda.amp.autocast(enabled=(args.dtype == 'fp16')):
            ft_per_sample_grads = ft_compute_sample_grad(*f_args)
            ft_per_sample_grads = vectorize(list(ft_per_sample_grads.values()))
            
            if normalize:
                non_zero_mask = torch.any(ft_per_sample_grads != 0, dim=1)
                ft_per_sample_grads_norm = ft_per_sample_grads.clone()
                ft_per_sample_grads_norm[non_zero_mask] = ft_per_sample_grads[non_zero_mask] / ft_per_sample_grads[non_zero_mask].norm(dim=-1, keepdim=True)
                ft_per_sample_grads = ft_per_sample_grads_norm
            
            if sample_idx==0 and index_t==0:
                emb = ft_per_sample_grads / args.num_samples / NFE
            else:
                emb += ft_per_sample_grads / args.num_samples / NFE

        if emb.isnan().any():
            print(f"emb is nan at task_idx: {args.task_idx}")
            exit()
            
    emb = project_func(emb.float())
    dstore_keys[0] = emb.cpu().numpy()
    print("emb: ", emb.shape, "emb: ", emb)

    del emb
    flush()
    exit()