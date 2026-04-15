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
    parser.add_argument('--layer', type=str, required=True)
    parser.add_argument('--f', type=str, required=True)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--NFE', type=int, default=10)
    parser.add_argument('--dtype', type=str, default='fp32')
    parser.add_argument('--ddim_inversion', action='store_true')
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default=str(experiment_dir / 'results' / 'grads'))
    parser.add_argument('--sd_model_path', type=str, default='/home/yonghyun.park/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b')
    parser.add_argument('--data_dir', type=str, default=str(experiment_dir / 'data'))
    parser.add_argument('--task_json', type=str, default=str(experiment_dir / 'configs' / 'all_tasks.json'))
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=256)
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
    if args.ddim_inversion:
        save_path += f'-ddim-gs_{args.guidance_scale}'
    save_path = Path(save_path)
    feature_path = f'{save_path}/concept_grad-{args.task_idx}-eta_{args.eta}.npy'
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
    pipe = DiffusionPipeline.from_pretrained(sd_version).to(device)
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

    if args.f == 'slider_local':
        def compute_f(params, buffers, xt, t, p_pos_, p_neg_):
            xt, t, p_pos_, p_neg_ = xt.unsqueeze(0), t.unsqueeze(0), p_pos_.unsqueeze(0), p_neg_.unsqueeze(0)
            et_pos = functional_call(unet, (params, buffers), args=xt, kwargs={'timestep': t, 'encoder_hidden_states': p_pos_})
            et_neg = functional_call(unet, (params, buffers), args=xt, kwargs={'timestep': t, 'encoder_hidden_states': p_neg_})
            
            et_pos, et_neg = et_pos.sample, et_neg.sample.detach()
            guidance = 3 * (et_pos - et_neg)
            f = F.mse_loss((et_pos + guidance).detach(), et_pos, reduction='none').mean(dim=(1,2,3)).sum()
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
        task['prompt'].replace('<new1> ', ''), # TODO
        batch_size=1,
        device='cuda',
        dtype=dtype,
    )
    x0, p_pos, p_neg, p_base = x0.repeat(batch_size, 1, 1, 1), p_pos.repeat(batch_size, 1, 1), p_neg.repeat(batch_size, 1, 1), p_base.repeat(batch_size, 1, 1)
    x0, p_pos, p_neg, p_base = x0.to(device, dtype=dtype), p_pos.to(device, dtype=dtype), p_neg.to(device, dtype=dtype), p_base.to(device, dtype=dtype)
    print('prompt:', task['prompt'])
    print('prompt_neg:', task['prompt'].replace('<new1> ', ''))

    # ------------------------------------------------------------
    # compute concept grad (concept slider loss)
    # ------------------------------------------------------------
    dstore_keys = np.memmap(feature_path, dtype=np.float32, mode='w+', shape=(1, proj_dim))
    selected_timesteps = torch.arange(1000 // args.NFE, 1000, 1000 // args.NFE, device=device)

    for batch_idx in tqdm(range(args.epochs), desc='concept grad slider...'):
        # Sample a random timestep for each image
        if ddim_inversion:
            from utils import CustomScheduler, invert
            scheduler = CustomScheduler(alphas_cumprod=pipe.scheduler.alphas_cumprod, device=device, dtype=dtype)
            scheduler.set_timesteps(51, device=device, is_inversion=True)
            traj_dict = invert(
                x0, 51, unet, p_pos, p_base, 
                scheduler=scheduler, device=device, dtype=dtype,
                eta=args.eta, guidance_scale=args.guidance_scale
            )[1]

            noise_list = traj_dict['noise'][4::5]
            xt_list = traj_dict['xt'][4::5]
            t_list = traj_dict['t'][4::5]
            t_list = [torch.tensor(batch_size * [t], device=device).long() for t in t_list]

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
            # with torch.cuda.amp.autocast(enabled=(args.dtype == 'fp16')):
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, xt, t, p_pos, p_neg)
            ft_per_sample_grads = vectorize(list(ft_per_sample_grads.values()))
            
            if normalize:
                non_zero_mask = torch.any(ft_per_sample_grads != 0, dim=1)
                ft_per_sample_grads_norm = ft_per_sample_grads.clone()
                ft_per_sample_grads_norm[non_zero_mask] = ft_per_sample_grads[non_zero_mask] / ft_per_sample_grads[non_zero_mask].norm(dim=-1, keepdim=True)
                ft_per_sample_grads = ft_per_sample_grads_norm
            
            if index_t==0:
                emb = ft_per_sample_grads.sum(dim=0, keepdim=True) 
                # if batch_idx==0:
                #     print(f"Max allocated GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
                #     print(emb[0])
            else:
                emb += ft_per_sample_grads.sum(dim=0, keepdim=True) 

    if emb.isnan().any():
        print(f"emb is nan at task_idx: {args.task_idx}")
        exit()

    emb = emb / args.batch_size / args.epochs
    emb = project_func(emb.float())
    dstore_keys[0] = emb.cpu().numpy()

    print(emb[0])
    del emb
    flush()