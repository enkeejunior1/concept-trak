# TRAK-based + LoGRA concept attribution
import gc
import random
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from utils import ExemplarDataset, seed_everything
from einops import einsum

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
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default=str(experiment_dir / 'results' / 'grads'))
    parser.add_argument('--sd_model_path', type=str, default='/home/yonghyun.park/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b')
    parser.add_argument('--data_dir', type=str, default=str(experiment_dir / 'data'))
    parser.add_argument('--task_json', type=str, default=str(experiment_dir / 'configs' / 'all_tasks.json'))
    parser.add_argument('--eta', type=float, default=0.1)
    return parser.parse_args()

# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L746
@torch.no_grad()
def reverse_process(
    unet,
    scheduler,
    latents,
    text_embeddings,
    total_timesteps=1000,
    start_timesteps=0,
    guidance_scale=3.0,
    eta=0.1,
):
    # latents_steps = []

    for timestep in scheduler.timesteps[start_timesteps:total_timesteps]:
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        # predict the noise residual
        noise_pred = unet(
            latent_model_input, timestep, encoder_hidden_states=text_embeddings,
        ).sample

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(
            noise_pred, timestep, latents, eta=eta, 
        ).prev_sample

    # return latents_steps
    return latents

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
    
    dtype = torch.float16 
    device = 'cuda'
    loss_rescale = 1e4
    
    # ------------------------------------------------------------
    # setting 
    # ------------------------------------------------------------
    save_path = f'{args.output_dir}/{args.layer}-slider_seed-NFE{args.NFE}'
    if args.normalize:
        save_path += '-norm'
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

    def compute_f(params, buffers, xt, t, p_pos_, p_neg_):
        xt, t, p_pos_, p_neg_ = xt.unsqueeze(0), t.unsqueeze(0), p_pos_.unsqueeze(0), p_neg_.unsqueeze(0)
        et_pos = functional_call(unet, (params, buffers), args=xt, kwargs={'timestep': t, 'encoder_hidden_states': p_pos_})
        et_neg = functional_call(unet, (params, buffers), args=xt, kwargs={'timestep': t, 'encoder_hidden_states': p_neg_})
        
        et_pos, et_neg = et_pos.sample, et_neg.sample.detach()
        guidance = 3 * (et_pos - et_neg)
        f = F.mse_loss((et_pos + guidance).detach(), et_pos, reduction='none').mean(dim=(1,2,3)).sum()
        return loss_rescale * f

    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0, 0, 0))

    # ------------------------------------------------------------
    # load test dataset
    # ------------------------------------------------------------
    from utils import get_synth_latent_text_embed_ti
    _, prompt_embs = get_synth_latent_text_embed_ti(
        sd_version,
        {
            'new1.bin': f'{data_path}/{task["model_path"]}',
        },
        image_path=None,
        captions=[
            task['prompt'],
            task['prompt'].replace('<new1> ', ''),
            '',
        ],
        device='cuda',
        dtype=dtype,
    )
    p_pos, p_neg, p_base = prompt_embs.split(1, dim=0)
    pipe.to(device, dtype=dtype)
    del pipe.vae
    del pipe.text_encoder
    flush()
    
    # ------------------------------------------------------------
    # compute concept grad (concept slider loss)
    # ------------------------------------------------------------
    dstore_keys = np.memmap(feature_path, dtype=np.float32, mode='w+', shape=(1, proj_dim))
    DDIM_NFE = 50

    test_image_save_path = f'{data_path}/{task["synth_image_path"]}'
    seed_image = int(test_image_save_path.split('/')[-1].split('.')[0])

    from diffusers import DDIMScheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    for i in tqdm(range(args.epochs), desc='concept grad slider...'):
        # xT -> xt (DDIM inversion, from T to tgt_t)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(DDIM_NFE, device=device)

            # sample target timestep
            tgt_t = (DDIM_NFE // args.NFE) * torch.randint(1, args.NFE, (1,)).item()
            xT = torch.randn(
                args.batch_size, 4, 64, 64, device=device, dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed_image) # TODO: replace with DDIM inversion 
            )
            xT = xT * pipe.scheduler.init_noise_sigma

            # xT -> xt
            (
                p_pos_, p_neg_, p_base_
            ) = (
                p_pos.repeat_interleave(args.batch_size, dim=0), 
                p_neg.repeat_interleave(args.batch_size, dim=0), 
                p_base.repeat_interleave(args.batch_size, dim=0),
            )
            p_pair = torch.cat([p_pos_, p_base_])

            xt = reverse_process(
                unet, pipe.scheduler, xT, p_pair,
                start_timesteps=0, total_timesteps=tgt_t, guidance_scale=args.guidance_scale,
                eta=args.eta,
            )
            assert args.guidance_scale == 1.0, "guidance_scale must be 1.0"
            
            pipe.scheduler.set_timesteps(1000)
            t = pipe.scheduler.timesteps[int(tgt_t * 1000 / DDIM_NFE)]
            scaled_xt = pipe.scheduler.scale_model_input(xt, t)

        vec_t = torch.tensor([t] * args.batch_size, device=device).long()
        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, xt, vec_t, p_pos_, p_neg_)
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

        del ft_per_sample_grads, ft_per_sample_grads_norm, p_base_, p_pos_, p_neg_, xt, vec_t
        flush()

    emb = project_func(emb.float()) / args.epochs
    dstore_keys[0] = emb.cpu().numpy()