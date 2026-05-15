import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from torch.func import functional_call, grad, vmap
from tqdm import tqdm

from utils import get_synth_latent_text_embed_ti, make_random_project_func, seed_everything


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
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--NFE', type=int, default=10)
    parser.add_argument('--dtype', type=str, default='fp16')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--proj_type', type=str, default='random_mask')
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default=str(experiment_dir / 'results' / 'grads'))
    parser.add_argument(
        '--sd_model_path',
        type=str,
        default='CompVis/stable-diffusion-v1-4',
        help='HF repo id or local snapshot directory.',
    )
    parser.add_argument('--data_dir', type=str, default=str(experiment_dir / 'data'))
    parser.add_argument('--task_json', type=str, default=str(experiment_dir / 'configs' / 'all_tasks.json'))
    return parser.parse_args()


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
    for timestep in scheduler.timesteps[start_timesteps:total_timesteps]:
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
        ).sample

        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents, eta=eta).prev_sample
    return latents


def main():
    args = arg_parser()
    seed_everything(42)

    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
    device = 'cuda'
    loss_rescale = 1e4 if dtype == torch.float16 else 1.0

    save_dir = Path(args.output_dir) / (
        f'{args.layer}-test_grad-NFE{args.NFE}'
        f'-gs_{args.guidance_scale}'
        f'-eta_{args.eta}'
    )
    if args.normalize:
        save_dir = Path(f'{save_dir}-norm')
    save_dir.mkdir(parents=True, exist_ok=True)
    feature_path = save_dir / f'test_grad-{args.task_idx}.npy'

    with open(args.task_json, 'r') as f:
        tasks = json.load(f)
    task = tasks[args.task_idx]
    task['model_path'] = task['model_path'].replace('models', 'models-ti')
    task['synth_image_path'] = task['synth_image_path'].replace('synth', 'synth-ti')
    print('current task:', task)

    pipe = DiffusionPipeline.from_pretrained(args.sd_model_path).to(device, dtype=dtype)
    unet = pipe.unet.to(device, dtype=dtype)
    unet.requires_grad_(False)
    unet.eval()
    for name, param in unet.named_parameters():
        param.requires_grad = args.layer in name
    proj_dim = 2**15

    project_func = make_random_project_func(
        count_parameters(unet),
        proj_dim=proj_dim,
        proj_max_batch_size=16,
        proj_type=args.proj_type,
        device=device,
    )

    params = {k: v.detach() for k, v in unet.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in unet.named_buffers() if v.requires_grad}

    def compute_f(params, buffers, xt, t, p_pos_, p_neg_):
        xt = xt.unsqueeze(0)
        t = t.unsqueeze(0)
        p_pos_ = p_pos_.unsqueeze(0)
        p_neg_ = p_neg_.unsqueeze(0)
        et_pos = functional_call(unet, (params, buffers), args=xt, kwargs={'timestep': t, 'encoder_hidden_states': p_pos_})
        et_neg = functional_call(unet, (params, buffers), args=xt, kwargs={'timestep': t, 'encoder_hidden_states': p_neg_})

        et_pos = et_pos.sample
        et_neg = et_neg.sample.detach()
        guidance = 3 * (et_pos - et_neg)
        f = F.mse_loss((et_pos + guidance).detach(), et_pos, reduction='none').mean(dim=(1, 2, 3)).sum()
        return loss_rescale * f

    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0, 0, 0))

    _, prompt_embs = get_synth_latent_text_embed_ti(
        args.sd_model_path,
        {'new1.bin': f'{args.data_dir}/{task["model_path"]}'},
        image_path=None,
        captions=[
            task['prompt'],
            task['prompt'].replace('<new1> ', ''),
            '',
        ],
        device=device,
        dtype=dtype,
    )
    p_pos, p_neg, p_base = prompt_embs.split(1, dim=0)

    del pipe.vae
    del pipe.text_encoder
    flush()

    feature_store = np.memmap(feature_path, dtype=np.float32, mode='w+', shape=(1, proj_dim))
    ddim_nfe = 50
    seed_image = int(Path(task['synth_image_path']).stem)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    emb = None
    for epoch_idx in tqdm(range(args.epochs), desc='compute test grad'):
        with torch.no_grad():
            pipe.scheduler.set_timesteps(ddim_nfe, device=device)
            tgt_t = (ddim_nfe // args.NFE) * torch.randint(1, args.NFE, (1,)).item()
            x_t_init = torch.randn(
                args.batch_size,
                4,
                64,
                64,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed_image),
            )
            x_t_init = x_t_init * pipe.scheduler.init_noise_sigma

            p_pos_ = p_pos.repeat_interleave(args.batch_size, dim=0)
            p_neg_ = p_neg.repeat_interleave(args.batch_size, dim=0)
            p_base_ = p_base.repeat_interleave(args.batch_size, dim=0)
            p_pair = torch.cat([p_pos_, p_base_])

            xt = reverse_process(
                unet,
                pipe.scheduler,
                x_t_init,
                p_pair,
                start_timesteps=0,
                total_timesteps=tgt_t,
                guidance_scale=args.guidance_scale,
                eta=args.eta,
            )
            pipe.scheduler.set_timesteps(1000)
            t = pipe.scheduler.timesteps[int(tgt_t * 1000 / ddim_nfe)]

        vec_t = torch.tensor([t] * args.batch_size, device=device).long()
        per_sample_grads = ft_compute_sample_grad(params, buffers, xt, vec_t, p_pos_, p_neg_)
        per_sample_grads = vectorize(list(per_sample_grads.values()))

        if args.normalize:
            non_zero_mask = torch.any(per_sample_grads != 0, dim=1)
            normalized = per_sample_grads.clone()
            normalized[non_zero_mask] = per_sample_grads[non_zero_mask] / per_sample_grads[non_zero_mask].norm(
                dim=-1,
                keepdim=True,
            )
            per_sample_grads = normalized

        batch_emb = per_sample_grads.sum(dim=0, keepdim=True)
        emb = batch_emb if emb is None else emb + batch_emb

        if epoch_idx == 0:
            print(emb[0])
            print(f'Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB')

        del per_sample_grads, batch_emb, p_pos_, p_neg_, p_base_, xt, vec_t
        flush()

    emb = project_func(emb.float()) / args.epochs
    feature_store[0] = emb.cpu().numpy()


if __name__ == '__main__':
    main()
