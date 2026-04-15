import argparse
import gc
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from einops import einsum
from PIL import Image
from tqdm import tqdm

from utils import ExemplarVisDataset, LAIONVisDataset, seed_everything


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def vectorize(g):
    return torch.cat([x.flatten(start_dim=1) for x in g], dim=-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def arg_parser():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_idx", type=int, required=True)
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument("--f", type=str, default="dpsv1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--ddim_inversion", action="store_true")
    parser.add_argument("--NFE", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=256)
    parser.add_argument("--num_split", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--train_guidance_scale", type=float, default=7.5)
    parser.add_argument("--concept_guidance_scale", type=float, default=1.0)
    parser.add_argument("--gen_guidance_scale", type=float, default=7.5)
    parser.add_argument("--gen_num_inference_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--results_dir", type=str, default=str(experiment_dir / "results" / "qual"))
    parser.add_argument("--grad_dir", type=str, default=str(experiment_dir / "results" / "grads"))
    parser.add_argument("--data_dir", type=str, default=str(experiment_dir / "data"))
    parser.add_argument("--task_json", type=str, default=str(experiment_dir / "configs" / "all_tasks.json"))
    parser.add_argument(
        "--sd_model_path",
        type=str,
        default="/home/yonghyun.park/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b",
    )
    parser.add_argument("--render_leastk", action="store_true")
    return parser.parse_args()


def load_tasks(task_json):
    with open(task_json, "r") as f:
        return json.load(f)


def resolve_seed(task, seed):
    if seed is not None:
        return seed
    synth_name = Path(task["synth_image_path"]).stem
    return int(synth_name)


def build_grad_dir(base_dir, layer, name, nfe, normalize=False, ddim_inversion=False, guidance_scale=7.5):
    path = Path(base_dir) / f"{layer}-{name}-NFE{nfe}"
    if normalize:
        path = Path(f"{path}-norm")
    if ddim_inversion:
        path = Path(f"{path}-ddim-gs_{guidance_scale}")
    return path


def build_loss_dir(base_dir, nfe):
    return Path(base_dir) / f"loss-NFE{nfe}"


def build_output_dir(args, seed):
    out_dir = Path(args.results_dir) / f"task_{args.task_idx}" / (
        f"{args.layer}-{args.f}-slider_seed"
        f"-train_gs_{args.train_guidance_scale}"
        f"-concept_gs_{args.concept_guidance_scale}"
        f"-eta_{args.eta}"
        f"-seed_{seed}"
    )
    if args.normalize:
        out_dir = Path(f"{out_dir}-norm")
    if args.ddim_inversion:
        out_dir = Path(f"{out_dir}-ddim")
    return out_dir


def render_grid(indices, scores, exemplar_ds, train_ds, num_exemplars, save_path):
    fig, axs = plt.subplots(1, len(indices), figsize=(20, 4))
    if len(indices) == 1:
        axs = [axs]
    for i, idx in enumerate(indices):
        if idx < num_exemplars:
            img = exemplar_ds[idx].resize((512, 512))
            axs[i].imshow(np.asarray(img))
            axs[i].add_patch(
                plt.Rectangle((0, 0), img.size[0], img.size[1], fill=False, edgecolor="red", linewidth=3)
            )
        else:
            img = train_ds[idx - num_exemplars][0].resize((512, 512))
            axs[i].imshow(np.asarray(img))
        axs[i].set_title(f"{scores[idx]:.2e}", fontsize=8)
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_prompt_summary(indices, exemplar_ds, train_ds, num_exemplars, save_path, header):
    with open(save_path, "w") as f:
        f.write(f"{header}\n")
        for rank, idx in enumerate(indices):
            if idx < num_exemplars:
                f.write(f"{rank}\texemplar\t{idx}\n")
            else:
                train_idx = idx - num_exemplars
                _, caption = train_ds[train_idx]
                f.write(f"{rank}\ttrain\t{train_idx}\t{caption}\n")


def encode_prompts(pipe, prompts, device):
    tokens = pipe.tokenizer(
        prompts,
        max_length=pipe.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].to(device)
    with torch.no_grad():
        return pipe.text_encoder(tokens)[0]


def generate_query_artifacts(pipe, prompt, seed, device, dtype, out_dir, num_steps, guidance_scale):
    generator = torch.Generator(device=device).manual_seed(seed)
    xT = torch.randn(1, 4, 64, 64, generator=generator, device=device, dtype=dtype)
    xT = xT * pipe.scheduler.init_noise_sigma
    image = pipe(
        prompt,
        latents=xT.clone(),
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    image.save(out_dir / "query.png")
    torch.save(xT.detach().cpu(), out_dir / "query_xT.pt")
    return xT, image


def compute_concept_grad(args, pipe, prompt, neg_prompt, xT, device, dtype):
    unet = pipe.unet.to(device, dtype=dtype)
    unet.requires_grad_(False)
    unet.eval()
    for name, param in unet.named_parameters():
        param.requires_grad = args.layer in name

    proj_dim = 2**15
    from dattri.func.projection import random_project
    from torch.func import functional_call, grad, vmap

    project_func = random_project(
        torch.randn(count_parameters(unet), device=device),
        count_parameters(unet),
        proj_max_batch_size=16,
        proj_dim=proj_dim,
        device=device,
    )

    params = {k: v.detach() for k, v in unet.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in unet.named_buffers() if v.requires_grad}
    prompt_embs = encode_prompts(pipe, [prompt, neg_prompt, ""], device).to(device, dtype=dtype)
    p_pos, p_neg, p_base = prompt_embs.split(1, dim=0)

    loss_rescale = 1e4 if dtype == torch.float16 else 1.0

    def compute_f(params, buffers, xt, t, p_pos_, p_neg_):
        xt = xt.unsqueeze(0)
        t = t.unsqueeze(0)
        p_pos_ = p_pos_.unsqueeze(0)
        p_neg_ = p_neg_.unsqueeze(0)
        et_pos = functional_call(unet, (params, buffers), args=xt, kwargs={"timestep": t, "encoder_hidden_states": p_pos_})
        et_neg = functional_call(unet, (params, buffers), args=xt, kwargs={"timestep": t, "encoder_hidden_states": p_neg_})
        et_pos = et_pos.sample
        et_neg = et_neg.sample.detach()
        guidance = 3 * (et_pos - et_neg)
        f = F.mse_loss((et_pos + guidance).detach(), et_pos, reduction="none").mean(dim=(1, 2, 3)).sum()
        return loss_rescale * f

    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0, 0, 0))

    emb = None
    ddim_nfe = 50
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    xT_batch = xT.repeat(args.batch_size, 1, 1, 1)

    for _ in tqdm(range(args.epochs), desc="concept grad slider_seed"):
        with torch.no_grad():
            pipe.scheduler.set_timesteps(ddim_nfe, device=device)
            tgt_t = (ddim_nfe // args.NFE) * torch.randint(1, args.NFE, (1,)).item()
            p_pos_ = p_pos.repeat_interleave(args.batch_size, dim=0)
            p_neg_ = p_neg.repeat_interleave(args.batch_size, dim=0)
            p_base_ = p_base.repeat_interleave(args.batch_size, dim=0)
            p_pair = torch.cat([p_pos_, p_base_])
            xt = reverse_process(
                unet,
                pipe.scheduler,
                xT_batch,
                p_pair,
                start_timesteps=0,
                total_timesteps=tgt_t,
                guidance_scale=args.concept_guidance_scale,
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
                dim=-1, keepdim=True
            )
            per_sample_grads = normalized

        batch_emb = per_sample_grads.sum(dim=0, keepdim=True)
        emb = batch_emb if emb is None else emb + batch_emb
        del per_sample_grads, batch_emb, p_base_, p_pos_, p_neg_, xt, vec_t
        flush()

    emb = emb / args.epochs
    concept_grad = project_func(emb.float())
    return concept_grad


def load_reference_grads(args, task, device):
    grad_dir = build_grad_dir(
        args.grad_dir,
        args.layer,
        args.f,
        args.NFE,
        normalize=args.normalize,
        ddim_inversion=args.ddim_inversion,
        guidance_scale=args.train_guidance_scale,
    )
    task_grad_path = grad_dir / f"task_grad-{args.task_idx}.npy"
    train_grad_paths = [grad_dir / f"train_grad-{split_idx}.npy" for split_idx in range(args.num_split)]

    exemplar_ds = ExemplarVisDataset(args.data_dir, task["test_case"], task["test_case_ind"])
    num_exemplars = len(exemplar_ds)
    task_grad = torch.from_numpy(np.memmap(task_grad_path, dtype=np.float32, mode="r", shape=(num_exemplars, 2**15))).to(device)
    train_grad = torch.cat(
        [
            torch.from_numpy(
                np.memmap(train_path, dtype=np.float32, mode="r", shape=(100_000 // args.num_split, 2**15))
            )
            for train_path in train_grad_paths
        ],
        dim=0,
    ).to(device)

    loss = None
    if args.f == "dasv1":
        loss_dir = build_loss_dir(args.grad_dir, args.NFE)
        task_loss_path = loss_dir / f"task_loss-{args.task_idx}.npy"
        train_loss_paths = [loss_dir / f"train_loss-{split_idx}.npy" for split_idx in range(args.num_split)]
        task_loss = torch.from_numpy(np.memmap(task_loss_path, dtype=np.float32, mode="r", shape=(num_exemplars,))).to(device)
        train_loss = torch.cat(
            [
                torch.from_numpy(np.memmap(loss_path, dtype=np.float32, mode="r", shape=(100_000 // args.num_split,)))
                for loss_path in train_loss_paths
            ],
            dim=0,
        ).to(device)
        loss = torch.cat([task_loss, train_loss], dim=0)

    return exemplar_ds, task_grad, train_grad, loss


def run_influence(args, out_dir, task, concept_grad, exemplar_ds, task_grad, train_grad, loss):
    device = concept_grad.device
    dtype = concept_grad.dtype
    train_ds = LAIONVisDataset(f"{args.data_dir}/laion_subset")
    num_exemplars = len(exemplar_ds)

    kernel = train_grad.T @ train_grad
    heuristic_lamb_dir = build_grad_dir(
        args.grad_dir,
        args.layer,
        args.f,
        args.NFE,
        normalize=args.normalize,
        ddim_inversion=args.ddim_inversion,
        guidance_scale=args.train_guidance_scale,
    )
    heuristic_lamb_path = heuristic_lamb_dir / "heuristic_lamb.pt"
    if heuristic_lamb_path.exists():
        heuristic_lamb = torch.load(heuristic_lamb_path, map_location=device)
    else:
        heuristic_lamb = 0.1 * torch.linalg.eigh(kernel).eigenvalues.mean().item()
        torch.save(heuristic_lamb, heuristic_lamb_path)

    lamb_r_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    recall_list = []
    reverse_recall_list = []
    top_k_idx_list = []
    least_k_idx_list = []

    for lamb_r in tqdm(lamb_r_list, desc="qual influence"):
        lamb = lamb_r * heuristic_lamb
        kernel_inv = torch.linalg.solve(
            kernel + lamb * torch.eye(kernel.shape[0], device=device, dtype=dtype),
            torch.eye(kernel.shape[0], device=device, dtype=dtype),
        )

        proj_train_grad = (train_grad @ kernel_inv).T
        train_scores = einsum(concept_grad, proj_train_grad, "i k, k j -> j")
        proj_task_grad = (task_grad @ kernel_inv).T
        task_scores = einsum(concept_grad, proj_task_grad, "i k, k j -> j")
        scores = torch.cat([task_scores, train_scores], dim=0)
        if loss is not None:
            scores = scores * loss

        top_k_idx = scores.argsort(descending=True)[: args.top_k].cpu()
        least_k_idx = scores.argsort(descending=False)[: args.top_k].cpu()
        recall = sum(1 for idx in top_k_idx.tolist() if idx < num_exemplars) / min(args.top_k, num_exemplars)
        reverse_recall = sum(1 for idx in least_k_idx.tolist() if idx < num_exemplars) / min(args.top_k, num_exemplars)

        top_k_idx_list.append(top_k_idx)
        least_k_idx_list.append(least_k_idx)
        recall_list.append(recall)
        reverse_recall_list.append(reverse_recall)

        render_grid(
            top_k_idx.tolist(),
            scores.detach().cpu(),
            exemplar_ds,
            train_ds,
            num_exemplars,
            out_dir / f"topk-lambda_{lamb_r:.1e}.jpg",
        )
        save_prompt_summary(
            top_k_idx.tolist(),
            exemplar_ds,
            train_ds,
            num_exemplars,
            out_dir / f"topk-prompts-lambda_{lamb_r:.1e}.txt",
            "TopK",
        )

        if args.render_leastk:
            render_grid(
                least_k_idx.tolist(),
                scores.detach().cpu(),
                exemplar_ds,
                train_ds,
                num_exemplars,
                out_dir / f"leastk-lambda_{lamb_r:.1e}.jpg",
            )
            save_prompt_summary(
                least_k_idx.tolist(),
                exemplar_ds,
                train_ds,
                num_exemplars,
                out_dir / f"leastk-prompts-lambda_{lamb_r:.1e}.txt",
                "LeastK",
            )

        del kernel_inv, proj_train_grad, train_scores, proj_task_grad, task_scores, scores
        flush()

    best_idx = int(np.argmax(recall_list))
    metrics = {
        "task_idx": args.task_idx,
        "prompt": task["prompt"],
        "seed": resolve_seed(task, args.seed),
        "lamb_r_list": [float(x) for x in lamb_r_list],
        "recall_list": [float(x) for x in recall_list],
        "reverse_recall_list": [float(x) for x in reverse_recall_list],
        "best_lambda_r": float(lamb_r_list[best_idx]),
        "best_recall": float(recall_list[best_idx]),
    }
    torch.save(
        {
            "recall_list": recall_list,
            "reverse_recall_list": reverse_recall_list,
            "lamb_r_list": lamb_r_list,
            "heuristic_lamb": heuristic_lamb,
            "top_k_idx_list": top_k_idx_list,
            "least_k_idx_list": least_k_idx_list,
        },
        out_dir / "recall_list.pt",
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    args = arg_parser()
    seed_everything(42)

    device = "cuda"
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    tasks = load_tasks(args.task_json)
    task = tasks[args.task_idx].copy()
    task["model_path"] = task["model_path"].replace("models", "models-ti")
    task["synth_image_path"] = task["synth_image_path"].replace("synth", "synth-ti")
    seed = resolve_seed(task, args.seed)
    out_dir = build_output_dir(args, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(args.sd_model_path).to(device, dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_textual_inversion(f'{args.data_dir}/{task["model_path"]}', weight_name="new1.bin")
    pipe.safety_checker = None
    pipe._safety_check = False

    neg_prompt = task["prompt"].replace("<new1> ", "")
    xT, _ = generate_query_artifacts(
        pipe,
        task["prompt"],
        seed,
        device,
        dtype,
        out_dir,
        args.gen_num_inference_steps,
        args.gen_guidance_scale,
    )
    torch.save(
        {
            "task_idx": args.task_idx,
            "seed": seed,
            "prompt": task["prompt"],
            "neg_prompt": neg_prompt,
            "model_path": task["model_path"],
        },
        out_dir / "query_meta.pt",
    )

    concept_grad = compute_concept_grad(args, pipe, task["prompt"], neg_prompt, xT, device, dtype)
    np.save(out_dir / "concept_grad.npy", concept_grad.detach().cpu().numpy())

    del pipe
    flush()

    exemplar_ds, task_grad, train_grad, loss = load_reference_grads(args, task, device)
    concept_grad = concept_grad.to(device, dtype=train_grad.dtype)
    metrics = run_influence(args, out_dir, task, concept_grad, exemplar_ds, task_grad, train_grad, loss)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
