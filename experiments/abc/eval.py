import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils import ExemplarVisDataset, LAIONVisDataset, normalize_objective_name, seed_everything


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def build_train_grad_dir(args, objective):
    path = Path(args.grad_dir) / f'{args.layer}-{objective}-NFE{args.NFE}'
    if args.normalize:
        path = Path(f'{path}-norm')
    if args.ddim_inversion:
        path = Path(f'{path}-ddim-gs_{args.train_guidance_scale}')
    return path


def build_test_grad_dir(args):
    path = Path(args.grad_dir) / (
        f'{args.layer}-test_grad-NFE{args.NFE}'
        f'-gs_{args.concept_guidance_scale}'
        f'-eta_{args.eta}'
    )
    if args.normalize:
        path = Path(f'{path}-norm')
    return path


def build_result_dir(args, objective):
    result_dir = Path(args.results_dir) / f'task_{args.task_idx}' / (
        f'{args.layer}-{objective}-test_grad'
        f'-train_gs_{args.train_guidance_scale}'
        f'-test_gs_{args.concept_guidance_scale}'
        f'-eta_{args.eta}'
    )
    if args.normalize:
        result_dir = Path(f'{result_dir}-norm')
    if args.ddim_inversion:
        result_dir = Path(f'{result_dir}-ddim')
    return result_dir


def arg_parser():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_idx', type=int, required=True)
    parser.add_argument('--layer', type=str, required=True)
    parser.add_argument('--num_split', type=int, default=8)
    parser.add_argument('--NFE', type=int, default=10)
    parser.add_argument('--f', type=str, required=True)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--ddim_inversion', action='store_true')
    parser.add_argument('--train_guidance_scale', type=float, default=7.5)
    parser.add_argument('--concept_guidance_scale', type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--grad_dir', type=str, default=str(experiment_dir / 'results' / 'grads'))
    parser.add_argument('--results_dir', type=str, default=str(experiment_dir / 'results' / 'eval'))
    parser.add_argument('--figures_dir', type=str, default=str(experiment_dir / 'figures' / 'eval'))
    parser.add_argument('--data_dir', type=str, default=str(experiment_dir / 'data'))
    parser.add_argument('--task_json', type=str, default=str(experiment_dir / 'configs' / 'all_tasks.json'))
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--render_leastk', action='store_true')
    return parser.parse_args()


def render_grid(indices, scores, exemplar_ds, train_ds, num_exemplars, save_path):
    fig, axs = plt.subplots(1, len(indices), figsize=(20, 4))
    if len(indices) == 1:
        axs = [axs]
    for i, idx in enumerate(indices):
        if idx < num_exemplars:
            img = exemplar_ds[idx].resize((512, 512))
            axs[i].imshow(np.asarray(img))
            axs[i].add_patch(
                plt.Rectangle((0, 0), img.size[0], img.size[1], fill=False, edgecolor='red', linewidth=3)
            )
        else:
            img = train_ds[idx - num_exemplars][0].resize((512, 512))
            axs[i].imshow(np.asarray(img))
        axs[i].set_title(f'{scores[idx]:.2e}', fontsize=8)
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_grads(args, objective, exemplar_count, device):
    train_grad_dir = build_train_grad_dir(args, objective)
    test_grad_dir = build_test_grad_dir(args)

    test_grad_path = test_grad_dir / f'test_grad-{args.task_idx}.npy'
    task_grad_path = train_grad_dir / f'task_grad-{args.task_idx}.npy'
    train_grad_paths = [train_grad_dir / f'train_grad-{split_idx}.npy' for split_idx in range(args.num_split)]

    test_grad = torch.from_numpy(
        np.memmap(test_grad_path, dtype=np.float32, mode='r', shape=(1, 2**15))
    ).to(device)
    task_grad = torch.from_numpy(
        np.memmap(task_grad_path, dtype=np.float32, mode='r', shape=(exemplar_count, 2**15))
    ).to(device)
    train_grad = torch.cat(
        [
            torch.from_numpy(
                np.memmap(train_path, dtype=np.float32, mode='r', shape=(100_000 // args.num_split, 2**15))
            )
            for train_path in train_grad_paths
        ],
        dim=0,
    ).to(device)

    loss = None
    if objective == 'das':
        task_loss_path = train_grad_dir / f'task_loss-{args.task_idx}.npy'
        train_loss_paths = [train_grad_dir / f'train_loss-{split_idx}.npy' for split_idx in range(args.num_split)]
        task_loss = torch.from_numpy(
            np.memmap(task_loss_path, dtype=np.float32, mode='r', shape=(exemplar_count,))
        ).to(device)
        train_loss = torch.cat(
            [
                torch.from_numpy(
                    np.memmap(loss_path, dtype=np.float32, mode='r', shape=(100_000 // args.num_split,))
                )
                for loss_path in train_loss_paths
            ],
            dim=0,
        ).to(device)
        loss = torch.cat([task_loss, train_loss], dim=0)

    return train_grad_dir, test_grad, task_grad, train_grad, loss


def validate_grads(test_grad, task_grad, train_grad):
    for name, grad in {
        'test_grad': test_grad,
        'task_grad': task_grad,
        'train_grad': train_grad,
    }.items():
        nan_count = torch.isnan(grad).any(dim=1).sum().item()
        zero_count = (grad == 0).all(dim=1).sum().item()
        assert nan_count == 0, f'{name} contains NaNs'
        assert zero_count == 0, f'{name} contains all-zero rows'


def main():
    args = arg_parser()
    seed_everything(42)
    objective = normalize_objective_name(args.f)

    device = 'cuda'
    dtype = torch.float32

    result_dir = build_result_dir(args, objective)
    figures_dir = Path(args.figures_dir) / result_dir.relative_to(Path(args.results_dir))
    result_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(args.task_json, 'r') as f:
        tasks = json.load(f)
    task = tasks[args.task_idx]

    with open(f'{args.data_dir}/json/{task["test_case"]}.json', 'r') as f:
        exemplar_paths = json.load(f)[task['test_case_ind']]['exemplar']

    train_ds = LAIONVisDataset(f'{args.data_dir}/laion_subset')
    exemplar_ds = ExemplarVisDataset(args.data_dir, task['test_case'], task['test_case_ind'])
    num_exemplars = len(exemplar_ds)
    assert num_exemplars == len(exemplar_paths)

    train_grad_dir, test_grad, task_grad, train_grad, loss = load_grads(args, objective, num_exemplars, device)
    test_grad = test_grad.to(device, dtype=dtype)
    task_grad = task_grad.to(device, dtype=dtype)
    train_grad = train_grad.to(device, dtype=dtype)
    if loss is not None:
        loss = loss.to(device, dtype=dtype)

    validate_grads(test_grad, task_grad, train_grad)

    kernel = train_grad.T @ train_grad
    heuristic_lamb_path = train_grad_dir / 'heuristic_lamb.pt'
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
    recall_denominator = min(args.top_k, num_exemplars)

    for lamb_r_idx, lamb_r in enumerate(tqdm(lamb_r_list, desc='eval')):
        lamb = lamb_r * heuristic_lamb
        kernel_inv_path = train_grad_dir / f'kernel_inv-{lamb_r_idx}.pt'
        if kernel_inv_path.exists():
            kernel_inv = torch.load(kernel_inv_path, map_location=device).to(kernel)
        else:
            eye = torch.eye(kernel.shape[0], device=device, dtype=dtype)
            kernel_inv = torch.linalg.solve(kernel + lamb * eye, eye)
            torch.save(kernel_inv, kernel_inv_path)

        proj_train_grad = (train_grad @ kernel_inv).T
        train_scores = (test_grad @ proj_train_grad).squeeze(0)
        proj_task_grad = (task_grad @ kernel_inv).T
        task_scores = (test_grad @ proj_task_grad).squeeze(0)
        scores = torch.cat([task_scores, train_scores], dim=0)
        if loss is not None:
            scores = scores * loss

        top_k_idx = scores.argsort(descending=True)[:args.top_k].cpu()
        least_k_idx = scores.argsort(descending=False)[:args.top_k].cpu()
        top_k_idx_list.append(top_k_idx)
        least_k_idx_list.append(least_k_idx)

        recall = sum(1 for idx in top_k_idx.tolist() if idx < num_exemplars) / recall_denominator
        reverse_recall = sum(1 for idx in least_k_idx.tolist() if idx < num_exemplars) / recall_denominator
        recall_list.append(recall)
        reverse_recall_list.append(reverse_recall)

        cpu_scores = scores.detach().cpu()
        render_grid(
            top_k_idx.tolist(),
            cpu_scores,
            exemplar_ds,
            train_ds,
            num_exemplars,
            figures_dir / f'topk-lambda_{lamb_r:.1e}.jpg',
        )
        if args.render_leastk:
            render_grid(
                least_k_idx.tolist(),
                cpu_scores,
                exemplar_ds,
                train_ds,
                num_exemplars,
                figures_dir / f'leastk-lambda_{lamb_r:.1e}.jpg',
            )

        del kernel_inv, proj_train_grad, train_scores, proj_task_grad, task_scores, scores, cpu_scores
        flush()

    best_idx = int(np.argmax(recall_list))
    stats = {
        'task_idx': args.task_idx,
        'lamb_r_list': [float(x) for x in lamb_r_list],
        'recall_list': [float(x) for x in recall_list],
        'reverse_recall_list': [float(x) for x in reverse_recall_list],
        'heuristic_lamb': float(heuristic_lamb),
        'top_k_idx_list': top_k_idx_list,
        'least_k_idx_list': least_k_idx_list,
    }
    torch.save(stats, result_dir / 'recall_list.pt')

    metrics = {
        'task_idx': args.task_idx,
        'lamb_r_list': [float(x) for x in lamb_r_list],
        'recall_list': [float(x) for x in recall_list],
        'reverse_recall_list': [float(x) for x in reverse_recall_list],
        'best_lambda_r': float(lamb_r_list[best_idx]),
        'best_recall': float(recall_list[best_idx]),
    }
    with open(result_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
