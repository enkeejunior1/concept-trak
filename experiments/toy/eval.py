import argparse
import concurrent.futures
import json
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from influence import build_result_dir
from utils import SyntheticClassDataset, num_class, num_conds, seed_everything


def arg_parser():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=0)
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--shape_idx', type=int, default=-1)
    parser.add_argument('--color_idx', type=int, default=-1)
    parser.add_argument('--target_concept_dim', type=int, default=0)
    parser.add_argument('--target_concept_idx', type=int, default=0)
    parser.add_argument('--NFE', type=int, default=10)
    parser.add_argument('--f', type=str, required=True)
    parser.add_argument('--concept_f', type=str, required=True)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--ddim_inversion', action='store_true')
    parser.add_argument('--num_split', type=int, default=8)
    parser.add_argument('--train_gs', type=float, default=7.5)
    parser.add_argument('--concept_gs', type=float, default=7.5)
    parser.add_argument('--grad_dir', type=str, default=str(experiment_dir / 'results' / 'grads'))
    parser.add_argument('--base_dir', type=str, default=str(experiment_dir))
    parser.add_argument('--results_dir', type=str, default=str(experiment_dir / 'results' / 'influence'))
    parser.add_argument('--eval_dir', type=str, default=str(experiment_dir / 'results' / 'eval'))
    parser.add_argument('--figures_dir', type=str, default=str(experiment_dir / 'figures' / 'eval'))
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=min(mp.cpu_count(), 8))
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--render_leastk', action='store_true')
    return parser.parse_args()


def create_figure_worker(args_tuple):
    lamb_r, recall, indices, imgs_path, attr_path, target_concept_dim, target_concept_idx, save_path = args_tuple
    ds = SyntheticClassDataset(imgs_path=imgs_path, attr_path=attr_path, num_conds=num_conds, num_class=num_class, transform=None)
    fig, axs = plt.subplots(1, len(indices), figsize=(15, 2))
    if len(indices) == 1:
        axs = [axs]
    for i, idx in enumerate(indices):
        img, label = ds[idx]
        axs[i].imshow(np.asarray(img))
        if label[target_concept_dim] == target_concept_idx:
            axs[i].add_patch(plt.Rectangle((0, 0), img.size[0], img.size[1], fill=False, edgecolor='red', linewidth=3))
        axs[i].axis('off')
    fig.suptitle(f'lambda={lamb_r:.1e}, recall={recall:.2f}', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(save_path)


seed_everything(42)
if __name__ == '__main__':
    args = arg_parser()
    if args.num_samples != 1:
        if args.shape_idx != 9:
            args.target_concept_dim = 0
            args.target_concept_idx = args.shape_idx
        if args.color_idx != 9:
            args.target_concept_dim = 1
            args.target_concept_idx = args.color_idx

    result_dir = build_result_dir(args)
    stats_path = result_dir / 'recall_list.pt'
    if not stats_path.exists():
        raise FileNotFoundError(f'Influence stats not found: {stats_path}')

    stats = torch.load(stats_path, map_location='cpu')
    lamb_r_list = stats['lamb_r_list']
    top_k_idx_list = [idx.tolist() if isinstance(idx, torch.Tensor) else list(idx) for idx in stats['top_k_idx_list']]
    least_k_idx_list = [idx.tolist() if isinstance(idx, torch.Tensor) else list(idx) for idx in stats['least_k_idx_list']]

    imgs_path = f"{args.base_dir}/data/images"
    attr_path = f"{args.base_dir}/data/labels/metadata.npy"
    train_ds = SyntheticClassDataset(imgs_path=imgs_path, attr_path=attr_path, num_conds=num_conds, num_class=num_class, transform=None)

    eval_dir = Path(args.eval_dir) / result_dir.relative_to(Path(args.results_dir))
    figures_dir = Path(args.figures_dir) / result_dir.relative_to(Path(args.results_dir))
    eval_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    recall_list = []
    reverse_recall_list = []
    figure_tasks = []
    for lamb_r, top_k_idx, least_k_idx in zip(lamb_r_list, top_k_idx_list, least_k_idx_list):
        top_k_idx = top_k_idx[:args.top_k]
        least_k_idx = least_k_idx[:args.top_k]
        recall_i = sum(1 for idx in top_k_idx if train_ds[idx][1][args.target_concept_dim] == args.target_concept_idx) / len(top_k_idx)
        reverse_recall_i = sum(1 for idx in least_k_idx if train_ds[idx][1][args.target_concept_dim] == args.target_concept_idx) / len(least_k_idx)
        recall_list.append(recall_i)
        reverse_recall_list.append(reverse_recall_i)
        figure_tasks.append((lamb_r, recall_i, top_k_idx, imgs_path, attr_path, args.target_concept_dim, args.target_concept_idx, figures_dir / f'topk-lambda_{lamb_r:.1e}.jpg'))
        if args.render_leastk:
            figure_tasks.append((lamb_r, reverse_recall_i, least_k_idx, imgs_path, attr_path, args.target_concept_dim, args.target_concept_idx, figures_dir / f'leastk-lambda_{lamb_r:.1e}.jpg'))

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(create_figure_worker, figure_tasks), total=len(figure_tasks), desc='Rendering figures'))

    best_idx = int(np.argmax(recall_list))
    metrics = {
        'recall_list': recall_list,
        'reverse_recall_list': reverse_recall_list,
        'lamb_r_list': lamb_r_list,
        'best_lambda_r': lamb_r_list[best_idx],
        'best_recall': recall_list[best_idx],
    }
    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
