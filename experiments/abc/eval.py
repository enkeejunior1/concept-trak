import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import ExemplarVisDataset, LAIONVisDataset


def build_result_dir(args):
    result_dir = f'{args.results_dir}/task_{args.task_idx}/{args.layer}-{args.f}-{args.concept_f}'
    if args.normalize:
        result_dir += '-norm'
    if args.ddim_inversion:
        result_dir += '-ddim'
    if args.concept_f in {'slider_local_1', 'slider_local_2', 'slider_seed'}:
        result_dir += f'-train_gs_{args.train_guidance_scale}'
        result_dir += f'-concept_gs_{args.concept_guidance_scale}'
        result_dir += f'-eta_{args.eta}'
    if args.prompt == 'special' and 'global' in args.concept_f and args.num_samples != 1 and 'slider' not in args.concept_f:
        result_dir += '-special'
    return Path(result_dir)


def arg_parser():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_idx', type=int, required=True)
    parser.add_argument('--num_samples', type=int, default=256)
    parser.add_argument('--layer', type=str, required=True)
    parser.add_argument('--NFE', type=int, default=10)
    parser.add_argument('--f', type=str, required=True)
    parser.add_argument('--concept_f', type=str, required=True)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--ddim_inversion', action='store_true')
    parser.add_argument('--train_guidance_scale', type=float, default=7.5)
    parser.add_argument('--concept_guidance_scale', type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--prompt', type=str, default='special')
    parser.add_argument('--results_dir', type=str, default=str(experiment_dir / 'results' / 'influence'))
    parser.add_argument('--eval_dir', type=str, default=str(experiment_dir / 'results' / 'eval'))
    parser.add_argument('--figures_dir', type=str, default=str(experiment_dir / 'figures' / 'eval'))
    parser.add_argument('--data_dir', type=str, default=str(experiment_dir / 'data'))
    parser.add_argument('--task_json', type=str, default=str(experiment_dir / 'configs' / 'all_tasks.json'))
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--render_leastk', action='store_true')
    return parser.parse_args()


def render_grid(indices, exemplar_ds, train_ds, num_exemplars, save_path):
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
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    args = arg_parser()
    result_dir = build_result_dir(args)
    stats_path = result_dir / 'recall_list.pt'
    if not stats_path.exists():
        raise FileNotFoundError(f'Influence stats not found: {stats_path}')

    with open(args.task_json, 'r') as f:
        tasks = json.load(f)
    task = tasks[args.task_idx]

    stats = torch.load(stats_path, map_location='cpu')
    recall_list = stats['recall_list']
    reverse_recall_list = stats['reverse_recall_list']
    lamb_r_list = stats['lamb_r_list']
    top_k_idx_list = stats['top_k_idx_list']
    least_k_idx_list = stats['least_k_idx_list']

    eval_dir = Path(args.eval_dir) / result_dir.relative_to(Path(args.results_dir))
    figures_dir = Path(args.figures_dir) / result_dir.relative_to(Path(args.results_dir))
    eval_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_ds = LAIONVisDataset(f'{args.data_dir}/laion_subset')
    exemplar_ds = ExemplarVisDataset(args.data_dir, task['test_case'], task['test_case_ind'])
    num_exemplars = len(exemplar_ds)

    for lamb_r, top_k_idx in zip(lamb_r_list, top_k_idx_list):
        render_grid(top_k_idx[:args.top_k], exemplar_ds, train_ds, num_exemplars, figures_dir / f'topk-lambda_{lamb_r:.1e}.jpg')
    if args.render_leastk:
        for lamb_r, least_k_idx in zip(lamb_r_list, least_k_idx_list):
            render_grid(least_k_idx[:args.top_k], exemplar_ds, train_ds, num_exemplars, figures_dir / f'leastk-lambda_{lamb_r:.1e}.jpg')

    best_idx = int(np.argmax(recall_list))
    metrics = {
        'task_idx': args.task_idx,
        'lamb_r_list': [float(x) for x in lamb_r_list],
        'recall_list': [float(x) for x in recall_list],
        'reverse_recall_list': [float(x) for x in reverse_recall_list],
        'best_lambda_r': float(lamb_r_list[best_idx]),
        'best_recall': float(recall_list[best_idx]),
    }
    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
