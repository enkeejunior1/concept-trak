import gc
import random
import argparse
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum
from utils import ExemplarDataset, seed_everything

def flush():
    torch.cuda.empty_cache()
    gc.collect()

def arg_parser():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_idx', type=int, required=True)
    parser.add_argument('--num_samples', type=int, default=256)
    parser.add_argument('--layer', type=str, required=True)
    parser.add_argument('--NFE', type=int, default=10)
    parser.add_argument('--f', type=str, default='')
    parser.add_argument('--concept_f', type=str, default='slider')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--ddim_inversion', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_split', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=256)
    parser.add_argument('--train_guidance_scale', type=float, default=7.5)
    parser.add_argument('--concept_guidance_scale', type=float, default=1.0)
    parser.add_argument('--p_pair', type=str, default='')
    parser.add_argument('--grad_dir', type=str, default=str(experiment_dir / 'results' / 'grads'))
    parser.add_argument('--results_dir', type=str, default=str(experiment_dir / 'results' / 'influence'))
    parser.add_argument('--figures_dir', type=str, default=str(experiment_dir / 'figures' / 'influence'))
    parser.add_argument('--data_dir', type=str, default=str(experiment_dir / 'data'))
    parser.add_argument('--task_json', type=str, default=str(experiment_dir / 'configs' / 'all_tasks.json'))
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--prompt', type=str, default='special')
    return parser.parse_args()

def resize_image(img, size):
    return img.resize((size, size), Image.LANCZOS)

def create_image_row(img_list):
    width = sum(img.width for img in img_list)
    height = max(img.height for img in img_list)
    row_image = Image.new('RGB', (width, height))
    
    x_offset = 0
    for img in img_list:
        row_image.paste(img, (x_offset, 0))
        x_offset += img.width
    return row_image

def create_image_grid(img_list, images_per_row=4):
    # Calculate number of rows needed
    num_images = len(img_list)
    num_rows = (num_images + images_per_row - 1) // images_per_row  # Ceiling division
    
    # Create rows
    rows = []
    for row_idx in range(num_rows):
        start_idx = row_idx * images_per_row
        end_idx = min(start_idx + images_per_row, num_images)
        row_images = img_list[start_idx:end_idx]
        
        # Create a single row
        row_width = sum(img.width for img in row_images)
        row_height = max(img.height for img in row_images)
        row_image = Image.new('RGB', (row_width, row_height))
        
        x_offset = 0
        for img in row_images:
            row_image.paste(img, (x_offset, 0))
            x_offset += img.width
        
        rows.append(row_image)
    
    # Combine all rows into a single image
    total_width = max(row.width for row in rows)
    total_height = sum(row.height for row in rows)
    grid_image = Image.new('RGB', (total_width, total_height))
    
    y_offset = 0
    for row in rows:
        grid_image.paste(row, (0, y_offset))
        y_offset += row.height
    
    return grid_image

seed_everything(42)
if __name__ == "__main__":
    # settings: args
    args = arg_parser()
    
    NFE = args.NFE
    f = args.f
    dtype = torch.float
    device = 'cuda'

    # settings: path
    task_json = args.task_json
    data_path = args.data_dir

    result_dir = f'{args.results_dir}/task_{args.task_idx}/{args.layer}-{args.f}-{args.concept_f}'
    if args.normalize:
        result_dir += '-norm'
    if args.ddim_inversion:
        result_dir += '-ddim'
    if args.concept_f == 'slider_local_1':
        result_dir += f'-train_gs_{args.train_guidance_scale}'
        result_dir += f'-concept_gs_{args.concept_guidance_scale}'
        result_dir += f'-eta_{args.eta}'
    if args.concept_f == 'slider_local_2':
        result_dir += f'-train_gs_{args.train_guidance_scale}'
        result_dir += f'-concept_gs_{args.concept_guidance_scale}'
        result_dir += f'-eta_{args.eta}'
    if args.concept_f == 'slider_seed':
        result_dir += f'-train_gs_{args.train_guidance_scale}'
        result_dir += f'-concept_gs_{args.concept_guidance_scale}'
        result_dir += f'-eta_{args.eta}'
    if args.prompt == 'special' and 'global' in args.concept_f and args.num_samples != 1 and not 'slider' in args.concept_f:
        result_dir += '-special'
    
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # load train, concept and task grad path
    grad_path_dict = {}
    num_split = args.num_split
    grad_path = f'{args.grad_dir}/{args.layer}-{args.f}-NFE{args.NFE}'
    if args.normalize:
        grad_path += '-norm'
    if args.ddim_inversion:
        grad_path += f'-ddim-gs_{args.train_guidance_scale}'
    grad_path_dict['task'] = f'{grad_path}/task_grad-{args.task_idx}.npy'
    grad_path_dict['train'] = [f'{grad_path}/train_grad-{split_idx}.npy' for split_idx in range(num_split)]

    concept_grad_path = f'{args.grad_dir}/{args.layer}-{args.concept_f}-NFE{args.NFE}'
    if args.normalize:
        concept_grad_path += '-norm'
    if args.ddim_inversion:
        concept_grad_path += f'-ddim-gs_{args.concept_guidance_scale}'
    if args.prompt == 'special' and 'global' in args.concept_f and args.num_samples != 1 and not 'slider' in args.concept_f:
        concept_grad_path += '-special'

    if args.concept_f == 'slider_local_1':
        grad_path_dict['concept'] = f'{concept_grad_path}/concept_grad-{args.task_idx}-eta_{args.eta}.npy'
    if args.concept_f == 'slider_local_2':
        grad_path_dict['concept'] = f'{concept_grad_path}/concept_grad-{args.task_idx}-eta_{args.eta}.npy'
    elif args.concept_f == 'slider_seed':
        grad_path_dict['concept'] = f'{concept_grad_path}/concept_grad-{args.task_idx}-eta_{args.eta}.npy'
    else:
        grad_path_dict['concept'] = f'{concept_grad_path}/concept_grad-{args.task_idx}.npy'

    # load task json 
    with open(task_json, 'r') as f:
        tasks = json.load(f)
    task = tasks[args.task_idx]
    test_case = task['test_case']
    test_case_ind = task['test_case_ind']

    with open(f'{data_path}/json/{test_case}.json', 'r') as f:
        exemplar_paths = json.load(f)[test_case_ind]['exemplar']

    # qualitative result of concept attribution 
    from utils import LAIONVisDataset, ExemplarVisDataset
    train_ds = LAIONVisDataset(f'{data_path}/laion_subset')
    examplar_ds = ExemplarVisDataset(data_path, task['test_case'], task['test_case_ind'])

    # load grad and compute kernel 
    concept_grad = torch.from_numpy(np.memmap(
        grad_path_dict['concept'], dtype=np.float32, mode='r',shape=(1, 2**15)
    )).to(device, dtype=dtype)
    task_grad = torch.from_numpy(np.memmap(
        grad_path_dict['task'], dtype=np.float32, mode='r',shape=(len(exemplar_paths), 2**15)
    )).to(device, dtype=dtype)
    train_grad = torch.cat([
        torch.from_numpy(np.memmap(
            grad_path_dict['train'][split_idx], dtype=np.float32, mode='r', shape=(100_000 // num_split, 2**15)
        ))
        for split_idx in range(num_split)
    ], dim=0).to(device, dtype=dtype)

    if args.f == 'dasv1':
        loss_path = f'{args.grad_dir}/loss-NFE{args.NFE}'
        loss_path_dict = {}
        loss_path_dict['task'] = f'{loss_path}/task_loss-{args.task_idx}.npy'
        loss_path_dict['train'] = [f'{loss_path}/train_loss-{split_idx}.npy' for split_idx in range(num_split)]
        train_loss = torch.cat([
            torch.from_numpy(np.memmap(
                loss_path_dict['train'][split_idx], dtype=np.float32, mode='r', shape=(100_000 // num_split)
            ))
            for split_idx in range(num_split)
        ], dim=0).to(device, dtype=dtype)

        task_loss = torch.from_numpy(np.memmap(
            loss_path_dict['task'], dtype=np.float32, mode='r',shape=(len(exemplar_paths))
        )).to(device, dtype=dtype)

        loss = torch.cat([task_loss, train_loss], dim=0)

    # check nan
    concept_nan_count = torch.isnan(concept_grad).any(dim=1).sum().item()
    task_nan_count = torch.isnan(task_grad).any(dim=1).sum().item()
    train_nan_count = torch.isnan(train_grad).any(dim=1).sum().item()
    assert concept_nan_count == 0 
    assert task_nan_count == 0 
    assert train_nan_count == 0

    concept_zero_count = (concept_grad == 0).all(dim=1).sum().item()
    task_zero_count = (task_grad == 0).all(dim=1).sum().item()
    train_zero_count = (train_grad == 0).all(dim=1).sum().item()
    assert concept_zero_count == 0 
    assert task_zero_count == 0 
    assert train_zero_count == 0, f'train_zero_count: {train_zero_count}'
    
    # heuristic lambda (saved in grad folder)
    kernel = train_grad.T @ train_grad
    heuristic_lamb_dir = f'{args.grad_dir}/{args.layer}-{args.f}-NFE{args.NFE}'
    if args.normalize:
        heuristic_lamb_dir += '-norm'
    if args.ddim_inversion:
        heuristic_lamb_dir += f'-ddim-gs_{args.train_guidance_scale}'
    heuristic_lamb_path = f'{heuristic_lamb_dir}/heuristic_lamb'
    heuristic_lamb_path += '.pt'

    if os.path.exists(heuristic_lamb_path):
        heuristic_lamb = torch.load(heuristic_lamb_path)
    else:
        heuristic_lamb = 0.1 * torch.linalg.eigh(kernel).eigenvalues.mean().item()
        torch.save(heuristic_lamb, heuristic_lamb_path)
    print(f'heuristic_lamb: {heuristic_lamb}')

    # Lambda sweeping 
    top_k = 10
    top_k_idx_list = []
    least_k_idx_list = []
    recall_list = []
    reverse_recall_list = []
    lamb_r_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    for lamb_r_idx, lamb_r in tqdm(enumerate(lamb_r_list), desc='Lambda sweeping'):
        lamb = lamb_r * heuristic_lamb

        # Add regularization to kernel
        kernel_inv_path = f'{heuristic_lamb_dir}/kernel_inv-{lamb_r_idx}.pt'
        try:
            kernel_inv = torch.load(kernel_inv_path, map_location=kernel.device).to(kernel)
        except:
            kernel_inv = torch.linalg.solve(
                kernel + lamb * torch.eye(kernel.shape[0]).to(kernel), 
                torch.eye(kernel.shape[0]).to(kernel)
            )
            torch.save(kernel_inv, kernel_inv_path)

        # compute influence score
        proj_train_grad = (train_grad @ kernel_inv).T
        train_scores = einsum(concept_grad, proj_train_grad, 'i k, k j -> j')

        # compute task influence score
        proj_task_grad = (task_grad @ kernel_inv).T
        task_scores = einsum(concept_grad, proj_task_grad, 'i k, k j -> j')

        # concat task and train scores
        scores = torch.cat([task_scores, train_scores], dim=0)
        if args.f == 'dasv1':
            scores = scores * loss

        # create figure
        fig, axs = plt.subplots(1, top_k, figsize=(20, 4))
        recall = 0
        top_k_idx = scores.argsort(descending=True)[:top_k]
        for i, idx in enumerate(top_k_idx):
            # Determine whether this idx is from exemplar or train dataset
            if idx < len(examplar_ds):
                img = examplar_ds[idx].resize((512, 512))
                axs[i].imshow(np.asarray(img))
                axs[i].add_patch(
                    plt.Rectangle(
                        (0, 0), img.size[0], img.size[1], 
                        fill=False, edgecolor='red', linewidth=3
                    )
                )
                axs[i].set_title(f'infl: {scores[idx]:.2e}', fontsize=8)
                recall += 1
            else:
                # This is from train dataset
                train_idx = idx - len(examplar_ds)
                img = train_ds[train_idx][0].resize((512, 512))
                axs[i].imshow(np.asarray(img))
                axs[i].set_title(f'infl: {scores[idx]:.2e}', fontsize=8)
            axs[i].axis('off')

        recall_i = recall / min(top_k, len(examplar_ds))
        recall_list.append(recall_i)
        print(f'task_idx_{args.task_idx} / Recall@K: {recall_i} / lambda_r: {lamb_r}')
        
        fig.suptitle(f'λ={lamb_r:.1e}', fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f'{result_dir}/concept_recall-{lamb_r:.1e}-recall_{recall_i:.2f}-topk.jpg', dpi=150, bbox_inches='tight')
        plt.close()

        # Create a figure with a specific size to ensure images are displayed larger
        fig, axs = plt.subplots(1, top_k, figsize=(20, 4))  # Increase figure width and set appropriate height
        reverse_recall = 0
        least_k_idx = scores.argsort(descending=False)[:top_k]
        for i, idx in enumerate(least_k_idx):
            # Determine whether this idx is from exemplar or train dataset
            if idx < len(examplar_ds):
                img = examplar_ds[idx].resize((512, 512))
                axs[i].imshow(np.asarray(img))
                axs[i].add_patch(
                    plt.Rectangle(
                        (0, 0), img.size[0], img.size[1], 
                        fill=False, edgecolor='red', linewidth=3
                    )
                )
                axs[i].set_title(f'infl: {scores[idx]:.2e}', fontsize=8)
                reverse_recall += 1
            else:
                # This is from train dataset
                train_idx = idx - len(examplar_ds)
                img = train_ds[train_idx][0].resize((512, 512))
                axs[i].imshow(np.asarray(img))
                axs[i].set_title(f'infl: {scores[idx]:.2e}', fontsize=8)
            axs[i].axis('off')

        reverse_recall_i = reverse_recall / min(top_k, len(examplar_ds))
        reverse_recall_list.append(reverse_recall_i)
        print(f'task_idx_{args.task_idx} / Reversed Recall@K: {reverse_recall_i} / lambda_r: {lamb_r}')
        
        fig.suptitle(f'λ={lamb_r:.1e}', fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f'{result_dir}/concept_recall-{lamb_r:.1e}-recall_{reverse_recall_i:.2f}-leastk.jpg', dpi=150, bbox_inches='tight')
        plt.close()

        top_k_idx_list.append(top_k_idx)
        least_k_idx_list.append(least_k_idx)

        del scores, proj_train_grad, proj_task_grad, kernel_inv
        flush()

    # save results
    torch.save({
        'recall_list': recall_list,
        'reverse_recall_list': reverse_recall_list,
        'lamb_r_list': lamb_r_list,
        'heuristic_lamb': heuristic_lamb,
        'top_k_idx_list': top_k_idx_list,
        'least_k_idx_list': least_k_idx_list,
    }, f'{result_dir}/recall_list.pt')