import gc
import random
import time
import argparse
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import multiprocessing as mp
import concurrent.futures
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum
from utils import SyntheticClassDataset, seed_everything, num_conds, num_class

def flush():
    torch.cuda.empty_cache()
    gc.collect()

def arg_parser():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, required=False, default=0)
    parser.add_argument('--sample_idx', type=int, required=False, default=0)
    parser.add_argument('--shape_idx', type=int, required=False, default=-1)
    parser.add_argument('--color_idx', type=int, required=False, default=-1)
    parser.add_argument('--target_concept_dim', type=int, required=False, default=0)
    parser.add_argument('--target_concept_idx', type=int, required=False, default=0)
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
    parser.add_argument('--save_figure', action='store_true')
    parser.add_argument('--save_stats', action='store_true')
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--figure', action='store_true')
    parser.add_argument('--num_workers', type=int, default=min(mp.cpu_count(), 8), help='Number of workers for parallel figure generation')
    return parser.parse_args()


def build_result_dir(args):
    result_dir = os.path.join(
        args.results_dir,
        f'shape_{args.shape_idx}_color_{args.color_idx}'
        f'_target_dim_{args.target_concept_dim}_target_idx_{args.target_concept_idx}',
        f'{args.f}-{args.concept_f}'
    )
    if args.num_samples == 1:
        result_dir += f'-sample_{args.sample_idx}'
    if args.normalize:
        result_dir += '-norm'
    if args.ddim_inversion:
        result_dir += f'-ddim-gs_{args.train_gs}'
    if args.concept_f == 'slider':
        result_dir += f'-concept_gs_{args.concept_gs}'
    if args.concept_f in {'slider_local_1', 'slider_local_2', 'slider_seed'}:
        result_dir += f'-eta_{args.eta}'
        result_dir += f'-train_gs_{args.train_gs}'
        result_dir += f'-concept_gs_{args.concept_gs}'
    return Path(result_dir)

def create_figure_worker(args_tuple):
    """Worker function for parallel figure generation"""
    (lamb_r, recall, figure_type, indices, imgs_path, attr_path, 
     target_concept_dim, target_concept_idx, result_dir) = args_tuple
    
    try:
        # Create temporary dataset instance for this worker
        temp_ds = SyntheticClassDataset(
            imgs_path=imgs_path, attr_path=attr_path, 
            num_conds=num_conds, num_class=num_class, transform=None
        )
        
        fig, axs = plt.subplots(1, len(indices), figsize=(15, 2))
        if len(indices) == 1:
            axs = [axs]
            
        for i, idx in enumerate(indices):
            img, label = temp_ds[idx]
            axs[i].imshow(np.asarray(img))
            
            # Highlight if this sample has the target attributes
            if (label[target_concept_dim] == target_concept_idx):
                axs[i].add_patch(
                    plt.Rectangle(
                        (0, 0), img.size[0], img.size[1], 
                        fill=False, edgecolor='red', linewidth=3
                    )
                )
            axs[i].axis('off')

        fig.suptitle(f'λ={lamb_r:.1e}', fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        filename = f'{result_dir}/concept_recall-{figure_type}-{lamb_r:.1e}-recall_{recall:.2f}.jpg'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        return f"Saved {filename}"
    except Exception as e:
        return f"Error creating figure for λ={lamb_r}: {str(e)}"

def precompute_kernel_inversions(kernel, heuristic_lamb, lamb_r_list, heuristic_lamb_dir):
    """Precompute all kernel inversions that don't exist"""
    print("Checking/precomputing kernel inversions...")
    
    missing_inversions = []
    for lamb_r_idx, lamb_r in enumerate(lamb_r_list):
        kernel_inv_path = f'{heuristic_lamb_dir}/kernel_inv-{lamb_r_idx}.npy'
        if not os.path.exists(kernel_inv_path):
            missing_inversions.append((lamb_r_idx, lamb_r))
    
    if not missing_inversions:
        print("All kernel inversions already exist.")
        return
    
    print(f"Computing {len(missing_inversions)} missing kernel inversions...")
    
    for lamb_r_idx, lamb_r in tqdm(missing_inversions, desc="Computing kernel inversions"):
        lamb = lamb_r * heuristic_lamb
        kernel_inv_path = f'{heuristic_lamb_dir}/kernel_inv-{lamb_r_idx}.npy'
        
        kernel_inv = torch.linalg.solve(
            kernel + lamb * torch.eye(kernel.shape[0]).to(kernel), 
            torch.eye(kernel.shape[0]).to(kernel)
        )
        np.save(kernel_inv_path, kernel_inv.cpu().numpy())
        print(f"Computed kernel_inv for λ_r={lamb_r}")

def compute_all_influence_scores(train_grad, concept_grad, heuristic_lamb, lamb_r_list, heuristic_lamb_dir, loss=None):
    """Compute influence scores for all lambda values efficiently"""
    print("Computing influence scores for all lambda values...")
    
    all_scores = []
    all_top_k_idx = []
    all_least_k_idx = []
    top_k = 10
    
    for lamb_r_idx, lamb_r in tqdm(enumerate(lamb_r_list), desc="Computing scores"):
        # Load kernel inverse
        # kernel_inv_path = f'{heuristic_lamb_dir}/kernel_inv-{lamb_r_idx}.npy'
        # kernel_inv = torch.from_numpy(np.load(kernel_inv_path)).to(train_grad)
        kernel_inv = torch.linalg.solve(
            kernel + lamb_r * heuristic_lamb * torch.eye(kernel.shape[0]).to(kernel), 
            torch.eye(kernel.shape[0]).to(kernel)
        )
        
        # Compute influence scores
        proj_train_grad = (train_grad.cuda() @ kernel_inv.cuda()).T
        train_scores = einsum(concept_grad.cuda(), proj_train_grad, 'i k, k j -> j')

        if loss is not None:
            train_scores = train_scores * loss.cuda()
        
        # Get top-k and least-k indices
        top_k_idx = train_scores.argsort(descending=True)[:top_k].cpu()
        least_k_idx = train_scores.argsort(descending=False)[:top_k].cpu()
        
        all_scores.append(train_scores.cpu().numpy())
        all_top_k_idx.append(top_k_idx.cpu().numpy())
        all_least_k_idx.append(least_k_idx.cpu().numpy())
        
        # Free memory
        del proj_train_grad, kernel_inv
        torch.cuda.empty_cache()
    
    return all_scores, all_top_k_idx, all_least_k_idx

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
    dtype = torch.float32
    device = 'cuda'

    if args.num_samples != 1:
        if args.shape_idx != 9:
            args.target_concept_dim = 0
            args.target_concept_idx = args.shape_idx
        if args.color_idx != 9:
            args.target_concept_dim = 1
            args.target_concept_idx = args.color_idx

    # settings: path
    result_dir = build_result_dir(args)
    result_dir.mkdir(parents=True, exist_ok=True)

    # if os.path.exists(f'{result_dir}/recall_list.pt'):
    #     print(f'{result_dir}/recall_list.pt exists. Will skip.')
    #     exit(0)
    
    # load train and concept grad paths
    grad_path_dict = {}
    num_split = args.num_split
    grad_path = f'{args.grad_dir}/{args.f}-NFE{args.NFE}'
    if args.normalize:
        grad_path += '-norm'
    if args.ddim_inversion:
        grad_path += f'-ddim-gs_{args.train_gs}'
    grad_path_dict['train'] = [f'{grad_path}/train_grad-{split_idx}.npy' for split_idx in range(num_split)]

    concept_grad_path = f'{args.grad_dir}/{args.concept_f}-NFE{args.NFE}'
    if args.normalize:
        concept_grad_path += '-norm'
    if args.ddim_inversion:
        concept_grad_path += f'-ddim-gs_{args.concept_gs}'

    if args.shape_idx == 9 or args.color_idx == 9:
        grad_path_dict['concept'] = f'{concept_grad_path}/concept_grad-shape_{args.shape_idx}-color_{args.color_idx}.npy'
    elif args.concept_f == 'slider':
        grad_path_dict['concept'] = f'{concept_grad_path}/concept_grad-target_dim_{args.target_concept_dim}-target_idx_{args.target_concept_idx}-shape_{args.shape_idx}-color_{args.color_idx}.npy'
    elif args.concept_f == 'slider_local_1':
        grad_path_dict['concept'] = f'{concept_grad_path}/concept_grad-target_dim_{args.target_concept_dim}-target_idx_{args.target_concept_idx}-shape_{args.shape_idx}-color_{args.color_idx}-sample_{args.sample_idx}-eta_{args.eta}.npy'
    elif args.concept_f == 'slider_local_2':
        grad_path_dict['concept'] = f'{concept_grad_path}/concept_grad-target_dim_{args.target_concept_dim}-target_idx_{args.target_concept_idx}-shape_{args.shape_idx}-color_{args.color_idx}-sample_{args.sample_idx}-eta_{args.eta}.npy'
    elif args.concept_f == 'slider_seed':
        grad_path_dict['concept'] = f'{concept_grad_path}/concept_grad-target_dim_{args.target_concept_dim}-target_idx_{args.target_concept_idx}-shape_{args.shape_idx}-color_{args.color_idx}-sample_{args.sample_idx}-eta_{args.eta}.npy'
    elif args.num_samples == 1:
        grad_path_dict['concept'] = f'{concept_grad_path}/concept_grad-shape_{args.shape_idx}-color_{args.color_idx}-sample_{args.sample_idx}.npy'
    elif args.num_samples == 0:
        grad_path_dict['concept'] = f'{concept_grad_path}/concept_grad-shape_{args.shape_idx}-color_{args.color_idx}.npy'
    else:
        raise ValueError(f'Invalid num_samples: {args.num_samples}')

    # load synthetic dataset for visualization
    import torchvision.transforms as transforms
    imgs_path = f"{args.base_dir}/data/images"
    attr_path = f"{args.base_dir}/data/labels/metadata.npy"
    train_ds = SyntheticClassDataset(
        imgs_path=imgs_path, attr_path=attr_path, num_conds=num_conds, num_class=num_class, transform=None
    )

    # load grad and compute kernel 
    split_size = len(train_ds) // num_split
    train_grad = torch.cat([
        torch.from_numpy(np.memmap(
            grad_path_dict['train'][split_idx], dtype=np.float32, mode='r', shape=(split_size, 2**15)
        ))
        for split_idx in range(num_split)
    ], dim=0).to(device, dtype=dtype)
    concept_grad = torch.from_numpy(np.memmap(
        grad_path_dict['concept'], dtype=np.float32, mode='r', shape=(1, 2**15)
    )).to(device, dtype=dtype)

    if args.f == 'dasv1':
        loss_path = f'{args.grad_dir}/loss-NFE{args.NFE}'
        loss_path_dict = {}
        loss_path_dict['train'] = [f'{loss_path}/train_loss-{split_idx}.npy' for split_idx in range(num_split)]
        train_loss = torch.cat([
            torch.from_numpy(np.memmap(
                loss_path_dict['train'][split_idx], dtype=np.float32, mode='r', shape=(split_size)
            ))
            for split_idx in range(num_split)
        ], dim=0).to(device, dtype=dtype)
        loss = train_loss

    # check all nan, zero rows
    concept_nan_count = torch.isnan(concept_grad).any(dim=1).sum().item()
    train_nan_count = torch.isnan(train_grad).any(dim=1).sum().item()
    if concept_nan_count == 0 and train_nan_count == 0:
        print(f'concept_nan_count: {concept_nan_count}, train_nan_count: {train_nan_count}')
    else:
        raise ValueError(f'concept_nan_count: {concept_nan_count}, train_nan_count: {train_nan_count}')
    
    concept_all_zero_count = (concept_grad==0).all(dim=1).sum().item()
    train_all_zero_count = (train_grad==0).all(dim=1).sum().item()
    if concept_all_zero_count == 0 and train_all_zero_count == 0:
        print(f'concept_all_zero_count: {concept_all_zero_count}, train_all_zero_count: {train_all_zero_count}')
    else:
        raise ValueError(f'concept_all_zero_count: {concept_all_zero_count}, train_all_zero_count: {train_all_zero_count}')
    
    # heuristic lambda (saved in grad folder)
    kernel = train_grad.T @ train_grad
    heuristic_lamb_dir = f'{args.grad_dir}/{args.f}-NFE{args.NFE}'
    if args.normalize:
        heuristic_lamb_dir += '-norm'
    if args.ddim_inversion:
        heuristic_lamb_dir += f'-ddim-gs_{args.train_gs}'
    heuristic_lamb_path = f'{heuristic_lamb_dir}/heuristic_lamb.pt'

    if os.path.exists(heuristic_lamb_path):
        heuristic_lamb = torch.load(heuristic_lamb_path)
    else:
        heuristic_lamb = 0.1 * torch.linalg.eigh(kernel).eigenvalues.mean().item()
        torch.save(heuristic_lamb, heuristic_lamb_path)
    print(f'heuristic_lamb: {heuristic_lamb}')

    # Lambda sweeping 
    top_k = 10
    lamb_r_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e+1, 1e+2, 1e+3, 1e+4]
    # if args.f == 'dtrakv1':
    #     lamb_r_list = [1e2]
    # if args.f == 'dasv1':
    #     lamb_r_list = [1e3]
    # if args.f == 'ttrakv1':
    #     lamb_r_list = [1e4]
    # if args.f == 'dpsv1':
    #     lamb_r_list = [1e2]
    
    target_attributes = [None, None]
    target_attributes[args.target_concept_dim] = args.target_concept_idx
    
    # if os.path.exists(f'{result_dir}/recall_list.pt'):
    #     print(f'{result_dir}/recall_list.pt exists. Will skip.')
    #     exit(0)
    
    # Precompute all kernel inversions
    # precompute_kernel_inversions(kernel, heuristic_lamb, lamb_r_list, heuristic_lamb_dir)
    
    # Compute all influence scores at once
    all_scores, all_top_k_idx, all_least_k_idx = compute_all_influence_scores(
        train_grad, concept_grad, heuristic_lamb, lamb_r_list, heuristic_lamb_dir, 
        loss if args.f == 'dasv1' else None
    )
    
    # Free memory from large tensors
    del train_grad, concept_grad, kernel
    if args.f == 'dasv1':
        del loss
    flush()
    
    # Process results
    recall_list = []
    reverse_recall_list = []
    figure_tasks = []
    
    for lamb_r_idx, (lamb_r, scores, top_k_idx, least_k_idx) in enumerate(zip(lamb_r_list, all_scores, all_top_k_idx, all_least_k_idx)):
        # Calculate recalls
        concept_samples_in_topk = sum(1 for idx in top_k_idx if train_ds[idx][1][args.target_concept_dim] == args.target_concept_idx)
        recall_i = concept_samples_in_topk / top_k
        recall_list.append(recall_i)
        
        concept_samples_in_leastk = sum(1 for idx in least_k_idx if train_ds[idx][1][args.target_concept_dim] == args.target_concept_idx)
        reverse_recall_i = concept_samples_in_leastk / top_k
        reverse_recall_list.append(reverse_recall_i)
        
        print(f'Target: {args.shape_idx}, {args.color_idx} / Recall@K: {recall_i} / lambda_r: {lamb_r}')
        print(f'Target: {args.shape_idx}, {args.color_idx} / Reversed Recall@K: {reverse_recall_i} / lambda_r: {lamb_r}')
        
        # Prepare figure generation tasks
        if args.save_figure:
            figure_tasks.extend([
                (lamb_r, recall_i, 'topk', top_k_idx, imgs_path, attr_path, 
                 args.target_concept_dim, args.target_concept_idx, result_dir),
                (lamb_r, reverse_recall_i, 'leastk', least_k_idx, imgs_path, attr_path, 
                 args.target_concept_dim, args.target_concept_idx, result_dir)
            ])
    
    # Generate figures in parallel
    if args.save_figure and figure_tasks:
        print(f"Generating {len(figure_tasks)} figures in parallel using {args.num_workers} workers...")
        
        # Use ProcessPoolExecutor for CPU-bound figure generation
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            figure_results = list(tqdm(
                executor.map(create_figure_worker, figure_tasks),
                total=len(figure_tasks),
                desc="Generating figures"
            ))
        
        for result in figure_results:
            if "Error" in result:
                print(result)

    # save results
    torch.save({
        'recall_list': recall_list,
        'reverse_recall_list': reverse_recall_list,
        'lamb_r_list': lamb_r_list,
        'heuristic_lamb': heuristic_lamb,
        'top_k_idx_list': [torch.from_numpy(idx) for idx in all_top_k_idx],
        'least_k_idx_list': [torch.from_numpy(idx) for idx in all_least_k_idx],
        'score_list': all_scores,
        'target_attributes': target_attributes,
    }, f'{result_dir}/recall_list.pt')
        
    print("Processing completed!")