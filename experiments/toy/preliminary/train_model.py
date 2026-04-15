import os
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
import torchvision.models as models

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = THIS_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

try:
    import wandb
except ImportError:
    wandb = None

from utils import (
    SyntheticClassDataset,
    create_model,
    check_gpu_health_and_set_device,
    load_pipeline,
    Muon,
    MultiLabelResNet,
    num_conds,
    num_class,
)
from torch.optim import Adam

# Set default dtype
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')
concept_list = ['shape', 'color']

def train_model(model, ds, dl, noise_scheduler, optimizers, num_epochs=10, device='cuda', epoch=0, p_cfg=0.0):
    """
    Train DDPM model with AMP.
    """
    model.train()
    for batch_id, batch in enumerate(dl):
        x0, c = batch
        x0, c = x0.to(device), c.to(device)
        
        c = torch.nn.functional.one_hot(c.long(), num_classes=ds.num_class).float().flatten(start_dim=1)
        if random.random() < p_cfg:
            assert False 
            c = torch.zeros_like(c)

        # x0 -> xt
        t = torch.randint(0, noise_scheduler.num_train_timesteps, (x0.shape[0],), device=device)
        et = torch.randn(x0.shape, device=device)
        xt = noise_scheduler.add_noise(x0, et, t)
        
        # Forward pass
        et_theta = model(xt, t, c)
        loss = F.mse_loss(et_theta, et)
        
        # Backward pass
        if isinstance(optimizers, list):
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Report to the wandb
        if wandb is not None and wandb.run is not None:
            wandb.log({"loss": loss.item()})
    return

@torch.no_grad()
def evaluate_generation_accuracy(model, classifier, ds, device, pipe):
    """
    Generate images with condition [0, 0] and evaluate classification accuracy.
    """
    model.eval()
    classifier.eval()
    
    # Generate images with condition [0, 0] (triangle, red)
    batch_size = 128
    generator = torch.Generator(device=device).manual_seed(42)
    cond = torch.tensor([
        [0, 0], # triangle, red
    ] * batch_size, device=device)
    cond = torch.nn.functional.one_hot(cond.long(), num_classes=ds.num_class).float().flatten(start_dim=1)
    cond_neg = torch.zeros_like(cond, dtype=torch.float32).flatten(start_dim=1)
    
    images = pipe(
        batch_size=batch_size, num_inference_steps=50, 
        conditions=cond.float(), null_cond=cond_neg.float(), generator=generator, guidance_scale=1.0
    )
    
    # Convert PIL images to tensors for classification
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    image_tensors = []
    for img in images:
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        image_tensors.append(img_tensor)
    
    image_batch = torch.cat(image_tensors, dim=0).to(device)
    
    # Classify generated images
    logits = classifier(image_batch)
    
    # Get predictions
    preds = [torch.argmax(logit, dim=1) for logit in logits]
    
    # Expected labels are [0, 0] for all images
    expected_labels = torch.tensor([0, 0], device=device)
    
    # Calculate accuracies
    acc_list = []
    for i in range(len(preds)):
        acc = (preds[i] == expected_labels[i]).float().mean().item()
        acc_list.append(acc)
    overall_acc = sum(acc_list) / len(acc_list)
    
    model.train()
    return acc_list + [overall_acc]

import random
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=str(EXPERIMENT_DIR), help='toy experiment directory')
    parser.add_argument('--output_path', type=str, default=str(EXPERIMENT_DIR / 'weights' / 'model.bin'), help='output path to save trained model')
    parser.add_argument('--task', type=str, default='generation', help='task name')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--classifier_path', type=str, default=str(EXPERIMENT_DIR / 'weights' / 'classifier.bin'), help='path to pretrained classifier')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--adam_lr', type=float, default=1e-4, help='adam learning rate')
    parser.add_argument('--muon_lr', type=float, default=1e-2, help='muon learning rate')
    parser.add_argument('--use_muon', action='store_true', help='use muon optimizer')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--use_wandb', action='store_true', help='enable Weights & Biases logging')
    args = parser.parse_args()

    # path
    save_dir = os.path.dirname(args.output_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # set device
    device = check_gpu_health_and_set_device(0)
    
    # seed
    seed_everything(42)

    # load data loader
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    imgs_path = f"{args.base_dir}/data/images"
    attr_path = f"{args.base_dir}/data/labels/metadata.npy"
    ds = SyntheticClassDataset(imgs_path=imgs_path, attr_path=attr_path, num_conds=num_conds, num_class=num_class, transform=transform)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, 
        pin_memory=True, num_workers=8, prefetch_factor=2, persistent_workers=True, timeout=100, 
    )

    # Load pretrained classifier
    classifier = MultiLabelResNet(num_classes_per_attribute=num_class, num_attributes=num_conds).to(device)
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))
    classifier.eval()
    print(f"Loaded pretrained classifier from {args.classifier_path}")

    # Initialize wandb
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed but --use_wandb was set.")
        wandb.init(project="toy-gen", name=f"gen-adam_lr_{args.adam_lr}-muon_lr_{args.muon_lr}")

    # load model and noise scheduler
    from diffusers import DDPMScheduler
    model, noise_scheduler = create_model(device, cond_dim=ds.num_conds*ds.num_class), DDPMScheduler(num_train_timesteps=1000)

    if args.use_muon:
        b_params = [p for p in model.parameters() if p.ndim < 2]
        w_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embedder" not in n and "final_layer" not in n]
        em_params = [p for n, p in model.named_parameters() if ("embedder" in n or "final_layer" in n) and p.ndim >= 2]

        optimizer1 = Adam(b_params + em_params, lr=args.adam_lr, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
        optimizer2 = Muon(w_params, lr=args.muon_lr, momentum=0.95, weight_decay=0.0)
        optimizers = [optimizer1, optimizer2]
    
    else:
        optimizer = Adam(model.parameters(), lr=args.adam_lr)
        optimizers = [optimizer]
    
    # Load the Diffusion pipeline once
    pipe = load_pipeline(model, device)
    
    for epoch in tqdm(range(args.num_epochs)):
        train_model(
            model, ds, dl, noise_scheduler, optimizers,
            num_epochs=args.num_epochs, device=device, epoch=epoch, p_cfg=0.0
        )

        # Evaluate generation accuracy every epoch
        gen_acc_list = evaluate_generation_accuracy(
            model, classifier, ds, device, pipe
        )
        
        # Log generation accuracy to wandb
        log_dict = {}
        for i, concept in enumerate(concept_list):
            log_dict[f"gen_{concept}_acc"] = gen_acc_list[i]
        log_dict["gen_overall_acc"] = gen_acc_list[-1]
        if wandb is not None and wandb.run is not None:
            wandb.log(log_dict)
        
        log_str = f"Epoch {epoch}: "
        for concept_idx, acc in enumerate(gen_acc_list[:-1]):
            concept = concept_list[concept_idx]
            log_str += f"Gen {concept} Acc={acc:.4f}, "
        log_str += f"Gen Overall Acc={gen_acc_list[-1]:.4f}"
        print(log_str)

        # Generate sample images every 25 epochs
        if epoch % 25 == 0:
            print(f"Generating sample images at epoch {epoch}...")
            
            batch_size = 4
            generator = torch.Generator(device=device).manual_seed(42)
            cond = torch.tensor([
                [0, 0], # triangle, red
                [0, 0], # triangle, red
                [0, 0], # triangle, red
                [0, 0], # triangle, red
            ], device=device)
            cond = torch.nn.functional.one_hot(cond.long(), num_classes=ds.num_class).float().flatten(start_dim=1)
            cond_neg = torch.zeros_like(cond, dtype=torch.float32).flatten(start_dim=1)
            images = pipe(
                batch_size=batch_size, num_inference_steps=50, 
                conditions=cond.float(), null_cond=cond_neg.float(), generator=generator, guidance_scale=1.0
            )

            captions = batch_size * ["triangle, red"]
            if wandb is not None and wandb.run is not None:
                wandb.log({"images_fake": [wandb.Image(img, caption=caption) for img, caption in zip(images, captions)]})
                
        if epoch % 100 == 0:
            checkpoint_dir = os.path.join(os.path.dirname(args.output_path), f"checkpoint_epoch_{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save the model
            model_path = os.path.join(checkpoint_dir, f"model-{epoch}.bin")
            torch.save(model.state_dict(), model_path)
            print(f"Model checkpoint saved to {model_path}")
            
            # Save optimizer state
            for i, optimizer in enumerate(optimizers):
                optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{i}-{epoch}.bin")
                torch.save(optimizer.state_dict(), optimizer_path)
                print(f"Optimizer state saved to {optimizer_path}")

    # Save final model
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), args.output_path)
    print(f"Final model saved to {args.output_path}")

if __name__ == "__main__":
    main()