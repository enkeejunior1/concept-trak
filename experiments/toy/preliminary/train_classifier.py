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
    check_gpu_health_and_set_device,
    Muon,
)
from torch.optim import Adam

# Set default dtype
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')
concept_list = ['shape', 'color']
num_conds = 2
num_class = 2

class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes_per_attribute=num_class, num_attributes=num_conds):
        super(MultiLabelResNet, self).__init__()
        # Use ResNet18 as base and modify for small input
        self.backbone = models.resnet18(pretrained=False)
        
        # Modify first conv for 3-channel 64x64 input
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity() # Remove maxpool for small images
        
        # Replace final layer with separate heads for each attribute
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove original fc layer
        
        # Create separate classification heads for each attribute
        self.head_list = nn.ModuleList([
            nn.Linear(feature_dim, num_classes_per_attribute) 
            for _ in range(num_attributes)
        ])
        
    def forward(self, x):
        features = self.backbone(x)
        logits = [head(features) for head in self.head_list]
        return logits

def train_classifier(model, ds, dl, dl_val, optimizers, num_epochs=10, device='cuda', epoch=0):
    """
    Train multi-label classifier model.
    """
    model.train()
    progress_bar = tqdm(dl, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
    for batch_id, batch in enumerate(progress_bar):
        x, labels = batch
        x = x.to(device)
        labels = labels.to(device)  # [batch_size, num_conds] - shape, color
        
        # Forward pass through the model
        logits = model(x)
        
        # Compute losses
        losses = [F.cross_entropy(logit, labels[:, i]) for i, logit in enumerate(logits)]
        loss = torch.sum(torch.stack(losses))
        
        # Backward pass
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
        })
    
    # Validation
    model.eval()
    val_total_correct_list = [0 for _ in range(num_conds)]
    val_total_samples = 0
    with torch.no_grad():
        for batch in dl_val:
            x, labels = batch
            x = x.to(device)
            labels = labels.to(device)  # [batch_size, num_conds] - shape, color
            
            # Forward pass through the model
            logits = model(x)
            
            # Calculate accuracies
            preds = [torch.argmax(logit, dim=1) for logit in logits]
            for i in range(num_conds):
                val_total_correct_list[i] += (preds[i] == labels[:, i]).sum().item()
            val_total_samples += x.size(0)
    
        # Calculate validation accuracy
        val_acc_list = [val_total_correct_list[i] / val_total_samples for i in range(len(val_total_correct_list))]
        
        # Log validation results
        if wandb is not None and wandb.run is not None:
            wandb.log({
                concept: acc for concept, acc in zip(concept_list, val_acc_list)
            })
    
    model.train()
    return val_acc_list

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
    parser.add_argument('--output_path', type=str, default=str(EXPERIMENT_DIR / 'weights' / 'classifier.bin'), help='output path to save trained model')
    parser.add_argument('--task', type=str, default='classification', help='task name')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
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
    imgs_path = f"{args.base_dir}/data/images-classifier"
    attr_path = f"{args.base_dir}/data/labels-classifier/metadata.npy"
    ds = SyntheticClassDataset(imgs_path=imgs_path, attr_path=attr_path, num_conds=num_conds, num_class=num_class, transform=transform, split='train')
    ds_val = SyntheticClassDataset(imgs_path=imgs_path, attr_path=attr_path, num_conds=num_conds, num_class=num_class, transform=transform, split='val')
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, 
        pin_memory=True, num_workers=8, prefetch_factor=2, persistent_workers=True, timeout=100, 
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False, 
        pin_memory=True, num_workers=8, prefetch_factor=2, persistent_workers=True, timeout=100, 
    )
    # Initialize wandb
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed but --use_wandb was set.")
        wandb.init(project="toy-classifier", name=f"classifier-adam_lr_{args.adam_lr}-muon_lr_{args.muon_lr}")

    # Create single multi-label classifier model
    model = MultiLabelResNet(num_classes_per_attribute=num_class, num_attributes=num_conds).to(device)

    if args.use_muon:
        # Separate parameters by type
        b_params = [p for p in model.parameters() if p.ndim < 2]
        w_params = [p for p in model.parameters() if p.ndim >= 2]
        
        optimizer1 = Adam(b_params, lr=args.adam_lr, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
        optimizer2 = Muon(w_params, lr=args.muon_lr, momentum=0.95, weight_decay=0.0)
        optimizers = [optimizer1, optimizer2]
    else:
        # Simple Adam optimizer for the entire model
        optimizer = Adam(model.parameters(), lr=args.adam_lr)
        optimizers = [optimizer]
    
    best_overall_acc = 0.0
    for epoch in range(args.num_epochs):
        val_acc_list = train_classifier(
            model, ds, dl, dl_val, optimizers,
            num_epochs=args.num_epochs, device=device, epoch=epoch
        )
        val_overall_acc = sum(val_acc_list) / len(val_acc_list)

        log_str = f"Epoch {epoch}: "
        for concept_idx, val_acc in enumerate(val_acc_list):
            concept = concept_list[concept_idx]
            log_str += f"Concept {concept} Val Acc={val_acc:.4f}, "
        print(log_str)
            
        # Save checkpoint every 100 epochs
        if epoch % 100 == 0:
            checkpoint_dir = os.path.join(os.path.dirname(args.output_path), f"checkpoint_epoch_{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save the model
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model-{epoch}.bin"))
            print(f"Model checkpoint saved to {checkpoint_dir}")
            
            # Save optimizer states
            for i, optimizer in enumerate(optimizers):
                optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{i}-{epoch}.bin")
                torch.save(optimizer.state_dict(), optimizer_path)
        
        # Keep track of best model
        if val_overall_acc > best_overall_acc:
            best_overall_acc = val_overall_acc

    # Save final model
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    
    print(f"Final model saved: {args.output_path}")
    print(f"Best overall accuracy: {best_overall_acc:.4f}")

if __name__ == "__main__":
    main()