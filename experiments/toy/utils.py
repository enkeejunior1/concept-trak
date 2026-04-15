import os
import gc
import random
import json
from pathlib import Path
from itertools import chain
from PIL import Image
import numpy as np
from tqdm import tqdm
from diffusers import UNet2DConditionModel, DDPMScheduler
from safetensors.torch import load_file

import torch.nn.functional as F
import torch
import pandas as pd
from torch.utils import data

def flush():
    torch.cuda.empty_cache()
    gc.collect()

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
# -----------------------------------------------------------------------------
# Model
import torch.nn as nn
import torchvision.models as models

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

# -----------------------------------------------------------------------------
# Dataset

class SyntheticClassDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path, attr_path, num_conds=2, num_class=2, transform=None, split='all'):
        self.imgs_path = imgs_path
        self.attr_path = attr_path
        self.transform = transform
        self.num_conds = num_conds
        self.num_class = num_class
        self.samples, self.targets = self.make_dataset()
        if split == 'all':
            self.samples = self.samples
            self.targets = self.targets
        elif split == 'train':
            self.samples = self.samples[:6666]
            self.targets = self.targets[:6666]
        elif split == 'val':
            self.samples = self.samples[6666:]
            self.targets = self.targets[6666:]
        else:
            raise ValueError(f"Invalid split: {split}")

    def make_dataset(self):
        # load image idx
        images_idx = os.listdir(self.imgs_path)
        images_idx = sorted(images_idx, key=lambda x: int(x.split('.')[0]))
        images = [os.path.join(self.imgs_path, id_) for id_ in images_idx]
        
        # load metadata 
        labels = np.load(self.attr_path, allow_pickle=True)
        labels = labels[:, :self.num_conds]
        # assert len(images) == len(labels) == 30000
        return images, labels

    def __getitem__(self, index):
        image_path = self.samples[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.targets[index])
        return image, label

    def __len__(self):
        assert len(self.samples) == len(self.targets)
        return len(self.targets)


class SyntheticTextDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path, attr_path, p_cfg=0.1, transform=None):
        self.imgs_path = imgs_path
        self.attr_path = attr_path
        self.transform = transform
        self.samples, self.targets = self.make_dataset()
        self.p_cfg = p_cfg

    def make_dataset(self):
        # load image idx
        images_idx = os.listdir(self.imgs_path)
        images_idx = sorted(images_idx, key=lambda x: int(x.split('.')[0]))
        images = [os.path.join(self.imgs_path, id_) for id_ in images_idx]
        
        # load metadata 
        labels = np.load(self.attr_path).astype(np.float32)
        assert len(images) == len(labels) == 30000
        return images, labels

    def __getitem__(self, index):
        image_path = self.samples[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.targets[index])
        if random.random() < self.p_cfg:
            label = torch.zeros_like(label)
        return image, label

    def __len__(self):
        assert len(self.samples) == len(self.targets)
        return len(self.targets)


from model import LightningDiT_XS_4
def create_model(device, is_t2i=False, **kwargs):
    model = LightningDiT_XS_4(
        input_size=64,
        in_channels=3,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        use_mlp_y_embedder=is_t2i,
        **kwargs,
    )
    model = model.to(device)
    return model

def check_gpu_health_and_set_device(rank):
    """Check if GPU is healthy and set device for this process."""
    try:
        # Test GPU health by performing a simple operation
        test_tensor = torch.tensor([1.0], device=f'cuda:{rank}')
        test_result = test_tensor * 2
        del test_tensor, test_result
        torch.cuda.empty_cache()
        
        # Set device for this process
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        print(f"GPU {rank} health check passed")
        return device
        
    except RuntimeError as e:
        print(f"GPU {rank} error: {e}")
        raise e



# -----------------------------------------------------------------------------
# DPS guidance
@torch.no_grad()
def get_dps_guidance(model, xt, c, t, x0, alpha_prod_t, loss_rescale=1e0):
    # set model to required_grad = False
    name_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad = False
            name_list.append(name)

    # xt -> x0_pred
    xt = xt.detach().requires_grad_(True)
    with torch.enable_grad():
        et = model(xt, t, c)
        x0_pred = (xt - (1 - alpha_prod_t) ** (0.5) * et) / alpha_prod_t ** (0.5)

        # x0_pred -> nabla_xt x0_pred
        loss = loss_rescale * F.mse_loss(x0_pred, x0)
        loss.backward()
        guidance = -xt.grad.detach().clone()

    # Clean up
    xt.grad = None
    del x0_pred, et, loss
    flush()

    # set model to required_grad = True
    for name in name_list:
        model.get_parameter(name).requires_grad = True
    return guidance

# -----------------------------------------------------------------------------
# DDIM inversion

@torch.no_grad()
def invert(
    x0, 
    NFE, 
    unet, 
    p_pos, 
    p_neg=None,
    scheduler=None, 
    timesteps=None, 
    device='cuda', 
    dtype=torch.float16,
    guidance_scale=7.5,
    eta=0.0,
    p_neg_gen=None,
):
    if scheduler is None:
        scheduler = CustomScheduler(alphas_cumprod=scheduler.alphas_cumprod, device=device, dtype=dtype)

    # set timesteps
    if timesteps is None:
        scheduler.set_timesteps(NFE, device=device, is_inversion=True)
        timesteps = scheduler.timesteps
    else:
        raise ValueError('timesteps must be _not_ provided')

    # start!
    t_traj = []
    xt_traj = []
    noise_traj = []
    xt = x0.to(device, dtype=dtype)
    for i, t in enumerate(timesteps):
        if i == len(timesteps) - 1:
            break

        # 1. predict noise model_output
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor([t] * xt.shape[0]).to(device, dtype=torch.long)
        elif isinstance(t, torch.Tensor):
            vec_t = t.repeat(xt.shape[0])
        
        # CFG 
        if guidance_scale == 1:
            et = unet(xt, vec_t, p_pos)
        elif guidance_scale == 0:
            et = unet(xt, vec_t, p_neg)
        else:
            p_neg = p_neg_gen(p_pos) if p_neg_gen is not None else p_neg
            et = unet(xt.repeat(2, 1, 1, 1), vec_t.repeat(2), torch.cat([p_pos, p_neg], dim=0))
            et_pos, et_neg = et.split(xt.shape[0], dim=0)
            et = et_neg + guidance_scale * (et_pos - et_neg)

        # 2. do x_t -> x_t-1
        output = scheduler.step(
            et, t, xt, eta=eta, x0=x0
        )
        xt = output.prev_sample.to(et)
        noise = output.et.to(et)
        
        t_traj.append(scheduler.timesteps_next[i].item())
        xt_traj.append(xt)
        noise_traj.append(noise)
        
    return xt, {'xt': xt_traj, 't': t_traj, 'noise': noise_traj}

@torch.no_grad()
def forward(
    xT, 
    NFE, 
    unet, 
    p_pos, 
    p_neg,
    scheduler=None, 
    timesteps=None, 
    device='cuda', 
    dtype=torch.float16,
    guidance_scale=7.5,
    eta=0.0,
):
    if scheduler is None:
        scheduler = CustomScheduler(alphas_cumprod=scheduler.alphas_cumprod, device=device, dtype=dtype)

    # set timesteps
    if timesteps is None:
        scheduler.set_timesteps(NFE, device=device, is_inversion=False)
        timesteps = scheduler.timesteps
    else:
        raise ValueError('timesteps must be _not_ provided')

    # start!
    t_traj = []
    xt_traj = []
    noise_traj = []
    xt = xT.to(device, dtype=dtype)
    for i, t in enumerate(timesteps):
        if i == len(timesteps) - 1:
            break

        # 1. predict noise model_output
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor([t] * xt.shape[0]).to(device, dtype=torch.long)
        elif isinstance(t, torch.Tensor):
            vec_t = t.repeat(xt.shape[0])
        
        # CFG 
        if guidance_scale == 1:
            et = unet(xt, vec_t, p_pos)
        elif guidance_scale == 0:
            et = unet(xt, vec_t, p_neg)
        else:
            assert False, 'CFG not supported'
            et = unet(xt.repeat(2, 1, 1, 1), vec_t.repeat(2), torch.cat([p_pos, p_neg], dim=0))
            et_pos, et_neg = et.split(xt.shape[0], dim=0)
            et = et_neg + guidance_scale * (et_pos - et_neg)

        # 2. do x_t -> x_t-1
        output = scheduler.step(
            et, t, xt, eta=eta if i != 0 else eta
        )
        xt = output.prev_sample.to(et)
        noise = output.et.to(et)
        
        t_traj.append(scheduler.timesteps_next[i].item())
        xt_traj.append(xt)
        noise_traj.append(noise)
        
    return xt, {'xt': xt_traj, 't': t_traj, 'noise': noise_traj}

class SchedulerOutput(object):
    def __init__(self, xt_next, P_xt, et=None):
        self.prev_sample = xt_next
        self.x0 = P_xt
        self.et = et
        
class CustomScheduler(object):
    def __init__(self, alphas_cumprod, device, dtype):
        # NOTE : verify this
        self.t_max = 999
        self.timesteps = None
        self.learn_sigma = False
        self.device = device
        self.dtype = dtype

        # get SNR schedule
        # self.get_alphas_cumprod()
        self.alphas_cumprod = alphas_cumprod.to(device, dtype=dtype)

    def set_timesteps(self, num_inferences, device=None, is_inversion=False):
        device = 'cpu' if device is None else device
        if is_inversion:
            seq = torch.linspace(0, 1, num_inferences, device=device) * self.t_max
            seq = seq + 1e-6
            seq_prev = torch.cat([torch.tensor([-1], device=device), seq[:-1]], dim = 0)
            self.timesteps = seq_prev[1:].long()
            self.timesteps_next = seq[1:].long()
            self.is_inversion = True
        else:
            seq = torch.linspace(0, 1, num_inferences, device=device) * self.t_max
            seq_prev = torch.cat([torch.tensor([-1], device=device), seq[:-1]], dim = 0)
            self.timesteps = reversed(seq[1:]).long()
            self.timesteps_next = reversed(seq_prev[1:]).long()
            self.is_inversion = False

    def step(self, et, t, xt, eta=0.0, x0=None, **kwargs):
        '''
        Notation
            - a : alpha / b : beta / e : epsilon
        '''
        if self.learn_sigma:
            et, logvar = torch.split(et, et.shape[1] // 2, dim=1)
        else:
            logvar = None
        assert et.shape == xt.shape, 'et, xt shape should be same'

        t_idx   = self.timesteps.tolist().index(t)
        t_next  = self.timesteps_next[t_idx]
        
        # extract need parameters : at, at_next
        at = extract(self.alphas_cumprod, t, xt.shape)
        at_next = extract(self.alphas_cumprod, t_next, xt.shape)

        # DDIM step ; xt-1 = sqrt(at-1 / at) (xt - sqrt(1-at)*e(xt, t)) + sqrt(1-at-1)*e(xt, t)
        P_xt = (xt - et * (1 - at).sqrt()) / at.sqrt()

        # Deterministic.
        if eta == 0:
            D_xt = (1 - at_next).sqrt() * et
            xt_next = at_next.sqrt() * P_xt + D_xt

        # Add noise. When eta is 1 and time step is 1000, it is equal to ddpm.
        elif logvar is None:
            if self.is_inversion:
                sigma_t = eta * ((1 - at) / (1 - at_next) * (1 - at_next / at)).sqrt()
            else:
                sigma_t = eta * ((1 - at_next) / (1 - at) * (1 - at / at_next)).sqrt()
            D_xt = (1 - at_next - sigma_t ** 2).sqrt() * et
            xt_next = at_next.sqrt() * P_xt + D_xt + sigma_t * torch.randn_like(xt)

        elif logvar is not None:
            bt = extract(self.betas, t, xt.shape)
            
            mean = 1 / torch.sqrt(1.0 - bt) * (xt - bt / torch.sqrt(1 - at) * et)
            xt_next = mean + torch.exp(0.5 * logvar) * torch.randn_like(xt, device=xt.device, dtype=xt.dtype)
            P_xt = None

        if x0 is not None:
            et = (xt - at.sqrt() * x0) / (1 - at).sqrt()

        return SchedulerOutput(xt_next, P_xt, et)

def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    if isinstance(t, int):
        t = torch.tensor([t])
        t = t.repeat(x_shape[0])
    elif isinstance(t, torch.Tensor):
        t = t.repeat(x_shape[0])
    else:
        raise ValueError(f"t must be int or torch.Tensor, got {type(t)}")
    bs, = t.shape
    assert x_shape[0] == bs, f"{x_shape[0]}, {t.shape}"
    out = torch.gather(a, 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out

# -----------------------------------------------------------------------------
# Pipeline

class CustomConditionalPipeline:
    def __init__(self, unet, scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.device = next(self.unet.parameters()).device
        self.dtype = next(self.unet.parameters()).dtype
        
    def __call__(self, 
        batch_size=1, num_inference_steps=50, conditions=None, null_cond=None, generator=None, guidance_scale=3.0,
        return_init_noise=False,
    ):
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Create random noise
        sample = torch.randn(
            (batch_size, 3, 64, 64),  # Assuming 64x64 RGB images
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        if return_init_noise:
            init_noise = sample
        conditions = conditions.to(device=self.device, dtype=self.dtype)
        null_cond = null_cond.to(device=self.device, dtype=self.dtype)

        # Denoising loop
        for t in self.scheduler.timesteps:
            t = t.to(self.device)

            # Predict noise
            if guidance_scale == 1.0:
                with torch.no_grad():
                    noise_pred = self.unet(
                        sample, 
                        t.unsqueeze(0).repeat(batch_size), 
                        conditions,
                    )
            elif guidance_scale == 0.0:
                with torch.no_grad():
                    noise_pred = self.unet(
                        sample, 
                        t.unsqueeze(0).repeat(batch_size), 
                        null_cond,
                    )
            else:
                with torch.no_grad():
                    noise_pred = self.unet(
                        torch.cat([sample, sample], dim=0),
                        t.unsqueeze(0).repeat(2*batch_size), 
                        torch.cat([conditions, null_cond], dim=0),
                    )
                    noise_cond, noise_null_cond = noise_pred.chunk(2)
                    noise_pred = noise_null_cond + guidance_scale * (noise_cond - noise_null_cond)
            
            # Compute previous sample
            sample = self.scheduler.step(noise_pred, t, sample).prev_sample

        sample = (sample + 1) / 2
        sample = (sample * 255).clamp(0, 255).to(torch.uint8)
        sample = sample.cpu().numpy().transpose(0, 2, 3, 1)
        
        images = []
        for i in range(sample.shape[0]):
            img = Image.fromarray(sample[i])
            images.append(img)
        
        if return_init_noise:
            return images, init_noise
        else:
            return images
    
def load_pipeline(unet, device):
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    pipeline = CustomConditionalPipeline(unet=unet, scheduler=scheduler)
    return pipeline


# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    
                momentum_buffer = state["momentum_buffer"]
                p.mul_(1 - eff_weight_decay)
                momentum_buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(momentum_buffer, momentum)
                
                # Only orthogonalize 2D parameters with ndim >= 2
                if grad.ndim >= 2:
                    v = zeropower_via_newtonschulz5(grad.float(), 5)  # Use float instead of bfloat16 for single GPU
                else:
                    v = grad
                    
                p.add_(other=v, alpha=-eff_lr)

def check_gpu_health_and_set_device(device_id=0):
    """Check GPU health and set device for single GPU training"""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = f'cuda:{device_id}'
        print(f"Using GPU: {device}")
        return device
    else:
        print("CUDA not available, using CPU")
        return 'cpu'