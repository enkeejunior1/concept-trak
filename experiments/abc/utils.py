import os
import numpy as np
import torch
import gc
import torch.nn.functional as F

def flush():
    gc.collect()
    torch.cuda.empty_cache()

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


OBJECTIVE_ALIASES = {
    'dsm': 'dsm',
    'ttrakv1': 'dsm',
    'dtrak': 'dtrak',
    'dtrakv1': 'dtrak',
    'das': 'das',
    'dasv1': 'das',
    'dps': 'dps',
    'dpsv1': 'dps',
}


def normalize_objective_name(name: str) -> str:
    try:
        return OBJECTIVE_ALIASES[name]
    except KeyError as exc:
        valid = ', '.join(sorted(set(OBJECTIVE_ALIASES.values())))
        raise ValueError(f'Invalid objective: {name}. Use one of: {valid}') from exc


def make_random_project_func(feature_dim, proj_dim, proj_max_batch_size, device, proj_type='random_mask', proj_seed=0):
    from dattri.func.projection import random_project

    sample_feature = torch.empty(1, feature_dim, device=device, dtype=torch.float32)
    return random_project(
        sample_feature,
        1,
        proj_dim=proj_dim,
        proj_max_batch_size=proj_max_batch_size,
        proj_seed=proj_seed,
        proj_type=proj_type,
        device=device,
    )
    
# ------------------------------------
# DPS
# ------------------------------------
@torch.no_grad()
def get_dps_guidance(model, xt, p_emb, t, x0, alpha_prod_t, loss_rescale=1e0):
    # set model to required_grad = False
    name_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad = False
            name_list.append(name)

    # xt -> x0_pred
    xt = xt.detach().requires_grad_(True)
    with torch.enable_grad():
        et = model(xt, t, p_emb).sample
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

# ------------------------------------
# Dataset
# ------------------------------------
class LaionDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot='data', subset_size=None, mode='no_flip'):
        self.dataroot = dataroot
        self.subset_size = subset_size

        self.latents = np.load(os.path.join(dataroot, f'laion_latents.npy'), mmap_mode='r')
        self.train_emb = np.load(os.path.join(dataroot, f'laion_text_embeddings.npy'), mmap_mode='r')
        self.num_captions = self.train_emb.shape[1]        

        self.mode = mode
        self.orig_length = len(self.train_emb)
        if mode == 'no_flip':
            self.length = self.orig_length if subset_size is None else subset_size
            self.latents = self.latents[:self.length]
        elif mode == 'flip':
            self.length = self.orig_length if subset_size is None else subset_size
            self.latents = self.latents[self.orig_length : self.orig_length + self.length]
        elif mode == 'no_flip_and_flip':
            self.length = self.orig_length * 2 if subset_size is None else subset_size * 2
            if subset_size is not None:
                latents_no_flip = self.latents[:subset_size]
                latents_flip = self.latents[self.orig_length : self.orig_length + subset_size]
                self.latents = np.concatenate([latents_no_flip, latents_flip], axis=0)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if subset_size is not None:
            self.train_emb = self.train_emb[:subset_size]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        latents_tensor = torch.from_numpy(self.latents[idx].copy()).float()

        hidx = idx % len(self.train_emb)
        hidden_states_tensor = torch.from_numpy(self.train_emb[hidx].copy()).float()

        return latents_tensor, hidden_states_tensor

import os
import re
import importlib
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple

class LAIONVisDataset:
    def __init__(self, path="data/laion_subset"):
        self.path_list = sorted([f'{path}/images/{x}' for x in os.listdir(f'{path}/images')])
        self.length = len(self.path_list)

        with open(f'{path}/captions.txt', 'r') as f:
            self.caption_list = [s.strip() for s in f.readlines()]

    def __getitem__(self, idx):
        # get image and caption
        img = Image.open(self.path_list[idx])
        img = img.convert('RGB')

        # center crop
        w, h = img.size
        if w > h:
            img = img.crop(((w - h) // 2, 0, (w + h) // 2, h))
        elif h > w:
            img = img.crop((0, (h - w) // 2, w, (h + w) // 2))

        caption = self.caption_list[idx]
        return img, caption

import json
import torch
from PIL import Image
from torchvision import transforms
from diffusers import DiffusionPipeline

class ExemplarDataset(torch.utils.data.Dataset):
    def __init__(self,
                 sd_version,
                 custom_diffusion_model_path,
                 test_case,
                 test_case_ind,
                 train_prompt,
                 vae_batch_size=20,
                 vae_device='cuda',
                 dataroot='data',
                 mode='no_flip_and_flip'
                 ):
        super().__init__()
        self.sd_version = sd_version
        self.model_path = custom_diffusion_model_path
        self.test_case = test_case
        self.test_case_ind = test_case_ind
        self.train_prompt = train_prompt
        self.dataroot = dataroot
        self.mode = mode

        self.exemplar_paths = self._get_exemplar_paths()
        self.latents, self.train_emb = self._prepare_exemplar_dataset(
            self.exemplar_paths, batch_size=vae_batch_size, device=vae_device
        )
        self.length = len(self.latents)
        self.train_emb_size = self.train_emb.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.latents[idx], self.train_emb[idx % self.train_emb_size]

    def _get_exemplar_paths(self):
        # load a dictionary of exemplars paths
        with open(f'{self.dataroot}/json/{self.test_case}.json', 'r') as f:
            exemplar_paths = json.load(f)[self.test_case_ind]['exemplar']
        return [s.replace('dataset', self.dataroot) for s in exemplar_paths]

    @torch.no_grad()
    def _prepare_exemplar_dataset(self, exemplar_paths, batch_size, device):
        # load pipeline
        pipeline = DiffusionPipeline.from_pretrained(self.sd_version).to(device)
        pipeline.load_textual_inversion(self.model_path, weight_name="new1.bin")
        pipeline.safety_checker = None
        vae = pipeline.vae
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder

        # get image transformation
        size = 512
        if self.mode == 'no_flip_and_flip' or self.mode == 'no_flip':
            image_transforms = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            images_noflip = [image_transforms(Image.open(exemplar_path).convert('RGB')) for exemplar_path in exemplar_paths]

        if self.mode == 'no_flip_and_flip' or self.mode == 'flip':
            image_transform_flip = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size),
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            images_flip = [image_transform_flip(Image.open(exemplar_path).convert('RGB')) for exemplar_path in exemplar_paths]

        if self.mode == 'no_flip_and_flip':
            images = torch.stack(images_noflip + images_flip).to(device)
        elif self.mode == 'no_flip':
            images = torch.stack(images_noflip).to(device)
        elif self.mode == 'flip':
            images = torch.stack(images_flip).to(device)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        num_exemplars = images.size(0)
        latents = []
        for start in range(0, num_exemplars, batch_size):
            end = min(start + batch_size, num_exemplars)
            batch_images = images[start:end]

            lats = vae.encode(batch_images).latent_dist.sample()
            lats = lats * vae.config.scaling_factor
            latents.append(lats.cpu())
        latents = torch.cat(latents, dim=0)

        # train text embedding
        tokens = tokenizer([self.train_prompt],
                            max_length=tokenizer.model_max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt")['input_ids'].to(device)
        train_emb = text_encoder(tokens)[0].cpu()
        train_emb = train_emb.expand(latents.size(0), -1, -1)

        del pipeline
        return latents, train_emb
    
class ExemplarVisDataset:
    def __init__(self, root, test_case, test_case_ind):
        self.path_list = self._get_exemplar_paths(root, test_case, test_case_ind)
        self.length = len(self.path_list)
    
    def _get_exemplar_paths(self, root, test_case, test_case_ind):
        # load a dictionary of exemplars paths
        with open(f'{root}/json/{test_case}.json', 'r') as f:
            exemplar_paths = json.load(f)[test_case_ind]['exemplar']
        return [s.replace('dataset', root) for s in exemplar_paths]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img = Image.open(self.path_list[idx])
        img = img.convert('RGB')

        # center crop
        w, h = img.size
        if w > h:
            img = img.crop(((w - h) // 2, 0, (w + h) // 2, h))
        elif h > w:
            img = img.crop((0, (h - w) // 2, w, (h + w) // 2))

        return img
    
from diffusers import DiffusionPipeline
def get_synth_latent_text_embed_ti(
        sd_version, custom_diffusion_paths, image_path=None, captions=None, device='cuda', dtype=torch.float):
    if isinstance(sd_version, str):
        pipeline = DiffusionPipeline.from_pretrained(sd_version).to(device, dtype=dtype)
        for weight_name, custom_diffusion_path in custom_diffusion_paths.items():
            pipeline.load_textual_inversion(custom_diffusion_path, weight_name=weight_name)
    else:
        pipeline = sd_version
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder

    # get image transformation
    if image_path is not None:
        size = 512
        image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        synth_image = image_transforms(Image.open(image_path)).unsqueeze(0).to(device, dtype=dtype)
        sample_latent = pipeline.vae.encode(synth_image).latent_dist.sample()[0].detach()
        latents = sample_latent.to(device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1, -1)
    else:
        latents = None

    if captions is not None:
        with torch.no_grad():
            tokens = tokenizer(
                captions, 
                max_length=tokenizer.model_max_length, padding="max_length", truncation=True, 
                return_tensors="pt"
            )['input_ids'].to(device)
            prompt_emb = text_encoder(tokens)[0]

        pipeline.to('cpu')
        del pipeline
        flush()
    else:
        prompt_emb = None
    return latents, prompt_emb

def get_synth_latent_text_embed(sd_version, custom_diffusion_path, image_path, train_caption, query_caption, batch_size, device='cuda', dtype=torch.float):
    if isinstance(sd_version, str):
        pipeline = DiffusionPipeline.from_pretrained(sd_version).to(device, dtype=dtype)
        pipeline.load_textual_inversion(custom_diffusion_path, weight_name="new1.bin")
    else:
        pipeline = sd_version
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder

    # get image transformation
    size = 512
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    synth_image = image_transforms(Image.open(image_path)).unsqueeze(0).to(device, dtype=dtype)
    sample_latent = pipeline.vae.encode(synth_image).latent_dist.sample()[0].detach()
    latents = sample_latent.to(device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1, -1)

    with torch.no_grad():
        tokens = tokenizer([train_caption, query_caption, ''],
                            max_length=tokenizer.model_max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt")['input_ids'].to(device)
        encoder_hidden_states = text_encoder(tokens)[0]
        train_emb, query_emb, base_emb = encoder_hidden_states.split(1, dim=0)
        train_emb, query_emb, base_emb = train_emb.expand(batch_size, -1, -1), query_emb.expand(batch_size, -1, -1), base_emb.expand(batch_size, -1, -1)

    pipeline.to('cpu')
    del pipeline
    flush()
    return latents, train_emb, query_emb, base_emb

# ------------------------------------
# Diffusion utils (Scheduler)
# ------------------------------------
@torch.no_grad()
def invert(
    x0, 
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
            et = unet(xt, vec_t, p_pos).sample
        elif guidance_scale == 0:
            et = unet(xt, vec_t, p_neg).sample
        else:
            et = unet(xt.repeat(2, 1, 1, 1), vec_t.repeat(2), torch.cat([p_pos, p_neg], dim=0)).sample
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
            et = unet(xt, vec_t, p_pos).sample
        elif guidance_scale == 0:
            et = unet(xt, vec_t, p_neg).sample
        else:
            et = unet(xt.repeat(2, 1, 1, 1), vec_t.repeat(2), torch.cat([p_pos, p_neg], dim=0)).sample
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
