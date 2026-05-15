conda create -n concept-trak python=3.10 -y
conda activate concept-trak
pip3 install --no-cache-dir \
    torch==2.8.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128
conda install conda-forge::cudatoolkit-dev -y
conda install nvidia::pytorch-cuda -y
conda install gcc_linux-64 gxx_linux-64 -y
pip3 install --no-cache-dir dattri[fast_jl]
pip3 install --no-cache-dir fast_jl

pip3 install --no-cache-dir \
    accelerate==0.26.1 \
    datasets==2.15.0 \
    "transformers>=4.25.1,<4.49.0" \
    diffusers==0.16.0 \
    numpy==1.26 \
    einops \
    pycocotools \
    tqdm \
    matplotlib \
    pandas \
    ml_collections \
    "huggingface_hub>=0.13.2,<0.26" \
    hf_transfer \
    timm==1.0.12 \
    torchdiffeq \
    pytorch_fid \
    fairscale \
    safetensors \
    wandb 

pip3 install einops numpy torch torchvision

pip3 install --no-cache-dir git+https://github.com/openai/CLIP.git
