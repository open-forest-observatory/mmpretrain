FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
# Accept conda terms of service
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true
RUN apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Create conda environment with Python, PyTorch, and CUDA toolkit
RUN conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y

# Install openmim and mmpretrain dependencies into the conda environment
RUN conda run -n open-mmlab pip install openmim tensorboard

# Set the container workdir
WORKDIR /app
# Copy files from the current directory into /app
COPY . /app

# Install the project with mim
RUN conda run -n open-mmlab mim install -e .

# Activate the conda environment by default for all subsequent commands
ENV PATH="/opt/conda/envs/open-mmlab/bin:$PATH"
