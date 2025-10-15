# 1. Base Image
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# 2. Environment Setup
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home/ubuntu

# 3. Copy all necessary files (no tar's to keep down bloat)
COPY ./config/Miniconda3-latest-Linux-x86_64.sh /home/ubuntu/

# 4. Install dependencies and Miniconda
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    curl \
    libgl1-mesa-glx \
    libosmesa6-dev \
    tmux \
    libglib2.0-0 \
    unzip \
    vim \ 
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/ubuntu/miniconda3

# Set Conda environment variables
ENV PATH=/home/ubuntu/miniconda3/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

SHELL ["/bin/bash", "-c"]
RUN echo "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc

WORKDIR /home/ubuntu


