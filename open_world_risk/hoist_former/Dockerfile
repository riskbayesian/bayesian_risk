# 1. Base Image
FROM doku88/my-shared-base:latest

# 2. Conda Environment Setup
# Get the tar's and put them in the right place 
# =============================================================================================
RUN mkdir -p /home/ubuntu/miniconda3/envs/hoist2
RUN curl -O http://127.0.0.1:8000/hoist2.tar && \ 
    tar -xvf hoist2.tar -C /home/ubuntu/miniconda3/envs/hoist2 --strip-components=1 && \
    rm /home/ubuntu/hoist2.tar 
# =============================================================================================

RUN conda config --append envs_dirs /home/ubuntu/miniconda3/envs

WORKDIR /workspace

