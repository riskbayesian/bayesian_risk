# 1. Base Image
FROM bayesianrisk/my-shared-base:latest

# 2. Conda Environment Setup
# Get the tar's and put them in the right place 
# =============================================================================================
RUN mkdir -p /home/ubuntu/miniconda3/envs/feature_splatting2

RUN curl -O http://127.0.0.1:8000/feature_splatting2.tar && \ 
    tar -xvf feature_splatting2.tar -C /home/ubuntu/miniconda3/envs/feature_splatting2 --strip-components=1 && \
    rm /home/ubuntu/feature_splatting2.tar 
# =============================================================================================

RUN conda config --append envs_dirs /home/ubuntu/miniconda3/envs

WORKDIR /home/ubuntu
