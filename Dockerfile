## Build Container:

# Start From Base OS
ARG UBUNTU_VERSION=20.04
# ARG OS=ubuntu2204
# ARG cudnn_version=8.6.0.*
# ARG cuda_version=cuda11.8

ARG dockername=nvidia/cuda:11.7.1-runtime-ubuntu20.04
FROM ${dockername} AS build
# LABEL build=true
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
# Install Miniconda + Create Separate Base & Pipeline Envs
COPY ["./pkglist-build", "./env.yml", "./"]
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates gnupg \
    && apt-get install -y software-properties-common
RUN xargs -a pkglist-build apt-get install -y --no-install-recommends \
    && apt-get install ffmpeg libsm6 libxext6  -y \
    && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && chmod +x ./miniconda.sh \
    && ./miniconda.sh -b -p /opt/conda \
    && /opt/conda/bin/conda update conda \
    && /opt/conda/bin/conda init bash 
    


ENV PATH=opt/conda/bin:$PATH

RUN conda create -n env python=3.9

SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]
RUN conda install pip
RUN conda install tqdm scipy pyyaml jupyterlab pillow Cython boto3 -c defaults -c anaconda
RUN pip install jax==0.4.2 https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.2+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
##KEY: The below lines are ideal, but there's a bug with the distribution of Torch 2 right now, we can start using it once the bug is fixed
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# RUN pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip install jax==0.4.7 -f https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.7+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl
RUN pip install jaxopt line-profiler pynwb "dash[diskcache]" dash-extensions Flask-Caching dash-bootstrap-components dash_daq dash-bootstrap-templates git+https://github.com/j-friedrich/OASIS.git@f3ae85e1225bfa4bfe098a3f119246ac1e4f8481#egg=oasis opencv-python 
RUN conda install -c "nvidia/label/cuda-11.7.1" cuda-nvcc
RUN pip install git+https://github.com/apasarkar/jnormcorre.git@main git+https://github.com/apasarkar/localmd.git@main git+https://github.com/apasarkar/masknmf.git@deconv_jax git+https://github.com/apasarkar/rlocalnmf.git@remove_nonneg git+https://github.com/facebookresearch/detectron2.git


SHELL ["/opt/conda/bin/conda", "run", "-n", "base", "/bin/bash", "-c"]
# Turn Pipeline Env Into Standalone Venv For Portability
RUN /opt/conda/bin/conda install -y -n base conda-pack \
    && /opt/conda/bin/conda run -n base conda-pack --ignore-missing-files -n env -o /tmp/env.tar \
    && mkdir /venv \
    && cd /venv \
    && tar xf /tmp/env.tar \
    && rm /tmp/env.tar \
    && /opt/conda/bin/conda run -n base /venv/bin/conda-unpack

# # Restart From New Base OS + Standalone Env From Build
ARG UBUNTU_VERSION
FROM ${dockername} AS run
COPY --from=build /venv /venv

# # Get Pipeline Entrypoint Script & List Of Runtime Dependencies 

COPY ["./pkglist-run", "./entrypoint.sh", "./"]
COPY ["./app/*", "./app/"]

# # Install Any Additional Pipeline-Specific Runtime Dependencies
RUN apt-get update && xargs -a pkglist-run apt-get install -y --no-install-recommends && rm -rf /var/lib/apt/lists/*  && rm pkglist-run
ENTRYPOINT ["./entrypoint.sh"]
