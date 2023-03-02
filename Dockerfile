## Build Container:

# Start From Base OS
ARG UBUNTU_VERSION=22.04
# ARG OS=ubuntu2204
# ARG cudnn_version=8.6.0.*
# ARG cuda_version=cuda11.8

ARG dockername=nvidia/cuda:11.6.0-devel-ubuntu20.04
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
    && /opt/conda/bin/conda install -c conda-forge mamba \
    && /opt/conda/bin/conda init bash 
    


ENV PATH=opt/conda/bin:$PATH

RUN conda create -n env python=3.9

SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]
RUN mamba install pytorch=1.11 torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
RUN mamba install pip
RUN mamba install pytorch-sparse -c pyg
RUN mamba install tqdm scipy pyyaml jupyterlab pillow Cython boto3 -c defaults -c anaconda
RUN pip install --upgrade "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install jaxopt torchnmf line-profiler pynwb "dash[diskcache]" dash-extensions Flask-Caching dash-bootstrap-components git+https://github.com/apasarkar/jnormcorre.git@full_jax git+https://github.com/apasarkar/localmd.git@random_svd_devel git+https://github.com/j-friedrich/OASIS.git@f3ae85e1225bfa4bfe098a3f119246ac1e4f8481#egg=oasis git+https://github.com/apasarkar/masknmf.git@deconv_jax git+https://github.com/apasarkar/rlocalnmf.git@remove_nonneg ipywidgets git+https://github.com/crahan/ipyfilechooser.git opencv-python git+https://github.com/facebookresearch/detectron2.git
# RUN /opt/conda/bin/conda clean --all --yes \
#     && apt-get -y remove wget ca-certificates gnupg \
#     && apt-get -y autoremove \
#     && apt-get autoclean \
#     && rm -rf /var/lib/apt/lists/* 


# /opt/conda/bin/conda clean --all --yes \
#     && apt-get -y remove wget ca-certificates gnupg \
#     && apt-get -y autoremove \
#     && apt-get autoclean \
#     && rm -rf /var/lib/apt/lists/* \


SHELL ["/opt/conda/bin/conda", "run", "-n", "base", "/bin/bash", "-c"]
# Turn Pipeline Env Into Standalone Venv For Portability
RUN /opt/conda/bin/conda install -y -n base conda-pack \
    && /opt/conda/bin/conda run -n base conda-pack --ignore-missing-files -n env -o /tmp/env.tar \
    && mkdir /venv \
    && cd /venv \
    && tar xf /tmp/env.tar \
    && rm /tmp/env.tar \
    && /opt/conda/bin/conda run -n base /venv/bin/conda-unpack

# # ## Runtime Container: 

# # Restart From New Base OS + Standalone Env From Build
ARG UBUNTU_VERSION
FROM ${dockername} AS run
COPY --from=build /venv /venv

# # Get Pipeline Entrypoint Script & List Of Runtime Dependencies 

COPY ["./pkglist-run", "./entrypoint.sh", "./"]
COPY ["./app/*", "./app/"]

# # Install Any Additional Pipeline-Specific Runtime Dependencies
RUN apt-get update && xargs -a pkglist-run apt-get install -y --no-install-recommends && rm -rf /var/lib/apt/lists/*  && rm pkglist-run

# ENTRYPOINT ["conda", "run", "-n", "env", "./run_notebook.sh"]
ENTRYPOINT ["./entrypoint.sh"]
