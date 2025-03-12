# Use an NVIDIA CUDA base image (with devel for building/compiling)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Use bash for all RUN commands so that conda activation works properly
SHELL ["/bin/bash", "-c"]

# Set noninteractive mode for apt-get to avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0.0

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Clone the vggsfm repository
RUN git clone https://github.com/Vlhermitte/vggsfm.git /opt/vggsfm

# Set working directory
WORKDIR /opt/vggsfm

# Run the installation script from the repository
# RUN bash install.sh

ENV ENV_NAME=vggsfm_tmp
ENV PYTHON_VERSION=3.10
ENV PYTORCH_VERSION=2.1.0
ENV CUDA_VERSION=12.1

# Create the conda environment with Python 3.10
RUN conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Update PATH so that commands run in the environment
ENV PATH=$CONDA_DIR/envs/$ENV_NAME/bin:$PATH

RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    conda install pytorch=$PYTORCH_VERSION torchvision pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia -y && \
    # conda install pytorch3d=0.7.5 -c pytorch3d -y && \
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y

RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME && \
    pip install hydra-core --upgrade

RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME && \
    pip install omegaconf opencv-python einops visdom tqdm scipy plotly scikit-learn imageio[ffmpeg] gradio trimesh huggingface_hub

RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME && \
    git clone https://github.com/jytime/LightGlue.git dependency/LightGlue && \
    pip install -e dependency/LightGlue/

RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME && \
    pip install numpy==1.26.3

RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME && \
    pip install pycolmap==3.10.0 pyceres==2.3

RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME && \
    pip install poselib==2.0.2

RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME && \
    pip install -e .

# Default command: activate the conda environment and launch bash
CMD ["bash", "-c", "source $CONDA_DIR/etc/profile.d/conda.sh && conda activate vggsfm_tmp && exec bash"]