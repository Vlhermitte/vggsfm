# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This Script Assumes Python 3.10, CUDA 12.1

conda deactivate

# Set environment variables
export ENV_NAME=vggsfm_tmp
export PYTHON_VERSION=3.10
export PYTORCH_VERSION=2.1.0
export CUDA_VERSION=12.1

# Create a new conda environment and activate it
conda create -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

# Install PyTorch, torchvision, and PyTorch3D using conda
# if macos
if [[ "$OSTYPE" == "darwin"* ]]; then
    conda install pytorch=$PYTORCH_VERSION torchvision -c pytorch
    MACOSX_DEPLOYMENT_TARGET=15.1 CC=clang CXX=clang++ pip install "git+https://github.com/facebookresearch/pytorch3d.git"
else
    conda install pytorch=$PYTORCH_VERSION torchvision pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia
    conda install pytorch3d=0.7.5 -c pytorch3d
fi
conda install -c fvcore -c iopath -c conda-forge fvcore iopath


# Install pip packages
pip install hydra-core --upgrade
pip install omegaconf opencv-python einops visdom tqdm scipy plotly scikit-learn imageio[ffmpeg] gradio trimesh huggingface_hub

# Install LightGlue
git clone https://github.com/jytime/LightGlue.git dependency/LightGlue

cd dependency/LightGlue/
python -m pip install -e .  # editable mode
cd ../../

# Force numpy <2
pip install numpy==1.26.3

# Ensure the version of pycolmap is 3.10.0 and pyceres is 2.3
# (pycolmap 3.10.0 needs pyceres 2.3 otherwise it will throw an error during bundle adjustment)
pip install pycolmap==3.10.0 pyceres==2.3

# (Optional) Install poselib 
pip install poselib==2.0.2

