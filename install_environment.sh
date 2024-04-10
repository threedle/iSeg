#!/usr/bin/env bash

# environment installation commands
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html # install PyTorch
pip install kaolin==0.14.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html  # install Kaolin
pip install ftfy==6.1.1 regex==2023.10.3 tqdm==4.66.1 # install required packages for CLIP
pip install git+https://github.com/openai/CLIP.git # install CLIP
pip install matplotlib==3.5.2 # install additional packages
pip install transformers==4.34.0 # install additional packages
pip install open3d==0.17.0 # install additional packages
pip install opencv-python==4.8.1.78 # install additional packages
pip install scikit-image==0.22.0 # install additional packages
pip install plyfile==1.0.1 # install additional packages
