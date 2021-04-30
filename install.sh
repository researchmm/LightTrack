#!/bin/bash

echo "****************** Installing lmdb ******************"
pip install lmdb

echo ""
echo ""
echo "****************** Installing pytorch with cuda10 ******************"
conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing cython ******************"
pip install cython==0.29.21

echo ""
echo ""
echo "****************** Installing skimage ******************"
pip install scikit-image

echo ""
echo ""
echo "****************** Installing pillow ******************"
pip install 'pillow<7.0.0'

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy

echo ""
echo ""
echo "****************** Installing shapely ******************"
pip install shapely

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py

echo ""
echo ""
echo '****************** Intalling mpi4py ******************'
conda install -y mpi4py

echo ""
echo ""
echo '****************** Intalling ray and hyperopt ******************'
pip uninstall --yes grpcio
pip install --upgrade setuptools
pip install --no-cache-dir grpcio>=1.28.1
conda install -c conda-forge -y grpcio
pip install ray==0.8.7
pip install ray[tune]
pip install hyperopt

echo ""
echo ""
echo "****************** Installing numba ******************"
pip install numba

echo ""
echo ""
echo "****************** Installing EfficientNet ******************"
cd lib/models/EfficientNet-PyTorch
pip install -e .
cd ../../..

echo ""
echo ""
echo "****************** Installing thop tool for FLOPs and Params computing ******************"
pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git

echo ""
echo ""
echo "****************** Installing GOT-10K related packages ******************"
cd toolkit
pip install -r requirements.txt
conda install -y matplotlib==3.0.2
cd ..

echo ""
echo ""
echo "****************** Installing Cream ******************"
pip install yacs
conda install -y tensorboard
pip install timm==0.1.20
pip install git+https://github.com/sovrasov/flops-counter.pytorch.git
pip install git+https://github.com/Tramac/torchscope.git

echo ""
echo ""
echo "****************** Installing tensorboardX, colorama ******************"
pip install tensorboardX
pip install colorama

echo ""
echo ""
echo "****************** Installation complete! ******************"
