#!/bin/bash

conda create --name midas python=3.6

eval "$(conda shell.bash hook)"
conda activate midas

pip install librosa
pip install matplotlib
pip install pickle-mixin
pip install tqdm
pip install -U scikit-learn      
pip install numpy
pip install glob2
pip install torch
pip install torchvision
pip install torch-utils
python test.py