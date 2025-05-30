#!/bin/bash

#SBATCH -p RTX3090
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres gpu:0

python inference.py 