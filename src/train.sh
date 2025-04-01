#!/bin/sh
### General options
#BSUB -q gpua100
#BSUB -J train
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -B
#BSUB -N
#BSUB -o ../outputs/train_%J.out
#BSUB -e ../outputs/train_%J.err

# Load the cuda module
module load cuda/11.6

source ~/Thesis/bin/activate

python train.py