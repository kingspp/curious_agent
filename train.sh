#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -o a2c.out 
#SBATCH --job-name=a2cbreakout
#SBATCH -C K80

python -u samples/a2c_pipeline.py ./samples/breakout_a2c_config.json 
