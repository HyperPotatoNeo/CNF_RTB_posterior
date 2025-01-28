#!/bin/bash
#SBATCH --constraint="80gb"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=unkillable

# or > salloc --gres=gpu:rtx8000:2 --cpus-per-task=8 --mem=32G  --time=12:00:00 --nodes=1 --partition=main

# Compute the output directory after the SBATCH directives
export OUTPUT_DIR=$HOME/script_outputs
# Update the output path of the SLURM output file
export SLURM_JOB_OUTPUT=${OUTPUT_DIR}/$(basename ${SLURM_JOB_OUTPUT})

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Ensure only anaconda/3 module loaded.
module --quiet purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3
module load cuda/11.7

# Creating the environment for the first time:
# conda create -y -n pytorch python=3.9 pytorch torchvision torchaudio \
#     pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge rich tqdm
# Other conda packages:
# conda install -y -n pytorch -c conda-forge rich tqdm

# Activate pre-existing environment.
conda activate pytorch

#git pull

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"

# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES

# run the following line to run prior code -- note, the path assume you run the files from the "scripts" folder
#python ../distill.py --exp cifar --distilled_ckpt_path ./distilled/ \
#                     --teacher_ckpt_filename cifar_pretrained.pth \
#                     --teacher_ckpt_path ./pretrained/ --wandb_track True

# run the following line to run prior code -- note, the path assume you run the files from the "scripts" folder
#python ../distill.py --exp gan_ffhq --distilled_ckpt_path ./distilled/ \
#                     --teacher_ckpt_filename gan_an_old_man.pth \
#                     --teacher_ckpt_path ./pretrained/ --wandb_track True


## run the following line to run prior code -- note, the path assume you run the files from the "scripts" folder
python ../distill.py --exp sd3_align --distilled_ckpt_path ./distilled/ \
                     --teacher_ckpt_filename sd3_a_green_car.pth \
                     --prompt "A car" --reward_prompt "A green car"
                     --teacher_ckpt_path ./pretrained/ --wandb_track True