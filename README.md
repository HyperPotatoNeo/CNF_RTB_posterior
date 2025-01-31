# README

## Training Scripts

This repository contains training scripts for finetuning various priors. Below are the commands to launch different training experiments.

## Launch Commands

### NVAE (FFHQ Dataset)
```bash
python3 train.py --exp "nvae_ffhq" --reward_prompt "A brown haired child." --replay_buffer 'uniform' --batch_size 64 --loss_batch_size 64 --wandb_track True --lr 3e-5 --prior_sample_prob 0.2 --replay_buffer_prob 0.2 --beta_start 30.0 --beta_end 100.0 --anneal_steps 5000 --diffusion_steps 25 --anneal False
```

### GAN (FFHQ Dataset)
```bash
python3 train.py --exp "gan_ffhq"  --reward_prompt "A brown haired child." --replay_buffer 'uniform' --batch_size 64 --loss_batch_size 64 --wandb_track True --lr 3e-5 --prior_sample_prob 0.2 --replay_buffer_prob 0.2 --beta_start 30.0 --beta_end 100.0 --anneal_steps 5000 --diffusion_steps 25 --anneal False
```

### GAN (CIFAR Dataset)
```bash
python3 train.py --exp "gan_cifar" --replay_buffer 'uniform' --batch_size 64 --loss_batch_size 64 --wandb_track True --lr 3e-5 --prior_sample_prob 0.2 --replay_buffer_prob 0.2 --beta_start 2.0 --beta_end 4.0 --anneal_steps 2000 --diffusion_steps 25 --anneal True --target_class 9 --compute_fid True
```

### NVAE (CIFAR Dataset)
```bash
python3 train.py --exp "nvae_cifar" --replay_buffer 'uniform' --batch_size 64 --loss_batch_size 64 --wandb_track True --lr 3e-5 --prior_sample_prob 0.4 --replay_buffer_prob 0.1 --beta_start 2.0 --beta_end 4.0 --anneal_steps 2000 --diffusion_steps 25 --anneal True --target_class 1 --compute_fid True
```

### CIFAR Training
```bash
python3 train.py --exp "cifar" --replay_buffer 'uniform' --batch_size 64 --loss_batch_size 64 --wandb_track True --lr 3e-5 --prior_sample_prob 0.4 --replay_buffer_prob 0.1 --beta_start 2.0 --beta_end 4.0 --anneal_steps 2000 --diffusion_steps 25 --anneal True --target_class 6 --compute_fid True
```

### Stable Diffusion 3 (SD3) Alignment
```bash
python3 train.py --exp "sd3_align" --prompt "A quiet village is disrupted by a meteor strike" --reward_prompt "A quiet village is disrupted by a meteor strike" --replay_buffer 'uniform' --batch_size 64 --loss_batch_size 64 --wandb_track True --lr 3e-5 --prior_sample_prob 0.2 --replay_buffer_prob 0.2 --beta_start 25.0 --beta_end 100.0 --anneal_steps 5000 --diffusion_steps 25 --anneal False
```

