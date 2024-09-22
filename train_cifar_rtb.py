import os
import sys
import matplotlib.pyplot as plt
import torch
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchcfm.models.unet.unet import UNetModelWrapper
from rtb_model import RTBModel
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()

parser.add_argument('--n_iters', default=10000, type=int, metavar='N', help='Number of training iterations')
parser.add_argument('-bs', '--batch_size', type=int, default=36, help="Training Batch Size.")
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float, help='Initial learning rate.')
parser.add_argument('--posterior_class', type=int, default=0, help="Posterior finetuning class")
parser.add_argument('--diffusion_steps', type=int, default=200)
parser.add_argument('--wandb_track', default=False, type=strtobool, help='Whether to track with wandb.')

args = parser.parse_args()

rtb_model = RTBModel(posterior_class=args.posterior_class, diffusion_steps=args.diffusion_steps)

rtb_model.finetune(shape=(args.batch_size, 3, 32, 32), n_iters = args.n_iters, wandb_track=args.wandb_track, learning_rate=args.lr)