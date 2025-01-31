import sys 
sys.path.append('./proteins/')

import torch 
import numpy as np
import random 

import argparse
from distutils.util import strtobool

import protein_rtb
import tb_sample_xt
#import tb 
import reward_models 
import prior_models
from replay_buffer import ReplayBuffer

from proteins.reward_ss_div import SSDivReward
from proteins.foldflow_prior import FoldFlowModel 

import rw_mcmc 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--exp', default="protein", type=str, help='Experiment name', choices=[ 'protein'])
parser.add_argument('--n_iters', default=50000, type=int, metavar='N', help='Number of training iterations')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help="Training Batch Size.")
parser.add_argument('--loss_batch_size', type=int, default=-1, help="Batched RTB loss batch size")
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float, help='Initial learning rate.')

parser.add_argument('--target_class', type=int, default=0, help='Target class for classifier-tuning methods')

parser.add_argument('--wandb_track', default=False, type=strtobool, help='Whether to track with wandb.')
parser.add_argument('--entity', default=None, type=str, help='Wandb entity')

parser.add_argument('--beta', default=1.0, type=float, help='Initial Inverse temperature for reward (Also used if anneal=False)')

parser.add_argument('--sample_save_path', default='~/scratch/cnf_rtb_prot_samples/rw_mcmc_samples/seed_', type=str, help='Path to save samples')
parser.add_argument('--load', default=False, type=strtobool, help='Whether to load checkpoint')
parser.add_argument('--seed', default=0, type=int, help='Random seed for training')

args = parser.parse_args()

# set seeds

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(args.seed)

sample_save_path = args.sample_save_path + str(args.seed) + "/"

if args.loss_batch_size == -1:
    args.loss_batch_size = args.batch_size


if args.exp == "protein":  
    #reward_model = FoldClassifier(device = device)
    #r_str = "TEMP_classifier_r"

    #reward_model = SheetPercentReward(device = device)
    #r_str = "TEMP_sheet_r"

    reward_model = SSDivReward(device = device)
    r_str = "RW_MCMC_temp_6_centered_r_reg_bias"
   

    reward_args = []

    seq_len = 64 #100 #256 #64 

    in_shape = (seq_len, 7)
    prior_model = FoldFlowModel(device = device)

    # only supported for batch_size 1 currently 
    #args.batch_size = 1
    #args.loss_batch_size = 1
    
    id = "protein_"+ r_str +"_len_" + str(seq_len)



mcmc = rw_mcmc.RW_MCMC(
        device = device,
        reward_model = reward_model,
        prior_model = prior_model,
        in_shape = in_shape,
        reward_args = reward_args,
        id = id,
        sample_save_path = sample_save_path,
        entity = args.entity,
        step_size = args.lr,
        beta = args.beta,
        coord_scaling = True,
        load = args.load
    )

mcmc.sample(batch_size = args.batch_size, 
            n_steps = args.n_iters, 
            wandb_track = args.wandb_track)
