import torch 
import argparse
from distutils.util import strtobool

import rtb 
import reward_models 
import prior_models 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--exp', default="sd3", type=str, help='Experiment name', choices=['sd3', 'cifar'])
parser.add_argument('--n_iters', default=50000, type=int, metavar='N', help='Number of training iterations')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help="Training Batch Size.")
parser.add_argument('--loss_batch_size', type=int, default=-1, help="Batched RTB loss batch size")
parser.add_argument('--lr', '--learning_rate', default=5e-5, type=float, help='Initial learning rate.')
parser.add_argument('--prompt', type=str, default="A photorealistic green rabbit on purple grass.", help='Prompt for finetuning')
parser.add_argument('--reward_prompt', type=str, default="", help='Prompt for reward model (defaults to args.prompt)')
parser.add_argument('--target_class', type=int, default=0, help='Target class for classifier-tuning methods')
parser.add_argument('--diffusion_steps', type=int, default=100)
parser.add_argument('--wandb_track', default=False, type=strtobool, help='Whether to track with wandb.')
parser.add_argument('--entity', default=None, type=str, help='Wandb entity')
parser.add_argument('--prior_sample', default=False, type=strtobool, help="Whether to use off policy samples from prior")
parser.add_argument('--beta_start', default=1.0, type=float, help='Initial Inverse temperature for reward (Also used if anneal=False)')
parser.add_argument('--beta_end', default=10.0, type=float, help='Final Inverse temperature for reward')
parser.add_argument('--anneal', default=False, type=strtobool, help='Whether to anneal beta (From beta_start to beta_end)')
parser.add_argument('--anneal_steps', default=15000, type=int, help="Number of steps for temperature annealing")

parser.add_argument('--save_path', default='~/scratch/CNF_RTB_ckpts/', type=str, help='Path to save model checkpoints')
parser.add_argument('--load_ckpt', default=False, type=strtobool, help='Whether to load checkpoint')
parser.add_argument('--load_path', default=None, type=str, help='Path to load model checkpoint')

args = parser.parse_args()

if args.reward_prompt == '':
    args.reward_prompt = args.prompt
if args.loss_batch_size == -1:
    args.loss_batch_size = args.batch_size

if args.exp == "sd3":
    reward_model = reward_models.ImageRewardPrompt(device = device, prompt = args.reward_prompt)
    reward_args = [args.reward_prompt]

    in_shape = (16, 64 ,64)
    prior_model = prior_models.StableDiffusion3(prompt = args.prompt, 
                                                num_inference_steps = 28, 
                                                device = device)
    id = args.prompt 

elif args.exp == "cifar":

    reward_model = reward_models.CIFARClassifier(device = device, target_class = args.target_class)
    reward_args  = [args.target_class]
    in_shape  = (3, 32, 32)
    prior_model = prior_models.CIFARModel(device = device,
                                          num_inference_steps=20) 
    id = "cifar_target_class_" + str(args.target_class)

rtb_model = rtb.RTBModel(
                     device = device, 
                     reward_model = reward_model,
                     prior_model = prior_model,
                     in_shape = in_shape, 
                     reward_args = reward_args, 
                     id = id,
                     model_save_path = args.save_path,
                     load_ckpt = args.load_ckpt,
                     load_ckpt_path = args.load_path,
                     entity = args.entity,
                     diffusion_steps = args.diffusion_steps, 
                     beta_start = args.beta_start, 
                     beta_end = args.beta_end,
                     loss_batch_size = args.loss_batch_size)

rtb_model.finetune(shape=(args.batch_size, *in_shape), n_iters = args.n_iters, wandb_track=args.wandb_track, learning_rate=args.lr, prior_sample=args.prior_sample, anneal=args.anneal, anneal_steps=args.anneal_steps)