import sys 
sys.path.append('./proteins/')

import torch 
import argparse
from distutils.util import strtobool

import rtb 
import protein_rtb
import tb_sample_xt
#import tb 
import reward_models 
import prior_models
from replay_buffer import ReplayBuffer

from proteins.reward_ss_div import FoldClassifier, SheetPercentReward
from proteins.conformation_energy import ConfEnergy 
from proteins.foldflow_prior import FoldFlowModel 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--exp', default="sd3_align", type=str, help='Experiment name', choices=['sd3_align', 'sd3_aes', 'cifar', 'protein'])
parser.add_argument('--tb', default=False, type=strtobool, help='Whether to use tb (vs rtb)')
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
#parser.add_argument('--prior_sample', default=False, type=strtobool, help="Whether to use off policy samples from prior")
parser.add_argument('--replay_buffer', default='none', type=str, help='Type of replay buffer to use', choices=['none','uniform','reward'])
parser.add_argument('--prior_sample_prob', default=0.0, type=float, help='Probability of using prior samples')
parser.add_argument('--replay_buffer_prob', default=0.0, type=float, help='Probability of using replay buffer samples')
parser.add_argument('--beta_start', default=1.0, type=float, help='Initial Inverse temperature for reward (Also used if anneal=False)')
parser.add_argument('--beta_end', default=10.0, type=float, help='Final Inverse temperature for reward')
parser.add_argument('--anneal', default=False, type=strtobool, help='Whether to anneal beta (From beta_start to beta_end)')
parser.add_argument('--anneal_steps', default=15000, type=int, help="Number of steps for temperature annealing")

parser.add_argument('--save_path', default='~/scratch/CNF_RTB_ckpts/', type=str, help='Path to save model checkpoints')
parser.add_argument('--load_ckpt', default=False, type=strtobool, help='Whether to load checkpoint')
parser.add_argument('--load_path', default=None, type=str, help='Path to load model checkpoint')

parser.add_argument('--langevin', default=False, type=strtobool, help="Whether to use Langevin dynamics for sampling")
parser.add_argument('--compute_fid', default=False, type=strtobool, help="Whether to compute FID score during training (every 1k steps)")

parser.add_argument('--inference', default='vpsde', type=str, help='Inference method for prior', choices=['vpsde', 'ddpm'])
parser.add_argument('--sampler_time', default = 0., type=float, help='Sampler learns to sample from p_t')

#parser.add_argument('--')

args = parser.parse_args()

if args.reward_prompt == '':
    args.reward_prompt = args.prompt
if args.loss_batch_size == -1:
    args.loss_batch_size = args.batch_size

if args.exp == "sd3_align":
    reward_model = reward_models.ImageRewardPrompt(device = device, prompt = args.reward_prompt)
    reward_args = [args.reward_prompt]

    in_shape = (16, 64 ,64)
    prior_model = prior_models.StableDiffusion3(prompt = args.prompt, 
                                                num_inference_steps = 28, 
                                                device = device)
    id = "align_" + args.prompt 

elif args.exp == "sd3_aes":
    reward_model = reward_models.AestheticPredictor(device = device)
    reward_args = []

    in_shape = (16, 64 ,64)
    prior_model = prior_models.StableDiffusion3(prompt = "", 
                                                num_inference_steps = 28, 
                                                device = device)
    id = "aes_no_prompt"

elif args.exp == "cifar":

    reward_model = reward_models.CIFARClassifier(device = device, target_class = args.target_class)
    reward_args  = [args.target_class]
    in_shape  = (3, 32, 32)
    prior_model = prior_models.CIFARModel(device = device,
                                          num_inference_steps=20) 
    id = "cifar_target_class_" + str(args.target_class)

elif args.exp == "protein":
    reward_model = ConfEnergy(device = device)
    r_str = "conf_energy"
    
    #reward_model = FoldClassifier(device = device)
    #r_str = "classifier_r"

    #reward_model = SheetPercentReward(device = device)
    #r_str = "sheet_r"

    reward_args = []

    seq_len = 100 #256 #64 

    in_shape = (seq_len, 7)
    prior_model = FoldFlowModel(device = device)

    # only supported for batch_size 1 currently 
    #args.batch_size = 1
    #args.loss_batch_size = 1
    
    id = "protein_foldflow_"+ r_str +"_len_" + str(seq_len)


replay_buffer = None    
if not args.replay_buffer == 'none':
    replay_buffer = ReplayBuffer(rb_size=10000, rb_sample_strategy=args.replay_buffer)


if args.sampler_time == 0.: 

    if args.exp == "protein":
        rtb_model = protein_rtb.ProteinRTBModel(
            device = device, 
            reward_model = reward_model,
            prior_model = prior_model,
            in_shape = in_shape, 
            reward_args = reward_args, 
            id = id,
            model_save_path = args.save_path,
            langevin = args.langevin,
            inference_type = args.inference,
            tb = args.tb,
            load_ckpt = args.load_ckpt,
            load_ckpt_path = args.load_path,
            entity = args.entity,
            diffusion_steps = args.diffusion_steps, 
            beta_start = args.beta_start, 
            beta_end = args.beta_end,
            loss_batch_size = args.loss_batch_size,
            replay_buffer = replay_buffer)
    else:
        rtb_model = rtb.RTBModel(
            device = device, 
            reward_model = reward_model,
            prior_model = prior_model,
            in_shape = in_shape, 
            reward_args = reward_args, 
            id = id,
            model_save_path = args.save_path,
            langevin = args.langevin,
            inference_type = args.inference,
            tb = args.tb,
            load_ckpt = args.load_ckpt,
            load_ckpt_path = args.load_path,
            entity = args.entity,
            diffusion_steps = args.diffusion_steps, 
            beta_start = args.beta_start, 
            beta_end = args.beta_end,
            loss_batch_size = args.loss_batch_size,
            replay_buffer = replay_buffer)
else:
    rtb_model = tb_sample_xt.TBModel(
            device = device, 
            reward_model = reward_model,
            prior_model = prior_model,
            in_shape = in_shape, 
            reward_args = reward_args, 
            id = id,
            model_save_path = args.save_path,
            sampler_time= args.sampler_time,  # time which gfn learns to sample at
            langevin = args.langevin,
            inference_type = args.inference,
            tb = args.tb,
            load_ckpt = args.load_ckpt,
            load_ckpt_path = args.load_path,
            entity = args.entity,
            diffusion_steps = args.diffusion_steps, 
            beta_start = args.beta_start, 
            beta_end = args.beta_end,
            loss_batch_size = args.loss_batch_size,
            replay_buffer = replay_buffer)

if args.langevin:
    rtb_model.pretrain_trainable_reward(n_iters = 20, batch_size = args.batch_size, learning_rate = args.lr, wandb_track = False) #args.wandb_track)

rtb_model.finetune(shape=(args.batch_size, *in_shape), n_iters = args.n_iters, wandb_track=args.wandb_track, learning_rate=args.lr, prior_sample_prob=args.prior_sample_prob, replay_buffer_prob=args.replay_buffer_prob, anneal=args.anneal, anneal_steps=args.anneal_steps)
