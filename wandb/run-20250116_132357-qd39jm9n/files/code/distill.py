import torch
import argparse
from distutils.util import strtobool

import rtb
# import tb
import reward_models
import prior_models
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

# Common arguments (similar to finetune)
parser.add_argument('--exp', default="sd3_align", type=str,
                    help='Experiment name',
                    choices=['sd3_align', 'sd3_aes', 'cifar', 'gan_ffhq'])
parser.add_argument('--tb', default=False, type=strtobool,
                    help='Whether to use tb (vs rtb)')
parser.add_argument('--n_iters', default=5000, type=int, metavar='N',
                    help='Number of distillation iterations')
parser.add_argument('-bs', '--batch_size', type=int, default=64,
                    help="Training Batch Size.")
parser.add_argument('--loss_batch_size', type=int, default=-1,
                    help="Batched RTB loss batch size")
parser.add_argument('--lr', '--learning_rate', default=3e-5, type=float,
                    help='Initial learning rate.')
parser.add_argument('--prompt', type=str,
                    default="A photorealistic green rabbit on purple grass.",
                    help='Prompt for model prior')
parser.add_argument('--reward_prompt', type=str, default="An old man",
                    help='Prompt for reward model (defaults to args.prompt)')
parser.add_argument('--target_class', type=int, default=0,
                    help='Target class for classifier-tuning methods')
parser.add_argument('--diffusion_steps', type=int, default=100)
parser.add_argument('--wandb_track', default=False, type=strtobool,
                    help='Whether to track with wandb.')
parser.add_argument('--entity', default=None, type=str,
                    help='Wandb entity')
parser.add_argument('--replay_buffer', default='none', type=str,
                    help='Type of replay buffer to use',
                    choices=['none', 'uniform', 'reward'])
parser.add_argument('--prior_sample_prob', default=0.0, type=float,
                    help='Probability of using prior samples')
parser.add_argument('--replay_buffer_prob', default=0.0, type=float,
                    help='Probability of using replay buffer samples')
parser.add_argument('--beta_start', default=1.0, type=float,
                    help='Initial Inverse temperature for reward')
parser.add_argument('--beta_end', default=10.0, type=float,
                    help='Final Inverse temperature for reward')
parser.add_argument('--anneal', default=False, type=strtobool,
                    help='Whether to anneal beta (From beta_start to beta_end)')
parser.add_argument('--anneal_steps', default=15000, type=int,
                    help="Number of steps for temperature annealing")

parser.add_argument('--save_path', default='~/scratch/CNF_RTB_ckpts/', type=str,
                    help='Path to save model checkpoints')
parser.add_argument('--load_ckpt', default=False, type=strtobool,
                    help='Whether to load checkpoint (if needed for teacher)')
parser.add_argument('--load_path', default=None, type=str,
                    help='Path to load model checkpoint for teacher')

parser.add_argument('--langevin', default=False, type=strtobool,
                    help="Whether to use Langevin dynamics for sampling")
parser.add_argument('--inference', default='vpsde', type=str,
                    help='Inference method for prior', choices=['vpsde', 'ddpm'])

# Distillation-specific arguments
parser.add_argument('--distilled_ckpt_path', default='~/scratch/CNF_RTB_ckpts/distilled/', type=str,
                    help='Where to save distilled checkpoints')
parser.add_argument('--teacher_ckpt_filename', default=None, type=str,
                    help=''
                         'Model checkpoint filename')
parser.add_argument('--teacher_ckpt_path', default=None, type=str,
                    help='Path to teacher (fine-tuned) model checkpoint')
parser.add_argument('--distill_iters', default=5000, type=int,
                    help='Number of distillation iterations (if separate from n_iters)')
parser.add_argument('--distill_lr', default=1e-4, type=float,
                    help='Distillation learning rate')
parser.add_argument('--distill_save_interval', default=500, type=int,
                    help='Save the distilled model every X iterations')

args = parser.parse_args()

posterior_architecture = 'unet'

if args.reward_prompt == '':
    args.reward_prompt = args.prompt
if args.loss_batch_size == -1:
    args.loss_batch_size = args.batch_size

if args.exp == "gan_ffhq":
    reward_model = reward_models.ImageRewardPrompt(device=device, prompt=args.reward_prompt)
    reward_args = [args.reward_prompt]

    in_shape = (2, 16, 16)
    prior_model = prior_models.GAN_FFHQ_Prompt(device=device)
    posterior_architecture = 'mlp'
    id = "gan_ffhq_" + args.reward_prompt

elif args.exp == "sd3_align":
    print('getting reward model...', end='')
    reward_model = reward_models.ImageRewardPrompt(device=device, prompt=args.reward_prompt)
    reward_args = [args.reward_prompt]
    print(' done!')

    in_shape = (16, 64, 64)
    print('getting prior model...', end='')
    prior_model = prior_models.StableDiffusion3(prompt=args.prompt,
                                                num_inference_steps=28,
                                                device=device)
    id = "align_" + args.prompt + '_' + args.reward_prompt
    print(' done!')

elif args.exp == "sd3_aes":
    reward_model = reward_models.AestheticPredictor(device=device)
    reward_args = []

    in_shape = (16, 64, 64)
    prior_model = prior_models.StableDiffusion3(prompt="",
                                                num_inference_steps=28,
                                                device=device)
    id = "aes_no_prompt"

elif args.exp == "cifar":
    reward_model = reward_models.CIFARClassifier(device=device, target_class=args.target_class)
    reward_args = [args.target_class]
    in_shape = (3, 32, 32)
    prior_model = prior_models.CIFARModel(device=device, num_inference_steps=20)
    id = "cifar_target_class_" + str(args.target_class)

if args.tb:
    print("Note: Using TB = True. (Though the code references RTB typically.)")

replay_buffer = None
if not args.replay_buffer == 'none':
    replay_buffer = ReplayBuffer(rb_size=10000, rb_sample_strategy=args.replay_buffer)

print('instantiating rtb model...', end='')
rtb_model = rtb.RTBModel(
    device=device,
    reward_model=reward_model,
    prior_model=prior_model,
    in_shape=in_shape,
    reward_args=reward_args,
    id=id,
    model_save_path=args.save_path,
    langevin=args.langevin,
    inference_type=args.inference,
    tb=args.tb,
    load_ckpt=False,  # We'll handle teacher load separately
    load_ckpt_path=None,  # same note
    entity=args.entity,
    diffusion_steps=args.diffusion_steps,
    beta_start=args.beta_start,
    beta_end=args.beta_end,
    loss_batch_size=args.loss_batch_size,
    replay_buffer=replay_buffer,
    posterior_architecture=posterior_architecture
)
print(' done!')

# Optional pretrain for Langevin if needed
if args.langevin:
    print('instantiating langevin model...', end='')
    rtb_model.pretrain_trainable_reward(
        n_iters=20,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        wandb_track=False
    )
    print(' done!')

# --------------------------------------------------------------------------
# DISTILL CALL
# --------------------------------------------------------------------------
# shape for distillation:
distill_shape = (args.batch_size, *in_shape)

# We use the method distill() previously shown in the RTBModel,
# which should be implemented something like:
#
# def distill(
#     self,
#     shape,
#     distilled_ckpt_path,
#     teacher_ckpt_path,
#     n_iters=10000,
#     learning_rate=1e-4,
#     save_interval=500,
#     wandb_track=False
# ):
#     ...
#     # your distillation logic
#     ...

rtb_model.distill(
    shape=distill_shape,
    distilled_ckpt_path=args.distilled_ckpt_path,
    teacher_ckpt_filename=args.teacher_ckpt_filename,
    teacher_ckpt_path=args.teacher_ckpt_path,
    n_iters=args.distill_iters,
    learning_rate=args.distill_lr,
    save_interval=args.distill_save_interval,
    wandb_track=args.wandb_track
)
