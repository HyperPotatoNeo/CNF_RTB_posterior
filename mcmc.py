import torch 
import argparse
import numpy as np
from distutils.util import strtobool
from torchvision import datasets, transforms
import rtb
import os
import wandb
from cleanfid import fid
#import tb 
import reward_models 
import prior_models
import hamiltorch
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--exp', default="cifar", type=str, help='Experiment name', choices=['sd3_align', 'sd3_aes', 'cifar', 'gan_ffhq'])
parser.add_argument('--method', default='hmc', type=str, help='MCMC method', choices=['langevin','hmc'])
parser.add_argument('--n_iters', default=1000, type=int, metavar='N', help='Number of mcmc iterations')
parser.add_argument('-bs', '--batch_size', type=int, default=10, help="Training Batch Size.")
parser.add_argument('--num_batches', type=int, default=10, help='Number of batches')
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float, help='Initial learning rate.')
parser.add_argument('--prompt', type=str, default="A photorealistic green rabbit on purple grass.", help='Prompt for finetuning')
parser.add_argument('--reward_prompt', type=str, default="", help='none')
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

parser.add_argument('--compute_fid', default=False, type=strtobool, help="Whether to compute FID score during training (every 1k steps)")

parser.add_argument('--inference', default='vpsde', type=str, help='Inference method for prior', choices=['vpsde', 'ddpm'])

#parser.add_argument('--')

args = parser.parse_args()

posterior_architecture = 'unet'

if args.reward_prompt == '':
    args.reward_prompt = args.prompt

if args.exp == "gan_ffhq":
    reward_model = reward_models.ImageRewardPrompt(device = device, prompt = args.reward_prompt, differentiable=True)
    reward_args = [args.reward_prompt]
    
    in_shape = (2, 16, 16)
    prior_model = prior_models.GAN_FFHQ_Prompt(device = device)
    prior_model.G.train()
    prior_model.G.requires_grad_(True)
    for param in prior_model.G.parameters():
        param.requires_grad = True
    posterior_architecture = 'mlp'
    id = "gan_ffhq_" + args.reward_prompt

if args.exp == "sd3_align":
    reward_model = reward_models.ImageRewardPrompt(device = device, prompt = args.reward_prompt)
    reward_args = [args.reward_prompt]

    in_shape = (16, 64 ,64)
    prior_model = prior_models.StableDiffusion3(prompt = args.prompt, 
                                                num_inference_steps = 28, 
                                                device = device)
    id = "align_" + args.prompt + '_' + args.reward_prompt

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.wandb_track:
    wandb.init(
        project='cfm_posterior',
        entity='swish',
        save_code=True,
        name='hmc_' + args.exp
    )

log_reward_sum = 0.0
log_prob_mean = 0.0
counter = 0
generated_images_dir = '/home/mila/s/siddarth.venkatraman/scratch/CNF_RTB_ckpts/hmc/' + args.exp + str(args.target_class) + args.reward_prompt
true_images_dir = 'fid/cifar10_class_' + str(args.target_class)
os.makedirs(generated_images_dir, exist_ok=True)

if args.method == 'hmc':
    def log_prob_func(x, return_reward=False, beta=100.0):
        x = x.view(1, in_shape[0], in_shape[1], in_shape[2])
        img = prior_model.differentiable_call(x)
        #reward_model(img).squeeze().backward()
        if return_reward:
            return reward_model(img).squeeze()
        else:
            norm_dist = torch.distributions.Normal(torch.zeros_like(x), torch.ones_like(x))
            return beta*reward_model(img).squeeze() + norm_dist.log_prob(x).sum()

    while True:
        num_samples = 1000#args.batch_size
        x_init = torch.randn(size=(np.prod(in_shape),)).cuda()
        x_hmc = hamiltorch.sample(log_prob_func=log_prob_func, params_init=x_init, num_samples=num_samples, step_size=args.lr, num_steps_per_sample=5)
        #print(x_hmc)
        element_idx = [49,99,149,100,249]
        with torch.no_grad():
            for i in range(100, 1000, 10):
                x = x_hmc[i]
                log_reward_sum = log_reward_sum + log_prob_func(x, return_reward=True).detach().cpu()
                counter += 1
                #if args.wandb_track:
                x = x.view(1, in_shape[0], in_shape[1], in_shape[2])
                img = prior_model(x)
                if 'cifar' in args.exp:
                    img_pil = transforms.ToPILImage()(img[0])
                else:
                    img_pil = img[0]
                img_pil.save(os.path.join(generated_images_dir, f'{counter}.png'))
                if counter % 100 == 0:
                    if args.wandb_track:
                        wandb.log({"hmc_image": wandb.Image(img[0])})
                    print(counter, 'log_reward_mean: ', log_reward_sum/counter)
                if counter % 500 == 0 and 'cifar' in args.exp:
                    fid_score = fid.compute_fid(generated_images_dir, true_images_dir)
                    print(counter, 'FID: ', )
                if counter == 6000:
                    break
                    
        print('log_reward_mean: ', log_reward_sum/counter)
            
            #wandb.log({"hmc_image": wandb.Image(img[0])})