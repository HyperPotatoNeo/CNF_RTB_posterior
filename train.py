import torch 
import argparse
from distutils.util import strtobool
import os
from torchvision import datasets, transforms
import rtb 
import non_latent_rtb
import memoryless_rtb 
import memoryless_am
#import tb 
import reward_models 
import prior_models
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--exp', default="sd3_align", type=str, help='Experiment name', choices=['sd3_align', 'sd3_aes', 'cifar', 'gan_ffhq', 'gan_cifar', 'gan_cifar_improve'])
parser.add_argument('--tb', default=False, type=strtobool, help='Whether to use tb (vs rtb)')
parser.add_argument('--n_iters', default=50000, type=int, metavar='N', help='Number of training iterations')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help="Training Batch Size.")
parser.add_argument('--loss_batch_size', type=int, default=-1, help="Batched RTB loss batch size")
parser.add_argument('--lr', '--learning_rate', default=3e-5, type=float, help='Initial learning rate.')
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

# am: implementation not finished yet. dpok, nl_rtb: code review needed
parser.add_argument('--method', default= 'rtb', type=str, help='Method for training', choices=['rtb', 'dpok', 'nl_rtb', 'am'])
parser.add_argument('--detach_freq', default=0.8, type=float, help='Number of samples to backpropagate through')

parser.add_argument('--posterior_ratio', default=0.2, type=float, help='Ratio of finetuning model for parameterizing drift')

parser.add_argument('--lora', default=False, type=strtobool, help='Whether to use lora or not')
parser.add_argument('--lora_rank', default=16, type=int, help='rank when using LORA peft')

parser.add_argument('--clip_x', default=False, type=strtobool, help='clip x between [-1,1] during sampling (for images)')
#parser.add_argument('--')

args = parser.parse_args()

posterior_architecture = 'unet'

if args.reward_prompt == '':
    args.reward_prompt = args.prompt
if args.loss_batch_size == -1:
    args.loss_batch_size = args.batch_size

if args.exp == "gan_ffhq":
    reward_model = reward_models.ImageRewardPrompt(device = device, prompt = args.reward_prompt)
    reward_args = [args.reward_prompt]
    
    in_shape = (2, 16, 16)
    prior_model = prior_models.GAN_FFHQ_Prompt(device = device)
    #posterior_architecture = 'mlp'
    id = "gan_ffhq_" + args.reward_prompt
    
if args.exp == "gan_cifar_improve":
    reward_model = reward_models.SNGANDiscriminatorReward(device = device)
    reward_args = []
    
    in_shape = (2, 8, 8)
    prior_model = prior_models.SNGANGenerator(device = device, sngan_improve = True)
    id = "gan_cifar_improve"
    
if args.exp == "gan_cifar":
    reward_model = reward_models.CIFARClassifier(device = device, target_class = args.target_class)
    reward_args  = [args.target_class]
    
    in_shape = (2, 8, 8)
    prior_model = prior_models.SNGANGenerator(device = device)
    id = "gan_cifar"
    
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
    prior_model = prior_models.StableDiffusion3(prompt = args.prompt, 
                                                num_inference_steps = 28, 
                                                device = device)
    id = "aes_" + args.prompt

elif args.exp == "cifar":

    reward_model = reward_models.CIFARClassifier(device = device, target_class = args.target_class)
    reward_args  = [args.target_class]
    in_shape  = (3, 32, 32)
    prior_model = prior_models.CIFARModel(device = device,
                                          num_inference_steps = args.diffusion_steps) 
    id = "cifar_target_class_" + str(args.target_class)

if 'cifar' in args.exp and args.compute_fid:
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    class_label = args.target_class
    output_dir = 'fid/cifar10_class_' + str(class_label)
    os.makedirs(output_dir, exist_ok=True)

    output_dir_all = 'fid/cifar10_class_all'
    os.makedirs(output_dir_all, exist_ok=True)

    generated_images_dir = 'fid/' + args.exp + '_cifar10_class_' + str(class_label)
    os.makedirs(generated_images_dir, exist_ok=True)

    count = 0

    # Filter and save images of class
    for i, (img, label) in enumerate(train_dataset):
        
        if label == class_label:
            img = transforms.ToPILImage()(img)    
            img.save(os.path.join(output_dir, f'{i}.png'))

            count += 1

    label_counts = {i: 0 for i in range(10)}
    for i, (img, label) in enumerate(train_dataset):
        if label_counts[label] < count // 10:
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(output_dir_all, f'label_{label}_{label_counts[label]}.png'))
            label_counts[label] += 1

        

if args.tb:
    print("Training with TB")

replay_buffer = None    
if not args.replay_buffer == 'none':
    replay_buffer = ReplayBuffer(rb_size=10000, rb_sample_strategy=args.replay_buffer)


# non-latent RTB is only implemented for CIFAR now.
if args.method == 'nl_rtb' and 'cifar' in args.exp:
    model = prior_models.CIFARModel(device = device, num_inference_steps=args.diffusion_steps).get_prior_model().train()   
    #rtb_model = non_latent_rtb
    rtb_model = memoryless_rtb.RTBModel(
            device = device, 
            reward_model = reward_model,
            prior_model = prior_model,
            model = model,
            in_shape = in_shape, 
            reward_args = reward_args, 
            id = id,
            model_save_path = args.save_path,
            langevin = args.langevin,
            lora = args.lora,
            lora_rank = args.lora_rank,
            clip_x = args.clip_x,
            inference_type = args.inference,
            tb = args.tb,
            load_ckpt = args.load_ckpt,
            load_ckpt_path = args.load_path,
            entity = args.entity,
            diffusion_steps = args.diffusion_steps, 
            beta_start = args.beta_start, 
            beta_end = args.beta_end,
            loss_batch_size = args.loss_batch_size,
            replay_buffer = replay_buffer,
            posterior_architecture = posterior_architecture,
            detach_freq=args.detach_freq,
            posterior_ratio=args.posterior_ratio)


elif args.method == 'am' and 'cifar' in args.exp:
    model = prior_models.CIFARModel(device = device).get_prior_model().train()   
    rtb_model = memoryless_am.RTBModel(
                device = device, 
                reward_model = reward_model,
                prior_model = prior_model,
                model = model,
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
                replay_buffer = replay_buffer,
                posterior_architecture = posterior_architecture,
                detach_freq=args.detach_freq)


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
                replay_buffer = replay_buffer,
                posterior_architecture = posterior_architecture)


if args.langevin:
    rtb_model.pretrain_trainable_reward(n_iters = 20, batch_size = args.batch_size, learning_rate = args.lr, wandb_track = False) #args.wandb_track)

rtb_model.finetune(shape=(args.batch_size, *in_shape), n_iters = args.n_iters, wandb_track=args.wandb_track, learning_rate=args.lr, prior_sample_prob=args.prior_sample_prob, replay_buffer_prob=args.replay_buffer_prob, anneal=args.anneal, anneal_steps=args.anneal_steps, exp=args.exp, compute_fid=args.compute_fid, class_label=args.target_class)
