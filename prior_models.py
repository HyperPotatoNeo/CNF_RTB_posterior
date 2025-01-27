import os
import types 
import torch
import torch.distributed as dist
from diffusers import StableDiffusion3Pipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from huggingface_hub import login
import numpy as np
# GAN FFHQ
#import GAN.stylegan3.dnnlib as dnnlib
#import GAN.stylegan3.legacy as legacy
#from GAN.stylegan3.torch_utils import misc
from NVAE.model import AutoEncoder
import NVAE.utils as utils
from sngan_cifar10.sngan_cifar10 import Generator, SNGANConfig 
import PIL.Image
from typing import List, Optional, Tuple, Union

from torchcfm.models.unet.unet import UNetModelWrapper
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

class MLP(torch.nn.Module):
    def __init__(self, dim, w=2048):
        super().__init__()
        self.dim = dim
        self.w = w
        self.time_dim = 64  # Dimension of the time embedding

        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + self.time_dim, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, dim),
        )

    def forward(self, t, x):
        """
        Forward pass of the MLP with time conditioning.

        :param x: Input tensor of shape [batch_size, dim]
        :param t: Time tensor of shape [batch_size]
        :return: Output tensor of shape [batch_size, out_dim]
        """
        time_emb = self.get_timestep_embedding(t, self.time_dim)
        x = torch.cat([x, time_emb], dim=1)  # Concatenate along the feature dimension
        return self.net(x)

    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        Generate sinusoidal embeddings for the given time steps.

        :param timesteps: 1-D tensor of time steps [batch_size]
        :param embedding_dim: Dimension of the time embeddings
        :return: Time embeddings of shape [batch_size, embedding_dim]
        """
        half_dim = embedding_dim // 2
        emb_scale = torch.log(torch.tensor(10000.0)).cuda() / (half_dim - 1)
        emb = torch.exp(-emb_scale * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32))
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

def set_bn(model, bn_eval_mode, num_samples=1, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
    else:
        model.train()
        for i in range(iter):
            if i % 10 == 0:
                print('setting BN statistics iter %d out of %d' % (i+1, iter))
            model.sample(num_samples, t)
        model.eval()

class NVAE_FFHQ_Prompt():
    def __init__(self, checkpoint, n_z, temp, device):
        self.temp = temp
        self.n_z = n_z
        self.ckpt = checkpoint
        checkpoint = torch.load('NVAE/' + checkpoint)
        args = checkpoint['args']
        if not hasattr(args, 'ada_groups'):
            args.ada_groups = False
        if not hasattr(args, 'min_groups_per_scale'):
            args.min_groups_per_scale = 1
        if not hasattr(args, 'num_mixture_dec'):
            args.num_mixture_dec = 10
        #if eval_args.batch_size > 0:
        args.batch_size = 16#eval_args.batch_size
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '6020'
        arch_instance = utils.get_arch_cells(args.arch_instance)
        with torch.no_grad():
            self.model = AutoEncoder(args, None, arch_instance).to(device)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            bn_eval_mode = False
            dist.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=1)
            set_bn(self.model, bn_eval_mode, num_samples=16, t=temp, iter=500)
            dist.destroy_process_group()
            print('BN done')
            #test sampling
            #set_bn(self.model, bn_eval_mode, num_samples=16, t=0.6, iter=500)
            #logits = self.model.sample(32, 0.6)
            #output = self.model.decoder_output(logits)
            #output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
            #    else output.sample()
        #img = (output_img.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
        #img = img[..., [1, 2, 0]]
        #pil_images = [PIL.Image.fromarray(img[i].detach().cpu().numpy()) for i in range(img.shape[0])]
        #for i, img_tensor in enumerate(pil_images):
        #    img_tensor.save(os.path.join('nvae_test_img', f'{i}.png'))

    def __call__(self, x):
        with torch.no_grad():
            logits = self.model.sample_z_post(x.shape[0], self.temp, x, self.n_z)
            output = self.model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
                else output.sample()
            if 'cifar' in self.ckpt:
                img = (output_img*255).clamp(0, 255).to(torch.uint8)
            else:
                img = (output_img.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
                img = [PIL.Image.fromarray(img[i].detach().cpu().numpy()) for i in range(img.shape[0])]
        return img


class GAN_FFHQ_Prompt():
    def __init__(self, device):
        self.device = device
        network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl'
        with dnnlib.util.open_url(network_pkl) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    def make_transform(translate: Tuple[float,float], angle: float):
        m = np.eye(3)
        s = np.sin(angle/360.0*np.pi*2)
        c = np.cos(angle/360.0*np.pi*2)
        m[0][0] = c
        m[0][1] = s
        m[0][2] = translate[0]
        m[1][0] = -s
        m[1][1] = c
        m[1][2] = translate[1]
        return m
    
    def __call__(self, x):
        x = x.view(-1, 512)
        label = torch.zeros([x.shape[0], self.G.c_dim], device=self.device)
        img = self.G(x, label)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil_images = [PIL.Image.fromarray(img[i].detach().cpu().numpy()) for i in range(img.shape[0])]
        return pil_images
    
    def differentiable_call(self, x):
        x = x.view(-1, 512)
        label = torch.zeros([x.shape[0], self.G.c_dim], device=self.device)
        img = self.G(x, label, force_fp32=True)
        img = img * 127.5 + 128#.clamp(0, 255).to(torch.uint8)
        return img

class StableDiffusion3():
    def __init__(self, prompt, num_inference_steps, device):
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps
        
        self.height = 512
        self.width = 512

        self.device = device

        cache_dir = os.path.expanduser("~/scratch/huggingface_cache/")
        print("Cache dir: ", cache_dir)

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            #cache_dir = cache_dir, 
            torch_dtype=torch.float16
        ).to(self.device)

    def __call__(self, x, return_logp = False):
        with torch.no_grad():
            img = self.pipe(
                prompt=self.prompt,
                latents=x,  # Pass the custom latents here
                num_inference_steps = self.num_inference_steps,
                guidance_scale = 5.0,
                height = self.height,
                width = self.width,
                num_images_per_prompt=x.shape[0]
            ).images
            
            # compute logp if needed
            if return_logp:
                raise NotImplementedError("Logp computation not implemented")
            else:
                return img


class CIFARModel():
    def __init__(self, device, num_inference_steps = 20, ot=False):
        
        self.num_inference_steps = num_inference_steps
        self.device = device

        # Prior flow model
        self.prior_model = UNetModelWrapper(
            dim=(3, 32, 32),
            num_res_blocks=2,
            num_channels=128,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        ).to(device)
        if ot:
            model_ckpt = "models/cifar10/otcfm_cifar10_weights_step_400000.pt"
        else:
            model_ckpt = "models/cifar10/cfm_cifar10_weights_step_400000.pt"
        checkpoint = torch.load(model_ckpt, map_location=device)

        state_dict = checkpoint["ema_model"]
        
        try:
            self.prior_model.load_state_dict(state_dict)
        except RuntimeError:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            self.prior_model.load_state_dict(new_state_dict)
        
        self.prior_model.eval()

        # Define the ODE
        tol = 1e-5
        self.neural_ode = NeuralODE(
            self.prior_model,
            sensitivity="adjoint",
            solver="euler",
            atol=tol,
            rtol=tol,
        )

    def __call__(self, x):
        t_span = torch.linspace(0, 1, self.num_inference_steps + 1, device=self.device)
        traj = self.neural_ode.trajectory(x, t_span=t_span)
        traj = traj[-1, :]
        img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)

        return img
    
    def differentiable_call(self, x):
        t_span = torch.linspace(0, 1, self.num_inference_steps + 1, device=self.device)
        traj = self.neural_ode.trajectory(x, t_span=t_span)
        traj = traj[-1, :]
        img = (traj * 127.5 + 128)#.clip(0, 255).to(torch.uint8)

        return img
    
class SNGANGenerator():
    def __init__(self, device, sngan_improve=False):
        self.device = device
        args = SNGANConfig()
        self.prior_model = Generator(args).to(device)
        #if sngan_improve:
        #    checkpoint = torch.load("./sngan_cifar10/checkpoint.pth")
        #    self.prior_model.load_state_dict(checkpoint['gen_state_dict'])
        #else:
        checkpoint = torch.load("./sngan_cifar10/sngan_cifar10.pth")
        self.prior_model.load_state_dict(checkpoint)
        
    def __call__(self, x):
        x = x.view(-1, 128)
        img = self.prior_model(x)
        return (img * 127.5 + 128).clip(0, 255).to(torch.uint8)
        