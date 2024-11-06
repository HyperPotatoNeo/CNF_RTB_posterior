import os 
import torch 
from diffusers import StableDiffusion3Pipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from huggingface_hub import login

from torchcfm.models.unet.unet import UNetModelWrapper
#from torchdiffeq import odeint
#from torchdyn.core import NeuralODE


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
    def __init__(self, device, num_inference_steps = 20):
        
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
        model_ckpt = "models/cifar10/otcfm_cifar10_weights_step_400000.pt"
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