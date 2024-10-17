import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Independent, Normal
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchcfm.models.unet.unet import UNetModelWrapper
from torchvision import datasets, transforms
from cifar10_models.vgg import vgg13_bn
import wandb
from tqdm import tqdm
from cleanfid import fid
import random

def create_batches(ids, batch_size):
    for i in range(0, len(ids), batch_size):
        yield ids[i:i + batch_size]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class VPSDE():
    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20,
        T: float = 1.0,
        epsilon: float = 1e-5,
        beta_schedule: str = 'linear',  # Added beta_schedule parameter
        **kwargs
    ):
        super().__init__()
        self.T = T
        self.epsilon = epsilon
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_schedule = beta_schedule  # Store the schedule type

    def beta(self, t: Tensor):
        if self.beta_schedule == 'linear':
            return self.beta_min + (self.beta_max - self.beta_min) * t
        elif self.beta_schedule == 'cosine':
            s = 0.008  # Small constant to prevent singularities
            f = (t / self.T + s) / (1 + s)
            tan_part = torch.tan(f * (np.pi / 2))
            beta_t = (2 * np.pi) / (self.T * (1 + s)) * tan_part
            # Clamp beta_t to be within [beta_min, beta_max]
            beta_t = torch.clamp(beta_t, min=self.beta_min, max=self.beta_max)
            return beta_t

    def sigma(self, t: Tensor) -> Tensor:
        return self.marginal_prob_scalars(t)[1]
        
    def prior(self, shape):
        mu = torch.zeros(shape).to(device)
        return Independent(Normal(loc=mu, scale=1., validate_args=False), len(shape))

    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        return torch.sqrt(self.beta(t)).view(-1, *[1]*len(D))

    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        return -0.5 * self.beta(t).view(-1, *[1]*len(D)) * x

    def marginal_prob_scalars(self, t: Tensor) -> tuple[Tensor, Tensor]:
        if self.beta_schedule == 'linear':
            log_coeff = 0.5 * (self.beta_max - self.beta_min) * t**2 + self.beta_min * t
            std = torch.sqrt(1. - torch.exp(-log_coeff))
            return torch.exp(-0.5 * log_coeff), std
        elif self.beta_schedule == 'cosine':
            s = 0.008  # Small constant to prevent singularities
            f = (t / self.T + s) / (1 + s)
            phi_t = f * (np.pi / 2)
            phi_0 = torch.tensor((s / (1 + s)) * (np.pi / 2), device=device)
            cos_phi_t = torch.cos(phi_t)
            cos_phi_0 = torch.cos(phi_0)
            coeff = cos_phi_t / cos_phi_0
            std = torch.sqrt(1. - coeff**2)
            return coeff, std

class RTBModel(nn.Module):
    def __init__(self, posterior_class=0, diffusion_steps=200, schedule='cosine', use_rb=False, temperature=1.0):
        super().__init__()
        self.device = device
        self.sde = VPSDE(beta_schedule='cosine')
        self.steps = diffusion_steps
        self.logZ = torch.nn.Parameter(torch.tensor(0.).to(self.device))
        self.use_rb = use_rb
        self.posterior_class = posterior_class
        self.temperature = temperature
        # Posterior noise model
        self.model = UNetModelWrapper(
            dim=(3, 32, 32),
            num_res_blocks=2,
            num_channels=128,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.0,
        ).to(device)
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
        PATH = "models/cifar10/otcfm_cifar10_weights_step_400000.pt"
        print("path: ", PATH)
        checkpoint = torch.load(PATH, map_location=device)
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
        # Classifier
        self.classifer = vgg13_bn(pretrained=True).to(device)
        self.classifier_mean = [0.4914, 0.4822, 0.4465]
        self.classifier_std = [0.2471, 0.2435, 0.2616]
        # Define the ODE
        tol = 1e-5
        self.neural_ode = NeuralODE(
            self.prior_model,
            sensitivity="adjoint",
            solver="euler",
            atol=tol,
            rtol=tol,
        )
        
    def log_reward(self, x, return_img=False):
        with torch.no_grad():
            t_span = torch.linspace(0, 1, 20 + 1, device=device)
            traj = self.neural_ode.trajectory(x, t_span=t_span)
            traj = traj[-1, :]
            img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
            logits = self.classifer((img.float() / 255 - torch.tensor(self.classifier_mean).cuda()[None, :, None, None]) / torch.tensor(self.classifier_std).cuda()[None, :, None, None])
            log_prob = nn.functional.log_softmax(logits/self.temperature, dim=1)
            log_r = log_prob[:, self.posterior_class]
        if return_img:
            return log_r, img
        return log_r
        
    def batched_rtb(self, shape, learning_cutoff=.1, prior_sample=False, iw_logz=False):
        # first pas through, get trajectory & loss for correction
        B, *D = shape

        with torch.no_grad():
            # run whole trajectory, and get PFs
            fwd_logs = self.forward(
                shape=shape,
                steps=self.steps,
                save_traj=True,  # save trajectory fwd
                prior_sample=prior_sample
            )
            x_mean_posterior, logpf_prior, logpf_posterior = fwd_logs['x_mean_posterior'], fwd_logs['logpf_prior'], fwd_logs['logpf_posterior']

            logr_x_prime = self.log_reward(x_mean_posterior)

            logr_prior_x_prime = logr_x_prime #+ prior_dist.log_prob(x_mean_posterior).sum(tuple(range(1, len(x_mean_posterior.shape)))).to(self.device).detach()
            # vargrad
            if iw_logz and prior_sample:
                #self.logZ.data = (logr_prior_x_prime).logsumexp(dim=0) - np.log(B)
                self.logZ.data = (-logpf_posterior + logpf_prior + logr_prior_x_prime).mean()
            elif iw_logz:
                self.logZ.data = (-logpf_posterior + logpf_prior + logr_prior_x_prime).logsumexp(dim=0) - np.log(B)
            else:
                self.logZ.data = (-logpf_posterior + logpf_prior + logr_prior_x_prime).mean()
            #print(logpf_posterior, logpf_prior, logr_prior_x_prime, self.logZ.data)
            #exit()

            rtb_loss = 0.5 * (((logpf_posterior + self.logZ - logpf_prior - logr_prior_x_prime) ** 2) - learning_cutoff).relu()

            # compute correction
            clip_idx = ((logpf_posterior + self.logZ - logpf_prior - logr_prior_x_prime) ** 2) < learning_cutoff
            correction = (logpf_posterior + self.logZ - logpf_prior - logr_prior_x_prime)
            correction[clip_idx] = 0.

        self.batched_forward(
            shape=shape,
            traj=fwd_logs['traj'],
            correction=correction,
            batch_size=B,
        )

        return rtb_loss.detach().mean(), logr_prior_x_prime.mean(), logr_x_prime.detach().mean()
    
    def finetune(self, shape, n_iters=100000, learning_rate=1e-5, clip=0.1, wandb_track=False, prior_sample=False, iw_logz=False, compute_fid=False, anneal=False, anneal_steps=20000):
        B, *D = shape
        param_list = [{'params': self.model.parameters()}]
        optimizer = torch.optim.Adam(param_list, lr=learning_rate)
        run_name = 'cifar_class_' + str(self.posterior_class) + '_temp_' + str(self.temperature) + '_logz_iw_' + str(iw_logz) + '_anneal_' + str(anneal)
        
        if wandb_track:
            wandb.init(
                project='cfm_posterior',
                entity='swish',
                save_code=True,
                name=run_name
            )
            hyperparams = {
                "learning_rate": learning_rate,
                "n_iters": n_iters,
                "posterior_class": self.posterior_class
            }
            wandb.config.update(hyperparams)
            with torch.no_grad():
                x = torch.randn(25, 3, 32, 32, device=device)
                t_span = torch.linspace(0, 1, 100 + 1, device=device)
                traj = self.neural_ode.trajectory(x, t_span=t_span)
                traj = traj[-1, :]
            img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
            fig, ax = plt.subplots(5, 5, figsize=(8, 8))
            for i in range(5):
                for j in range(5):
                    ax[i, j].imshow(img[i * 5 + j].permute(1, 2, 0).cpu().numpy())
                    ax[i, j].axis("off")
            wandb.log({"prior_samples": fig})
            
        prior_traj = False
        start_temp = self.temperature
        for it in range(n_iters):
            if anneal and it < anneal_steps:
                self.temperature = ((anneal_steps - it)/anneal_steps) * start_temp + (it / anneal_steps) * 0.2
            optimizer.zero_grad()
            loss, logr, classifer_log_prob = self.batched_rtb(shape=shape, prior_sample=prior_traj, iw_logz=iw_logz)
            #loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
            optimizer.step()
            if prior_sample:
                prior_traj = not prior_traj    
            
            if wandb_track: 
                if not it%100 == 0:
                    wandb.log({"loss": loss.item(), "logZ": self.logZ.detach().cpu().numpy(), "log_r": logr.item(), "classifier_log_prob": classifer_log_prob.item(), "epoch": it})
                else:
                    with torch.no_grad():
                        logs = self.forward(
                            shape=(25, *D),
                            steps=self.steps
                            )
                        x = logs['x_mean_posterior']
                        normal_dist = torch.distributions.Normal(torch.zeros((25,) + tuple(D), device=self.device),
                                                 torch.ones((25,) + tuple(D), device=self.device))
                        logp = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
                        log_rew, img = self.log_reward(x, return_img=True)
                        log_pr = logp + log_rew
                        #t_span = torch.linspace(0, 1, 20 + 1, device=device)
                        #traj = self.neural_ode.trajectory(x, t_span=t_span)
                        #traj = traj[-1, :]
                        #img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
                        fig, ax = plt.subplots(5, 5, figsize=(8, 8))
                        for i in range(5):
                            for j in range(5):
                                ax[i, j].imshow(img[i * 5 + j].permute(1, 2, 0).cpu().numpy())
                                ax[i, j].axis("off")
                                ax[i, j].set_title(str(log_pr[i*5 + j].cpu().numpy()) + "; " + str(np.round(log_rew[i*5 + j].cpu().numpy(), 2)), fontsize=6)
                    
                    if it%1000 == 0 and compute_fid and it>0:
                        print('COMPUTING FID:')
                        generated_images_dir = 'generated_cifar10_class_' + str(self.posterior_class)
                        true_images_dir = 'cifar10_class_' + str(self.posterior_class)
                        for k in range(60):
                            with torch.no_grad():
                                logs = self.forward(
                                    shape=(100, *D),
                                    steps=self.steps
                                    )
                                x = logs['x_mean_posterior']
                                t_span = torch.linspace(0, 1, 20 + 1, device=device)
                                traj = self.neural_ode.trajectory(x, t_span=t_span)
                                traj = traj[-1, :]
                                img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
                                for i, img_tensor in enumerate(img):
                                    img_pil = transforms.ToPILImage()(img_tensor)
                                    img_pil.save(os.path.join(generated_images_dir, f'airplane_{k*100 + i}.png'))
                        fid_score = fid.compute_fid(generated_images_dir, true_images_dir)
                        wandb.log({"loss": loss.item(), "logZ": self.logZ.detach().cpu().numpy(), "log_r": logr.item(), "epoch": it, "classifier_log_prob": classifer_log_prob.item(), "posterior_samples": fig, "fid_score": fid_score})
                    else:
                        wandb.log({"loss": loss.item(), "logZ": self.logZ.detach().cpu().numpy(), "log_r": logr.item(), "epoch": it, "classifier_log_prob": classifer_log_prob.item(), "posterior_samples": fig})
            
    def forward(
            self,
            shape,
            steps,
            condition: list=[],
            likelihood_score_fn=None,
            guidance_factor=0.,
            detach_freq=0.0,
            backward=False,
            x_1=None,
            save_traj=False,
            prior_sample=False,
            time_discretisation='uniform' #uniform/random
    ):
        """
        An Euler-Maruyama integration of the model SDE with GFN for RTB

        shape: Shape of the tensor to sample (including batch size)
        steps: Number of Euler-Maruyam steps to perform
        likelihood_score_fn: Add an additional drift to the sampling for posterior sampling. Must have the signature f(t, x)
        guidance_factor: Multiplicative factor for the likelihood drift
        detach_freq: Fraction of steps on which not to train
        """
        # if not isinstance(condition, (list, tuple)):
        #     raise ValueError(f"condition must be a list or tuple or torch.Tensor, received {type(condition)}")
        B, *D = shape
        sampling_from = "prior" if likelihood_score_fn is None else "posterior"
        if likelihood_score_fn is None:
            likelihood_score_fn = lambda t, x: 0.

        if backward:
            x = x_1
            timesteps = np.flip(timesteps)
            t = torch.zeros(B).to(self.device) + self.sde.epsilon
        else:
            x = self.sde.prior(D).sample([B]).to(self.device)
            t = torch.ones(B).to(self.device) * self.sde.T

        # assume x is gaussian noise
        normal_dist = torch.distributions.Normal(torch.zeros((B,) + tuple(D), device=self.device),
                                                 torch.ones((B,) + tuple(D), device=self.device))

        logpf_posterior = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        logpb = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)#torch.zeros_like(logpf_posterior)
        dt = -1/(steps+1)

        #####
        if save_traj:
            traj = [x.clone()]

        for step, _ in enumerate((pbar := tqdm(range(steps)))):
            pbar.set_description(
                f"Sampling from the {sampling_from} | t = {t[0].item():.1f} | sigma = {self.sde.sigma(t)[0].item():.1e}"
                f"| scale ~ {x.std().item():.1e}")
            if backward:
                g = self.sde.diffusion(t, x)
                std = g * (np.abs(dt)) ** (1 / 2)
                x_prev = x.detach()
                x = (x + self.sde.drift(t, x) * dt) + (std * torch.randn_like(x))
            t += dt
            if t[0] < self.sde.epsilon:  # Accounts for numerical error in the way we discretize t.
                continue # continue instead of break because it works for forward and backward
            
            x_prev = x.detach()
            g = self.sde.diffusion(t, x)

            posterior_drift = -self.sde.drift(t, x) - (g ** 2) * self.model(t, x) / self.sde.sigma(t).view(-1, *[1]*len(D))
            f_posterior = posterior_drift
            # compute parameters for denoising step (wrt posterior)
            x_mean_posterior = x + f_posterior * dt * (-1.0 if backward else 1.0)
            std = g * (np.abs(dt)) ** (1 / 2)

            # compute step
            if prior_sample:
                x = x - self.sde.drift(t, x) * dt + std * torch.randn_like(x)
            else:
                x = x_mean_posterior + std * torch.randn_like(x)
            x = x.detach()
            
            # compute parameters for pb
            #t_next = t + dt
            #pb_drift = self.sde.drift(t_next, x)
            #x_mean_pb = x + pb_drift * (-dt)
            pb_drift = -self.sde.drift(t, x_prev)
            x_mean_pb = x_prev + pb_drift * (dt)
            pb_std = g * (np.abs(dt)) ** (1 / 2)

            if save_traj:
                traj.append(x.clone())
                
            pf_post_dist = torch.distributions.Normal(x_mean_posterior, std)
            pb_dist = torch.distributions.Normal(x_mean_pb, pb_std)

            # compute log-likelihoods of reached pos wrt to prior & posterior models
            #logpb += pb_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
            logpb += pb_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))
            logpf_posterior += pf_post_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))

            if torch.any(torch.isnan(x)):
                print("Diffusion is not stable, NaN were produced. Stopped sampling.")
                break
        
        logs = {
            'x_mean_posterior': x,#,x_mean_posterior,
            'logpf_prior': logpb,
            'logpf_posterior': logpf_posterior,
            'traj': traj if save_traj else None
        }

        return logs
    
    def batched_forward(
            self,
            shape,
            traj,
            correction,
            batch_size=64,
            condition: list = [],
            likelihood_score_fn=None,
            guidance_factor=0.,
            detach_freq=0.0,
            backward=False,
    ):
        """
        Batched implementation of self.forward. See self.forward for details.
        """

        B, *D = shape
        # compute batch size wrt traj (since each node in the traj is already a batch)
        traj_batch = batch_size // traj[0].shape[0]

        steps = len(traj) - 1
        timesteps = np.linspace(1, 0, steps + 1)

        if likelihood_score_fn is None:
            likelihood_score_fn = lambda t, x: 0.

        if backward:
            x = traj[-1]
            timesteps = np.flip(timesteps)
        else:
            x = traj[0].to(self.device)

        no_grad_steps = random.sample(range(steps), int(steps * 0.0))  # Sample detach_freq fraction of timesteps for no grad

        # # assume x is gaussian noise
        # normal_dist = torch.distributions.Normal(torch.zeros((B,) + tuple(D), device=self.device),
        #                                          torch.ones((B,) + tuple(D), device=self.device))
        #
        # logpf_posterior = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        # logpf_prior = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)

        # we iterate through the traj
        steps = list(range(len(traj)))
        steps = [step for step in steps[:-1] if step not in no_grad_steps]

        for i, batch_steps in enumerate(create_batches(steps, traj_batch)):

            #pbar.set_description(f"Sampling from the posterior | batch = {i}/{int(len(steps)//batch_size)} - {i*100/len(steps)//batch_size:.2f}%")

            dt = timesteps[np.array(batch_steps) + 1] - timesteps[batch_steps]

            t_ = []
            xs = []
            xs_next = []
            dts = []
            bs = 0  # this might be different than traj_batch
            for step in batch_steps:
                if timesteps[step + 1] < self.sde.epsilon:
                    continue
                t_.append(torch.full((x.shape[0],), timesteps[step + 1]).float())
                dts.append(torch.full((x.shape[0],), timesteps[step + 1] - timesteps[step]).float())
                xs.append(traj[step])
                xs_next.append(traj[step+1])
                bs += 1

            if len(t_) == 0:
                continue
            
            t_ = torch.cat(t_, dim=0).to(self.device).view(-1, 1, 1, 1)
            dts = torch.cat(dts, dim=0).to(self.device).view(-1, 1, 1, 1)
            xs = torch.cat(xs, dim=0).to(self.device)
            xs_next = torch.cat(xs_next, dim=0).to(self.device)

            g = self.sde.diffusion(t_, xs).to(self.device)

            f_posterior = -self.sde.drift(t_, xs) - g ** 2 * self.model(t_[:,0,0,0], xs) / self.sde.sigma(t_).view(-1, *[1]*len(D))
            
            # compute parameters for denoising step (wrt posterior)
            x_mean_posterior = xs + f_posterior * dts
            std = g * (-dts) ** (1 / 2)

            # compute step
            if backward:
                # retrieve original variance noise & compute step back
                variance_noise = (self.sde.drift(t_, xs) * dt) / std
                xs = (xs + self.sde.drift(t_, xs) * dt) + (std * variance_noise)
            else:
                # retrieve original variance noise & compute step fwd
                variance_noise = (xs_next - x_mean_posterior) / std
                xs = x_mean_posterior + std * variance_noise

            xs = xs.detach()

            # define distributions wrt posterior score model
            pf_post_dist = torch.distributions.Normal(x_mean_posterior, std)

            # compute log-likelihoods of reached pos wrt to posterior model
            logpf_posterior = pf_post_dist.log_prob(xs).sum(tuple(range(1, len(xs.shape))))

            # compute loss for posterior & accumulate gradients.
            partial_rtb = ((logpf_posterior + self.logZ) * correction.repeat(bs)).mean()
            partial_rtb.backward()

            if torch.any(torch.isnan(x)):
                print("Diffusion is not stable, NaN were produced. Stopped sampling.")
                break

        return True