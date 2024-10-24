import os 
import numpy as np
import torch
import torch.nn as nn
from torchcfm.models.unet.unet import UNetModelWrapper
import wandb
from tqdm import tqdm
import random

from sde import VPSDE
import utils

class RTBModel(nn.Module):
    def __init__(self, 
                 device,
                 reward_model,
                 prior_model, 
                 in_shape,
                 reward_args, 
                 id,
                 model_save_path,
                 load_ckpt = False,
                 load_ckpt_path = None,
                 entity = 'swish',
                 diffusion_steps=100, 
                 beta_start=1.0, 
                 beta_end=10.0):
        super().__init__()
        self.device = device
        self.sde = VPSDE(device = self.device, beta_schedule='cosine')
        self.steps = diffusion_steps
        self.reward_args = reward_args 
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = beta_start
        self.in_shape = in_shape 

        # for run name
        self.id = id 
        self.entity = entity 

        # Posterior noise model
        self.logZ = torch.nn.Parameter(torch.tensor(0.).to(self.device))
        self.model = UNetModelWrapper(
            dim = self.in_shape,
            num_res_blocks = 2,
            num_channels = 128,
            channel_mult = [1, 2, 2, 2],
            num_heads = 4,
            num_head_channels = 64,
            attention_resolutions = "16",
            dropout = 0.0,
        ).to(self.device)
        
        # Prior flow model pipeline
        self.prior_model = prior_model 

        self.reward_model = reward_model 

        self.model_save_path = os.path.expanduser(model_save_path)
        
        self.load_ckpt = load_ckpt 
        if load_ckpt_path is not None:
            self.load_ckpt_path = os.path.expanduser(load_ckpt_path)
        else:
            self.load_ckpt_path = load_ckpt_path 

    def save_checkpoint(self, model, optimizer, epoch, run_name):
        if self.model_save_path is None:
            print("Model save path not provided. Checkpoint not saved.")
            return
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        savedir = self.model_save_path + run_name + '/'
        os.makedirs(savedir, exist_ok=True)
        
        filepath = savedir + 'checkpoint_'+str(epoch)+'.pth'
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at {filepath}")

    def load_checkpoint(self, model, optimizer):
        if self.load_ckpt_path is None:
            print("Checkpoint path not provided. Checkpoint not loaded.")
            return model, optimizer
        
        checkpoint = torch.load(self.load_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {self.load_ckpt_path}")

        # get iteration number (number before .pth)
        it = int(self.load_ckpt_path.split('/')[-1].split('_')[-1].split('.')[0])
        print("Epoch number: ", it)
        return model, optimizer, it

    # linearly anneal between beta_start and beta_end 
    def get_beta(self, it, anneal, anneal_steps):
        if anneal and it < anneal_steps:
            beta = ((anneal_steps - it)/anneal_steps) * self.beta_start + (it / anneal_steps) * self.beta_end
        else:
            beta = self.beta_start 

        return beta 

    def log_reward(self, x, return_img=False):
        with torch.no_grad():
            img = self.prior_model(x)
            
            log_r = self.reward_model(img, *self.reward_args).to(self.device)
        if return_img:
            return log_r, img
        return log_r
        
    def batched_rtb(self, shape, learning_cutoff=.1, prior_sample=False):
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

            # vargrad
            self.logZ.data = (-logpf_posterior + logpf_prior + self.beta*logr_x_prime).mean()

            rtb_loss = 0.5 * (((logpf_posterior + self.logZ - logpf_prior - self.beta*logr_x_prime) ** 2) - learning_cutoff).relu()

            # compute correction
            clip_idx = ((logpf_posterior + self.logZ - logpf_prior - self.beta*logr_x_prime) ** 2) < learning_cutoff
            correction = (logpf_posterior + self.logZ - logpf_prior - self.beta*logr_x_prime)
            correction[clip_idx] = 0.

        self.batched_forward(
            shape=shape,
            traj=fwd_logs['traj'],
            correction=correction,
            batch_size=B,
        )

        return rtb_loss.detach().mean(), logr_x_prime.mean()
    
    def finetune(self, shape, n_iters=100000, learning_rate=5e-5, clip=0.1, wandb_track=False, prior_sample=False, anneal=False, anneal_steps=15000):
        B, *D = shape
        param_list = [{'params': self.model.parameters()}]
        optimizer = torch.optim.Adam(param_list, lr=learning_rate)
        run_name = self.id + '_beta_start_' + str(self.beta_start) + '_beta_end_' + str(self.beta_end) + '_anneal_' + str(anneal) + '_prior_sample_' + str(prior_sample)
        
        if self.load_ckpt:
            self.model, optimizer, load_it = self.load_checkpoint(self.model, optimizer)
        else:
            load_it = 0

        if wandb_track:
            wandb.init(
                project='cfm_posterior',
                entity=self.entity,
                save_code=True,
                name=run_name
            )
            hyperparams = {
                "learning_rate": learning_rate,
                "n_iters": n_iters,
                "reward_args": self.reward_args,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "anneal": anneal,
                "anneal_steps": anneal_steps
            }
            wandb.config.update(hyperparams)
            with torch.no_grad():
                x = torch.randn(10, *self.in_shape, device=self.device)
                img = self.prior_model(x)
                prior_reward = self.reward_model(img, *self.reward_args)
            wandb.log({"prior_samples": [wandb.Image(img[k], caption = prior_reward[k]) for k in range(len(img))]})
            
        prior_traj = False
        for it in range(load_it, n_iters):
            self.beta = self.get_beta(it, anneal, anneal_steps)
            optimizer.zero_grad()
            loss, logr = self.batched_rtb(shape=shape, prior_sample=prior_traj)

            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
            optimizer.step()
            if prior_sample:
                prior_traj = not prior_traj    
            
            if wandb_track: 
                if not it%100 == 0:
                    wandb.log({"loss": loss.item(), "logZ": self.logZ.detach().cpu().numpy(), "log_r": logr.item(), "epoch": it})
                else:
                    with torch.no_grad():
                        logs = self.forward(
                            shape=(10, *D),
                            steps=self.steps
                            )
                        x = logs['x_mean_posterior']
                        img = self.prior_model(x)
                        post_reward = self.reward_model(img, *self.reward_args)
                        wandb.log({"loss": loss.item(), "logZ": self.logZ.detach().cpu().numpy(), "log_r": logr.item(), "epoch": it, 
                                   "posterior_samples": [wandb.Image(img[k], caption=post_reward[k]) for k in range(len(img))]})

                        # save model and optimizer state
                        self.save_checkpoint(self.model, optimizer, it, run_name)
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

        for i, batch_steps in enumerate(utils.create_batches(steps, traj_batch)):

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