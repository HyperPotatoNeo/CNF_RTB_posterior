import copy
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchcfm.models.unet.unet import UNetModelWrapper
import wandb
from tqdm import tqdm
import random
from prior_models import MLP
from cleanfid import fid
import torchvision.utils as vutils
import torchvision.transforms as tf
import matplotlib.pyplot as plt

from sde import VPSDE, DDPM
import reward_models
import rtb_utils as utils

class RTBModel(nn.Module):
    def __init__(self, 
                 device,
                 reward_model,
                 prior_model, 
                 in_shape,
                 reward_args, 
                 id,
                 model_save_path,
                 langevin=False,
                 inference_type='vpsde',
                 tb=False,
                 load_ckpt=False,
                 load_ckpt_path=None,
                 entity='swish',
                 diffusion_steps=100, 
                 beta_start=1.0, 
                 beta_end=10.0,
                 loss_batch_size=64,
                 replay_buffer=None,
                 distilled_model_path=None,
                 posterior_architecture='unet'
                 ):
        super().__init__()
        self.device = device
        
        if inference_type == 'vpsde':
            self.sde = VPSDE(device = self.device, beta_schedule='cosine')
        else:
            self.sde = DDPM(device = self.device, beta_schedule='cosine')
        self.sde_type = self.sde.sde_type

        # load distilled version of the model if we are given a path
        if distilled_model_path is not None and os.path.exists(distilled_model_path):
            self.distilled_model = UNetModelWrapper(
                dim=self.in_shape,
                num_res_blocks=2,
                num_channels=128,
                channel_mult=[1, 2, 2, 2],
                num_heads=4,
                num_head_channels=64,
                attention_resolutions="16",
                dropout=0.0,
            ).to(self.device)  # random initialization
            distilled_checkpoint = torch.load(os.path.expanduser(distilled_model_path), map_location=self.device)
            self.distilled_model.load_state_dict(distilled_checkpoint['model_state_dict'])
        else:
            self.distilled_model = None

        self.steps = diffusion_steps
        self.reward_args = reward_args 
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = beta_start
        self.in_shape = in_shape
        self.loss_batch_size = loss_batch_size
        self.replay_buffer = replay_buffer
        self.use_rb = False if replay_buffer is None else True

        # for run name
        self.id = id 
        self.entity = entity 

        self.latent_prior = torch.distributions.Normal(torch.tensor(0.).to(self.device), torch.tensor(1.).to(self.device))
        self.tb = tb 

        # Posterior noise model
        self.logZ = torch.nn.Parameter(torch.tensor(0.).to(self.device))
        #if posterior_architecture == 'unet':
        self.model = UNetModelWrapper(
            dim=self.in_shape,
            num_res_blocks=2,
            num_channels=128,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.0,
        ).to(self.device)
        #elif posterior_architecture == 'mlp':
        #    self.model = MLP(dim = self.in_shape[0]).to(self.device)
        
        # Prior flow model pipeline
        self.prior_model = prior_model 

        self.reward_model = reward_model 

        if langevin: 
            self.num_classes = 10
            self.trainable_reward = reward_models.TrainableClassifierReward(in_shape = self.in_shape, 
                                                                  device = self.device, num_classes = 10)
            self.cls_optimizer = torch.optim.Adam(self.trainable_reward.parameters(), lr=5e-5)
        else:
            self.trainable_reward = None 

        self.langevin = langevin 
        
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

    def classifier_reward(self, x):
        if self.num_classes == 1:
            log_r_target = self.log_reward(x)
            log_r_pred = self.trainable_reward(x)
            loss = ((log_r_pred - log_r_target)**2).mean() 
        else:
            im = self.prior_model(x)
            target_log_probs = self.reward_model.get_class_logits(im, *self.reward_args)
            pred_log_probs = torch.nn.functional.log_softmax(self.trainable_reward(x))
            loss = torch.nn.CrossEntropyLoss()(pred_log_probs, target_log_probs.argmax(dim=-1)).mean()
        return loss

    def update_trainable_reward(self, x):
        if not self.langevin:
            print("Trainable reward not initialized.")
            return

        self.cls_optimizer.zero_grad()
        loss = self.classifier_reward(x)
        loss.backward()
        self.cls_optimizer.step()
        return

    def pretrain_trainable_reward(self, batch_size, n_iters = 100, learning_rate = 5e-5, wandb_track = False):
        if not self.langevin:
            print("Trainable reward not initialized.")
            return

        B = batch_size
        D = self.in_shape

        run_name = self.id + '_pretrain_reward_lr_' + str(learning_rate)

        if wandb_track:
            wandb.init(
                project='cfm_posterior',
                entity=self.entity,
                save_code=True,
                name=run_name
            )
            hyperparams = {
                "learning_rate": learning_rate,
                "reward_args": self.reward_args,
                "training reward": self.langevin
            }
            wandb.config.update(hyperparams)

        for i in range(n_iters):
            self.cls_optimizer.zero_grad()

            x = torch.randn(B, *D, device=self.device)

            if self.num_classes == 1:
                log_r_target = self.log_reward(x)
                log_r_pred = self.trainable_reward(x)
                loss = ((log_r_pred - log_r_target)**2).mean()
            
            else:
                im = self.prior_model(x)
                target_log_probs = self.reward_model.get_class_logits(im, *self.reward_args)
                pred_log_probs = torch.nn.functional.log_softmax(self.trainable_reward(x))
                loss =  torch.nn.CrossEntropyLoss()(pred_log_probs, target_log_probs.argmax(dim=-1)).mean()
            loss.backward()
            self.cls_optimizer.step()

            if wandb_track:
                wandb.log({"loss": loss.item(), "iter": i})
                if i%100 == 0:
                    with torch.no_grad():
                        x = torch.randn(20, *self.in_shape, device=self.device)
                        img = self.prior_model(x)
                        prior_reward = self.reward_model(img, *self.reward_args)
                        trained_reward = self.trainable_reward(x).log_softmax(dim=-1)
                        wandb.log({"prior_samples": [wandb.Image(img[k], caption = "logR(x1) = {}, TrainlogR(z) = {}".format(prior_reward[k], trained_reward[k])) for k in range(len(img))]})



            if i%10 == 0:
                print("Iter: ", i, "Loss: ", loss.item())

        return 

    # linearly anneal between beta_start and beta_end 
    def get_beta(self, it, anneal, anneal_steps):
        if anneal and it < anneal_steps:
            beta = ((anneal_steps - it)/anneal_steps) * self.beta_start + (it / anneal_steps) * self.beta_end
        elif anneal:
            beta = self.beta_end
        else:
            beta = self.beta_start 

        return beta 

    # return shape is: (B, *D)
    def prior_log_prob(self, x):
        return self.latent_prior.log_prob(x).sum(dim = tuple(range(1, len(x.shape))))

    def log_reward(self, x, return_img=False):
        with torch.no_grad():
            img = self.prior_model(x)
            
            log_r = self.reward_model(img, *self.reward_args).to(self.device)
        if return_img:
            return log_r, img
        return log_r
        
    def batched_rtb(self, shape, learning_cutoff=.1, prior_sample=False, rb_sample=False):
        # first pas through, get trajectory & loss for correction
        B, *D = shape
        x_1 = None
        if rb_sample:
            x_1, logr_x_prime = self.replay_buffer.sample(shape[0])

        with torch.no_grad():
            # run whole trajectory, and get PFs
            if self.sde_type == 'vpsde':
                fwd_logs = self.forward(
                    shape=shape,
                    steps=self.steps,
                    save_traj=True,  # save trajectory fwd
                    prior_sample=prior_sample,
                    x_1 = x_1,
                    backward = rb_sample
                )
            elif self.sde_type == 'ddpm':
                fwd_logs = self.forward_ddpm(
                    shape=shape,
                    steps=self.steps,
                    save_traj=True,  # save trajectory fwd
                    prior_sample=prior_sample,
                    x_1 = x_1,
                    backward = rb_sample
                )
            x_mean_posterior, logpf_prior, logpf_posterior = fwd_logs['x_mean_posterior'], fwd_logs['logpf_prior'], fwd_logs['logpf_posterior']

            if not rb_sample:
                logr_x_prime = self.log_reward(x_mean_posterior)

            self.logZ.data = (-logpf_posterior + logpf_prior + self.beta*logr_x_prime).mean()

            rtb_loss = 0.5 * (((logpf_posterior + self.logZ - logpf_prior - self.beta*logr_x_prime) ** 2) - learning_cutoff).relu()

            # Add to replay_buffer
            if not rb_sample:
                self.replay_buffer.add(x_mean_posterior.detach(), logr_x_prime.detach(), rtb_loss.detach())
            
            # compute correction
            clip_idx = ((logpf_posterior + self.logZ - logpf_prior - self.beta*logr_x_prime) ** 2) < learning_cutoff
            correction = (logpf_posterior + self.logZ - logpf_prior - self.beta*logr_x_prime)
            correction[clip_idx] = 0.

        if self.sde_type == 'vpsde':
            self.batched_forward(
                shape=shape,
                traj=fwd_logs['traj'],
                correction=correction,
                batch_size=B
            )
        elif self.sde_type == 'ddpm':

            self.batched_forward_ddpm(
                shape=shape,
                traj=fwd_logs['traj'],
                correction=correction,
                batch_size=B
            )

        return rtb_loss.detach().mean(), logr_x_prime.mean()
    
    def finetune(self, shape, n_iters=100000, learning_rate=5e-5, clip=0.1, wandb_track=False, prior_sample_prob=0.0, replay_buffer_prob=0.0, anneal=False, anneal_steps=15000, exp='sd3_align', compute_fid=False, class_label=0):
        B, *D = shape
        param_list = [{'params': self.model.parameters()}]
        optimizer = torch.optim.Adam(param_list, lr=learning_rate)
        run_name = self.id + '_sde_' + self.sde_type +'_steps_' + str(self.steps) + '_lr_' + str(learning_rate) + '_beta_start_' + str(self.beta_start) + '_beta_end_' + str(self.beta_end) + '_anneal_' + str(anneal) + '_prior_prob_' + str(prior_sample_prob) + '_rb_prob_' + str(replay_buffer_prob) + '_langevin_' + str(self.langevin)
        
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
                num_log = 25 #if 'cifar' in exp else 10
                x = torch.randn(num_log, *self.in_shape, device=self.device)
                img = self.prior_model(x)
                prior_reward = self.reward_model(img, *self.reward_args)
            log_dict = {"prior_reward": prior_reward.mean(),"prior_samples": [wandb.Image(img[k], caption = prior_reward[k]) for k in range(len(img))]}
            if 'cifar' in exp:
                if 'gan' in exp:
                    title = 'SN-GAN Prior'
                else:
                    title = 'Flow Prior'
                grid = vutils.make_grid(img, nrow=5, padding=1)
                # The grid is of shape (3, H, W). We need to permute for matplotlib.
                plt.figure(figsize=(8, 8))
                plt.imshow(grid.permute(1, 2, 0).cpu())
                plt.axis('off')
                plt.title(title, fontsize=20)
                log_dict['prior_grid'] = wandb.Image(plt)
                plt.close()
            wandb.log(log_dict)
            
        if 'sd3' in exp or 'ffhq' in exp:
            print('SAVING IMAGES:')
            generated_images_dir = self.model_save_path + run_name + '/' + 'prior_images'
            os.makedirs(generated_images_dir, exist_ok=True)
            for k in range(10):
                with torch.no_grad():
                    logs = self.forward(
                            shape=(10, *D),
                            steps=self.steps
                            )
                    x = logs['x_mean_posterior']
                    img_x = self.prior_model(x)
                for i, img_tensor in enumerate(img_x):
                    img_tensor.save(os.path.join(generated_images_dir, f'{k*10 + i}.png'))
            
        for it in range(load_it, n_iters):
            prior_traj = False
            rb_traj = False
            rand_n = np.random.uniform()
            # No replay buffer for first 10 iters
            if rand_n < prior_sample_prob:
                prior_traj = True
            elif it > 5 and rand_n < prior_sample_prob + replay_buffer_prob:
                rb_traj = True
                
            self.beta = self.get_beta(it, anneal, anneal_steps)
            optimizer.zero_grad()
            loss, logr = self.batched_rtb(shape=shape, prior_sample=prior_traj, rb_sample=rb_traj)

            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
            optimizer.step() 
            
            # trainable reward classifier
            if self.langevin:
                x_1, logr_x_prime = self.replay_buffer.sample(shape[0])
                self.update_trainable_reward(x_1)

            if wandb_track: 
                if not it % 100 == 0:
                    wandb.log({"loss": loss.item(), "logZ": self.logZ.detach().cpu().numpy(), "log_r": logr.item(), "epoch": it})
                else:
                    with torch.no_grad():
                        num_log = 25 #if 'cifar' in exp else 10
                        if self.sde_type == 'vpsde':
                            logs = self.forward(
                                shape=(num_log, *D),
                                steps=self.steps
                            )
                        elif self.sde_type == 'ddpm':
                            logs = self.forward_ddpm(
                                shape=(num_log, *D),
                                steps=self.steps
                            )

                        x = logs['x_mean_posterior']
                        img = self.prior_model(x)
                        post_reward = self.reward_model(img, *self.reward_args)
                        if self.langevin:
                            log_dict = {"prior_samples": [wandb.Image(img[k], caption = "logR(x1) = {}, TrainlogR(z) = {}".format(prior_reward[k], trained_reward[k])) for k in range(len(img))]}
                        else:
                            log_dict = {"posterior_reward": post_reward.mean(), "loss": loss.item(), "logZ": self.logZ.detach().cpu().numpy(), "log_r": logr.item(), "epoch": it,
                                   "posterior_samples": [wandb.Image(img[k], caption=post_reward[k]) for k in range(len(img))]}
                        if 'cifar' in exp:
                            if 'gan' in exp:
                                title = 'SN-GAN Posterior (Bird)'
                            else:
                                title = 'Flow Posterior (Truck)'
                            grid = vutils.make_grid(img, nrow=5, padding=1)
                            # The grid is of shape (3, H, W). We need to permute for matplotlib.
                            plt.figure(figsize=(8, 8))
                            plt.imshow(grid.permute(1, 2, 0).cpu())
                            plt.axis('off')
                            plt.title(title, fontsize=20)
                            log_dict['posterior_grid'] = wandb.Image(plt)
                            plt.close()
                        
                        if it%500 == 0 and it > 0 and ('sd3' in exp or 'ffhq' in exp):
                            print('SAVING IMAGES:')
                            generated_images_dir = self.model_save_path + run_name + '/' + 'posterior_images'
                            os.makedirs(generated_images_dir, exist_ok=True)
                            for k in range(10):
                                with torch.no_grad():
                                    logs = self.forward(
                                            shape=(10, *D),
                                            steps=self.steps
                                            )
                                    x = logs['x_mean_posterior']
                                    img_x = self.prior_model(x)
                                for i, img_tensor in enumerate(img_x):
                                    img_tensor.save(os.path.join(generated_images_dir, f'{k*10 + i}.png'))

                        if it%1000 == 0 and 'cifar' in exp and compute_fid:# and it>0:
                            print('COMPUTING FID:')
                            if 'improve' in exp:
                                class_label = 20
                            generated_images_dir = 'fid/' + exp + '_cifar10_class_' + str(class_label)
                            true_images_dir = 'fid/cifar10_class_' + str(class_label)
                            K = 600 if 'improve' in exp else 60
                            for k in range(K):
                                with torch.no_grad():
                                    logs = self.forward(
                                        shape=(100, *D),
                                        steps=self.steps
                                        )
                                    x = logs['x_mean_posterior']
                                    img_fid = self.prior_model(x)
                                    for i, img_tensor in enumerate(img_fid):
                                        img_pil = transforms.ToPILImage()(img_tensor)
                                        img_pil.save(os.path.join(generated_images_dir, f'{k*100 + i}.png'))
                            fid_score = fid.compute_fid(generated_images_dir, true_images_dir)
                            log_dict['fid'] = fid_score
                        wandb.log(log_dict)

                        # save model and optimizer state
                        self.save_checkpoint(self.model, optimizer, it, run_name)
    
    def get_langevin_correction(self, x):
        # add gradient wrt x of trainable reward to model
        if self.langevin:
            with torch.set_grad_enabled(True):
                x.requires_grad = True
                log_rx = self.trainable_reward(x)
                grad_log_rx = torch.autograd.grad(log_rx.sum(), x, create_graph=True)[0]
                
            lp_correction = grad_log_rx#.detach()
        else:
            lp_correction = torch.zeros_like(x)
        return lp_correction.detach()

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
            time_discretisation='uniform', #uniform/random
            distilled=False
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
            t = torch.zeros(B).to(self.device) + self.sde.epsilon
        else:
            x = self.sde.prior(D).sample([B]).to(self.device)
            t = torch.ones(B).to(self.device) * self.sde.T

        # assume x is gaussian noise
        normal_dist = torch.distributions.Normal(torch.zeros((B,) + tuple(D), device=self.device),
                                                 torch.ones((B,) + tuple(D), device=self.device))

        xT = copy.copy(x)

        if not distilled:

            logpf_posterior = 0*normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
            logpb = 0*normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)#torch.zeros_like(logpf_posterior)
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
                    x = (x - self.sde.drift(t, x) * dt) + (std * torch.randn_like(x))
                else:
                    x_prev = x.detach()

                t += dt * (-1.0 if backward else 1.0)
                if t[0] < self.sde.epsilon:  # Accounts for numerical error in the way we discretize t.
                    continue # continue instead of break because it works for forward and backward

                g = self.sde.diffusion(t, x)

                lp_correction = self.get_langevin_correction(x)
                posterior_drift = -self.sde.drift(t, x) - (g ** 2) * (self.model(t, x) + lp_correction) / self.sde.sigma(t).view(-1, *[1]*len(D))

                f_posterior = posterior_drift
                # compute parameters for denoising step (wrt posterior)
                x_mean_posterior = x + f_posterior * dt# * (-1.0 if backward else 1.0)
                std = g * (np.abs(dt)) ** (1 / 2)

                # compute step
                if prior_sample and not backward:
                    x = x - self.sde.drift(t, x) * dt + std * torch.randn_like(x)
                elif not backward:
                    x = x_mean_posterior + std * torch.randn_like(x)
                x = x.detach()

                # compute parameters for pb
                #t_next = t + dt
                #pb_drift = self.sde.drift(t_next, x)
                #x_mean_pb = x + pb_drift * (-dt)
                if backward:
                    pb_drift = -self.sde.drift(t, x)
                    x_mean_pb = x + pb_drift * (dt)
                else:
                    pb_drift = -self.sde.drift(t, x_prev)
                    x_mean_pb = x_prev + pb_drift * (dt)
                #x_mean_pb = x_prev + pb_drift * (dt)
                pb_std = g * (np.abs(dt)) ** (1 / 2)

                if save_traj:
                    traj.append(x.clone())

                pf_post_dist = torch.distributions.Normal(x_mean_posterior, std)
                pb_dist = torch.distributions.Normal(x_mean_pb, pb_std)

                # compute log-likelihoods of reached pos wrt to prior & posterior models
                #logpb += pb_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
                if backward:
                    logpb += pb_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
                    logpf_posterior += pf_post_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
                else:
                    logpb += pb_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))
                    logpf_posterior += pf_post_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))

                if torch.any(torch.isnan(x)):
                    print("Diffusion is not stable, NaN were produced. Stopped sampling.")
                    break
            if backward:
                traj = list(reversed(traj))
            logs = {
                'x_mean_posterior': x,  #,x_mean_posterior,
                'logpf_prior': logpb,
                'logpf_posterior': logpf_posterior,
                'traj': traj if save_traj else None,
                'x0': xT
            }
        else:
            x_distilled = self.distilled_model(
                t=torch.zeros(B, device=self.device),
                x=xT
            )
            logs = {
                'x_mean_posterior': x_distilled,  #,x_mean_posterior,
                'x0': xT
            }
        return logs
    
    def forward_ddpm(
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
        DDPM Update for SDE

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
            # timesteps = np.flip(timesteps)
            t = torch.zeros(B).to(self.device) + self.sde.epsilon
        else:
            x = self.sde.prior(D).sample([B]).to(self.device)
            t = torch.ones(B).to(self.device) * self.sde.T

        # assume x is gaussian noise
        normal_dist = torch.distributions.Normal(torch.zeros((B,) + tuple(D), device=self.device),
                                                 torch.ones((B,) + tuple(D), device=self.device))

        logpf_posterior = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        logpb = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
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
                x = x + self.sde.drift(t, x, np.abs(dt)) + std * torch.randn_like(x)
            else:
                x_prev = x.detach()
                
            t += dt * (-1.0 if backward else 1.0)
            if t[0] < self.sde.epsilon:  # Accounts for numerical error in the way we discretize t.
                continue # continue instead of break because it works for forward and backward
            
            std = self.sde.diffusion(t, x, np.abs(dt))
            
            lp_correction = self.get_langevin_correction(x)
            posterior_drift = self.sde.drift(t, x, np.abs(dt)) + self.model(t, x) + lp_correction #/ self.sde.sigma(t).view(-1, *[1]*len(D))
            f_posterior = posterior_drift
            # compute parameters for denoising step (wrt posterior)
            x_mean_posterior = x + f_posterior 
            #std = g * (np.abs(dt)) ** (1 / 2)

            # compute step
            if prior_sample and not backward:
                x = x + self.sde.drift(t, x, np.abs(dt)) + std * torch.randn_like(x)
            elif not backward:
                x = x_mean_posterior + std * torch.randn_like(x)
            x = x.detach()
            
            # compute parameters for pb
            #t_next = t + dt
            #pb_drift = self.sde.drift(t_next, x)
            #x_mean_pb = x + pb_drift * (-dt)
            if backward:
                pb_drift = self.sde.drift(t, x, np.abs(dt))
            else:
                pb_drift = self.sde.drift(t, x_prev, np.abs(dt))
            x_mean_pb = x_prev + pb_drift
            pb_std = self.sde.diffusion(t, x_prev, np.abs(dt)) #g * (np.abs(dt)) ** (1 / 2)

            if save_traj:
                traj.append(x.clone())
                
            pf_post_dist = torch.distributions.Normal(x_mean_posterior, std)
            pb_dist = torch.distributions.Normal(x_mean_pb, pb_std)

            # compute log-likelihoods of reached pos wrt to prior & posterior models
            #logpb += pb_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
            if backward:
                logpb += pb_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
                logpf_posterior += pf_post_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
            else:
                logpb += pb_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))
                logpf_posterior += pf_post_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))

            if torch.any(torch.isnan(x)):
                print("Diffusion is not stable, NaN were produced. Stopped sampling.")
                break
        
        if backward:
            traj = list(reversed(traj))
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

            lp_correction = self.get_langevin_correction(xs)
            f_posterior = -self.sde.drift(t_, xs) - g ** 2 * (self.model(t_[:,0,0,0], xs) + lp_correction) / self.sde.sigma(t_).view(-1, *[1]*len(D))
            
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
    
    
    def batched_forward_ddpm(
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

            std = self.sde.diffusion(t_, xs, -dts).to(self.device)

            lp_correction = self.get_langevin_correction(xs)
            f_posterior = self.sde.drift(t_, xs, -dts) + self.model(t_[:,0,0,0], xs) + lp_correction #/ self.sde.sigma(t_).view(-1, *[1]*len(D))
            
            # compute parameters for denoising step (wrt posterior)
            x_mean_posterior = xs + f_posterior 
            std = std #g * (-dts) ** (1 / 2)

            
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

    def distill(
            self,
            shape,
            distilled_ckpt_path,
            teacher_ckpt_filename,
            teacher_ckpt_path,
            n_iters=10000,
            learning_rate=1e-4,
            save_interval=500,
            wandb_track=False
    ):
        """
        Distills a fine-tuned diffusion model (self.model) into a single-step generator,
        but uses the exact same architecture class as self.model (e.g., a UNet).

        1) Loads teacher (fine-tuned) weights from teacher_ckpt_path into self.model.
        2) Instantiates a new random UNet (same class as self.model) to become our 'distilled_model'.
        3) Trains this distilled UNet to replicate teacher samples (RTB style).
        4) Saves the distilled model every few epochs (save_interval).
        5) Logs training metrics to wandb if wandb_track=True.

        Args:
            shape:             (B, *D) shape for training batches
            distilled_ckpt_path: Path to save distilled checkpoints
            teacher_ckpt_path:   Path to the fine-tuned teacher weights
            n_iters:          Number of distillation iterations
            learning_rate:    Learning rate for the distilled model
            save_interval:    Interval for saving the distilled model
            wandb_track:      Whether to track metrics in wandb
        """
        import copy
        print('Started distillation.')
        # -------------------------------------------------------------------------
        # 1) LOAD TEACHER (FINE-TUNED) WEIGHTS
        # -------------------------------------------------------------------------
        if teacher_ckpt_path is not None and teacher_ckpt_filename is not None:
            teacher_checkpoint = torch.load(os.path.expanduser(teacher_ckpt_path + teacher_ckpt_filename), map_location=self.device)
            self.model.load_state_dict(teacher_checkpoint['model_state_dict'])
            print(f"Loaded teacher (fine-tuned) checkpoint from {teacher_ckpt_path}")
        else:
            print("No teacher_ckpt_path provided; using self.model as-is.")

        # -------------------------------------------------------------------------
        # 2) INSTANTIATE A NEW, RANDOMLY INITIALIZED UNET (SAME CLASS AS self.model)
        # -------------------------------------------------------------------------
        # For demonstration, let's assume self.model is a UNetModelWrapper.
        # We'll create a fresh instance below:
        self.distilled_model = UNetModelWrapper(
            dim=self.in_shape,
            num_res_blocks=2,
            num_channels=128,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.0,
        ).to(self.device)  # random initialization

        # Define optimizer
        distilled_optimizer = torch.optim.Adam(self.distilled_model.parameters(), lr=learning_rate)

        # -------------------------------------------------------------------------
        # 3) TRAIN THE SINGLE-STEP GENERATOR TO REPLICATE TEACHER SAMPLES
        # -------------------------------------------------------------------------
        run_name = f"{self.id}_distillation_{n_iters}_it_{learning_rate}_lr"

        if wandb_track:
            wandb.init(
                project='cfm_posterior',
                entity=self.entity,
                save_code=True,
                name=run_name
            )
            wandb.config.update({
                "learning_rate": learning_rate,
                "n_iters": n_iters,
                "distill_run_name": run_name,
                "student_arch": "same_unet_as_teacher",
                "sde_type": self.sde_type
            })

        for it in range(n_iters):
            # (a) Sample random noise for teacher's multi-step generation
            B = shape[0]

            # (b) Generate teacher sample (multi-step) using VPSDE or DDPM
            with torch.no_grad():
                if self.sde_type == 'vpsde':
                    teacher_logs = self.forward(
                        shape=(B, *self.in_shape),
                        steps=self.steps
                    )
                elif self.sde_type == 'ddpm':
                    teacher_logs = self.forward_ddpm(
                        shape=(B, *self.in_shape),
                        steps=self.steps
                    )
                else:
                    raise ValueError(f"Unknown sde_type: {self.sde_type}")

                x_teacher = teacher_logs['x_mean_posterior']
                z = teacher_logs['x0']

            # # ---------------------------------------------------------------------
            # # (b) Sample random noise for the student (one-step)
            # # ---------------------------------------------------------------------
            # z = torch.randn(B, *self.in_shape, device=self.device)

            # (c) Distilled model forward pass in ONE step
            x_distilled = self.distilled_model(
                t=torch.zeros(B, device=self.device),
                x=z
            )

            # (e) Distillation loss: e.g. MSE between teacher sample & distilled sample
            mse = torch.nn.functional.mse_loss(x_distilled, x_teacher)

            # # Flatten for pairwise distance (B, C*H*W)
            # B = x_distilled.size(0)
            # x_flat = x_distilled.view(B, -1)  # shape: (B, D)
            # # Compute pairwise L2 distances; shape: (B, B)
            # distances = torch.cdist(x_flat, x_flat, p=2)
            #
            # # Optional: exclude diagonal (distance from a sample to itself = 0)
            # non_diag_mask = ~torch.eye(B, dtype=bool, device=x_distilled.device)
            # non_diag_distances = distances[non_diag_mask]
            # mean_pairwise_dist = non_diag_distances.mean()
            #
            # mean_pairwise_dist = distances.mean()

            dvar = x_distilled.var(0).mean()
            mean_pairwise_dist = torch.relu(-dvar + .9)

            # We want to maximize pairwise diversity, so we subtract it from the total loss
            diversity_weight = 1.0  # hyperparameter to tune
            diversity_loss = diversity_weight * mean_pairwise_dist

            # Combine the two losses
            loss = mse + diversity_loss


            distilled_optimizer.zero_grad()
            loss.backward()
            distilled_optimizer.step()

            # ---------------------------------------------------------------------
            # (f) Logging
            # ---------------------------------------------------------------------
            if wandb_track:
                wandb.log({
                    "distill_loss": loss.item(),
                    "iter": it,
                    "regularizer_loss": mean_pairwise_dist,
                    "dist_variance": dvar,
                    "mse": mse
                })

            if it % 100 == 0:
                print(f"[Distill Iter {it}/{n_iters}] Loss = {loss.item():.6f}")

                # Generate and log final images from both teacher & student
                with torch.no_grad():
                    # Teacher images
                    teacher_img_batch = self.prior_model(x_teacher)
                    # Student images
                    student_img_batch = self.prior_model(x_distilled)

                if wandb_track:
                    # Log side-by-side samples: teacher vs student

                    if wandb_track:
                        # Take the first 9 images from student_img_batch (or teacher_img_batch)
                        # and arrange them in a 3x3 grid.
                        grid_student = vutils.make_grid([safe_to_tensor(img).float() for img in student_img_batch[:9]], nrow=3, padding=2, normalize=True)
                        grid_teacher = vutils.make_grid([safe_to_tensor(img).float() for img in teacher_img_batch[:9]], nrow=3, padding=2, normalize=True)

                        # Log that grid image to wandb
                        wandb.log({
                            "student_3x3_grid": wandb.Image(grid_student, caption="Distilled Samples"),
                            "teacher_3x3_grid": wandb.Image(grid_teacher, caption="Diffusion Samples"),
                            "iter": it
                        })

            # ---------------------------------------------------------------------
            # 4) SAVE CHECKPOINT EVERY `save_interval`
            # ---------------------------------------------------------------------
            if it % save_interval == 0 and it > 0:
                os.makedirs(distilled_ckpt_path, exist_ok=True)
                ckpt_filename = os.path.join(distilled_ckpt_path, f"distilled_checkpoint_{it}.pth")
                torch.save({
                    'distilled_model_state_dict': self.distilled_model.state_dict(),
                    'optimizer_state_dict': distilled_optimizer.state_dict()
                }, ckpt_filename)
                print(f"Distilled UNet checkpoint saved at iteration {it} -> {ckpt_filename}")

        print("Distillation complete. Your student model (same class as teacher) is in self.distilled_model.")

def safe_to_tensor(img):
    if isinstance(img, torch.Tensor):
        # Already a tensor, so no conversion needed
        return img
    else:
        # Convert PIL Image or ndarray to tensor
        return tf.ToTensor()(img)