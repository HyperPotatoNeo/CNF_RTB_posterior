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
from sde import VPSDE, DDPM, MemorylessSDE
import reward_models
import utils



# Lean adjoint matching (am) with memoryless flow matching SDE
# Implemeted based of paper "Adjoint Matching (Domingo-Enrich et al., 2024)"

class RTBModel(nn.Module):
    def __init__(self, 
                 device,
                 reward_model,
                 prior_model, 
                 model,
                 in_shape,
                 reward_args, 
                 id,
                 model_save_path,
                 langevin = False,
                 inference_type = 'vpsde',
                 tb = False,
                 load_ckpt = False,
                 load_ckpt_path = None,
                 entity = 'swish',
                 diffusion_steps=100, 
                 beta_start=1.0, 
                 beta_end=10.0,
                 loss_batch_size=64,
                 replay_buffer=None,
                 posterior_architecture='unet',
                 detach_freq=0.8):
        super().__init__()
        self.device = device
        
        self.sde = MemorylessSDE(device = self.device)
        self.sde_type = self.sde.sde_type

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

        self.tb = tb 

        # Posterior noise model
        self.logZ = torch.nn.Parameter(torch.tensor(0.).to(self.device))
        self.model = model


        self.prior_model = prior_model
        tol = 1e-5

        self.reward_model = reward_model 
        self.num_backprop_steps = int((1-detach_freq) * self.steps) 

        self.trainable_reward = None 
        
        self.model_save_path = os.path.expanduser(model_save_path)
        
        self.load_ckpt = load_ckpt 
        if load_ckpt_path is not None:
            self.load_ckpt_path = os.path.expanduser(load_ckpt_path)
        else:
            self.load_ckpt_path = load_ckpt_path 

    @staticmethod
    def select_random_time_steps(total_steps, num_backprop_steps=4):
        """
        Randomly selects `num_backprop_steps` unique time steps from `0` to `total_steps - 1`.
        
        Args:
            total_steps (int): Total number of time steps.
            num_backprop_steps (int): Number of time steps to select for backpropagation.
        
        Returns:
            set: A set containing the indices of selected time steps.
        """
        if num_backprop_steps > total_steps:
            raise ValueError("Number of backpropagation steps cannot exceed total steps.")
        
        selected_steps = set(random.sample(range(total_steps), num_backprop_steps))
        return selected_steps


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

    # return shape is: (B, *D)
    def prior_log_prob(self, x):
        return self.latent_prior.log_prob(x).sum(dim = tuple(range(1, len(x.shape))))

    def log_reward(self, x, return_img=False, return_grad=False):
        if return_grad:
            img = (x * 127.5 + 128).clip(0, 255)
            
            log_r = self.reward_model(img, *self.reward_args).to(self.device)           
            return log_r
        else:
            with torch.no_grad():
                img = (x * 127.5 + 128).clip(0, 255)
                
                log_r = self.reward_model(img, *self.reward_args).to(self.device)
        if return_img:
            return log_r, img
        return log_r

  
    

    def batched_adjoint_matching(self, shape, steps=20, dt=None, num_backprop_steps=4):
        """
        Perform one iteration of Adjoint Matching with selective backpropagation through randomly selected time steps.
        
        Args:
            shape (tuple): Shape of the input tensor (including batch size).
            steps (int): Total number of time steps.
            dt (float, optional): Time step size. If None, defaults to 1.0 / steps.
            num_backprop_steps (int): Number of time steps to backpropagate through.
        
        Returns:
            tuple: (loss, mean_reward)
        """
        B, *D = shape
        
        # 1) Randomly select time steps for memory-efficient backpropagation
        selected_steps = self.select_random_time_steps(steps, num_backprop_steps)
        
        # 2) Forward pass with "save_traj=True" to store all states
        with torch.no_grad():
            fwd_logs = self.forward(
                shape=shape,
                steps=steps,
                save_traj=True,   
                prior_sample=False 
            )

        traj = fwd_logs['traj']  # List of tensors from X_0 to X_steps
        x_final = traj[-1]       # X_steps
        
        if dt is None:
            dt = 1.0 / steps

 
        # 4) Initialize adjoint states
        adj = [None] * (steps + 1)
        
        # Final boundary condition: a_steps = -lambda * grad(r(X_final))
        x_final.requires_grad_(True)

        # We want to sample proportional to the "reward": R(x): P(X) \propto R(x) = exp(r(x)).
        # r(x) = log R(x)
        reward_final = self.log_reward(x_final, return_grad=True)  
        
        lambda_scale = self.beta  # Hyperparameter lambda
        g_final = (-lambda_scale) * reward_final.sum()
        
        # Compute gradient of g_final w.r.t x_final
        grad_final = torch.autograd.grad(g_final, x_final, create_graph=False)[0]
        adj[steps] = grad_final.detach()  # a_steps

        # 5) Backward pass to compute adjoint states
        for k in reversed(range(steps)):
            x_curr = traj[k+1].requires_grad_(True)
            
            # b_base = 2 * prior_model(t_{k+1}, x_curr) + sde.drift(t_{k+1}, x_curr)
            # For simplicity, treat time_{k+1} = (k+1)/steps
            tval = torch.ones(B, device=x_curr.device) * ((k+1)/steps)
            
            b_base = 2.*self.prior_model.drift(tval, x_curr) + self.sde.drift(tval, x_curr)
            # compute Jacobian-vector product: a_{k+1}^T grad_x b_base
            # => we can do a manual trick using torch.autograd.grad:
            Jv = (b_base * adj[k+1].detach()).sum()  # sum over features
            dJdx = torch.autograd.grad(Jv, x_curr, create_graph=False)[0]  # shape [B, *D]
            # now accumulate:
            a_k = adj[k+1] + dt * dJdx
            adj[k] = a_k.detach()

        # 6) Compute the Mean Squared Error (MSE) loss only for selected steps
        MSE = torch.zeros([], device=x_final.device)

        for k in selected_steps:
            X_k = traj[k].detach()          # X_k (stopgrad)
            tval = torch.ones(B, device=X_k.device) * (k / steps)
            sigma_k = self.sde.sigma(tval)  # sigma(t_k)

            # Compute posterior and prior drifts
            f_post = 2.0 * self.model(tval, X_k) + self.sde.drift(tval, X_k)
            
            with torch.no_grad():
                f_prior = 2.0 * self.prior_model.drift(tval, X_k) + self.sde.drift(tval, X_k)
            

            # Compute control
            ctrl = (f_post - f_prior) / (sigma_k.view(-1, *[1]*len(D)) + 1e-8)
            
            # Compute the difference between control and target (-sigma * adj)
            lean_adj = adj[k].detach() # stopgrad
            diff = ctrl + sigma_k.view(-1, *[1]*len(D)) * lean_adj
            
            # Accumulate MSE loss
            MSE_k = 0.5 * diff.square().sum(dim=tuple(range(1, diff.ndim))).mean()  # Sum over spatial dimensions
            MSE_k.backward()
            
            MSE += MSE_k.item()  # 7) Backpropagate the loss
        MSE = MSE/len(selected_steps)
        
        
        return MSE.detach(), reward_final.detach(), fwd_logs
    
                
    
    def finetune(self, shape, n_iters=100000, learning_rate=5e-5, clip=0.1, wandb_track=False, 
                prior_sample_prob=0.0, replay_buffer_prob=0.0, anneal=False, anneal_steps=15000, 
                exp='sd3_align', compute_fid=False, class_label=0):
        """
        Fine-tunes the model using Adjoint Matching with selective backpropagation through random time steps.
        
        Args:
            shape (tuple): Shape of the input tensor (including batch size).
            steps (int): Total number of time steps.
            n_iters (int): Number of training iterations.
            learning_rate (float): Learning rate for the optimizer.
            clip (float): Maximum norm for gradient clipping.
            wandb_track (bool): Whether to track experiments with Weights & Biases.
            prior_sample_prob (float): Probability of sampling from the prior model.
            replay_buffer_prob (float): Probability of sampling from the replay buffer.
            anneal (bool): Whether to anneal beta.
            anneal_steps (int): Number of steps for annealing.
            exp (str): Experiment name.
            compute_fid (bool): Whether to compute FID scores.
            class_label (int): Class label for FID computation (if applicable).
            num_backprop_steps (int): Number of time steps to backpropagate through (default: 4).
        """
        
        B, *D = shape
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        run_name = (f"{self.id}_AdjMatching_steps_{self.steps}_lr_{learning_rate}"
                    f"_beta_start_{self.beta_start}_beta_end_{self.beta_end}"
                    f"_anneal_{anneal}")
        
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
                "anneal_steps": anneal_steps,
                "num_backprop_steps": self.num_backprop_steps
            }
            wandb.config.update(hyperparams)
            with torch.no_grad():
                img = self.integration(self.prior_model, 10, self.in_shape, steps=self.steps, ode = True)
                prior_reward = self.reward_model(img, *self.reward_args)
            wandb.log({"prior_samples": [wandb.Image(img[k], caption=prior_reward[k]) for k in range(len(img))]})

        for it in range(load_it, n_iters):
            optimizer.zero_grad()
            
            # Pass the number of backprop steps
            loss, logr, logs = self.batched_adjoint_matching(shape, steps=self.steps, num_backprop_steps=self.num_backprop_steps)
            
            logZ = (self.beta * logr + logs['logpf_prior'] - logs['logpf_posterior']).mean()

            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
            self.beta = self.get_beta(it, anneal, anneal_steps)
            optimizer.step() 
            
            if wandb_track: 
                if it % 100 != 0:
                    wandb.log({"loss": loss.item(), "log_r": logr.mean().item(), "epoch": it, "logZ": logZ.item()})
                else:
                    with torch.no_grad():
                        logs = self.forward(
                            shape=(10, *D),
                            steps=self.steps,
                            ode = True,
                        )
                        x = logs['x_mean_posterior']
                        img = (x * 127.5 + 128).clip(0, 255)
                        post_reward = self.reward_model(img, *self.reward_args)
  
                        logZ_ode = (self.beta * post_reward + logs['logpf_prior'] - logs['logpf_posterior']).mean() 

                        log_dict = {
                            "loss": loss.item(),
                            "log_r_ode": post_reward.mean().item(),
                            "epoch": it,
                            "logZ_ode": logZ_ode.item(),
                            "posterior_samples": [wandb.Image(img[k], caption=post_reward[k]) for k in range(len(img))]
                        }

                        if it % 1000 == 0 and 'cifar' in exp and compute_fid:
                            print('COMPUTING FID:')
                            generated_images_dir = f'fid/{exp}_am_ot_cifar10_class_{class_label}'
                            true_images_dir = f'fid/cifar10_class_{class_label}'
                            os.makedirs(generated_images_dir, exist_ok=True)
                            os.makedirs(true_images_dir, exist_ok=True)  # Ensure the directory exists
                            
                            post_reward_test = 0
                            logZ_test = 0

                            for k in range(60):
                                logs = self.forward(
                                    shape=(100, *D),
                                    steps=self.steps,
                                    ode = True,
                                )
                                x = logs['x_mean_posterior']
                                img_fid = (x * 127.5 + 128).clip(0, 255).to(torch.uint8)

                                post_reward_test += self.reward_model(img_fid, *self.reward_args)
                                logZ_test += (self.beta * post_reward_test + logs['logpf_prior'] - logs['logpf_posterior']).mean()
                                
                                for i, img_tensor in enumerate(img_fid):
                                    img_pil = transforms.ToPILImage()(img_tensor)
                                    img_pil.save(os.path.join(generated_images_dir, f'{k*100 + i}.png'))
                            
                            post_reward_test /= 60
                            logZ_test /= 60

                            fid_score = fid.compute_fid(generated_images_dir, true_images_dir)

                            log_dict['log_r_test'] = post_reward_test.mean().item()
                            log_dict['logZ_test'] = logZ_test.item()
                            log_dict['fid'] = fid_score

                        wandb.log(log_dict)

                        # Save model and optimizer state
                        self.save_checkpoint(self.model, optimizer, it, run_name)
            
        print("Done Adjoint Matching fine-tuning!")    

    


    def integration(
            self,
            model,
            batch_size,
            shape,
            steps = 20,
            ode = False,
    ):

        x = self.sde.prior(shape).sample([batch_size]).to(self.device)
        t = torch.zeros(batch_size).to(self.device) + self.sde.epsilon  
        dt = 1/(steps) - self.sde.epsilon/(steps-1)
        
        for step, _ in enumerate(range(steps)):

            x_prev = x.detach()
            
            t += dt
            if ode:
                x = x + model.drift(t, x) * dt
            else:
                g = self.sde.diffusion(t, x)

                learned_drift = model.drift(t, x)

                std = g * (np.abs(dt)) ** (1 / 2)

                x = x + dt * (2*learned_drift + self.sde.drift(t,x)) + std * torch.randn_like(x)
            x = x.detach()
        
        img = (x * 127.5 + 128).clip(0, 255).to(torch.uint8)
        return img




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
            time_discretisation='uniform',
            ode = False #uniform/random
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
            t = torch.ones(B).to(self.device) 
        else:
            
            x = self.sde.prior(D).sample([B]).to(self.device)
            x_0 = x
            t = torch.zeros(B).to(self.device) + self.sde.epsilon

        # assume x is gaussian noise
        normal_dist = torch.distributions.Normal(torch.zeros((B,) + tuple(D), device=self.device),
                                                 torch.ones((B,) + tuple(D), device=self.device))

        logpf_posterior = 0*normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        logpb = 0*normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)#torch.zeros_like(logpf_posterior)
        dt = 1/(steps) - self.sde.epsilon/(steps-1)
        #####
        if save_traj:
            traj = [x.clone()]

        for step, _ in enumerate((pbar := tqdm(range(steps)))):
            pbar.set_description(
                f"Sampling from the {sampling_from} | t = {t[0].item():.1f} | sigma = {self.sde.sigma(t)[0].item():.1e}"
                f"| scale ~ {x.std().item():.1e}")
            
            # back-sampling
            if backward:
                g = self.sde.diffusion(t, x)
                std = g * (np.abs(dt)) ** (1 / 2)
                x_prev = x.detach()
                x = (x + self.sde.drift(t, x) * dt) + (std * torch.randn_like(x))
            else:
                x_prev = x.detach()
            
            t += dt * (-1.0 if backward else 1.0)
            if t[0] < self.sde.epsilon:  # Accounts for numerical error in the way we discretize t.
                continue # continue instead of break because it works for forward and backward
            g = self.sde.diffusion(t, x)


            if ode:
                x_mean_posterior = x + self.model(t, x) * dt
                x = x_mean_posterior
            else:

                posterior_drift =  2*self.model(t, x) + self.sde.drift(t, x)
                
                f_posterior = posterior_drift
                # compute parameters for denoising step (wrt posterior)
                x_mean_posterior = x + f_posterior * dt # * (-1.0 if backward else 1.0)
                std = g * (np.abs(dt)) ** (1 / 2)

                # compute step
                if prior_sample and not backward:
                    x = x + (2*self.prior_model.drift(t, x) + self.sde.drift(t, x)) * dt + std * torch.randn_like(x)
                elif not backward:
                    x = x_mean_posterior + std * torch.randn_like(x)
            
            x = x.detach()
            
            if ode:
                x_mean_pb = x_prev + self.prior_model.drift(t, x) * dt

            else:
                if backward:

                    prior_drift = (2*self.prior_model.drift(t, x) + self.sde.drift(t, x))
                    x_mean_prior = x + pb_drift * (dt)

                    pb_drift = (self.sde.drift(t, x))
                    x_mean_pb = x_prev + pb_drift * (dt)
                else:
                    pb_drift = (2*self.prior_model.drift(t,x) + self.sde.drift(t, x))
                    x_mean_pb = x_prev + pb_drift * (dt)
            #x_mean_pb = x_prev + pb_drift * (dt)
            pb_std = g * (np.abs(dt)) ** (1 / 2)

            if save_traj:
                traj.append(x.clone())
                
            if ode:
                # x_0 ~ N(0, 1), x_1 = ODE(x_0), logpf(x_1) = N(0, 1)
                # logpb = log p(x_0|x_1) = 0, because x_1 -> x_0 is deterministic
                logpf_posterior = torch.distributions.Normal(torch.zeros_like(x_0), torch.ones_like(x_0)).log_prob(x_0).sum(tuple(range(1, len(x.shape))))
                logpb = torch.zeros_like(logpf_posterior)
            

            else:
            # compute log-likelihoods of reached pos wrt to prior & posterior models
            #logpb += pb_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
                pf_post_dist = torch.distributions.Normal(x_mean_posterior, std)
                pb_dist = torch.distributions.Normal(x_mean_pb, pb_std)
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
    

