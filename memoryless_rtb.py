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
import copy 

from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

def get_target_modules(model):
    # Create a list to store the names of target modules
    target_modules = []

    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of Linear or Conv layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)):
            target_modules.append(name)  # Append the name of the layer

    return target_modules

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
                 lora = True,
                 lora_rank=16,
                 clip_x = False,
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
                 detach_freq=0.0,
                 posterior_ratio=0.2,
                 ):
        super().__init__()
        self.device = device
        
        # if inference_type == 'vpsde':
        #     self.sde = VPSDE(device = self.device, beta_schedule='cosine', beta_max = 0.05, beta_min = 0.0001)
        # else:
        #     self.sde = DDPM(device = self.device, beta_schedule='cosine')


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

        self.langevin = langevin 

        # for run name
        self.id = id 
        self.entity = entity 

        self.tb = tb 

        self.clip_x = clip_x 

        # Posterior noise model
        self.logZ = torch.nn.Parameter(torch.tensor(0.).to(self.device))
        
        self.lora = lora 
        self.lora_rank = lora_rank 

        if lora:
            #targets = get_target_modules(model)
            self.targets = ["qkv", "proj_out"]
            unet_lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_rank,
                init_lora_weights = True, #"gaussian", # True does Kaiming uniform, to make it no-op
                target_modules = ["qkv", "proj_out"] #targets,  #["Dense_0", "conv"],
            )
            self.model = get_peft_model(copy.deepcopy(model.train()), unet_lora_config)

            posterior_params = sum(p.numel() for p in model.parameters())
            lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
             
            print(f"\nTotal params: "
                  f"\nPOSTERIOR model: {posterior_params / 1e6:.2f}M"
                  f"\nTrainable posterior parameters: {lora_params / 1e6:.2f}M/{posterior_params / 1e6:.2f}M  ({lora_params * 100 / posterior_params:.2f}%)\n")
            
            #print([name for name, _ in self.model.named_modules()])
        
            print("lora params size: ", lora_params)

            # set requires_grad for all parameters to enable grad flow (optimizer will only update lora ones)
            for name,param in self.model.named_parameters():
                if not param.requires_grad:
                    param.requires_grad = True

            self.param_list = [{'params': [p for name, p in self.model.named_parameters() if ("lora" in name)]}]

            num_opt_params = sum(p.numel() for name, p in self.model.named_parameters() if ("lora" in name))
            
            if num_opt_params != lora_params:
                print(f"Number of lora_params: {lora_params}, Number of optimized params: {num_opt_params}")
                assert num_opt_params == lora_params
          
        else:
            self.model = model
            self.param_list = [{'params': self.model.parameters()}]

        self.model = self.model.train()

        self.prior_model = prior_model
        tol = 1e-5
        self.posterior_ratio = posterior_ratio
        self.reward_model = reward_model 
        self.detach_freq = detach_freq

        self.trainable_reward = None 
        
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

        # for lora
        if self.lora:
            lora_path = savedir + 'checkpoint_'+str(epoch) + "_lora" 
            self.model.save_pretrained(lora_path)
            print(f"Lora peft saved at: {lora_path}")


    def load_checkpoint(self, model, optimizer):
        if self.load_ckpt_path is None:
            print("Checkpoint path not provided. Checkpoint not loaded.")
            return model, optimizer
        
        checkpoint = torch.load(self.load_ckpt_path)

        if not self.lora:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            set_peft_model_state_dict(model, checkpoint, adapter_name = "lora")
        
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
            loss =  torch.nn.CrossEntropyLoss()(pred_log_probs, target_log_probs.argmax(dim=-1)).mean()
        return loss

    def get_langevin_correction(self, x):
        # add gradient wrt x of reward model
        if self.langevin:
            with torch.set_grad_enabled(True):
                x.requires_grad = True
                log_rx = self.log_reward(x)
                grad_log_rx = torch.autograd.grad(log_rx.sum(), x, create_graph=True)[0]
                
            lp_correction = grad_log_rx#.detach()
        else:
            lp_correction = torch.zeros_like(x)
        return lp_correction.detach()


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

    def log_reward(self, x, return_img=False):
        with torch.no_grad():
            img = (x * 127.5 + 128).clip(0, 255)
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

            fwd_logs = self.forward(
                    shape=shape,
                    steps=self.steps,
                    save_traj=True,  # save trajectory fwd
                    prior_sample=prior_sample,
                    x_1 = x_1,
                    backward = rb_sample,
                    ode = False
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


        self.batched_forward(
                shape=shape,
                traj=fwd_logs['traj'],
                correction=correction,
                batch_size=B,
                detach_freq=self.detach_freq)   


        return rtb_loss.detach().mean(), logr_x_prime.mean()
    
    
    def finetune(self, shape, n_iters=100000, learning_rate=5e-5, clip=0.1, wandb_track=False, prior_sample_prob=0.0, replay_buffer_prob=0.0, anneal=False, anneal_steps=15000, exp='sd3_align', compute_fid=False, class_label=0):
        B, *D = shape

        # only get params with grad:
        #param_list = [{'params': self.model.parameters()}]
        

        optimizer = torch.optim.Adam(self.param_list, lr=learning_rate)
        run_name = self.id + '_sde_' + self.sde_type + '_NL_RTB' +'_steps_' + str(self.steps) + '_lr_' + str(learning_rate) + '_beta_start_' + str(self.beta_start) + '_beta_end_' + str(self.beta_end) + '_anneal_' + str(anneal) + '_prior_prob_' + str(prior_sample_prob) + '_rb_prob_' + str(replay_buffer_prob) + "_lora_" + str(self.lora) + "_rank_"+ str(self.lora_rank) + "_clip_x_"+str(self.clip_x) + "_post_ratio_" + str(self.posterior_ratio)
        
        
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

                #img = self.integration(self.prior_model, 10, self.in_shape, steps = self.steps, ode = True)
                
                x = torch.randn(B, *D, device=self.device)
                img = self.prior_model(x)
                prior_reward = self.reward_model(img, *self.reward_args)
            wandb.log({"prior_samples": [wandb.Image(img[k], caption = prior_reward[k]) for k in range(len(img))]})
            
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
            self.model.zero_grad() 

            loss, logr = self.batched_rtb(shape=shape, prior_sample=prior_traj, rb_sample=rb_traj)

            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
            optimizer.step() 
            
            if ((rb_traj == False) and (prior_traj == False)) or (it == load_it):
                self.logZ_on_pol = self.logZ.detach().cpu().numpy()

            if wandb_track: 
                if not it%100 == 0:
                   
                    wandb.log({"loss": loss.item(), "logZ": self.logZ.detach().cpu().numpy(), "logZ_on_pol": self.logZ_on_pol,
                                "log_r": logr.item(), "epoch": it})
                else:
                    with torch.no_grad():
   
                        logs = self.forward(
                                shape=(10, *D),
                                steps=self.steps, ode=True)

                        x = logs['x_mean_posterior']
                        img = (x * 127.5 + 128)
                        post_reward = self.reward_model(img, *self.reward_args)
                        logZ_ode = (self.beta * post_reward + logs['logpf_prior'] - logs['logpf_posterior']).mean()

                        
                        logs_sde = self.forward(
                            shape = (100, *D),
                            steps = self.steps, ode=False
                        )
                        x_sde = logs_sde['x_mean_posterior']
                        img_sde = (x_sde * 127.5 + 128)
                        post_sde_reward = self.reward_model(img_sde, *self.reward_args)
                        logZ_sde = (self.beta * post_sde_reward + logs_sde['logpf_prior'] - logs_sde['logpf_posterior']).mean()            
                        
                        
                        log_dict = {"loss": loss.item(), "logZ": self.logZ.detach().cpu().numpy(), "logZ_on_pol": self.logZ_on_pol,
                                    "log_r": logr.item(), "epoch": it, 
                                   "posterior_samples": [wandb.Image(img[k], caption=post_reward[k]) for k in range(len(img))],
                                   "logZ_ode": logZ_ode.item(), "log_r_ode": post_reward.mean().item(), 
                                   "logZ_sde": logZ_sde.item(), "log_r_sde": post_sde_reward.mean().item()}



                        if it%1000 == 0 and 'cifar' in exp and compute_fid and it>0:
                            print('COMPUTING FID:')
                            generated_images_dir = f'fid/{exp}_nl_rtb_cifar10_class_{class_label}'

                            if not os.path.exists(generated_images_dir):
                                os.makedirs(generated_images_dir)

                            true_images_dir = 'fid/cifar10_class_' + str(class_label)

                            true_images_dir_all = 'fid/cifar10_class_all'

                            post_reward_test = 0
                            logZ_test = 0

                            for k in range(60):
                                with torch.no_grad():
                                    logs = self.forward(
                                        shape=(100, *D),
                                        steps=self.steps,
                                        ode=True
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

                            #fid_score_all = fid.compute_fid(generated_images_dir, true_images_dir_all) 
                            #log_dict['fid_uncond'] = fid_score_all

                        wandb.log(log_dict)

                        # save model and optimizer state
                        self.save_checkpoint(self.model, optimizer, it, run_name)
    
    # Euler-Maruyama integration of memoryless flow matching SDE 
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
            ode=False
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

        logpf_posterior = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        logpf_prior = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device) #torch.zeros_like(logpf_posterior)
        dt = 1/(steps+1)
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

            # Euler intergration: x = x + v(x) * dt + g * dW
            if ode:
                x_mean_posterior = x + (self.posterior_ratio * self.model(t, x) + (1-self.posterior_ratio)*self.prior_model.drift(t, x).detach()) * dt
                x = x_mean_posterior
            # Equivalent SDE that has the same marginal as the ODE
            # x = (2 * v(x) - kappa(x) * x) * dt + g * dW
            else:

                posterior_drift = 2*(self.posterior_ratio * self.model(t, x) + (1-self.posterior_ratio) * self.prior_model.drift(t, x).detach()) + self.sde.drift(t, x)

                f_posterior = posterior_drift
                x_mean_posterior = x + f_posterior * dt # * (-1.0 if backward else 1.0)
            
            
                std = g * (np.abs(dt)) ** (1 / 2)

                # compute step
                if prior_sample and not backward:
                    x = x + (2*self.prior_model.drift(t, x) + self.sde.drift(t, x)) * dt + std * torch.randn_like(x)
                elif not backward:
                    x = x_mean_posterior + std * torch.randn_like(x)
            
            x = x.detach()
            
            if self.clip_x:
                x = torch.clamp(x, -2.0, 2.0)

            if ode:
                x_mean_prior = x_prev + self.prior_model.drift(t, x_prev) * dt
            else:
                if backward:
                    prior_drift = (2*self.prior_model.drift(t, x) + self.sde.drift(t, x))
                    x_mean_prior = x + prior_drift * (dt*-1)
                else:
                    prior_drift = (2*self.prior_model.drift(t,x_prev) + self.sde.drift(t, x_prev))
                    x_mean_prior = x_prev + prior_drift * (dt)
            
            prior_std = g * (np.abs(dt)) ** (1 / 2)

            if save_traj:
                traj.append(x.clone())


            if ode:
                # x_0 ~ N(0, 1), x_1 = ODE(x_0), logpf(x_1) = N(0, 1)
                # logpb = log p(x_0|x_1) = 0, because x_1 -> x_0 is deterministic
                logpf_posterior = torch.distributions.Normal(torch.zeros_like(x_0), torch.ones_like(x_0)).log_prob(x_0).sum(tuple(range(1, len(x.shape))))
                logpf_prior = torch.zeros_like(logpf_posterior)
            

            else:
            # compute log-likelihoods of reached pos wrt to prior & posterior models
            #logpb += pb_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
                pf_post_dist = torch.distributions.Normal(x_mean_posterior, std)
                pf_prior_dist = torch.distributions.Normal(x_mean_prior, prior_std)
                
                assert torch.all(prior_std == std)

                if backward:
                    logpf_prior += pf_prior_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
                    logpf_posterior += pf_post_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
                else:
                    logpf_prior += pf_prior_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))
                    logpf_posterior += pf_post_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))

                if torch.any(torch.isnan(x)):
                    print("Diffusion is not stable, NaN were produced. Stopped sampling.")
                    break

        if backward:
            traj = list(reversed(traj))
        logs = {
            'x_mean_posterior': x,#,x_mean_posterior,
            'logpf_prior': logpf_prior,
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
            # timesteps 0 -> 1
            #timesteps = np.flip(timesteps)

        no_grad_steps = random.sample(range(steps), int(steps * detach_freq))  # Sample detach_freq fraction of timesteps for no grad



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
                t_.append(torch.full((x.shape[0],), 1 - timesteps[step + 1]).float())
                dts.append(torch.full((x.shape[0],), - timesteps[step + 1] + timesteps[step]).float())
                xs.append(traj[step])
                xs_next.append(traj[step+1])
                bs += 1
                
                
                #print("dts: ", (-timesteps[step + 1] + timesteps[step]))
                assert ((-timesteps[step+1] + timesteps[step]) > 0.0)


            if len(t_) == 0:
                continue
            
            t_ = torch.cat(t_, dim=0).to(self.device).view(-1, 1, 1, 1)
            dts = torch.cat(dts, dim=0).to(self.device).view(-1, 1, 1, 1)
            xs = torch.cat(xs, dim=0).to(self.device)
            xs_next = torch.cat(xs_next, dim=0).to(self.device)


            g = self.sde.diffusion(t_, xs).to(self.device)
            
            f_posterior = 2*((1-self.posterior_ratio)*self.prior_model.drift(t_[:,0,0,0], xs).detach() + self.posterior_ratio*self.model(t_[:,0,0,0], xs)) + self.sde.drift(t_, xs)


            #f_posterior = (2 * self.model(t_[:,0,0,0], xs)) + self.sde.drift(t_, xs)

            # compute parameters for denoising step (wrt posterior)
            x_mean_posterior = xs + f_posterior * dts
            std = g * (dts) ** (1 / 2)

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