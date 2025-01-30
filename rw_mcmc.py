import os 
import torch 
import wandb
import pickle 

class RW_MCMC():
    def __init__(self, 
                 device,
                 reward_model,
                 prior_model, 
                 in_shape,
                 reward_args, 
                 id,
                 sample_save_path = "~/scratch/cnf_rtb_prot_samples/rw_mcmc_samples/seed_",
                 load= False,
                 entity = 'swish',
                 step_size = 0.01, 
                 beta = 1.0, 
                 coord_scaling = True):
        
        self.sample_length = in_shape[0]
        self.device = device 
        self.reward_model = reward_model

        self.prior_model = prior_model

        self.load = load
        if self.load:
            self.load_path = sample_save_path + "sample_dict_rw_mcmc_it_{}.pkl"
            self.load_path = os.path.expanduser(self.load_path)

        if coord_scaling:
            self.latent_std = 10.0 
        else:
            self.latent_std = 1.0

        self.latent_prior = torch.distributions.Normal(torch.tensor(0.).to(self.device), torch.tensor(self.latent_std).to(self.device))

        # for run name
        self.id = id 
        self.entity = entity 

        self.beta = beta 
        self.step_size = step_size

        self.sample_save_path = os.path.expanduser(sample_save_path)

        self.reward_args = reward_args

    def target_density(self, x):
        # x assumed to be [B, N, 7]  - last 3 coordinates are translations
        #x_shape = x.shape 
        

        quats = x[:,:,:4]
        translations = x[:, :, 4:]

        if ((quats.norm(dim=-1) - 1.0)**2).mean() > 1e-5:
            return torch.tensor(float('-inf')).to(self.device)

        # get reward
        with torch.no_grad():
            # get log prior (over quats, latent prior is -inf if not norm 1, 0 otherwise)
            log_prior = self.latent_prior.log_prob(translations).sum(dim=(1,2))

            prior_out = self.prior_model(x)
            log_r = self.reward_model(prior_out, *self.reward_args).to(self.device)

            log_prob = log_prior + self.beta * log_r

        return log_prob
    
    def proposal(self, x):
        # x assumed to be [B, N, 7]  - last 3 coordinates are translations
        quats = x[:,:,:4]
        translations = x[:, :, 4:]

        # for quaternions, sample gaussian and project
        quats_proposal = quats + self.step_size * torch.randn_like(quats)
        #quats_proposal = torch.distributions.Normal(quats, self.step_size).rsample(quats.shape)
        quats_proposal = quats_proposal / quats_proposal.norm(dim=-1, keepdim=True)

        # for translations, sample gaussian
        trans_proposal = translations + self.step_size * torch.randn_like(translations)
        #trans_proposal = torch.distributions.Normal(translations, torch.ones_like(translations) * self.step_size).rsample(translations.shape)

        # return proposal of shape [B, N, 7]
        return torch.cat([quats_proposal, trans_proposal], dim=-1)

    def acceptance_ratio(self, x, x_proposal):
        # x assumed to be [B, N, 7]  - last 3 coordinates are translations
        # x_proposal assumed to be [B, N, 7]  - last 3 coordinates are translations
        log_prob = self.target_density(x)
        log_prob_proposal = self.target_density(x_proposal)

        # accept with probability min(1, p(x_proposal) / p(x))
        return torch.exp(log_prob_proposal - log_prob)
    
    def sample(self, batch_size, n_steps, wandb_track = False, load=False):
        
        run_name = self.id + '_beta_' + str(self.beta) + '_step_size_' + str(self.step_size) +'_batch_size_' + str(batch_size) 

        if wandb_track:
            wandb.init(
                project='cfm_posterior',
                entity=self.entity,
                save_code=True,
                name=run_name
            )
            hyperparams = {
                "step_size": self.step_size,
                "batch_size": batch_size,
                "n_steps": n_steps,
                "reward_args": self.reward_args,
                "beta": self.beta
            }
            wandb.config.update(hyperparams)
        
        # x assumed to be [B, N, 7]  - last 3 coordinates are translations
        samples_array = []
        
        if load:
            print("Loading samples from: ", self.load_path)

            samples_prev = pickle.load(open(self.load_path, "rb"))
            x = samples_prev["rigids_t"].to(self.device)
            cur_it = 301 
        else:
            # initialize from prior model 
            x = self.prior_model.sample_prior(batch_size= batch_size, sample_length = self.sample_length)
            x = x["rigids_t"].to(self.device)
            cur_it = 0
        #x = x.to(self.device)

        samples_array.append(x.cpu().detach().numpy())

        num_samples_total = 0
        num_accepted_total = 0 
        num_saved_samples = 0

        for i in range(cur_it, n_steps):
            x_proposal = self.proposal(x)
            accept = self.acceptance_ratio(x, x_proposal)
            u = torch.rand(x.shape[0]).to(self.device)

            accept_mask = (u < accept).float()

            accept_mask = accept_mask.unsqueeze(-1).unsqueeze(-1)
            #accept_mask = accept_mask.expand(x.shape)
            #u = torch.rand(x.shape).to(self.device)

            x = x_proposal * accept_mask + x * (1. - accept_mask)

            num_accepted = accept_mask[:, 0, 0].float().sum().item()

            num_samples_total += batch_size 
            num_accepted_total += num_accepted 

            acceptance_rate_batch = num_accepted/batch_size
            acceptance_rate_total = num_accepted_total/num_samples_total 

            prior_out = self.prior_model(x)
            log_r = self.reward_model(prior_out, *self.reward_args).to(self.device)
            log_r_mean = log_r.mean()
            
            print("\nlog_r: {}, accept_rate_batch:{}, accept_rate_total: {}".format(log_r_mean, acceptance_rate_batch, acceptance_rate_total))
            print("batch size: {}, num accepted: {}, num_samples_total: {}".format(batch_size, num_accepted, num_samples_total))

            if wandb_track and i%100 != 0:
                wandb.log({"log_r": log_r_mean.item(), "epoch": i, "accept_rate_batch": acceptance_rate_batch, "accept_rate_total": acceptance_rate_total})

            if i % 100 == 0:
                samples_array.append(x.cpu().detach().numpy())
                num_saved_samples += batch_size 
                # save as PDB in sample save path
                
                if wandb_track:
                    log_r  = log_r.cpu().detach().numpy()


                    imgs_pil = self.reward_model.get_prot_image(prior_out)
                    wandb.log({"log_r": log_r_mean.item(), "epoch": i, "accept_rate_batch": acceptance_rate_batch, "accept_rate_total": acceptance_rate_total, 
                                "posterior_samples":  [wandb.Image(imgs_pil[k], caption=log_r[k]) for k in range(len(imgs_pil))], 
                                "num_saved_samples": num_saved_samples})
                     
                if self.sample_save_path is not None:
                    self.reward_model.save_samples(samples = prior_out, save_path = self.sample_save_path, it = i)

                    
                
        return samples_array 