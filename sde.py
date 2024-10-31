import numpy as np 
import torch 
from torch import Tensor
from torch.distributions import Normal, Independent 

class VPSDE():
    def __init__(
        self,
        device, 
        beta_min: float = 0.1,
        beta_max: float = 20,
        T: float = 1.0,
        epsilon: float = 1e-5,
        beta_schedule: str = 'linear',  # Added beta_schedule parameter
        **kwargs
    ):
        super().__init__()
        self.sde_type = 'vpsde'
        self.device = device 
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
        mu = torch.zeros(shape).to(self.device)
        return Independent(Normal(loc=mu, scale=1., validate_args=False), len(shape))

    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        return torch.sqrt(self.beta(t)).view(-1, *[1]*len(D))

    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        return -0.5 * self.beta(t).view(-1, *[1]*len(D)) * x

    def marginal_prob_scalars(self, t: Tensor):
        if self.beta_schedule == 'linear':
            log_coeff = 0.5 * (self.beta_max - self.beta_min) * t**2 + self.beta_min * t
            std = torch.sqrt(1. - torch.exp(-log_coeff))
            return torch.exp(-0.5 * log_coeff), std
        elif self.beta_schedule == 'cosine':
            s = 0.008  # Small constant to prevent singularities
            f = (t / self.T + s) / (1 + s)
            phi_t = f * (np.pi / 2)
            phi_0 = torch.tensor((s / (1 + s)) * (np.pi / 2), device=self.device)
            cos_phi_t = torch.cos(phi_t)
            cos_phi_0 = torch.cos(phi_0)
            coeff = cos_phi_t / cos_phi_0
            std = torch.sqrt(1. - coeff**2)
            return coeff, std
        


# use DDPM updates (x0 and x1 are both N(0,1))
class DDPM():
    def __init__(
        self,
        device, 
        beta_min: float = 0.1,
        beta_max: float = 20,
        T: float = 1.0,
        epsilon: float = 1e-5,
        beta_schedule: str = 'linear',  # Added beta_schedule parameter
        **kwargs
    ):
        super().__init__()
        self.sde_type = 'ddpm'
        self.device = device 
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
        mu = torch.zeros(shape).to(self.device)
        return Independent(Normal(loc=mu, scale=1., validate_args=False), len(shape))

    def diffusion(self, t: Tensor, x: Tensor, dt: Tensor) -> Tensor:
        _, *D = x.shape
        beta = self.beta(t)/self.beta_max
        return torch.sqrt(beta * dt).view(-1, *[1]*len(D))

    def drift(self, t: Tensor, x: Tensor, dt: Tensor) -> Tensor:
        _, *D = x.shape
        beta = self.beta(t)/self.beta_max
        
        #print("dt: ", dt)
        #print("beta: ", beta)
        #print("beta: ", beta * dt)
        #print("1 - beta*dt", 1.0 - beta*dt)
        coeff = torch.sqrt(1.0 - beta*dt).view(-1, *[1]*len(D))
        
        #print("\nbeta(t): ", self.beta(t))
        #print("coeff: ", coeff)

        return (coeff - 1.0) * x   
        #return -0.5 * self.beta(t).view(-1, *[1]*len(D)) * x

    def marginal_prob_scalars(self, t: Tensor):
        if self.beta_schedule == 'linear':
            log_coeff = 0.5 * (self.beta_max - self.beta_min) * t**2 + self.beta_min * t
            std = torch.sqrt(1. - torch.exp(-log_coeff))
            return torch.exp(-0.5 * log_coeff), std
        elif self.beta_schedule == 'cosine':
            s = 0.008  # Small constant to prevent singularities
            f = (t / self.T + s) / (1 + s)
            phi_t = f * (np.pi / 2)
            phi_0 = torch.tensor((s / (1 + s)) * (np.pi / 2), device=self.device)
            cos_phi_t = torch.cos(phi_t)
            cos_phi_0 = torch.cos(phi_0)
            coeff = cos_phi_t / cos_phi_0
            std = torch.sqrt(1. - coeff**2)
            return coeff, std