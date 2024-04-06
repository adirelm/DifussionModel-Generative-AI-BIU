import torch
import numpy as np
import torch.nn.functional as F

class Diffusion():
    def __init__(self, unet, scheduler='regular', device='cpu'):
        
        self.device = device
        self.unet = unet

        # Pre-calculate values for the diffusion process based on a set number of timesteps
        self.T = 300 # Total number of timesteps

        # switch case of the scheduler
        if scheduler == 'regular':
            self.betas = self.linear_beta_schedule(timesteps=self.T) # Beta values for each timestep
        elif scheduler == 'cosine':
            self.betas = self.cosine_schedule(num_timesteps=self.T, s=-0.5)
        else:
            raise NotImplementedError

        # Calculate alpha values and their cumulative product for diffusion scaling

        # Alpha values derived from betas
        self.alphas = 1. - self.betas

        # Cumulative product of alphas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        # Shifted alphas for variance calculation
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Square root of reciprocal alphas for scaling during the reverse diffusion
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Square root of alphas cumulative product for forward diffusion scaling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod) 

        # Square root of the complement of alphas cumulative product for noise scaling
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Variance of the posterior distribution
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def cosine_schedule(self, num_timesteps, s=-0.5):
        def f(t):
            return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
        x = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alphas_cumprod = f(x) / f(torch.tensor([0]))
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clip(betas, 0.0001, 0.999)
        return betas

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        """
        Generates a linear schedule for beta values over a specified number of timesteps.
        
        Parameters:
        - timesteps (int): The total number of timesteps in the diffusion process.
        - start (float): The starting value of beta.
        - end (float): The ending value of beta.
        
        Returns:
        - A tensor of beta values linearly spaced between the start and end values.
        """

        # Generates linearly spaced beta values between start and end
        return torch.linspace(start, end, timesteps)
    
    def get_index_from_list(self, vals, t, x_shape):
        """
        Retrieves a specific index from a list of values, accounting for the batch dimension.
        
        Parameters:
        - vals (tensor): The tensor from which to gather values.
        - t (tensor): The indices of the values to gather.
        - x_shape (tuple): The shape of the input tensor.
        
        Returns:
        - A tensor containing the values at the specified indices.
        """

        # Get batch size from timestep tensor
        batch_size = t.shape[0]

        # Gather values based on timesteps, moving to CPU if necessary
        out = vals.gather(-1, t.cpu())

        # Reshape output to match the batch dimension and maintain the shape for broadcasting
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """
        Applies the forward diffusion process to an image for a given timestep.
        
        Parameters:
        - x_0 (tensor): The original image tensor.
        - t (tensor): The current timestep.
        - device (str): The device to perform computations on.
        
        Returns:
        - A tuple containing the noisy image and the noise applied.
        """

        # Generate random noise with the same shape as the image
        noise = torch.randn_like(x_0)

        # Calculate the scale for the original image and the noise based on alpha values
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)

        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # Combine the original image with noise based on the calculated scales
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
    
    def get_loss(self, x_0, t):
        """
        Calculates the loss for a single step of the diffusion process.
        
        This function computes the L1 loss between the actual noise used to corrupt an input image
        and the noise predicted by the model. This loss guides the model in learning to accurately
        predict the noise at each timestep, which is essential for generating clear images from noisy ones
        during the reverse diffusion process.

        Parameters:
        - model (torch.nn.Module): The diffusion model that predicts noise from noisy images.
        - x_0 (torch.Tensor): The original, clean images from the dataset.
        - t (torch.Tensor): The timesteps at which the images are noised. This tensor should have
          the same batch size as x_0 and contain values indicating the diffusion step for each image.

        Returns:
        - torch.Tensor: The L1 loss between the actual noise and the predicted noise.
        """

        # Generate a noisy version of the input images and the corresponding real noise used
        x_noisy, noise = self.forward_diffusion_sample(x_0, t, self.device)

        # Predict the noise from the noisy images using the model
        noise_pred = self.unet(x_noisy, t)

        # Calculate and return the L1 loss between the actual and predicted noise
        return F.l1_loss(noise, noise_pred)
    
    @torch.no_grad()
    def sample_timestep(self, x, t):
        """
        Performs a single timestep of the reverse diffusion process on a batch of images.
        
        This function uses the model to predict the noise in a noisy image and then calculates
        a less noisy version of the image. If not at the final step, it also adds a controlled
        amount of noise back to the image, simulating a step of the reverse diffusion process.
        
        Parameters:
        - x (torch.Tensor): The batch of noisy images at a certain timestep.
        - t (torch.Tensor): The current timestep for each image in the batch.
        
        Returns:
        - torch.Tensor: The batch of images after a single reverse diffusion step,
          potentially less noisy than the input if not at the final step.
        """

        # Retrieve the beta, sqrt(1-alpha_cumprod), and sqrt(1/alpha) values for the current timestep
        betas_t = self.get_index_from_list(self.betas, t, x.shape)

        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        # Predict the noise and compute the model's mean estimate of the clean image
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.unet(x, t) / sqrt_one_minus_alphas_cumprod_t)

        # Calculate the variance of the posterior distribution for the current timestep
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
        if t == 0:
            # If at the first timestep, return the model's mean as the final denoised image
            # This step corresponds to the final reconstruction in the reverse diffusion process.
            return model_mean
        else:
            # If not at the first timestep, simulate the reverse diffusion by adding scaled noise
            noise = torch.randn_like(x) # Generate random noise with the same shape as the input
            return model_mean + torch.sqrt(posterior_variance_t) * noise # Add scaled noise to the model's mean estimate