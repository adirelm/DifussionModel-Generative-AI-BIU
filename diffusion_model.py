import torch
import torchvision
import matplotlib.pyplot as plt

def show_images(dataset, num_samples=20, cols=4):
    """
    Plots a selection of images from a given dataset.
    
    Parameters:
    - dataset: An iterable dataset where each element is expected to be a tuple,
               with the first element being an image.
    - num_samples: The total number of images to display. Default is 20.
    - cols: The number of columns in the display grid. Default is 4.
    
    The function calculates the required number of rows based on num_samples and cols,
    and displays the images in a grid layout within a matplotlib figure.
    """

    # Initialize a figure with a specified size in inches
    plt.figure(figsize=(15,15))

    # Loop through the dataset, extracting each image up to num_samples
    for i, img in enumerate(dataset):
        # Stop the loop if the number of desired samples is reached
        if i == num_samples:
            break

        # Calculate the position of the current image in the grid and create a subplot for it
        # The number of rows is dynamically calculated based on num_samples and cols
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)

        # Display the current image. img[0] assumes the dataset returns a tuple (image, label)
        plt.imshow(img[0])

# Load the StanfordCars dataset with automatic download if not present in the current directory
data = torchvision.datasets.StanfordCars(root=".", download=True)

# Visualize a selection of images from the dataset using the show_images function
show_images(data)

"""Later in this notebook we will do some additional modifications to this dataset, for example make the images smaller, convert them to tensors ect.

# Building the Diffusion Model

## Step 1: The forward process = Noise scheduler

We first need to build the inputs for our model, which are more and more noisy images. Instead of doing this sequentially, we can use the closed form provided in the papers to calculate the image for any of the timesteps individually.

**Key Takeaways**:
- The noise-levels/variances can be pre-computed
- There are different types of variance schedules
- We can sample each timestep image independently (Sums of Gaussians is also Gaussian)
- No model is needed in this forward step
"""

import torch.nn.functional as F

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
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

def get_index_from_list(vals, t, x_shape):
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

def forward_diffusion_sample(x_0, t, device="cpu"):
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
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)

    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    # Combine the original image with noise based on the calculated scales
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Pre-calculate values for the diffusion process based on a set number of timesteps
T = 300 # Total number of timesteps
betas = linear_beta_schedule(timesteps=T) # Beta values for each timestep

# Calculate alpha values and their cumulative product for diffusion scaling

# Alpha values derived from betas
alphas = 1. - betas

# Cumulative product of alphas
alphas_cumprod = torch.cumprod(alphas, axis=0)

# Shifted alphas for variance calculation
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

# Square root of reciprocal alphas for scaling during the reverse diffusion
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# Square root of alphas cumulative product for forward diffusion scaling
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) 

# Square root of the complement of alphas cumulative product for noise scaling
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# Variance of the posterior distribution
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

"""Let's test it on our dataset ..."""

from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

# Set parameters for data loading and image processing

# Target size for resizing images
IMG_SIZE = 64

# Batch size for data loading
BATCH_SIZE = 128

def load_transformed_dataset():
    """
    Loads and transforms the StanfordCars dataset for use in the diffusion model.
    
    Returns:
    - A concatenated dataset of the transformed training and test sets.
    """

    # Define transformations: resize, horizontal flip, convert to tensor, normalize
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize images to a fixed size
        transforms.RandomHorizontalFlip(), # Randomly flip images horizontally
        transforms.ToTensor(), # Convert images to PyTorch tensors
        transforms.Lambda(lambda t: (t * 2) - 1) # Normalize tensors to range [-1, 1]
    ]

    # Combine all transforms into a single operation
    data_transform = transforms.Compose(data_transforms)

    # Load training and test datasets with transformations applied
    train = torchvision.datasets.StanfordCars(root=".", download=True, transform=data_transform)
    test = torchvision.datasets.StanfordCars(root=".", download=True, transform=data_transform, split='test')

    # Concatenate training and test sets for a larger dataset
    return torch.utils.data.ConcatDataset([train, test])

def show_tensor_image(image):
    """
    Converts a tensor image to a displayable format and shows it.
    
    Parameters:
    - image (tensor): The image tensor to display.
    """
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2), # Rescale tensor values to [0, 1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # Rearrange channels from CxHxW to HxWxC for plotting
        transforms.Lambda(lambda t: t * 255.),  # Scale pixel values up to 255
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)), # Convert to numpy array and adjust datatype
        transforms.ToPILImage(), # Convert numpy array to PIL image for display
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] # Select the first image in a batch if batch dimension exists
    plt.imshow(reverse_transforms(image)) # Display the image

# Load preprocessed dataset
data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # Create DataLoader for batching and shuffling

# Simulate forward diffusion

# Prepare to visualize the forward diffusion process
image = next(iter(dataloader))[0] # Extract a single batch and select the first image

plt.figure(figsize=(15,15)) # Initialize figure for plotting
plt.axis('off') # Hide axis for cleaner visualization
num_images = 10 # Number of images to display
stepsize = int(T/num_images) # Calculate step size to evenly distribute timesteps

# Generate and display images at various timesteps during the forward diffusion
for idx in range(0, T, stepsize):

    # Current timestep as a tensor
    t = torch.Tensor([idx]).type(torch.int64)

    # Create subplot for each timestep
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)

    # Apply forward diffusion to the image
    img, noise = forward_diffusion_sample(image, t)

    # Display the resulting noisy image
    show_tensor_image(img)

"""## Step 2: The backward process = U-Net

**Key Takeaways**:
- We use a simple form of a UNet for to predict the noise in the image
- The input is a noisy image, the ouput the noise in the image
- Because the parameters are shared accross time, we need to tell the network in which timestep we are
- The Timestep is encoded by the transformer Sinusoidal Embedding
- We output one single value (mean), because the variance is fixed
"""

from torch import nn
import math

class Block(nn.Module):
    """
    A building block for a U-Net architecture, capable of performing both
    convolutional operations and time-conditional transformations on input data.

    Attributes:
    - in_ch (int): Number of input channels.
    - out_ch (int): Number of output channels.
    - time_emb_dim (int): Dimensionality of the time embedding.
    - up (bool): Indicates whether the block performs upsampling. If False, the block
      performs downsampling or maintains the input size.

    Methods:
    - forward(x, t): Propagates an input tensor x and a time embedding t through the block.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        """
        Initializes the Block with convolutional and normalization layers,
        along with a time embedding linear transformation.

        Parameters:
        - in_ch (int): The number of channels in the input tensor.
        - out_ch (int): The desired number of channels in the output tensor.
        - time_emb_dim (int): The dimensionality of the time embedding vector.
        - up (bool, optional): If True, the block is configured to upsample the input tensor. Defaults to False.
        """
        super().__init__() # Initialize the superclass (nn.Module) to inherit its properties and methods

        # Define a linear layer to process time embeddings
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch) 

        # If the block is specified as an upsampling block...
        if up: 
            # Convolution to halve the channels after concatenation in U-Net
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)

            # Transposed convolution for upsampling
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            # Regular convolution for downsampling or maintaining size
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

            # Convolution with stride for downsampling
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        # Additional convolution layer to further process the tensor
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Batch normalization layers to stabilize and accelerate training
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)

        # ReLU activation function for non-linearity
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        """
        Forward pass of the block, applying convolutions, time embedding, and normalization.

        Parameters:
        - x (Tensor): The input tensor with shape (batch_size, in_ch, height, width).
        - t (Tensor): The time embedding tensor with shape (batch_size, time_emb_dim).

        Returns:
        - Tensor: The output tensor after applying convolutions, time embedding, and
          either upsampling or downsampling based on the block's configuration.
        """
        # Apply the first convolution, followed by batch normalization and ReLU activation
        h = self.bnorm1(self.relu(self.conv1(x)))

       # Process the time embedding through a fully connected layer and ReLU
        time_emb = self.relu(self.time_mlp(t))

        # Reshape time embedding to be broadcastable with the feature map dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]

        # Add the time embedding to the feature map, introducing the time condition
        h = h + time_emb

        # Apply the second convolution, batch normalization, and ReLU activation
        h = self.bnorm2(self.relu(self.conv2(h)))

        # Apply the transformation layer (either upsampling or downsampling based on initialization)
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Generates sinusoidal embeddings for a given time input.
    
    The sinusoidal embeddings are useful for models that need to encode the notion of time
    or sequence without relying on learned embeddings. This approach is particularly useful
    in tasks like diffusion models where the notion of 'time' or 'step' is crucial.
    
    Attributes:
    - dim (int): The dimensionality of the embeddings to be generated.
    """
    def __init__(self, dim):
        """
        Initializes the SinusoidalPositionEmbeddings module.
        
        Parameters:
        - dim (int): The dimensionality of the output embeddings.
        """
        super().__init__() # Initialize the superclass (nn.Module)

        # Store the dimensionality for the embeddings
        self.dim = dim 

    def forward(self, time):
        """
        Generates the sinusoidal embeddings for the given time tensor.
        
        Parameters:
        - time (Tensor): A tensor containing time indices of shape (batch_size,).
        
        Returns:
        - Tensor: The sinusoidal embeddings of shape (batch_size, dim).
        """
        # Get the device of the input tensor (CPU or GPU)
        device = time.device

        # Calculate half of the embedding dimension
        half_dim = self.dim // 2

        # Calculate the scaling factor for the sinusoidal function
        embeddings = math.log(10000) / (half_dim - 1)

        # Generate the base embeddings using a range and the scaling factor
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # Apply the sinusoidal function to the scaled time indices
        embeddings = time[:, None] * embeddings[None, :]

        # Concatenate the sine and cosine embeddings to form the final embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # Return the generated sinusoidal embeddings
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified U-Net architecture designed for processing images with the incorporation of time embeddings.
    This model is particularly suited for tasks that require understanding both spatial features and temporal
    dynamics, such as in diffusion models for image generation.

    The architecture follows the classic U-Net design with downsampling and upsampling paths and skip connections,
    but it also integrates time embeddings at each block to enable time-conditioned image processing.
    """
    def __init__(self):
        """
        Initializes the SimpleUnet model with predefined channel dimensions and a time embedding layer.
        """
        super().__init__() # Initialize the nn.Module superclass

         # Define the number of input channels for images (e.g., 3 for RGB images)
        image_channels = 3

        # Define the channel dimensions for the downsampling path
        down_channels = (64, 128, 256, 512, 1024)

        # Define the channel dimensions for the upsampling path, mirroring the downsampling path
        up_channels = (1024, 512, 256, 128, 64)

        # Define the output dimension, which typically matches the number of input channels for tasks like image reconstruction
        out_dim = 3

        # Set the dimensionality of the time embeddings
        time_emb_dim = 32

        # Create a sequential model for processing time embeddings
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim), # Generates sinusoidal time embeddings
                nn.Linear(time_emb_dim, time_emb_dim), # Linear layer to process the embeddings
                nn.ReLU() # ReLU activation function for non-linearity
            )

        # Initial convolutional layer to project input images into the model's channel dimensions
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Create downsampling layers using the Block class
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        
        # Create upsampling layers, specifying the 'up' parameter as True to reverse the operation
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        # Final convolutional layer to project the output back to the desired output dimension
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        """
        Propagates an input image and a timestep through the U-Net, applying time-conditioned transformations.

        Parameters:
        - x (Tensor): The input image tensor.
        - timestep (Tensor): A tensor representing the current timestep or phase in the diffusion process.

        Returns:
        - Tensor: The transformed image tensor.
        """

        # Generate time embeddings from the input timestep
        t = self.time_mlp(timestep)

        # Apply the initial convolution to the input image
        x = self.conv0(x)

        # Initialize a list to store the outputs of the downsampling path for skip connections
        residual_inputs = []

        # Downsample the input while applying time-conditioned transformations
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x) # Store the output for later use in skip connections

        # Upsample the output and combine with the stored outputs from the downsampling path   
        for up in self.ups:
            residual_x = residual_inputs.pop() # Retrieve the corresponding output from the downsampling path
            x = torch.cat((x, residual_x), dim=1) # Concatenate for skip connection
            x = up(x, t) # Apply time-conditioned upsampling
        
        # Apply the final convolution to produce the output image
        return self.output(x)

# Instantiate the SimpleUnet model
model = SimpleUnet()

# Calculate the total number of parameters in the model
# This is done by iterating over all parameters (weights and biases) of the model,
# calculating the number of elements in each parameter using numel(),
# and summing these numbers to get the total count of trainable parameters.
num_params = sum(p.numel() for p in model.parameters())

# Print the total number of parameters to give an idea of the model's size and complexity
print("Num params: ", num_params)

# Display the model's architecture by printing the model object
# This output includes the structure of the model showing the defined layers and their parameters,
# which helps in understanding the model's design and debugging if necessary.
model

"""
## Step 3: The loss

**Key Takeaways:**
- After some maths we end up with a very simple loss function
- There are other possible choices like L2 loss ect.
"""

def get_loss(model, x_0, t):
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
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)

    # Predict the noise from the noisy images using the model
    noise_pred = model(x_noisy, t)

    # Calculate and return the L1 loss between the actual and predicted noise
    return F.l1_loss(noise, noise_pred)

"""## Sampling
- Without adding @torch.no_grad() we quickly run out of memory, because pytorch tacks all the previous images for gradient calculation
- Because we pre-calculated the noise variances for the forward pass, we also have to use them when we sequentially perform the backward process
"""

@torch.no_grad()
def sample_timestep(x, t):
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
    betas_t = get_index_from_list(betas, t, x.shape)

    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Predict the noise and compute the model's mean estimate of the clean image
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    # Calculate the variance of the posterior distribution for the current timestep
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    if t == 0:
        # If at the first timestep, return the model's mean as the final denoised image
        # This step corresponds to the final reconstruction in the reverse diffusion process.
        return model_mean
    else:
        # If not at the first timestep, simulate the reverse diffusion by adding scaled noise
        noise = torch.randn_like(x) # Generate random noise with the same shape as the input
        return model_mean + torch.sqrt(posterior_variance_t) * noise # Add scaled noise to the model's mean estimate

@torch.no_grad()
def save_sample_progression(epoch, filename_prefix="progression"):
    """
    Visualizes and saves the progression of generated images over a series of timesteps during the reverse diffusion process.

    This function generates an image starting from random noise and progressively refines it through the reverse diffusion steps.
    The generated images at specified intervals are plotted and saved to visually demonstrate the model's generative process over time.

    Parameters:
    - epoch (int): The current training epoch. Used in the filename to identify the stage of training.
    - filename_prefix (str, optional): A prefix for the saved filename. Defaults to "progression".
    """

    img_size = IMG_SIZE # Use the globally defined image size
    img = torch.randn((1, 3, img_size, img_size), device=device) # Start with a random noise image

    plt.figure(figsize=(15,15)) # Initialize a figure for plotting
    plt.axis('off') # Turn off the axis for a cleaner visualization

    num_images = 10 # Number of images to display in the progression
    stepsize = int(T/num_images) # Calculate the interval between timesteps to display

    # Iterate over timesteps in reverse order to simulate the reverse diffusion process
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long) # Create a tensor for the current timestep
        img = sample_timestep(img, t) # Apply the reverse diffusion step
        img = torch.clamp(img, -1.0, 1.0)  # Clamp the image values to maintain the expected range

        # Plot the image at specified intervals
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1) # Determine the subplot position
            show_tensor_image(img.detach().cpu()) # Show the image (move to CPU if necessary)

    # Construct the save path using the provided filename prefix and the current epoch
    save_path = f"Imgs/{filename_prefix}_epoch_{epoch}.png"
    plt.savefig(save_path)  # Save the figure to the specified path
    plt.close()  # Close the figure to free memory
    print(f"Progression image saved to {save_path}") # Print a confirmation message

"""## Save And Load Functions"""

def save_model(model, filepath='model_state_dict.pth'):
    """
    Saves the model's state dictionary to a file.

    This function is useful for checkpointing during training, allowing the model's learned parameters
    to be saved at various stages or after training completes.

    Parameters:
    - model (torch.nn.Module): The model to save.
    - filepath (str, optional): The path to the file where the model's state dictionary should be saved.
      Defaults to 'model_state_dict.pth'.
    """

    # Save the model's state dictionary to the specified file path
    torch.save(model.state_dict(), filepath)

    # Print a confirmation message
    print(f'Model saved to {filepath}')

def load_model(model, isNew=True, filepath='model_state_dict.pth'):
    """
    Loads a model's state dictionary from a file.

    This function can initialize a model with a previously saved state, facilitating the resumption of training
    or the use of the model for inference without needing to retrain from scratch.

    Parameters:
    - model (torch.nn.Module): The model into which the state dictionary should be loaded.
    - isNew (bool, optional): A flag indicating whether the model is newly instantiated (True) and should be
      moved to the appropriate device without loading weights, or if it should load an existing state dictionary (False).
      Defaults to True.
    - filepath (str, optional): The path to the file from which the model's state dictionary should be loaded.
      Defaults to 'model_state_dict.pth'.
    """
    # If the model is new, simply move the model to the appropriate device without loading weights
    if (isNew):
        model.to(device)
    # Otherwise, load the state dictionary from the specified file and move the model to the appropriate device
    else:
        model.load_state_dict(torch.load(filepath, map_location=device))
        model.to(device)
        # Print a confirmation message
        print(f'Model loaded from {filepath}')

"""## Training"""

from tqdm import tqdm
from torch.optim import Adam

# Determine the device to use based on whether a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a previously trained model state or initialize a new model
load_model(model=model, isNew=False, filepath='Models/model_epoch_96.pth')

# Initialize the optimizer with the model parameters and a learning rate
optimizer = Adam(model.parameters(), lr=0.001)

# Define the total number of epochs for training
epochs = 600

for epoch in range(96, epochs):
    # Create a tqdm progress bar for visual feedback
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")

    # Iterate over the dataloader to process each batch
    for step, batch in progress_bar:
      optimizer.zero_grad() # Zero the gradients to prevent accumulation

      # Randomly select timesteps for each image in the batch
      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()

      # Calculate the loss for the current batch and timestep
      loss = get_loss(model, batch[0], t)

      loss.backward() # Perform backpropagation to calculate gradients
      
      optimizer.step() # Update model parameters

      # Update the tqdm progress bar with the current loss
      progress_bar.set_postfix(loss=loss.item())
      
      # Optionally, save the model and generate sample progressions at the start of each epoch
      if step == 0:
        save_model(model, filepath=f'Models/model_epoch_{epoch}.pth')

        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        save_sample_progression(epoch, filename_prefix=f"progression_step_{step}")


      