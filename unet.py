import math
import torch
from torch import nn
from torch.nn.utils import spectral_norm

"""
**Key Takeaways**:
- We use a simple form of a UNet for to predict the noise in the image
- The input is a noisy image, the ouput the noise in the image
- Because the parameters are shared accross time, we need to tell the network in which timestep we are
- The Timestep is encoded by the transformer Sinusoidal Embedding
- We output one single value (mean), because the variance is fixed
"""

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
            self.conv1 = spectral_norm(nn.Conv2d(2*in_ch, out_ch, 3, padding=1))

            # Transposed convolution for upsampling
            self.transform = spectral_norm(nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1))
        else:
            # Regular convolution for downsampling or maintaining size
            self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1))

            # Convolution with stride for downsampling
            self.transform = spectral_norm(nn.Conv2d(out_ch, out_ch, 4, 2, 1))

        # Additional convolution layer to further process the tensor
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1))

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