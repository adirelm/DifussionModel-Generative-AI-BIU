import torch
import torchvision
from tqdm import tqdm
from unet import SimpleUnet
from torch.optim import Adam
from diffusion import Diffusion
from torch.utils.data import DataLoader
from utils import show_images, load_model, save_model, load_transformed_dataset, save_sample_progression, simulate_forward_diffusion

# Set parameters for data loading and image processing

# Target size for resizing images
IMG_SIZE = 64

# Batch size for data loading
BATCH_SIZE = 128

# Determine the device to use based on whether a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate the Diffusion model with SimpleUnet model
diffusion_model = Diffusion(SimpleUnet(), device)

# Calculate the total number of parameters in the model
# This is done by iterating over all parameters (weights and biases) of the model,
# calculating the number of elements in each parameter using numel(),
# and summing these numbers to get the total count of trainable parameters.
num_params = sum(p.numel() for p in diffusion_model.unet.parameters())

# Print the total number of parameters to give an idea of the model's size and complexity
print("Num params: ", num_params)

# Display the model's architecture by printing the model object
# This output includes the structure of the model showing the defined layers and their parameters,
# which helps in understanding the model's design and debugging if necessary.
diffusion_model.unet
    

# Load the StanfordCars dataset with automatic download if not present in the current directory
data = torchvision.datasets.StanfordCars(root=".", download=True)

# Visualize a selection of images from the dataset using the show_images function
show_images(data)

# Load preprocessed dataset
data = load_transformed_dataset(IMG_SIZE)
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # Create DataLoader for batching and shuffling

# Simulate forward diffusion
simulate_forward_diffusion(dataloader, diffusion_model)

"""## Training"""

# Load a previously trained model state or initialize a new model
load_model(model=diffusion_model.unet, isNew=False, filepath='Models/Improvements/Cosine_Scheduler/model_epoch_72.pth', device=device)

# Initialize the optimizer with the model parameters and a learning rate
optimizer = Adam(diffusion_model.unet.parameters(), lr=0.001)

# Define the total number of epochs for training
epochs = 100
print(f'Device: {diffusion_model.device}')

for epoch in range(72, epochs):
    # Create a tqdm progress bar for visual feedback
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{epochs}")

    # Iterate over the dataloader to process each batch
    for step, batch in progress_bar:
      optimizer.zero_grad() # Zero the gradients to prevent accumulation

      # Randomly select timesteps for each image in the batch
      t = torch.randint(0, diffusion_model.T, (BATCH_SIZE,), device=device).long()

      # Calculate the loss for the current batch and timestep
      loss = diffusion_model.get_loss(batch[0], t)

      loss.backward() # Perform backpropagation to calculate gradients
      
      optimizer.step() # Update model parameters

      # Update the tqdm progress bar with the current loss
      progress_bar.set_postfix(loss=loss.item())
      
      # Optionally, save the model and generate sample progressions at the start of each epoch
      if step == 0:
        save_model(diffusion_model.unet, filepath=f'Models/Improvements/Cosine_Scheduler/model_epoch_{epoch}.pth')

        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        save_sample_progression(epoch, IMG_SIZE, diffusion_model.device, diffusion_model, filename_prefix=f"progression_step_{step}")


      