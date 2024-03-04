import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

PATH = 'Improvements/Spectral_Norm_and_Cosine_Scheduler'

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

def load_model(model, isNew=True, filepath='model_state_dict.pth', device='cpu'):
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
    
def load_transformed_dataset(img_size):
    """
    Loads and transforms the StanfordCars dataset for use in the diffusion model.

    Parameters:
    - img_size (int)
    
    Returns:
    - A concatenated dataset of the transformed training and test sets.
    """

    # Define transformations: resize, horizontal flip, convert to tensor, normalize
    data_transforms = [
        transforms.Resize((img_size, img_size)), # Resize images to a fixed size
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

@torch.no_grad()
def save_sample_progression(epoch, img_size, device, diffusion_model, filename_prefix="progression"):
    """
    Visualizes and saves the progression of generated images over a series of timesteps during the reverse diffusion process.

    This function generates an image starting from random noise and progressively refines it through the reverse diffusion steps.
    The generated images at specified intervals are plotted and saved to visually demonstrate the model's generative process over time.

    Parameters:
    - epoch (int): The current training epoch. Used in the filename to identify the stage of training.
    - img_size (int)
    - device
    - difussion_model
    - filename_prefix (str, optional): A prefix for the saved filename. Defaults to "progression".
    """

    img = torch.randn((1, 3, img_size, img_size), device=device) # Start with a random noise image

    plt.figure(figsize=(15,15)) # Initialize a figure for plotting
    plt.axis('off') # Turn off the axis for a cleaner visualization

    num_images = 10 # Number of images to display in the progression
    stepsize = int(diffusion_model.T/num_images) # Calculate the interval between timesteps to display

    # Iterate over timesteps in reverse order to simulate the reverse diffusion process
    for i in range(0, diffusion_model.T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long) # Create a tensor for the current timestep
        img = diffusion_model.sample_timestep(img, t) # Apply the reverse diffusion step
        img = torch.clamp(img, -1.0, 1.0)  # Clamp the image values to maintain the expected range

        # Plot the image at specified intervals
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1) # Determine the subplot position
            show_tensor_image(img.detach().cpu()) # Show the image (move to CPU if necessary)

    # Construct the save path using the provided filename prefix and the current epoch
    save_path = f"Imgs/{PATH}/{filename_prefix}_epoch_{epoch}.png"
    plt.savefig(save_path)  # Save the figure to the specified path
    plt.close()  # Close the figure to free memory
    print(f"Progression image saved to {save_path}") # Print a confirmation message

def simulate_forward_diffusion(dataloader, diffusion_model):
    # Prepare to visualize the forward diffusion process
    image = next(iter(dataloader))[0] # Extract a single batch and select the first image

    plt.figure(figsize=(15,15)) # Initialize figure for plotting
    plt.axis('off') # Hide axis for cleaner visualization
    num_images = 10 # Number of images to display
    stepsize = int(diffusion_model.T/num_images) # Calculate step size to evenly distribute timesteps

    # Initialize the figure outside the loop
    plt.figure(figsize=(20, 4)) # Adjust the size as needed

    # Generate and display images at various timesteps during the forward diffusion
    for idx, step in enumerate(range(0, diffusion_model.T, stepsize), start=1):
        # Current timestep as a tensor
        t = torch.Tensor([step]).type(torch.int64)

        # Apply forward diffusion to the image
        img, noise = diffusion_model.forward_diffusion_sample(image, t)

        # Create subplot for each timestep
        plt.subplot(1, num_images + 1, idx)

        # Display the resulting noisy image
        show_tensor_image(img)

        # Optionally, you can set the title for each subplot to indicate the timestep
        plt.title(f"Timestep: {step}")

    # After preparing all subplots, display the figure with all images
    plt.show()