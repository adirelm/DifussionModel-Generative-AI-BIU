import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib.ticker import FuncFormatter

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

def load_model(model, optimizer, filepath='model_checkpoint.pth', device='cpu'):
    """
    Loads the model's state dictionary and other training information from a file if it exists.
    If the file does not exist, it initializes the model on the specified device without loading any state.

    Parameters:
    - model (torch.nn.Module): The model into which the state dictionary will be loaded.
    - optimizer (torch.optim.Optimizer): The optimizer to which the state will be loaded.
    - filepath (str, optional): The path to the file from which the model and training information should be loaded.
    - device (str, optional): The device to load the model onto.

    Returns:
    - int: The epoch to start training from. Returns 1 if the checkpoint file does not exist.
    - list: The average loss history to date.
    """
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        average_loss_history = checkpoint.get('average_loss_history', [])

        print(f"Checkpoint loaded from {filepath}. Resuming from epoch {epoch}.")
        return epoch + 1, average_loss_history
    else:
        print("No checkpoint file found. Starting from scratch.")
        model.to(device)
        return 1, []

def save_model(model, optimizer, epoch, average_loss_history, filepath='model_checkpoint.pth'):
    """
    Saves the model's state dictionary, optimizer state, and training loss history to a file.

    Parameters:
    - model (torch.nn.Module): The model to save.
    - optimizer (torch.optim.Optimizer): The optimizer used for training.
    - epoch (int): The current epoch at the moment of saving.
    - average_loss_history (list): The list of average loss values recorded at each epoch.
    - filepath (str, optional): The path to the file where the model and training information should be saved.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'average_loss_history': average_loss_history
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath} at epoch {epoch}.")
    
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

@torch.no_grad()
def generate_images(diffusion_model, num_images, img_size, device):
    # Initialize a tensor to hold the generated images
    generated_images = torch.zeros((num_images, 3, img_size, img_size), device=device)

    # Generate each image individually
    for n in range(num_images):
        # Start with random noise for each image
        img = torch.randn((1, 3, img_size, img_size), device=device)

        # Iteratively apply the reverse diffusion steps over all timesteps
        for i in reversed(range(diffusion_model.T)):
            t = torch.full((1,), i, device=device, dtype=torch.long)  # Current timestep tensor
            img = diffusion_model.sample_timestep(img, t)  # Apply reverse diffusion step
        

        # Store the generated image
        generated_images[n] = torch.clamp(img.squeeze(0), 0.0, 1.0)  # Remove batch dimension added by sample_timestep

    return generated_images

def prepare_real_images_subset(dataset, num_images=100, img_size=299, batch_size=128, device='cpu'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    real_images = []
    for images, _ in loader:
        real_images.append(images)
        if len(real_images) * batch_size >= num_images:
            break
    real_images = torch.cat(real_images)[:num_images]
    real_images = F.interpolate(real_images, size=(img_size, img_size)).to(device)
    real_images = real_images / 2.0 + 0.5  # Rescale images from [-1, 1] to [0, 1]
    return real_images

def plot_loss(average_loss_history, epoch, total_epochs, PLOT_INTERVAL, PATH):
    if epoch % PLOT_INTERVAL == 0 or epoch == total_epochs:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, epoch + 1), average_loss_history, marker='o', linestyle='-', label='Average Loss over Epochs')
      
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss Progression')
        
        # Improve readability by showing every nth tick mark on x-axis
        # If there are many epochs, adjust n to a larger number
        n = max(1, int(epoch / 20))  # Adjust '20' to show fewer tick marks as needed
        plt.xticks(np.arange(1, epoch + 1, step=n))
        
        # Format the y-axis with three decimal places
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.3f}'))
        
        plt.legend()
        plt.grid(True)
        
        # Ensure the plots directory exists
        os.makedirs(f'Plots/{PATH}', exist_ok=True)
        plt.savefig(f'Plots/{PATH}/loss_plot_epoch_{epoch}.png')
        plt.close()