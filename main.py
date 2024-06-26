import torch
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
from unet import SimpleUnet
from torch.optim import Adam
import matplotlib.pyplot as plt
from diffusion import Diffusion
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from utils import prepare_real_images_subset, generate_images, plot_loss
from utils import PATH, show_images, load_model, save_model, load_transformed_dataset, save_sample_progression, simulate_forward_diffusion

def main(args):
  # Set parameters for data loading and image processing

  # Target size for resizing images
  IMG_SIZE = args.image_size

  # Batch size for data loading
  BATCH_SIZE = args.batch_size

  FID_INTERVAL = args.fid_interval
  PLOT_INTERVAL = args.plot_interval

  # Determine the device to use based on whether a GPU is available
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Instantiate the Diffusion model with SimpleUnet model
  diffusion_model = Diffusion(SimpleUnet(activation=args.activation, spectral_norm_on=args.spectral_norm_on), scheduler=args.scheduler, device=device)

  # Initialize FID metric
  fid_metric = FrechetInceptionDistance().to(device)

  # Calculate the total number of parameters in the model
  # This is done by iterating over all parameters (weights and biases) of the model,
  # calculating the number of elements in each parameter using numel(),
  # and summing these numbers to get the total count of trainable parameters.
  num_params = sum(p.numel() for p in diffusion_model.unet.parameters())

  # Print the total number of parameters to give an idea of the model's size and complexity
  print(f'Device: {diffusion_model.device}, Num params: {num_params}')

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
  # simulate_forward_diffusion(dataloader, diffusion_model)

  data_transform = Compose([
      Resize((IMG_SIZE, IMG_SIZE)), 
      ToTensor(),
      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  dataset = torchvision.datasets.StanfordCars(root=".", download=True, transform=data_transform)
  real_images_subset = prepare_real_images_subset(dataset, device=device)

  """## Training"""

  # Initialize the optimizer with the model parameters and a learning rate
  optimizer = Adam(diffusion_model.unet.parameters(), lr=args.lr)

  # Load a previously trained model state or initialize a new model
  epoch, average_loss_history = load_model(diffusion_model.unet, optimizer, filepath=args.load_model_path, device=device)

  # Define the total number of epochs for training
  epochs = args.epochs

  for epoch in range(epoch, epochs + 1):
      total_loss = 0

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

        total_loss += loss.item()

        # Update the tqdm progress bar with the current loss
        progress_bar.set_postfix(loss=loss.item())
        
        # Optionally, save the model and generate sample progressions at the start of each epoch
        # Check if this is the last step/batch of the epoch
        is_last_step = (step == len(dataloader) - 1)
        if is_last_step:
          save_model(diffusion_model.unet, optimizer, epoch, average_loss_history, filepath=f'{args.save_model_path}')

          print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
          save_sample_progression(epoch, IMG_SIZE, diffusion_model.device, diffusion_model, filename_prefix=f"progression_step_{step}")

      # Calculate and store the average loss for the epoch
      avg_loss = total_loss / len(dataloader)
      average_loss_history.append(avg_loss)

      # Save a plot of the average loss over epochs
      plot_loss(average_loss_history, epoch=epoch, total_epochs=epochs, PLOT_INTERVAL=PLOT_INTERVAL, PATH=PATH)

      if epoch % FID_INTERVAL == 0:
        print('Calculating FID...')
        fid_metric.reset()  # Prepare for a new FID calculation

        # Generate fake images
        fake_images = generate_images(diffusion_model, len(real_images_subset), IMG_SIZE, device)
        
        # If the images are normalized to [0, 1], rescale them to [0, 255]
        fake_images_rescaled = fake_images * 255.0

        # Convert to torch.uint8
        fake_images_uint8 = fake_images_rescaled.type(torch.uint8)

        # Resize fake images to match the size expected by the FID metric
        fake_images_resized = F.interpolate(fake_images_uint8.float(), size=(299, 299), mode='bilinear', align_corners=False).type(torch.uint8)

        # Similarly, ensure real_images_subset is in torch.uint8 and correct size
        real_images_subset_resized = F.interpolate(real_images_subset.float(), size=(299, 299), mode='bilinear', align_corners=False).type(torch.uint8)

        # Update FID metric with real and fake images
        fid_metric.update(real_images_subset_resized, real=True)
        fid_metric.update(fake_images_resized, real=False)

        # Compute FID score
        fid_score = fid_metric.compute()
        print(f"Epoch {epoch}: FID score = {fid_score.item()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=201)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)
    parser.add_argument('--scheduler',
                        help='type of scheduler to use.',
                        type=str,
                        choices=['regular', 'cosine'],
                        default='cosine')
    parser.add_argument('--spectral_norm_on',
                        help='whether to apply spectral normalization to the model.',
                        type=bool,
                        default=True)
    parser.add_argument('--activation',
                        help='type of activation to use.',
                        type=str,
                        choices=['relu', 'silu', 'gelu'],
                        default='relu')
    parser.add_argument('--image_size',
                        help='size of each image dimension.',
                        type=int,
                        default=64)
    parser.add_argument('--load_model_path',
                        help='path to load model from.',
                        type=str,
                        default=f'Models/{PATH}/model_epoch_200.pth')
    parser.add_argument('--save_model_path',
                        help='path to save model to.',
                        type=str,
                        default=f'Models/{PATH}')
    parser.add_argument('--plot_interval',
                        help='number of epochs between each plot.',
                        type=int,
                        default=2)
    parser.add_argument('--fid_interval',
                        help='number of epochs between each FID calculation.',
                        type=int,
                        default=100)

    args = parser.parse_args()
    main(args)