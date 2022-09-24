from diffusers import UNet2DModel
from torchvision.transforms.transforms import InterpolationMode
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, Normalize

import diffusers
import numpy as np
import torch
import datasets
from dataclasses import dataclass
import requests
from datetime import datetime

@dataclass
class TrainingConfig:
  # Data
  dataset_path: str = 'https://aimote-datasets.s3.us-west-1.amazonaws.com/aimote_ffz_images_1x_cleaned_15k.tar.gz'
  #    dataset_name: str = 'aimote_ffz_images_1x_cleaned_15k'

  # Training
  image_size: int = 32
  train_batch_size: int = 64
  eval_batch_size: int = 64
  num_workers: int = 1
  num_epochs: int = 3000
  learning_rate: float = 1e-4
  lr_warmup_steps: int = 500
  save_image_epochs: int = 50
  save_model_epochs: int = 50
  project_name = 'aimote-ddpm'
  gradient_accumulation_steps = 1
  base_output_dir = 'output'
  eval_inference_iterations = 1000

  # Misc
  device: str = 'cuda'
  seed: int = 1337
  mixed_precision: str = 'no'

  #hack
  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

  def output_dir(self):
    return os.path.join(self.base_output_dir, self.project_name, self.timestamp)

config = TrainingConfig()

dlm = datasets.utils.download_manager.DownloadManager()
extracted_dataset_path = dlm.download_and_extract(config.dataset_path)
dataset = datasets.load_dataset(extracted_dataset_path)
dataset_1k = torch.utils.data.Subset(dataset['train'], list(range(1000)))


preprocess_transform = Compose([
    Resize((32, 32), InterpolationMode.LANCZOS),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def transform(examples):
  images = [preprocess_transform(img.convert('RGB')) for img in examples['image']]
  return {'images': images}
dataset.set_transform(transform)

train_dataloader = torch.utils.data.DataLoader(dataset_1k, batch_size=config.train_batch_size, shuffle=True)


# TODO: understand the structure of this more completely
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 128, 256, 512),  # the number of output channes for each UNet block
    down_block_types=(
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
      ),
)

sample_image = dataset['train'][0]['images'].unsqueeze(0)
print('Input shape:', sample_image.shape)
print('Output shape:', model(sample_image, timestep=0).sample.shape)

num_params = sum(p.numel() for p in model.parameters())
print(f'Num params: {num_params}')

from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, tensor_format="pt")

import torchvision.transforms.functional as F
from PIL import Image

def to_pil(im):
  im = ((im + 1.0) * 0.5) * 255.0 # Convert back to range 0-255
  return F.to_pil_image(im.squeeze().type(torch.uint8))

def display_im_tens(im):
  display(to_pil(im).resize((128, 128), Image.NEAREST))

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

from diffusers.optimization import get_cosine_schedule_with_warmup

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

from accelerate import Accelerator

from tqdm.auto import tqdm
import os

from diffusers import DDPMPipeline

import math
import os

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

@torch.no_grad()
def call_model(
        pipe,
        batch_size: int = 1,
        generator=None,
        output_type = "pil",
        inference_steps=1000,
        **kwargs,):
        # Sample gaussian noise to begin loop
    image = torch.randn(
        (batch_size, pipe.unet.in_channels, pipe.unet.sample_size, pipe.unet.sample_size),
        generator=generator,
    )
    image = image.to(pipe.device)

    # set step values
    pipe.scheduler.set_timesteps(inference_steps)

    for t in pipe.progress_bar(pipe.scheduler.timesteps):
        # 1. predict noise model_output
        model_output = pipe.unet(image, t).sample#['sample']

        # 2. compute previous image: x_t -> t_t-1
        image = pipe.scheduler.step(model_output, t, image, generator=generator).prev_sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == "pil":
        image = pipe.numpy_to_pil(image)
    return image

def evaluate(config, epoch, pipeline, inference_steps=1000):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = call_model(
        pipeline,
        batch_size = config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        inference_steps=inference_steps,
    )

    # Make a grid out of the images
    grid_size = int(math.ceil(math.sqrt(config.eval_batch_size)))
    image_grid = make_grid(images, rows=grid_size, cols=grid_size)

    # Save the images
    test_dir = os.path.join(config.output_dir(), "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    global_step = 0

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir(), "logs")
    )
    if accelerator.is_main_process:
      accelerator.init_trackers(config.project_name)

    model.to(config.device)
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Now you train the model
    for epoch in range(config.num_epochs):
        #progress_bar = tqdm(
        #    total=len(train_dataloader),
        #    disable=not accelerator.is_local_main_process,
        #    desc=f"Epoch {epoch}")
        print(f'epoch {epoch} / {config.num_epochs}')
        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            #progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            #progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline, config.eval_inference_iterations)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir())

from accelerate import notebook_launcher
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args, num_processes=1)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir output