import os
import numpy as np
import torch.nn as nn
import torch
import math
import torch.utils.data as data
import MyTorchWrapper as mtw
from architectures import ImageCaptioner
import torchinfo
from typing import List
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer



def train(
    data_loader: data.DataLoader,
    model: ImageCaptioner,
    num_epochs: int,
    criterion=nn.CrossEntropyLoss(),
    checkpoint_folder="models/",
    log_folder="logs/",
    save_every=1,
    print_every=20,
):
    # Move models to device
    device = mtw.get_torch_device(use_gpu=True, debug=True)
    model.to(device)
    criterion.to(device)

    # Defining the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Ensure that output folders exist, otherwise create them
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if not os.path.exists(checkpoint_folder + model.name):
        os.makedirs(checkpoint_folder + model.name)

    # Set the total number of training steps per epoch
    total_step = math.ceil(
        len(data_loader.dataset) / data_loader.batch_sampler.batch_size
    )
    print("Total number of steps per epoch:", total_step)
    f = open(log_folder + model.name + ".log", "w")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})    

    for epoch in range(1, num_epochs + 1):
        for i_step in range(1, total_step + 1):
            # Obtain the batch.
            images, captions, _ = next(iter(data_loader))

            # Tokenizing the captions of the batch
            captions = [tokenizer(caption, return_tensors="pt").input_ids for caption in captions ]

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions = captions.to(device)

            # Forward - backward pass
            image_captioner.zero_grad()  # Zero the gradients.
            outputs = image_captioner.forward(
                images, captions
            )  # Passing the inputs through the CNN-RNN model

            batch_loss = 0
            for caption_input, caption_output in zip(captions, outputs):
                caption_input = caption_input.type(torch.float)
                caption_input.requires_grad_()
                caption_output = caption_output.type(torch.float)
                caption_output.requires_grad_()
                
                # Determine the difference in dimensions and add padding to compute the loss properly
                dim_diff = abs(caption_input.shape[1] - caption_output.shape[1])
                if (caption_input.shape[1] < caption_output.shape[1]):
                    caption_input = nn.functional.pad(caption_input, (0, dim_diff))
                elif (caption_input.shape[1] > caption_output.shape[1]):
                    caption_output = nn.functional.pad(caption_output, (0, dim_diff))

                loss = criterion(caption_output, caption_input)
                batch_loss += loss

                loss.backward()
                optimizer.step()

            batch_loss = batch_loss / len(captions)
            # Get training statistics
            stats = (
                f"Epoch [{epoch}/{num_epochs}], Step [{i_step}/{total_step}], "
                f"Loss: {batch_loss:.4f}, Perplexity: {np.exp(batch_loss):.4f}"
            )

            # Print training statistics to file.
            f.write(stats + "\n")
            f.flush()

            # Print training statistics (on different line).
            if i_step % print_every == 0:
                print("\r" + stats)

        # Save the weights.
        if epoch % save_every == 0:
            torch.save(
                image_captioner.RNN.state_dict(),
                os.path.join(f"{checkpoint_folder}{model.name}", f"decoder-epoch{epoch}.pkl"),
            )
            torch.save(
                image_captioner.CNN.state_dict(),
                os.path.join(f"{checkpoint_folder}{model.name}", f"encoder-epoch{epoch}.pkl"),
            )

    # Close the training log file.
    f.close()

    # Save a summary of the model for future reference
    model_stats = torchinfo.summary(
        model,
        input_data=[images, captions],
        device=device,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        mode="train",
        verbose=0
    )
    with open(checkpoint_folder + model.name + "/model_summary.txt", "w") as file:
        file.write(str(model_stats))




if __name__ == "__main__":
    from architectures.ViT_GPT2 import ViT_GPT2
    from architectures.VGG_GPT2 import CocoDataset

    # Build data loader.
    cocoapi_year = "2017"

    train_image_dir = f"../data/cocoapi/images/train{cocoapi_year}"
    train_captions_file = f"../data/cocoapi/annotations/captions_train{cocoapi_year}.json"
    train_dataset = CocoDataset(train_image_dir, train_captions_file, f"../clean_data/train{cocoapi_year}/")

    # Data loaders for training and validation
    batch_size = 32
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initializing image captioning models
    models: List[ImageCaptioner] = []

    models.append(ViT_GPT2())

    # Train the model
    for image_captioner in models:
        train(
            data_loader=data_loader,
            model=image_captioner,
            num_epochs=3,  # number of training epochs
            criterion=nn.CrossEntropyLoss(),
            checkpoint_folder="../models/", # folder in which to store checkpoints of the training weights
            log_folder="../logs/",
            save_every=1,  # determines frequency (epochs) of saving model weights
            print_every=20,  # determines frequency (steps) for printing average loss
        )
