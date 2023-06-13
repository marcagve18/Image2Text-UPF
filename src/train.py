import os
import numpy as np
import torch.nn as nn
import torch
from data_loader import get_loader
import math
import torch.utils.data as data
import MyTorchWrapper as mtw
from architectures import ImageCaptioner
import torchinfo
from typing import List


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

    for epoch in range(1, num_epochs + 1):
        for i_step in range(1, total_step + 1):
            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader.dataset.get_train_indices()

            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler

            # Obtain the batch.
            images, captions, _ = next(iter(data_loader))

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions = captions.to(device)

            # Forward - backward pass
            image_captioner.zero_grad()  # Zero the gradients.
            outputs = image_captioner.forward(
                images, captions
            )  # Passing the inputs through the CNN-RNN model
            # exit()
            loss = criterion(
                outputs.view(-1, vocab_size), captions.view(-1)
            )  # Calculating the batch loss.
            loss.backward()  # Backwarding pass
            optimizer.step()  # Updating the parameters in the optimizer

            # Get training statistics
            stats = (
                f"Epoch [{epoch}/{num_epochs}], Step [{i_step}/{total_step}], "
                f"Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):.4f}"
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
    from architectures import EB7_LSTM, R50_LSTM, ViT_LSTM

    # Build data loader.
    cocoapi_year = "2017"
    data_loader = get_loader(
        image_folder=f"../clean_data/train{cocoapi_year}/",
        annotations_file=f"../data/cocoapi/annotations/captions_train{cocoapi_year}.json",
        batch_size=128,
        vocab_threshold=5, # minimum word count threshold
        vocab_from_file=True, # if True, load existing vocab file
        ratio=0.001, # proportion of coco dataset to use
    )

    # Initializing image captioning models
    vocab_size = len(data_loader.dataset.vocab)  # The size of the vocabulary
    models: List[ImageCaptioner] = []

    models.append(R50_LSTM(
        embed_size=256, # dimensionality of image and word embeddings
        hidden_size=512, # number of features in hidden state of the RNN decoder
        lstm_layers=1, # Number of hidden layers of each lstm cell
        vocabulary_size=vocab_size,
        bidirectional_lstm=False,
    ))
    models.append(EB7_LSTM(
        embed_size=256, # dimensionality of image and word embeddings
        hidden_size=512, # number of features in hidden state of the RNN decoder
        lstm_layers= 3, # Number of hidden layers of each lstm cell
        vocabulary_size=vocab_size,
        bidirectional_lstm=False,
    ))
    models.append(EB7_LSTM(
        embed_size=256,  # dimensionality of image and word embeddings
        hidden_size=512,  # number of features in hidden state of the RNN decoder
        lstm_layers=3, # Number of hidden layers of each lstm cell
        vocabulary_size=vocab_size,
        bidirectional_lstm=True,
    ))
    models.append(ViT_LSTM(
        embed_size=256, # dimensionality of image and word embeddings
        hidden_size=512, # number of features in hidden state of the RNN decoder
        lstm_layers=3, # Number of hidden layers of each lstm cell
        vocabulary_size=vocab_size,
        bidirectional_lstm=False,
    ))
    models.append(ViT_LSTM(
        embed_size=256, # dimensionality of image and word embeddings
        hidden_size=256, # number of features in hidden state of the RNN decoder
        lstm_layers=5, # Number of hidden layers of each lstm cell
        vocabulary_size=vocab_size,
        bidirectional_lstm=False,
    ))

    # Train the model
    for image_captioner in models:
        train(
            data_loader=data_loader,
            model=image_captioner,
            num_epochs=3,  # number of training epochs
            criterion=nn.CrossEntropyLoss(),
            checkpoint_folder="../models2/", # folder in which to store checkpoints of the training weights
            log_folder="../logs2/",
            save_every=1,  # determines frequency (epochs) of saving model weights
            print_every=20,  # determines frequency (steps) for printing average loss
        )
