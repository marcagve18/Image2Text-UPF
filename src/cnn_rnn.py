import os
import sys
# sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models
import torch
from torchvision import transforms
from data_loader import get_loader
import math
import torch.utils.data as data
import MyTorchWrapper as mtw


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # disable learning for parameters
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocabulary_size, num_layers=1, bidirectional_lstm=False):
        super().__init__()
        
        self.bidirectional = bidirectional_lstm
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocabulary_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional_lstm)
        self.last_linear = nn.Linear(hidden_size * 2 if self.bidirectional else hidden_size, vocabulary_size)
        
        
        # self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size)) # Initializing values for hidden and cell state

    def forward(self, features, captions):    
        # Remove <end> token from captions and embed captions
        cap_embedding = self.vocabulary_embedding(captions[:, :-1])

        # Concatenate the images features to the first of caption embeddings.
        embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1)

        lstm_out, _ = self.lstm(embeddings)
        outputs = self.last_linear(lstm_out)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """
        accepts pre-processed image tensor (inputs) and returns predicted
        sentence (list of tensor ids of length max_len)
        Args:
            inputs: shape is (1, 1, embed_size)
            states: initial hidden state of the LSTM
            max_len: maximum length of the predicted sentence

        Returns:
            res: list of predicted words indices
        """
        res = []

        # Now we feed the LSTM output and hidden states back into itself to get the caption
        for i in range(max_len):
            lstm_out, states = self.lstm(
                inputs, states
            )  # lstm_out: (1, 1, hidden_size)
            outputs = self.last_linear(lstm_out.squeeze(dim=1))  # outputs: (1, vocab_size)
            _, predicted_idx = outputs.max(dim=1)  # predicted: (1, 1)
            res.append(predicted_idx.item())
            # if the predicted idx is the stop index, the loop stops
            if predicted_idx == 1:
                break
            inputs = self.vocabulary_embedding(predicted_idx)  # inputs: (1, embed_size)
            # prepare input for next iteration
            inputs = inputs.unsqueeze(1)  # inputs: (1, 1, embed_size)

        return res

class ImageCaptioner(nn.Module):
    def __init__(self, embed_size, hidden_size, vocabulary_size, num_layers=1, bidirectional_lstm=False) -> None:
        super().__init__()
        self.CNN = EncoderCNN(embed_size)
        self.RNN = DecoderRNN(hidden_size, embed_size, vocabulary_size, num_layers=num_layers, bidirectional_lstm=bidirectional_lstm)
    

    def forward(self, images, captures):
        out_features = self.CNN(images)
        output = self.RNN(out_features, captures)

        return output
    




if __name__ == '__main__':
    cocoapi_year = "2017"

    batch_size = 128  # batch size
    vocab_threshold = 5  # minimum word count threshold
    vocab_from_file = True  # if True, load existing vocab file
    embedding_size = 256  # dimensionality of image and word embeddings
    hidden_size = 512  # number of features in hidden state of the RNN decoder
    num_epochs = 3  # number of training epochs
    save_every = 1  # determines frequency of saving model weights
    print_every = 20  # determines window for printing average loss
    log_file = "training_log.txt"  # name of file with saved training loss and perplexity


    # Build data loader.
    data_loader = get_loader(
        image_folder=f"../clean_data/train{cocoapi_year}/",
        annotations_file=f"../data/cocoapi/annotations/captions_train{cocoapi_year}.json",
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        vocab_from_file=vocab_from_file,
        ratio=0.5
    )

    # The size of the vocabulary.
    vocab_size = len(data_loader.dataset.vocab)

    # Initializing the encoder and decoder
    image_captioner = ImageCaptioner(embedding_size, hidden_size, vocab_size)

    # Move models to device
    device = mtw.get_torch_device(use_gpu=True, debug=True)
    image_captioner.to(device)

    # Defining the loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Specifying the learnable parameters of the mode
    params = list(image_captioner.RNN.parameters()) + list(image_captioner.CNN.embed.parameters())

    # Defining the optimize
    optimizer = torch.optim.Adam(params, lr=0.001)

    # Set the total number of training steps per epoch
    total_step = math.ceil(len(data_loader.dataset) / data_loader.batch_sampler.batch_size)

    print(total_step)

    f = open(log_file, "w")

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

            # Zero the gradients.
            image_captioner.zero_grad()

            # Passing the inputs through the CNN-RNN model
            outputs = image_captioner.forward(images, captions)

            # Calculating the batch loss.
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

            

            # Backwarding pass
            loss.backward()

            # Updating the parameters in the optimizer
            optimizer.step()

            # Getting training statistics
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
                image_captioner.RNN.state_dict(), os.path.join("../models", "decoder-%d.pkl" % epoch)
            )
            torch.save(
                image_captioner.CNN.state_dict(), os.path.join("../models", "encoder-%d.pkl" % epoch)
            )

    # Close the training log file.
    f.close()