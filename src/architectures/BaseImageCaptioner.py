from typing import Iterator
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter


class EncoderCNN(nn.Module):
    def forward(self, images):
        features = self.pretrained_cnn(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

    @property
    def fc(self):
        """Returns fully-connected layer(s) added to the pretrained CNN.
        """

    @property
    def pretrained_cnn(self) -> nn.Module:
        """Returns the pretrained CNN used as a base to extract image features.
        """


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocabulary_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.vocabulary_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.last_linear = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, vocabulary_size)

    def forward(self, features, captions):    
        # Remove <end> token from captions and use embedding to always have the
        # same input size for the LSTM.
        cap_embedding = self.vocabulary_embedding(captions[:, :-1])

        # Concatenate the images features to the first of caption embeddings.
        embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1)

        # Pass the combined input data through the network
        lstm_out, _ = self.lstm(embeddings)
        lstm_out_concat = lstm_out.reshape(-1, self.hidden_size * 2 if self.bidirectional else self.hidden_size)
        outputs = self.last_linear(lstm_out_concat)
        
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
    def forward(self, images, captures):
        out_features = self.CNN(images)
        output = self.RNN(out_features, captures)
        return output
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Returns the learnable parameters of the model
        """
        return list(self.RNN.parameters(recurse)) + list(self.CNN.fc.parameters(recurse))

    @property
    def name(self) -> str:
        """Returns the name of the model.
        """

    @property
    def CNN(self) -> EncoderCNN:
        """The CNN encoder used by the model.
        """

    @property
    def RNN(self) -> DecoderRNN:
        """The RNN decoder used by the model.
        """

    