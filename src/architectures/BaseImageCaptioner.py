from typing import Iterator
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from abc import abstractmethod


class EncoderCNN(nn.Module):
    def forward(self, images):
        features = self.pretrained_cnn(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

    @property
    @abstractmethod
    def fc(self):
        """Returns fully-connected layer(s) added to the pretrained CNN."""

    @abstractmethod
    def pretrained_cnn(self, input: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass with the pretrained CNN (only the layers that do
        not need to be re-trained).
        """


class DecoderRNN(nn.Module):
    def __init__(
        self,
        hidden_size,
        embedding_size,
        vocabulary_size,
        num_layers=1,
        bidirectional=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.vocabulary_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.last_linear = nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, vocabulary_size
        )

    def forward(self, features, captions):
        # Remove <end> token from captions and use embedding to always have the
        # same input size for the LSTM.
        cap_embedding = self.vocabulary_embedding(captions[:, :-1])

        # Concatenate the images features to the first of caption embeddings.
        embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1)

        # Pass the combined input data through the network
        lstm_out, _ = self.lstm(embeddings)
        lstm_out_concat = lstm_out.reshape(
            -1, self.hidden_size * 2 if self.bidirectional else self.hidden_size
        )
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
            outputs = self.last_linear(
                lstm_out.squeeze(dim=1)
            )  # outputs: (1, vocab_size)
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
        """Returns the learnable parameters of the model"""
        return list(self.RNN.parameters(recurse)) + list(self.CNN.fc.parameters(recurse))

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the model."""

    @property
    @abstractmethod
    def CNN(self) -> EncoderCNN:
        """The CNN encoder used by the model."""

    @property
    @abstractmethod
    def RNN(self) -> DecoderRNN:
        """The RNN decoder used by the model."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary used by the model."""









def beam_search(decoder_output, beam_width, max_length, start_token, end_token, device):
    """
    Perform beam search to generate the most likely sequence.

    Args:
        decoder_output (Tensor): The output from the decoder model.
        beam_width (int): The number of hypotheses to consider at each step.
        max_length (int): The maximum length of the generated sequence.
        start_token (int): The token representing the start of the sequence.
        end_token (int): The token representing the end of the sequence.

    Returns:
        list: The most likely sequence of tokens.
    """

    # Initialize beam search
    hypotheses = [[start_token] for _ in range(beam_width)]
    scores = torch.zeros(beam_width, device=device)

    # Beam search loop
    for t in range(max_length):
        # Expand hypotheses and scores
        hypotheses_expanded = [h for h in hypotheses for _ in range(beam_width)]
        scores_expanded = scores.repeat(beam_width)

        # Calculate scores for all candidate tokens
        decoder_output = decoder_output.view(beam_width, -1)  # (beam_width, vocab_size)
        candidate_scores = scores_expanded + decoder_output.view(-1)  # (beam_width * vocab_size)

        # Select top-K candidates
        top_scores, top_indices = torch.topk(candidate_scores, k=beam_width, dim=0)

        # Update hypotheses and scores
        new_hypotheses = []
        new_scores = []

        for score, index in zip(top_scores, top_indices):
            hyp_index = index // decoder_output.size(1)  # Index of the hypothesis
            token_index = index % decoder_output.size(1)  # Index of the token

            new_hypotheses.append(hypotheses_expanded[hyp_index] + [token_index.item()])
            new_scores.append(score.item())

        hypotheses = new_hypotheses
        scores = torch.tensor(new_scores, device=device)

        # Check if all hypotheses have ended
        all_ended = all(h[-1] == end_token for h in hypotheses)
        if all_ended:
            break

    # Select the hypothesis with the highest score
    best_hypothesis = hypotheses[scores.argmax()]

    return best_hypothesis
