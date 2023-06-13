import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

from .BaseImageCaptioner import EncoderCNN, DecoderRNN, ImageCaptioner


class ViT_LSTM_CNN(EncoderCNN):
    def __init__(self, embed_size):
        super().__init__()
        # According to the original paper, smaller ViT models behave better
        # for smaller input datasets than larger ViT models. Since COCO dataset
        # is small, we use the smallest ViT model.
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        # Disable learning for pretrained parameters
        for param in vit.parameters():
            param.requires_grad_(False)

        # Remove the last classification layer of the model
        # https://discuss.pytorch.org/t/assertionerror-expected-batch-size-seq-length-hidden-dim-got-torch-size-1-768-24-31/150869/4
        self.__feature_extractor = create_feature_extractor(
            vit, return_nodes=["getitem_5"]
        )

        # Add new Linear layers to adapt model to image captioning.
        self.__fc = nn.Sequential(
            nn.Linear(vit.heads[-1].in_features, embed_size, bias=True),
        )

    @property
    def fc(self):
        return self.__fc

    def pretrained_cnn(self, input: torch.Tensor) -> torch.Tensor:
        return self.__feature_extractor(input)["getitem_5"]


class ViT_LSTM(ImageCaptioner):
    def __init__(
        self,
        embed_size,
        hidden_size,
        lstm_layers,
        vocabulary_size,
        bidirectional_lstm=False,
    ) -> None:
        super().__init__()
        self.__name = f"ViT_LSTM_e{embed_size}_h{hidden_size}_l{lstm_layers}"
        if bidirectional_lstm:
            self.__name += "_bidirectional"

        self.__CNN = ViT_LSTM_CNN(embed_size)
        self.__RNN = DecoderRNN(
            hidden_size,
            embed_size,
            vocabulary_size,
            num_layers=lstm_layers,
            bidirectional=bidirectional_lstm,
        )

    @property
    def name(self) -> str:
        return self.__name

    @property
    def CNN(self) -> EncoderCNN:
        return self.__CNN

    @property
    def RNN(self) -> DecoderRNN:
        return self.__RNN
