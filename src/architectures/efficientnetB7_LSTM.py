import torch
import torch.nn as nn
import torchvision.models as models
from .BaseImageCaptioner import EncoderCNN, DecoderRNN, ImageCaptioner


class EB7_LSTM_CNN(EncoderCNN):
    def __init__(self, embed_size):
        super().__init__()
        efficientnet_b7 = models.efficientnet_b7(
            weights=models.EfficientNet_B7_Weights.DEFAULT
        )

        # Disable learning for pretrained parameters
        for param in efficientnet_b7.parameters():
            param.requires_grad_(False)

        # Remove the last classification layer of the model
        modules = list(efficientnet_b7.children())[:-1]
        self.__pretrained_cnn = nn.Sequential(*modules)

        # Add new Linear layers to adapt model to image captioning.
        self.__fc = nn.Sequential(
            nn.Dropout(0.5, inplace=True),
            nn.Linear(efficientnet_b7.classifier[-1].in_features, 1000, bias=True),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(1000, embed_size, bias=True),
        )

    @property
    def fc(self):
        return self.__fc

    def pretrained_cnn(self, input: torch.Tensor) -> torch.Tensor:
        return self.__pretrained_cnn(input)


class EB7_LSTM(ImageCaptioner):
    def __init__(
        self,
        embed_size,
        hidden_size,
        lstm_layers,
        vocab_size,
        bidirectional_lstm=False,
    ) -> None:
        super().__init__()
        self.__name = f"efficientnetB7_LSTM_e{embed_size}_h{hidden_size}_l{lstm_layers}"
        if bidirectional_lstm:
            self.__name += "_bidirectional"

        self.__vocab_size = vocab_size
        self.__CNN = EB7_LSTM_CNN(embed_size)
        self.__RNN = DecoderRNN(
            hidden_size,
            embed_size,
            vocab_size,
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
    
    @property
    def vocab_size(self) -> int:
        return self.__vocab_size
