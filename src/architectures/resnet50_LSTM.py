import torch
import torch.nn as nn
import torchvision.models as models
from .BaseImageCaptioner import EncoderCNN, DecoderRNN, ImageCaptioner


class R50_LSTM_CNN(EncoderCNN):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Disable learning for pretrained parameters
        for param in resnet.parameters():
            param.requires_grad_(False)

        # Change the last classification layer of the model
        modules = list(resnet.children())[:-1]
        self.__pretrained_cnn = nn.Sequential(*modules)

        # Add new Linear layer to adapt model to image captioning.
        self.__fc = nn.Linear(resnet.fc.in_features, embed_size)

    @property
    def fc(self):
        return self.__fc

    def pretrained_cnn(self, input: torch.Tensor) -> torch.Tensor:
        return self.__pretrained_cnn(input)


class R50_LSTM(ImageCaptioner):
    def __init__(
        self,
        embed_size,
        hidden_size,
        lstm_layers,
        vocab_size,
        bidirectional_lstm=False,
    ) -> None:
        super().__init__()
        self.__name = f"resnet50_LSTM_e{embed_size}_h{hidden_size}_l{lstm_layers}"
        if bidirectional_lstm:
            self.__name += "_bidirectional"

        self.__vocab_size = vocab_size
        self.__CNN = R50_LSTM_CNN(embed_size)
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
