import torch.nn as nn
import torchvision.models as models
from architectures.BaseImageCaptioner import EncoderCNN, DecoderRNN, ImageCaptioner


class EB7_defaultLSTM_CNN(EncoderCNN):
    def __init__(self, embed_size):
        super().__init__()
        efficientnet_b7 = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)

        # Disable learning for pretrained parameters
        for param in self.pretrained_cnn.parameters():
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
    
    @property
    def pretrained_cnn(self) -> nn.Module:
        return self.__pretrained_cnn


class EB7_defaultLSTM(ImageCaptioner):
    def __init__(self, embed_size, hidden_size, vocabulary_size) -> None:
        super().__init__()
        self.name = 'efficientnetB7_bidirectionalLSTM'
        self.CNN = EB7_defaultLSTM_CNN(embed_size)
        self.RNN = DecoderRNN(hidden_size, embed_size, vocabulary_size, num_layers=1, bidirectional=True)
