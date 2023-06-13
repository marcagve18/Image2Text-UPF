import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import GPT2LMHeadModel


from .BaseImageCaptioner import EncoderCNN, DecoderRNN, ImageCaptioner

class GPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_model.config.add_cross_attention = True

    def forward(self, features, captions):
        # prediction_outputs = []

        # Step 1: Extracting the embeddings of the captions
        embeds = self.gpt2_model.transformer.wte.weight[captions, :]

        # Step 2: Combine image features and caption tokens
        # We add the image features at the start of each sequence (image_features are tensors of the appropriate size)
        embeds = torch.cat((features.unsqueeze(dim=1), embeds), dim=1)
        

        # Step 3: Forward pass through GPT-2 model
        outputs = self.gpt2_model.generate(inputs_embeds = embeds)
        # predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        # prediction_outputs.append(predicted_token_ids)

        # max_output = max([caption.shape[0] for caption in prediction_outputs])

        # outputs = torch.Tensor()
        # for predicted_caption in prediction_outputs: 
        #     output_tensor = predicted_caption   
        #     if predicted_caption.shape[0] < max_output:
        #         needed_padding = max_output - predicted_caption.shape[0]
        #         output_tensor = torch.functional.pad(predicted_caption, (needed_padding, 0, 0, 0), value=)
        #     outputs.cat()

        return outputs.tensor


class ViT(EncoderCNN):
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
        self.__feature_extractor = create_feature_extractor(vit, return_nodes=['getitem_5'])

        # Add new Linear layers to adapt model to image captioning.
        self.__fc = nn.Sequential(
            nn.Linear(vit.heads[-1].in_features, embed_size, bias=True),
        )

    @property
    def fc(self):
        return self.__fc

    def pretrained_cnn(self, input: torch.Tensor) -> torch.Tensor:
        return self.__feature_extractor(input)['getitem_5']


class ViT_GPT2(ImageCaptioner):
    def __init__(self, vocab_size, embed_size=768) -> None:
        super().__init__()
        self.__name = "ViT_GPT2"
        self.__vocab_size = vocab_size
        self.__CNN = ViT(embed_size=embed_size)
        self.__RNN = GPT2()

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
