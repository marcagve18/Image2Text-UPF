import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights
from pycocotools.coco import COCO
from PIL import Image
from transformers import GPT2LMHeadModel
from torch.utils.data import Dataset
import skimage.io as io


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # We use pretrained VGG
        self.features = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # We linearize the output
        x = x.view(x.size(0), -1)
        return x



class CaptionGenerator(nn.Module):
    def __init__(self):
        super(CaptionGenerator, self).__init__()
        self.vgg = VGG()
        self.fc = nn.Linear(25088, 768) # Fully connected to match GPT2 embedding size
        
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_model.config.add_cross_attention = True

    def forward(self, images, captions):
        # Step 1: Extract image features using VGG
        image_features = self.vgg(images)

        prediction_outputs = []
        for image_feature, caption in zip(image_features, captions):
            # Step 2: Extracting the embeddings of the captions
            embeds = self.gpt2_model.transformer.wte.weight[caption, :]

            # Step 3: Pass image features through a feed-forward network to get a size suitable for GPT-2
            image_feature = self.fc(image_feature)

            # Step 4: Combine image features and caption tokens
            # We add the image features at the start of each sequence (image_features are tensors of the appropriate size)
            embeds = torch.cat((image_feature.unsqueeze(0).unsqueeze(0), embeds), dim=1)

            # Step 5: Forward pass through GPT-2 model
            outputs = self.gpt2_model(inputs_embeds = embeds)
            predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
            prediction_outputs.append(predicted_token_ids)
            

        return prediction_outputs


class CocoDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(captions_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        url = image_info["coco_url"]
        image = Image.fromarray(io.imread(url))
        
        if self.transform:
            image = self.transform(image)

        caption_ids = self.coco.getAnnIds(image_id)
        caption = [self.coco.anns[cid]['caption'] for cid in caption_ids]
        caption = self.coco.anns[caption_ids[0]]['caption']
        
        return image, caption