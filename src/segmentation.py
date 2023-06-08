import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg16, VGG16_Weights
import torch.optim as optim
from torchvision import transforms
from pycocotools.coco import COCO
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader

# Step 1: Set up the environment

# Step 2: Preprocess the data
# Define the transformation to be applied to each image
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to a fixed size
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize the image
])

# Step 3: Build the VGG model
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

# # Step 4: Build the language model
class CaptionGenerator(nn.Module):
    def __init__(self):
        super(CaptionGenerator, self).__init__()
        self.vgg = VGG()
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_tokenizer.add_special_tokens({'pad_token': self.gpt2_tokenizer.eos_token})
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_model.config.add_cross_attention = True
        # self.gpt2_model.config.return_dict = False

    def forward(self, images, captions):
        # Step 1: Extract image features using VGG
        image_features = self.vgg(images)

        for image_feature, caption in zip(image_features, captions):
            # Step 2: Tokenize captions and prepare input tensors
            inputs = self.gpt2_tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=50)
            embeds = self.gpt2_model.transformer.wte.weight[inputs, :]
            print(embeds)
            # Step 3: Pass image features through a feed-forward network to get a format suitable for GPT-2
            # Note: You might need to adjust the dimensions and other details according to your implementation.
            #image_features = self.ffn(image_features.view(image_features.size(0), -1))

            # Step 4: Combine image features and caption tokens
            # We add the image features at the start of each sequence (assuming image_features are tensors of the appropriate size)
            inputs["input_ids"] = torch.cat((image_feature, inputs["input_ids"]), dim=1)
            inputs["attention_mask"] = torch.cat((torch.ones_like(image_feature), inputs["attention_mask"]), dim=1)

            # Step 5: Forward pass through GPT-2 model
            outputs = self.gpt2_model(**inputs)

        return outputs

# # Step 5: Prepare the data
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
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        caption_ids = self.coco.getAnnIds(image_id)
        # caption = [self.coco.anns[cid]['caption'] for cid in caption_ids]
        caption = self.coco.anns[caption_ids[0]]['caption']
        
        return image, caption

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_image_dir = "../data/train2014"
train_captions_file = "../data/annotations/captions_train2014.json"
train_dataset = CocoDataset(train_image_dir, train_captions_file, transform=image_transform)

val_image_dir = "../data/val2014"
val_captions_file = "../data/annotations/captions_val2014.json"
val_dataset = CocoDataset(val_image_dir, val_captions_file, transform=image_transform)

# Create train and validation data loaders
batch_size = 32
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Obtain two random samples from the training dataset
samples = np.random.choice(len(train_dataset), size=2, replace=False)
image1, caption_inputs1 = train_dataset[samples[0]]
image2, caption_inputs2 = train_dataset[samples[1]]

# print(caption_inputs1[0])
# print(train_dataset.tokenizer.encode(caption_inputs1[0]))
# print(caption_inputs2)

# Convert the tensor images to numpy arrays
image1 = image1.permute(1, 2, 0).numpy()
image2 = image2.permute(1, 2, 0).numpy()

# # Decode the tokenized captions
# captions1 = [train_dataset.tokenizer.decode(token_ids) for token_ids in caption_inputs1.input_ids]
# # captions2 = [train_dataset.tokenizer.decode(token_ids) for token_ids in caption_inputs2.input_ids]

# # Plot the images with their captions
# plt.imshow(image1)
print(caption_inputs1)

# plt.show()

# Instantiate the pre-trained models
cnn_backbone = vgg16(weights=VGG16_Weights.DEFAULT)
language_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Freeze the parameters of the CNN backbone
for param in cnn_backbone.parameters():
    param.requires_grad = False


# Create an instance of the Image Captioning model
caption_generator = CaptionGenerator()

# Set up the optimizer and loss function
learning_rate = 0.001
optimizer = optim.Adam(caption_generator.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
num_epochs = 1000


# # Step 7: Perform inference

# Main code
caption_generator.train()

# Training loop
for epoch in range(num_epochs):
    for images, captions in train_data_loader:
        optimizer.zero_grad()
        # print(captions)
        outputs = caption_generator(images, captions)
        loss = criterion(outputs, captions)
        print(loss, "WASA")
        loss.backward()
        optimizer.step()

# Inference

