import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from architectures.vgg_gpt2 import CocoDataset, CaptionGenerator


cocoapi_year = 2017


train_image_dir = f"../data/cocoapi/images/train{cocoapi_year}"
train_captions_file = f"../data/cocoapi/annotations/captions_train{cocoapi_year}.json"
train_dataset = CocoDataset(train_image_dir, train_captions_file)

val_image_dir = f"../data/cocoapi/images/val{cocoapi_year}"
val_captions_file = f"../data/cocoapi/annotations/captions_val{cocoapi_year}.json"
val_dataset = CocoDataset(val_image_dir, val_captions_file)

# Data loaders for training and validation
batch_size = 32
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Create an instance of the Image Captioning model
caption_generator = CaptionGenerator()

# Freeze the parameters of the CNN backbone
for param in caption_generator.vgg.parameters():
    param.requires_grad = False

# Set up the optimizer and loss function
learning_rate = 0.001
optimizer = optim.Adam(caption_generator.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
num_epochs = 1000

# We set train mode and load the GPT2 tokenizer
caption_generator.train()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Training loop
for epoch in range(num_epochs):
    for images, captions in train_data_loader:
        optimizer.zero_grad()
        # Tokenizing the captions of the batch
        caption_tokens = [tokenizer(caption, return_tensors="pt").input_ids for caption in captions ]
        outputs = caption_generator(images, caption_tokens)

        batch_loss = 0
        # For each caption we compute the loss
        for caption_input, caption_output in zip(caption_tokens, outputs):
            caption_input = caption_input.type(torch.float)
            caption_input.requires_grad_()
            caption_output = caption_output.type(torch.float)
            caption_output.requires_grad_()
            
            # Determine the difference in dimensions and add padding to compute the loss properly
            dim_diff = abs(caption_input.shape[1] - caption_output.shape[1])
            if (caption_input.shape[1] < caption_output.shape[1]):
                caption_input = nn.functional.pad(caption_input, (0, dim_diff))
            elif (caption_input.shape[1] > caption_output.shape[1]):
                caption_output = nn.functional.pad(caption_output, (0, dim_diff))

            loss = criterion(caption_output, caption_input)
            batch_loss += loss

            loss.backward()
            optimizer.step()
        print(f"Batch loss : {batch_loss/len(captions)}")