import json
from torch.utils import data as data
from vocabulary import Vocabulary
from PIL import Image
import torch
from torchvision import transforms
import skimage.io as io
import MyTorchWrapper as mtw
from tqdm import tqdm
import os
from tokenizer import NltkTokenizer

class NoCaps(data.Dataset):
    def __init__(
        self,
        data_path,
        annotations_file_for_vocab,
        vocab_threshold,
        vocab_file,
        start_word,
        end_word,
        unk_word,
        vocab_from_file,
        transformations,
        load_from_internet=True
    ):
        
        self.data_path = data_path
        # create vocabulary from the captions
        self.vocab = Vocabulary(
            vocab_threshold,
            vocab_file,
            start_word,
            end_word,
            unk_word,
            annotations_file_for_vocab,
            vocab_from_file,
        )

        self.load_from_internet = load_from_internet
        
        self.transformations = transformations
        
        self.json = None

        if load_from_internet:
            with open(self.data_path, "r") as json_file:
                self.json = json.load(json_file)
                
            self.images = self.json['images']
            self.num_images = len(self.images)
        else:
            self.num_images = len(os.listdir(data_path))
        
        print(f"Total number of images: {self.num_images}")
        print("Loading images from internet:", self.load_from_internet)


    
      

    def __getitem__(self, index):

        ########### LOAD IMAGES FROM URL ############
        # print(f"ID {img_id}, URL: {url}")
        if self.load_from_internet:
            url = self.images[index]["coco_url"]
            image = Image.fromarray(io.imread(url)).convert("RGB")
            image = self.transformations(image)
        else:
            image_path = f"{self.data_path}/{index}.pt"
            image = torch.load(image_path)

        # return pre-processed image
        return image
  
    
    def __len__(self):
        return self.num_images



def clean_sentence(output, idx2word):
    sentence = ""
    for i in output:
        word = idx2word[i]
        if i == 0:
            continue
        if i == 1:
            break
        if i == 18:
            sentence = sentence + word
        else:
            sentence = sentence + " " + word
    return sentence



if __name__ == '__main__':
    from architectures import *
    from typing import List


    device = mtw.get_torch_device(use_gpu=True, debug=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), 
            (0.229, 0.224, 0.225),
        ),
    ])

    cocoapi_year = 2017
    vocab_file="./vocab.pkl",
    start_word="<start>",
    end_word="<end>",
    unk_word="<unk>",
    vocab_from_file=True
    vocab_threshold = 5
    
    
    nocaps = NoCaps(
        data_path="../clean_data/nocaps",
        vocab_threshold=vocab_threshold,
        vocab_file="./vocab.pkl",
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        annotations_file_for_vocab=f"../data/cocoapi/annotations/captions_train{cocoapi_year}.json",
        vocab_from_file=True,
        transformations=transform,
        load_from_internet=False
        )


    models: List[ImageCaptioner] = []
    vocab_size = len(nocaps.vocab)  # The size of the vocabulary

    models.append(R50_LSTM(
        embed_size=256, # dimensionality of image and word embeddings
        hidden_size=512, # number of features in hidden state of the RNN decoder
        lstm_layers=1, # Number of hidden layers of each lstm cell
        vocab_size=vocab_size,
        bidirectional_lstm=False,
    ))
 
    models.append(EB7_LSTM(
        embed_size=256, # dimensionality of image and word embeddings
        hidden_size=512, # number of features in hidden state of the RNN decoder
        lstm_layers= 3, # Number of hidden layers of each lstm cell
        vocab_size=vocab_size,
        bidirectional_lstm=False,
    ))
    models.append(EB7_LSTM(
        embed_size=256,  # dimensionality of image and word embeddings
        hidden_size=512,  # number of features in hidden state of the RNN decoder
        lstm_layers=3, # Number of hidden layers of each lstm cell
        vocab_size=vocab_size,
        bidirectional_lstm=True,
    ))
    models.append(ViT_LSTM(
        embed_size=256, # dimensionality of image and word embeddings
        hidden_size=512, # number of features in hidden state of the RNN decoder
        lstm_layers=3, # Number of hidden layers of each lstm cell
        vocab_size=vocab_size,
        bidirectional_lstm=False,
    ))
    

    data_loader = data.DataLoader(dataset=nocaps, batch_size=1, pin_memory=True)
    num_epochs = 4
    predictions_folder = "../predictions/"

    for image_captioner in models:
        if not os.path.exists(predictions_folder + image_captioner.name):
            os.makedirs(predictions_folder + image_captioner.name)

        for epoch in range(1, num_epochs + 1):
            image_captioner.eval()

            image_captioner.to(device)
            image_captioner.CNN.load_state_dict(torch.load(f"../models/{image_captioner.name}/encoder-epoch{epoch}.pkl", map_location=device))
            image_captioner.RNN.load_state_dict(torch.load(f"../models/{image_captioner.name}/decoder-epoch{epoch}.pkl", map_location=device))
            
            # Run inference and store the result
            predictions = []
            info_every = 5
            print(f"Len data loader {len(data_loader)}")
            
            for i, image in tqdm(enumerate(data_loader)):
                image = image.squeeze(0)
                image = image.to(device)
                features = image_captioner.CNN(image).unsqueeze(1)
                output = image_captioner.RNN.sample(features)
                predicted_caption = clean_sentence(output, data_loader.dataset.vocab.idx2word)
                # print(f"PREDICTED CAPTION: {predicted_caption}")
                predictions.append({"image_id": i, "caption": predicted_caption})
            
            # # Print first 25 captions with their Image ID.
            # for k in range(25):
            #     print(predictions[k]["image_id"], predictions[k]["caption"])

            
            # Save the predictions in a JSON file
            json.dump(predictions, open(f"{predictions_folder}{image_captioner.name}/predictions-epoch{epoch}.json", "w"))
                                                


