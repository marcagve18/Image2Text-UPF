from metrics import NoCaps
import json
from torch.utils import data as data
from vocabulary import Vocabulary
from PIL import Image
import torch
from torchvision import transforms
import skimage.io as io
import MyTorchWrapper as mtw
from cnn_rnn import ImageCaptioner
from tqdm import tqdm

if __name__ == '__main__':
    device = mtw.get_torch_device(use_gpu=True, debug=True)

    model_name = "efficientnetB7_defaultRNN"
    model_epoch = "12"

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
        data_path="../data/nocaps/nocaps_val_image_info.json",
        vocab_threshold=vocab_threshold,
        vocab_file="./vocab.pkl",
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        annotations_file_for_vocab=f"../data/cocoapi/annotations/captions_train{cocoapi_year}.json",
        vocab_from_file=True,
        transformations=transform
        )
    
    

    data_loader = data.DataLoader(dataset=nocaps, batch_size=1, pin_memory=True)

    predictions = []
    info_every = 5
    print(f"Len data loader {len(data_loader)}")
    
    for i, image in tqdm(enumerate(data_loader)):
        image = image.to(device)
        torch.save(image, f"../data/nocaps/images/{i}.pt")
    

    
    
