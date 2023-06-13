import torch
from data_loader import get_loader
from PIL import Image
import matplotlib.pyplot as plt
import MyTorchWrapper as mtw
from  architectures import *
import torch.utils.data as data
from typing import List
from architectures.BaseImageCaptioner import ImageCaptioner


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
    from architectures import EB7_LSTM, R50_LSTM, ViT_LSTM
    
    device = mtw.get_torch_device(use_gpu=True, debug=True)
    
    # Build data loader.
    cocoapi_year = "2017"
    data_loader = get_loader(
        image_folder=f"../clean_data/val{cocoapi_year}/",
        annotations_file=f"../data/cocoapi/annotations/captions_val{cocoapi_year}.json",
        batch_size=1,
        vocab_threshold=5, # minimum word count threshold
        vocab_from_file=True, # if True, load existing vocab file
        ratio=0.01, # proportion of coco dataset to use
    )

    # Initializing image captioning models
    vocab_size = len(data_loader.dataset.vocab)  # The size of the vocabulary
    models: List[ImageCaptioner] = []

    models.append(R50_LSTM(
        embed_size=256, # dimensionality of image and word embeddings
        hidden_size=512, # number of features in hidden state of the RNN decoder
        lstm_layers=1, # Number of hidden layers of each lstm cell
        vocabulary_size=vocab_size,
        bidirectional_lstm=False,
    ))
    models.append(EB7_LSTM(
        embed_size=256, # dimensionality of image and word embeddings
        hidden_size=512, # number of features in hidden state of the RNN decoder
        lstm_layers= 3, # Number of hidden layers of each lstm cell
        vocabulary_size=vocab_size,
        bidirectional_lstm=False,
    ))
    models.append(EB7_LSTM(
        embed_size=256,  # dimensionality of image and word embeddings
        hidden_size=512,  # number of features in hidden state of the RNN decoder
        lstm_layers=3, # Number of hidden layers of each lstm cell
        vocabulary_size=vocab_size,
        bidirectional_lstm=True,
    ))
    models.append(ViT_LSTM(
        embed_size=256, # dimensionality of image and word embeddings
        hidden_size=512, # number of features in hidden state of the RNN decoder
        lstm_layers=3, # Number of hidden layers of each lstm cell
        vocabulary_size=vocab_size,
        bidirectional_lstm=False,
    ))
    models.append(ViT_LSTM(
        embed_size=256, # dimensionality of image and word embeddings
        hidden_size=256, # number of features in hidden state of the RNN decoder
        lstm_layers=5, # Number of hidden layers of each lstm cell
        vocabulary_size=vocab_size,
        bidirectional_lstm=False,
    ))

    # Load pre-trained weights
    for image_captioner in models:
        image_captioner.eval()
        image_captioner.to(device)
        image_captioner.CNN.load_state_dict(torch.load(f"../models/{image_captioner.name}/encoder-epoch3.pkl", map_location=torch.device(device)))
        image_captioner.RNN.load_state_dict(torch.load(f"../models/{image_captioner.name}/decoder-epoch3.pkl", map_location=torch.device(device)))


    # Check the captions for a small sample of images
    original_image_folder = f"../data/cocoapi/images/val{cocoapi_year}/"
    images_to_load = min(5, data_loader.dataset.num_images)
    for i in range(images_to_load):
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader.dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler

        # Obtain the batch.
        image, token_caption, filename = next(iter(data_loader))
        token_caption = token_caption.tolist()[0]
        caption = clean_sentence(token_caption, data_loader.dataset.vocab.idx2word)
        image = image.to(device)

        predicted_captions = {}
        for image_captioner in models:
            features = image_captioner.CNN(image).unsqueeze(1)
            output = image_captioner.RNN.sample(features)
            predicted_caption = clean_sentence(output, data_loader.dataset.vocab.idx2word)
            predicted_captions[image_captioner.name] = predicted_caption

        label = f"ORIGINAL CAPTION: {caption}\n"
        for model in predicted_captions.keys():
            label += f"{model.upper()}: {predicted_captions[model]}\n"
        
        
        original_image = Image.open(original_image_folder + filename[0])
        plt.imshow(original_image)
        plt.xlabel(label)
        plt.show()

     