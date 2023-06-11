from torchvision import transforms
import torch
from data_loader import get_loader
from PIL import Image
import matplotlib.pyplot as plt
import MyTorchWrapper as mtw
from  cnn_rnn import ImageCaptioner
import torch.utils.data as data


cocoapi_year = "2017"
device = mtw.get_torch_device(use_gpu=True, debug=True)

# Creating the data loader.
data_loader = get_loader(
    image_folder= f"../clean_data/val{cocoapi_year}/",
    annotations_file=f"../data/cocoapi/annotations/captions_val{cocoapi_year}.json",
    batch_size=1,
    vocab_threshold=5, # minimum word count threshold
    vocab_from_file=True, # if True, load existing vocab file
    ratio=1
)

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

embedding_size = 256
hidden_size = 512
vocab_size = len(data_loader.dataset.vocab)

image_captioner = ImageCaptioner(embedding_size, hidden_size, vocab_size)
image_captioner.eval()

image_captioner.to(device)
image_captioner.CNN.load_state_dict(torch.load("../models/encoder-10.pkl", map_location=torch.device(device)))
image_captioner.RNN.load_state_dict(torch.load("../models/decoder-10.pkl", map_location=torch.device(device)))


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
    print(f"ORIGINAL CAPTION: {caption}")

    original_image = Image.open(original_image_folder + filename[0])
    plt.imshow(original_image)
    plt.show()

    image = image.to(device)
    features = image_captioner.CNN(image).unsqueeze(1)
    output = image_captioner.RNN.sample(features)
    predicted_caption = clean_sentence(output, data_loader.dataset.vocab.idx2word)
    print(f"PREDICTED CAPTION {predicted_caption}")

     