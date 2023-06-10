from torchvision import transforms
import torch
from data_loader import get_loader
from PIL import Image
import matplotlib.pyplot as plt
import MyTorchWrapper as mtw
from  cnn_rnn import ImageCaptioner

print("hello")

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

image_captioner.CNN.load_state_dict(torch.load("../models/encoder-3.pkl"))
image_captioner.RNN.load_state_dict(torch.load("../models/decoder-3.pkl"))

image_captioner.to(device)

original_image_folder = f"../data/cocoapi/images/val{cocoapi_year}/"

images_to_load = 5
i = 0
print(len(data_loader))
for image, token_caption, filename in data_loader:
    token_caption = token_caption.tolist()[0]
    caption = clean_sentence(token_caption, data_loader.dataset.vocab.idx2word)
    print(f"ORIGINAL CAPTION: {caption}")

    original_image = Image.open(original_image_folder + filename[0])
    plt.imshow(original_image)
    plt.show()

    i += 1
    if i == images_to_load:
        break



     