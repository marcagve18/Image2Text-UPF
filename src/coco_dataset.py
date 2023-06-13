import nltk
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils import data as data
from tqdm import tqdm
import random

from vocabulary import Vocabulary


class CoCoDataset(data.Dataset):
    def __init__(
        self,
        img_folder,
        annotations_file,
        batch_size,
        vocab_threshold,
        vocab_file,
        start_word,
        end_word,
        unk_word,
        vocab_from_file,
        ratio=0.1,
    ):
        self.img_folder = img_folder
        self.batch_size = batch_size
        # create vocabulary from the captions
        self.vocab = Vocabulary(
            vocab_threshold,
            vocab_file,
            start_word,
            end_word,
            unk_word,
            annotations_file,
            vocab_from_file,
        )
        
        self.coco = COCO(annotations_file)
        num_images = len(self.coco.imgs)
        num_captions = len(self.coco.anns)
        print(f"Total number of images: {num_images}")
        print(f"Total number of captions: {num_captions}")
        
        # Reduce the dataset to the desired percentage
        imgIds = random.sample(self.coco.getImgIds(), int(num_images * ratio))
        self.ids = self.coco.getAnnIds(imgIds=imgIds)
        self.num_images = len(imgIds)
        print(f"Reduced number of images: {self.num_images}")
        print(f"Reduced number of captions: {len(self.ids)}")

        # Get list of tokens for each caption
        print("Obtaining caption lengths...")
        tokenized_captions = [
            nltk.tokenize.word_tokenize(
                str(self.coco.anns[self.ids[index]]["caption"]).lower()
            )
            for index in tqdm(np.arange(len(self.ids)))
        ]

        # get len of each caption
        self.caption_lengths = [len(token) for token in tokenized_captions]

        # Pad all captions with <end> token so that all of them have the same length
        self.max_length = max(self.caption_lengths)
      

    def __getitem__(self, index):
        # obtain image and caption if in training mode
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]["caption"]
        img_id = self.coco.anns[ann_id]["image_id"]
        
        ############ LOAD IMAGES FROM FILES ############
        # NOTE: We have decided to first convert the images to tensors to save
        # computation time in the training phase. Therefore, we load the
        # tensors from a separate folder
        path = self.coco.loadImgs(img_id)[0]["file_name"] + ".pt"
        image = torch.load(self.img_folder+path)

        
        ############ LOAD IMAGES FROM URL ############
        # NOTE: this should probably be on if executing in google colab. Take into consideration that you will need to transofrm the images appropriately
        # path = self.coco.loadImgs(img_id)[0]
        # url = path["coco_url"]
        # # print(f"ID {img_id}, URL: {url}")
        # image = Image.fromarray(io.imread(url)).convert("RGB")

        # Convert caption to tensor of word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [self.vocab(self.vocab.start_word)]
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        caption = torch.Tensor(caption).long()

        # return pre-processed image, caption tensors and image path for testing the model
        image_filename = path.split('.pt')[0]
        return image, caption, image_filename
  
    
    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        
        # find indices of captions having specific length
        all_indices = np.where(
            [
                self.caption_lengths[i] == sel_length
                for i in np.arange(len(self.caption_lengths))
            ]
        )[0]
        # select only limited (batch size) number of them
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices


    def __len__(self):
        return len(self.ids)

