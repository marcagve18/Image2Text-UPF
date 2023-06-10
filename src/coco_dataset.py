import json
import os

import nltk
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils import data as data
from tqdm import tqdm
import skimage.io as io

from vocabulary import Vocabulary


class CoCoDataset(data.Dataset):
    def __init__(
        self,
        batch_size,
        vocab_threshold,
        vocab_file,
        start_word,
        end_word,
        unk_word,
        annotations_file,
        vocab_from_file,
        img_folder,
        ratio=0.1,
    ):
        self.batch_size = batch_size
        self.img_folder = img_folder
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
        self.ids = list(self.coco.anns.keys())
        print(f"Number of images: {len(self.ids)}")
        images_amount = int(len(self.ids) * ratio)
        self.ids = self.ids[0:images_amount]
        print("Obtaining caption lengths...")

        #  get list of tokens for each caption
        tokenized_captions = [
            nltk.tokenize.word_tokenize(
                str(self.coco.anns[self.ids[index]]["caption"]).lower()
            )
            for index in tqdm(np.arange(len(self.ids)))
        ]

        # get len of each caption
        self.caption_lengths = [len(token) for token in tokenized_captions]
      

    def __getitem__(self, index):
        # obtain image and caption if in training mode
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]["caption"]
        img_id = self.coco.anns[ann_id]["image_id"]
        
        ############ LOAD IMAGES FROM FILES ############
        path = self.coco.loadImgs(img_id)[0]["file_name"] + ".pt"
        image = torch.load(self.img_folder+path) # (os.path.join(self.img_folder, path)).convert("RGB") # FIXME: Pass folder as parameter

        
        ############ LOAD IMAGES FROM URL ############ NOTE: this should probably be on if executing in google colab.
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

        image_filename = path.split('.pt')[0]
        
        # return pre-processed image, caption tensors and image path for testing the model
        return image, caption, image_filename
  
    
    def get_train_indices(self):
        # select random len
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

