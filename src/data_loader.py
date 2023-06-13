import os
from tokenizer import Tokenizer
import nltk
import torch.utils.data as data

from coco_dataset import CoCoDataset

nltk.download("punkt")


def get_loader(
    image_folder,
    annotations_file,
    tokenizer: Tokenizer,
    batch_size=1,
    ratio=0.1,
): 
    
    # COCO caption dataset.
    dataset = CoCoDataset(
        batch_size=batch_size,
        img_folder=image_folder,
        annotations_file=annotations_file,
        tokenizer=tokenizer,
        ratio=ratio,
    )

    # Randomly sample a caption length, and sample indices with that length.
    indices = dataset.get_train_indices()
    # Create and assign a batch sampler to retrieve a batch with the sampled indices.
    initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    # data loader for COCO dataset.
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_sampler=data.sampler.BatchSampler(
            sampler=initial_sampler, batch_size=dataset.batch_size, drop_last=False
        ),
        pin_memory=True,
    )

    return data_loader
