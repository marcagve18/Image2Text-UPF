import os

import nltk
import torch.utils.data as data

from coco_dataset import CoCoDataset

nltk.download("punkt")


def get_loader(
    image_folder,
    annotations_file,
    batch_size=1,
    vocab_threshold=None,
    vocab_file="./vocab.pkl",
    start_word="<start>",
    end_word="<end>",
    unk_word="<unk>",
    vocab_from_file=True,
    ratio=0.1,
):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary.
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch and override any existing vocab_file.
                       If True, load vocab from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading.
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    
    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    if vocab_from_file:
        assert os.path.exists(
            vocab_file
        ), "vocab_file does not exist. Change vocab_from_file to False to create vocab_file."


    # COCO caption dataset.
    dataset = CoCoDataset(
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        vocab_file=vocab_file,
        start_word=start_word,
        end_word=end_word,
        unk_word=unk_word,
        annotations_file=annotations_file,
        vocab_from_file=vocab_from_file,
        img_folder=image_folder,
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
