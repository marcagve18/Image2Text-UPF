from abc import ABC, abstractmethod
import nltk
import torch
from vocabulary import Vocabulary
import os
from transformers import GPT2Tokenizer

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, sentence: str) -> torch.Tensor:
        """Returns a sentence tokenized. Also adds a `beginning of sentence`
        token, an `end of sentence` token and `unknown` tokens if a word is not
        found in the vocabulary.
        """
    
    @abstractmethod
    def clean_sentence(self, token_sequence: torch.Tensor) -> str:
        pass

    @abstractmethod
    def vocab_size(self) -> int:
        pass


class Gpt2Tokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({
            'eos_token': '<EOS>',
            'bos_token': '<BOS>',
            'unk_token': '<UNK>',
            'pad_token': '<PAD>',
        })
    
    def tokenize(self, sentence: str) -> torch.Tensor:
        return self.tokenizer(sentence, return_tensors="pt").input_ids

    def clean_sentence(self, token_sequence : torch.Tensor) -> str:
        return self.tokenizer.decode(token_sequence, skip_special_tokens=True)
    
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size
    


class NltkTokenizer(Tokenizer):
    def __init__(self, 
            vocab_threshold,
            start_word = "<start>",
            end_word = "<end>",
            unk_word = "<unk>",
            vocab_file = None,
        ):
        if vocab_file:
            assert os.path.exists(vocab_file), "vocab_file does not exist"

        self.vocab = Vocabulary(
            vocab_threshold,
            vocab_file,
            start_word,
            end_word,
            unk_word,
            vocab_from_file=True if vocab_file else False,
        )

    def tokenize(self, sentence : str) -> torch.Tensor:
        tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
        caption = [self.vocab(self.vocab.start_word)]
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        caption = torch.Tensor(caption).long()
        return caption

    
    def clean_sentence(self, token_sequence: torch.Tensor) -> str:
        sentence = ""
        for i in token_sequence:
            word = self.vocab.idx2word[i]
            if i == 0:
                continue
            if i == 1:
                break
            if i == 18:
                sentence = sentence + word
            else:
                sentence = sentence + " " + word
        return sentence
    
    def vocab_size(self) -> int:
        return len(self.vocab)
        