import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]
    

def get_or_build_tokenizer(config, dataset, lang):
    # Same as {'tokenizer_file': f"tokenizer_file_{lang}.json"}
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Maps word it hasnt seen to unknown
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) 
        # Defining split as whitespace
        tokenizer.pre_tokenizer = Whitespace()
        # defining a trainer to train with special tokens
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        # train the tokenizer
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else: 
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    dataset_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_target"]}', 
                            split="train")
    
    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_target = get_or_build_tokenizer(config, dataset_raw, config['lang_target'])

    train_ds_size = int(0.9*len(dataset_raw))
    val_ds_size = len(dataset_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(dataset_raw, [train_ds_size, val_ds_size])