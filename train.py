import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import BilungialDataset, casual_mask

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
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_target"]}', 
                            split="train")
    
    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_target = get_or_build_tokenizer(config, ds_raw, config['lang_target'])

    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilungialDataset(train_ds_raw, tokenizer_src, tokenizer_target, 
                                config['lang_src'], config['lang_target'], config['seq_len'])
    val_ds = BilungialDataset(val_ds_raw, tokenizer_src, tokenizer_target, 
                                config['lang_src'], config['lang_target'], config['seq_len'])
    
    max_len_src = 0
    max_len_target = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        target_ids = tokenizer_src.decode(item['translation'][config['lang_target']]).ids
        max_len_src = max(max_len_src, src_ids)
        max_len_target = max(max_len_target, target_ids)
    
    print(f"Max length of source sequence: {max_len_src}")
    print(f"Max length of target sequence: {max_len_target}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_target