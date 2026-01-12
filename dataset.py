import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class BilungialDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_target, 
                src_lang, target_lang, seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_target = tokenizer_target
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_lang]
        target_text = src_target_pair['translation'][self.target_lang]
        