import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_or_build_tokenizer(config, dataset, lang):
    # Same as {'tokenizer_file': f"tokenizer_file_{lang}.json"}
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Maps word it hasnt seen to unknown
        tokenizer = Tokenizer(WordLevel(unk_token=['UNK'])) 
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])