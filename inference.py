from pathlib import Path
import torch
import sys
from config import get_config, get_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer

def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_target, max_len, device):
    sos_idx = tokenizer_target.token_to_id('[SOS]')
    eos_idx = tokenizer_target.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.proj(out[:, -1])
        
        _, next_word = torch.max(prob, dim=1)
        
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def get_translation(sentence: str):
    config = get_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_target']))
    
    if not tokenizer_src_path.exists() or not tokenizer_tgt_path.exists():
        print(f"Error: Tokenizer files not found at {tokenizer_src_path}")
        return

    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_target = Tokenizer.from_file(str(tokenizer_tgt_path))

    model = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_target.get_vocab_size(), 
        config['seq_len'], 
        config['seq_len'], 
        config['d_model']
    ).to(device)

    model_filename = get_weights_file_path(config, "latest") 
    print(f"Loading weights from: {model_filename}")
    
    if not Path(model_filename).exists():
        print(f"Error: Weights file not found at {model_filename}")
        return

    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    print(f"Translating: '{sentence}'")
    with torch.no_grad():
        source = tokenizer_src.encode(sentence)
        
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (config['seq_len'] - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)

        source = source.unsqueeze(0)

        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        
        model_out = greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_target, config['seq_len'], device)
        
        model_out_text = tokenizer_target.decode(model_out.detach().cpu().numpy())
        
        return model_out_text

if __name__ == '__main__':
    if len(sys.argv) > 1:
        sentence = sys.argv[1]
        result = get_translation(sentence)
        print(f"\nPREDICTED: {result}")
    else:
        print("Enter a sentence to translate (or '_' to exit):")
        while True:
            text = input("> ")
            if text.lower() == '_': break
            try:
                result = get_translation(text)
                print(f"PREDICTED: {result}")
                print("-" * 30)
            except Exception as e:
                print(f"Error: {e}")