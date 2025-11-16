import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt', quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Data processing class for the T5 model.
        
        Uses 'google-t5/t5-small' tokenizer for both encoder and decoder.
        Uses PAD token (ID=0) as the decoder start token.
        Behavior is different for the test set (no SQL targets).
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        '''
        Process the data for T5 model.
        
        For train/dev:
            - Load NL queries and SQL queries
            - Tokenize both
            - Create decoder inputs (shifted right) and targets
        
        For test:
            - Load only NL queries
            - Tokenize encoder inputs only
        '''
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_lines = load_lines(nl_path)

        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql_lines = load_lines(sql_path)
            assert len(nl_lines) == len(sql_lines), f"Mismatch: {len(nl_lines)} NL vs {len(sql_lines)} SQL"
        else:
            sql_lines = None

        pad_id = tokenizer.pad_token_id  # T5 uses pad_token_id = 0

        examples = []
        for i, nl in enumerate(nl_lines):
            # Encoder input: natural language query
            enc_ids = tokenizer.encode(nl, add_special_tokens=True)
            enc_tensor = torch.tensor(enc_ids, dtype=torch.long)

            if split != "test":
                sql = sql_lines[i]
                
                # Tokenize SQL with EOS
                sql_ids = tokenizer.encode(sql, add_special_tokens=True)
                
                # Standard T5 shifting:
                # decoder_input: [PAD, tok1, tok2, ..., tokN-1]
                # decoder_target: [tok1, tok2, ..., tokN, EOS]
                dec_in_ids = [pad_id] + sql_ids[:-1]
                dec_tgt_ids = sql_ids

                example = {
                    "encoder_ids": enc_tensor,
                    "decoder_input_ids": torch.tensor(dec_in_ids, dtype=torch.long),
                    "decoder_target_ids": torch.tensor(dec_tgt_ids, dtype=torch.long),
                    "initial_decoder_input": torch.tensor([pad_id], dtype=torch.long),
                }
            else:
                # Test set: no targets
                example = {
                    "encoder_ids": enc_tensor,
                    "initial_decoder_input": torch.tensor([pad_id], dtype=torch.long),
                }

            examples.append(example)

        return examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def normal_collate_fn(batch):
    '''
    Collation function for train/dev with dynamic padding.
    
    Returns:
        encoder_ids: BxT - input to encoder
        encoder_mask: BxT - attention mask for encoder
        decoder_inputs: BxT' - input to decoder (shifted right)
        decoder_targets: BxT' - target tokens for decoder
        initial_decoder_inputs: Bx1 - initial decoder token (for generation)
    '''
    # Encoder side
    encoder_seqs = [ex["encoder_ids"] for ex in batch]
    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # Decoder side
    decoder_in_seqs = [ex["decoder_input_ids"] for ex in batch]
    decoder_tgt_seqs = [ex["decoder_target_ids"] for ex in batch]

    decoder_inputs = pad_sequence(decoder_in_seqs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_tgt_seqs, batch_first=True, padding_value=PAD_IDX)

    # Initial decoder input (BOS token)
    init_dec = torch.stack([ex["initial_decoder_input"] for ex in batch])

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, init_dec


def test_collate_fn(batch):
    '''
    Collation function for test set (no decoder targets).
    
    Returns:
        encoder_ids: BxT
        encoder_mask: BxT
        initial_decoder_inputs: Bx1
    '''
    encoder_seqs = [ex["encoder_ids"] for ex in batch]
    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    init_dec = torch.stack([ex["initial_decoder_input"] for ex in batch])

    return encoder_ids, encoder_mask, init_dec


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    '''
    Load data for LLM prompting experiments.
    
    Returns:
        train_x: list of train NL queries
        train_y: list of train SQL
        dev_x: list of dev NL queries
        dev_y: list of dev SQL
        test_x: list of test NL queries
    '''
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    
    return train_x, train_y, dev_x, dev_y, test_x
