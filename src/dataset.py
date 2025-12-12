import torch
import pandas as pd
import re
from collections import Counter
from torch.utils.data import Dataset

class SimpleTokenizer:
    def __init__(self, min_freq=2, pad_token="<PAD>", unk_token="<UNK>"):
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'([.,!?()"\'-])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()

    def build_vocab(self, texts):
        print(f"Building vocabulary from {len(texts)} sentences...")
        counter = Counter()
        for text in texts:
            tokens = self.clean_text(text)
            counter.update(tokens)
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        idx = 2
        ignored_count = 0
        for word, count in counter.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1
            else:
                ignored_count += 1
                
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary built: {self.vocab_size} unique tokens.")
        print(f"Ignored {ignored_count} rare words (freq < {self.min_freq}).")

    def encode(self, text, max_length):
        tokens = self.clean_text(text)
        token_ids = []
        
        for token in tokens:
            if token in self.word2idx:
                token_ids.append(self.word2idx[token])
            else:
                token_ids.append(self.word2idx[self.unk_token])
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        padding_length = max_length - len(token_ids)
        token_ids = token_ids + [self.word2idx[self.pad_token]] * padding_length
        
        return torch.tensor(token_ids, dtype=torch.long)

class QuoraDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=64):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        q1_text = str(row['question1'])
        q2_text = str(row['question2'])
        label = int(row['is_duplicate'])
        q1_ids = self.tokenizer.encode(q1_text, self.max_length)
        q2_ids = self.tokenizer.encode(q2_text, self.max_length)
        
        return {
            'q1': q1_ids,
            'q2': q2_ids,
            'label': torch.tensor(label, dtype=torch.float)
        }