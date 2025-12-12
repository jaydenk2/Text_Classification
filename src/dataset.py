import torch
import pandas as pd
from torch.utils.data import Dataset

class QuoraDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        """
        Args:
            csv_file (str): Path to the TSV file.
            tokenizer: Any tokenizer (e.g., BERT tokenizer or a simple dictionary).
            max_length (int): Max sequence length after concatenation.
        """
        self.data = pd.read_csv(csv_file)
        self.data = self.data.dropna(subset=['question1', 'question2'])
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        q1_text = str(row['question1'])
        q2_text = str(row['question2'])
        label = int(row['is_duplicate'])
        encoded_q1 = self.tokenizer(
            q1_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoded_q2 = self.tokenizer(
            q2_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'q1': encoded_q1['input_ids'].flatten(),
            'q2': encoded_q2['input_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }