import torch
import pandas as pd
from torch.utils.data import Dataset

class QuoraDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128, mode='separate'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        # 1. Load Data
        print(f"Loading {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # 2. Critical Preprocessing for Quora Dataset
        initial_len = len(df)
        # Drop rows where essential columns are missing
        df = df.dropna(subset=['question1', 'question2', 'is_duplicate'])
        if len(df) < initial_len:
            print(f"Dropped {initial_len - len(df)} rows with missing values.")
        
        # 3. Type Casting
        self.questions1 = df['question1'].astype(str).tolist()
        self.questions2 = df['question2'].astype(str).tolist()
        self.labels = df['is_duplicate'].astype(int).tolist()

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        q1 = self.questions1[idx]
        q2 = self.questions2[idx]
        label = self.labels[idx]

        if self.mode == 'concat':
            # --- TRANSFORMER MODE (Single Input) ---
            encoding = self.tokenizer.encode_plus(
                q1,
                q2,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding.get('token_type_ids', torch.tensor([])).flatten(), 
                'label': torch.tensor(label, dtype=torch.float)
            }

        elif self.mode == 'separate':
            # --- LSTM MODE (Dual Input) ---
            # tokenize Q1
            enc1 = self.tokenizer.encode_plus(
                q1,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            # tokenize Q2
            enc2 = self.tokenizer.encode_plus(
                q2,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'q1_input_ids': enc1['input_ids'].flatten(),
                'q1_attention_mask': enc1['attention_mask'].flatten(),
                'q2_input_ids': enc2['input_ids'].flatten(),
                'q2_attention_mask': enc2['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.float)
            }