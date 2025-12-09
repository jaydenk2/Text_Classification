import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Import your custom modules
from src.dataset import QuoraDataset 
from src.model import SiameseLSTM, BertClassifier
from src.train import train_model
from src.utils import plot_training_history  # <--- ADDED THIS IMPORT
import logging
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# --- CONFIGURATION ---
CONFIG = {
    'seed': 42,
    # CHECK THIS PATH: Your uploaded file was named "quora_duplicate_questions (1).tsv"
    # You must either rename the file or update this string below.
    'data_path': 'data/quora_duplicate_questions (1).tsv', 
    'batch_size': 64,
    'epochs': 5,
    'learning_rate': 0.001,
    'max_length': 64,
    'test_size': 0.2,
    'model_type': 'bert',    
    'dataset_mode': 'concat' 
}

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 1. PREPARE DATA
    # ---------------------------------------------------------
    print("Loading and splitting data...")
    try:
        df = pd.read_csv(CONFIG['data_path'], sep='\t', on_bad_lines='skip')
    except FileNotFoundError:
        print(f"ERROR: Could not find file at {CONFIG['data_path']}")
        print("Please rename your file or update 'data_path' in CONFIG.")
        return

    df = df.sample(frac=0.1, random_state=CONFIG['seed']) 
    print(f"Subsampled dataset to {len(df)} rows for faster CPU training.")
    train_df, val_df = train_test_split(
        df, 
        test_size=CONFIG['test_size'], 
        stratify=df['is_duplicate'], 
        random_state=CONFIG['seed']
    )
    
    train_df.to_csv('data/train_split.csv', index=False)
    val_df.to_csv('data/val_split.csv', index=False)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # ---------------------------------------------------------
    # 2. DATASETS & DATALOADERS
    # ---------------------------------------------------------
    print("Initializing Tokenizer and Datasets...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = QuoraDataset(
        csv_file='data/train_split.csv',
        tokenizer=tokenizer,
        max_length=CONFIG['max_length'],
        mode=CONFIG['dataset_mode']
    )
    
    val_dataset = QuoraDataset(
        csv_file='data/val_split.csv',
        tokenizer=tokenizer,
        max_length=CONFIG['max_length'],
        mode=CONFIG['dataset_mode']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # ---------------------------------------------------------
    # 3. INITIALIZE MODEL
    # ---------------------------------------------------------
    print(f"Initializing {CONFIG['model_type'].upper()} model...")
    
    if CONFIG['model_type'] == 'lstm':
        model = SiameseLSTM(
            vocab_size=tokenizer.vocab_size, 
            embedding_dim=128, 
            hidden_dim=256
        )
    elif CONFIG['model_type'] == 'bert':
        model = BertClassifier()
    else:
        raise ValueError(f"Unknown model type: {CONFIG['model_type']}")

    # ---------------------------------------------------------
    # 4. TRAIN
    # ---------------------------------------------------------
    print("Starting training...")
    history = train_model(
        model=model,
        train_iterator=train_loader,
        valid_iterator=val_loader,
        epochs=CONFIG['epochs'],
        lr=CONFIG['learning_rate'],
        save_path=f"best_model_{CONFIG['model_type']}.pt"
    )
    
    print("Training complete!")
    
    # ---------------------------------------------------------
    # 5. PLOT RESULTS
    # ---------------------------------------------------------
    plot_training_history(history, model_name=CONFIG['model_type'])

if __name__ == '__main__':
    main()