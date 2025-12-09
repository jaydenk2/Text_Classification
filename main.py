import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# --- IMPORTS FROM YOUR SRC ---
from src.dataset import QuoraDataset 
from src.train import train_model
from src.utils import plot_training_history

# --- IMPORTS FOR YOUR NEW MODELS ---
# Ensure your folder is named "Models" (Capital M)
from Models.CNN import TextCNN
from Models.Transformer import TransformerClassifier

import logging
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'seed': 42,
    'data_path': 'data/quora_duplicate_questions (1).tsv', 
    'batch_size': 64,
    'epochs': 5,
    'learning_rate': 0.001,
    'max_length': 64,
    'test_size': 0.2,
    'model_type': 'cnn',   # Default, will be overwritten by args
    'dataset_mode': 'concat'
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    # ---------------------------------------------------------
    # 0. PARSE ARGUMENTS
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train Quora Duplicate Question Model")
    
    # Add arguments
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'transformer', 'lstm', 'bert'],
                        help='Choose model architecture: cnn, transformer, lstm, or bert')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Overwrite CONFIG with command line arguments
    CONFIG['model_type'] = args.model
    CONFIG['epochs'] = args.epochs
    
    print(f"--- Configuration ---")
    print(f"Model: {CONFIG['model_type']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"---------------------")

    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 1. PREPARE DATA
    # ---------------------------------------------------------
    print("Loading and splitting data...")
    try:
        # Try reading with tabs first (standard Quora format)
        df = pd.read_csv(CONFIG['data_path'], sep='\t', on_bad_lines='skip')
    except FileNotFoundError:
        # Fallback to standard CSV if user converted it
        try:
            df = pd.read_csv(CONFIG['data_path'])
        except:
            print(f"ERROR: Could not find file at {CONFIG['data_path']}")
            return

    # --- CRITICAL FIX: CLEAN DATA ---
    # Drop rows where 'is_duplicate' is NaN or missing
    df = df.dropna(subset=['is_duplicate'])
    # Convert labels to integers (handles cases where they might be floats like 1.0)
    df['is_duplicate'] = df['is_duplicate'].astype(int)
    # --------------------------------

    # Optional: Subsample for speed during debugging
    # df = df.sample(frac=0.1, random_state=CONFIG['seed']) 
    
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
    # We use BERT tokenizer for everything to get consistent word-to-id mapping
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    train_dataset = QuoraDataset(
        csv_file='data/train_split.csv',
        tokenizer=tokenizer,
        max_length=CONFIG['max_length']
    )
    
    val_dataset = QuoraDataset(
        csv_file='data/val_split.csv',
        tokenizer=tokenizer,
        max_length=CONFIG['max_length']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # ---------------------------------------------------------
    # 3. INITIALIZE MODEL
    # ---------------------------------------------------------
    print(f"Initializing {CONFIG['model_type'].upper()} model...")
    vocab_size = tokenizer.vocab_size
    num_classes = 2 # 0 = Not Duplicate, 1 = Duplicate
    
    if CONFIG['model_type'] == 'cnn':
        model = TextCNN(
            vocab_size=vocab_size, 
            embed_dim=128, 
            num_classes=num_classes
        )
    elif CONFIG['model_type'] == 'transformer':
        model = TransformerClassifier(
            num_embeddings=vocab_size, 
            embedding_dim=128, 
            hidden_dim=256, 
            num_classes=num_classes,
            n_heads=4
        )
    else:
        # If user asks for bert/lstm but you haven't implemented them yet
        raise ValueError(f"Model type '{CONFIG['model_type']}' is not set up in main.py yet.")

    model.to(device)

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
        save_path=f"best_model_{CONFIG['model_type']}.pt",
        device=device
    )
    
    print("Training complete!")
    
    # ---------------------------------------------------------
    # 5. PLOT RESULTS
    # ---------------------------------------------------------
    plot_training_history(history, model_name=CONFIG['model_type'])

if __name__ == '__main__':
    main()