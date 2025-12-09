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
# Make sure your folder is named 'Models' and has an __init__.py
from models.CNN import TextCNN
from models.Transformer import TransformerClassifier
# Keep existing models if you want

import logging
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# --- CONFIGURATION ---
CONFIG = {
    'seed': 42,
    'data_path': 'data/quora_duplicate_questions (1).tsv', 
    'batch_size': 64,
    'epochs': 5,
    'learning_rate': 0.001,
    'max_length': 64, # CNN/Transformer handle longer sequences well, but 64 is faster
    'test_size': 0.2,
    
    # CHANGE THIS to 'cnn', 'transformer', 'bert', or 'lstm'
    'model_type': 'cnn',  
    
    # Use 'concat' for CNN/Transformer/BERT, 'separate' for Siamese LSTM
    'dataset_mode': 'concat' 
}

def set_seed(seed):
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
        return

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
    
    # If using CNN/Transformer (non-siamese), ensure dataset_mode is 'concat'
    # so the two questions are joined into one sequence.
    mode = CONFIG['dataset_mode']
    if CONFIG['model_type'] in ['cnn', 'transformer', 'bert']:
        mode = 'concat' # Force concat for these architectures
        
    train_dataset = QuoraDataset(
        csv_file='data/train_split.csv',
        tokenizer=tokenizer,
        max_length=CONFIG['max_length'],
    )
    
    val_dataset = QuoraDataset(
        csv_file='data/val_split.csv',
        tokenizer=tokenizer,
        max_length=CONFIG['max_length'],
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
    elif CONFIG['model_type'] == 'lstm':
        model = SiameseLSTM(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256)
    elif CONFIG['model_type'] == 'bert':
        model = BertClassifier()
    else:
        raise ValueError(f"Unknown model type: {CONFIG['model_type']}")

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