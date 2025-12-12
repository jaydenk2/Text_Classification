import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.dataset import QuoraDataset, SimpleTokenizer
from src.train import train_model
from src.utils import plot_training_history
from models.CNN import TextCNN
from models.Transformer import TransformerClassifier

CONFIG = {
    'seed': 42,
    'data_path': 'data/quora_duplicate_questions (1).tsv', 
    'batch_size': 512,
    'epochs': 5,
    'learning_rate': 0.001,
    'max_length': 64,
    'min_freq': 2,
    'test_size': 0.2,
    'model_type': 'cnn'
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Train Quora Model Manually")
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'transformer'])
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    
    CONFIG['model_type'] = args.model
    CONFIG['epochs'] = args.epochs

    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    print("Loading data...")
    try:
        df = pd.read_csv(CONFIG['data_path'], sep='\t', on_bad_lines='skip')
    except FileNotFoundError:
        df = pd.read_csv('data/quora_duplicate_questions.csv')
    df = df.dropna(subset=['question1', 'question2', 'is_duplicate'])
    df['is_duplicate'] = df['is_duplicate'].astype(int)
    print(f"Total samples: {len(df)}")
    train_df, val_df = train_test_split(
        df, 
        test_size=CONFIG['test_size'], 
        stratify=df['is_duplicate'], 
        random_state=CONFIG['seed']
    )
    print("Initializing Manual Tokenizer...")
    tokenizer = SimpleTokenizer(min_freq=CONFIG['min_freq'])
    all_train_text = pd.concat([train_df['question1'], train_df['question2']]).tolist()
    tokenizer.build_vocab(all_train_text)
    print("Encoding datasets...")
    train_dataset = QuoraDataset(
        df=train_df,
        tokenizer=tokenizer,
        max_length=CONFIG['max_length']
    )
    
    val_dataset = QuoraDataset(
        df=val_df,
        tokenizer=tokenizer,
        max_length=CONFIG['max_length']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    print(f"Initializing {CONFIG['model_type'].upper()} model...")
    vocab_size = tokenizer.vocab_size
    num_classes = 2
    
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
            num_classes=num_classes,
            n_heads=4
        )

    model.to(device)
    history = train_model(
        model=model,
        train_iterator=train_loader,
        valid_iterator=val_loader,
        epochs=CONFIG['epochs'],
        lr=CONFIG['learning_rate'],
        save_path=f"best_model_{CONFIG['model_type']}.pt",
        device=device
    )
    
    plot_training_history(history, model_name=CONFIG['model_type'])
    print("Done!")

if __name__ == '__main__':
    main()