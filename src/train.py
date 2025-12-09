import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # For progress bars
import numpy as np

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8
    """
    # Round predictions to the closest integer (0 or 1)
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

def train_one_epoch(model, iterator, optimizer, criterion, device):
    """
    Runs one epoch of training.
    """
    epoch_loss = 0
    epoch_acc = 0
    
    model.train() # Set model to training mode (enables Dropout, BatchNorm)
    
    # tqdm creates a nice progress bar in your terminal
    for batch in tqdm(iterator, desc="Training", leave=False):
        
        # 1. Move all batch data to the GPU/CPU
        # Your dataset returns a dictionary, so we move each value to device
        batch_data = {k: v.to(device) for k, v in batch.items()}
        
        # Get labels specifically
        labels = batch_data['label']
        
        # 2. Zero gradients
        optimizer.zero_grad()
        
        # 3. Forward Pass
        # We pass the WHOLE batch_data dict to the model
        # The model handles extracting what it needs (q1_input_ids, etc.)
        predictions = model(batch_data).squeeze(1)
        
        # 4. Compute Loss
        loss = criterion(predictions, labels)
        
        # 5. Backward Pass
        loss.backward()
        
        # 6. Optimizer Step
        optimizer.step()
        
        # Track metrics
        acc = binary_accuracy(predictions, labels)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    """
    Evaluates the model on validation data.
    """
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval() # Set model to eval mode (disables Dropout)
    
    with torch.no_grad(): # Do not calculate gradients (saves memory)
        for batch in tqdm(iterator, desc="Evaluating", leave=False):
            
            # Move data to device
            batch_data = {k: v.to(device) for k, v in batch.items()}
            labels = batch_data['label']

            # Forward pass
            predictions = model(batch_data).squeeze(1)
            
            # Compute loss/acc
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train_model(model, train_iterator, valid_iterator, epochs=5, lr=0.001, save_path='best_model.pt'):
    """
    Main training loop that orchestrates epochs.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    
    # BCEWithLogitsLoss is standard for Binary Classification
    # It combines a Sigmoid layer and the BCELoss in one single class
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_valid_loss = float('inf')
    
    history = {
        'train_loss': [], 'train_acc': [],
        'valid_loss': [], 'valid_acc': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f"\t--> New Best Model Saved! (Val Loss: {valid_loss:.3f})")
            
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
        
    return history