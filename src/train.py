import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def calculate_accuracy(preds, y):
    """
    Accuracy for 2-class output (Logits -> Argmax vs Label)
    """
    # preds shape: [batch_size, 2]
    # y shape: [batch_size]
    
    # Get the index of the max logit (0 or 1)
    top_pred = preds.argmax(1, keepdim=True)
    
    # Check if it matches the label
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    
    # Calculate percentage
    acc = correct.float() / y.shape[0]
    return acc

def train_one_epoch(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in tqdm(iterator, desc="Training", leave=False):
        # 1. Unpack batch
        # Extract 'input_ids' specifically for CNN/Transformer
        inputs = batch['input_ids'].to(device)
        labels = batch['label'].to(device).long() # Labels must be Long/Int for CrossEntropy
        
        # 2. Zero Grads
        optimizer.zero_grad()
        
        # 3. Forward (Pass only the inputs, not the dict)
        predictions = model(inputs)
        
        # 4. Loss & Backward
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        # 5. Metrics
        acc = calculate_accuracy(predictions, labels)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating", leave=False):
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device).long()

            predictions = model(inputs)
            
            loss = criterion(predictions, labels)
            acc = calculate_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train_model(model, train_iterator, valid_iterator, epochs=5, lr=0.001, save_path='best_model.pt', device=None):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # CHANGED: Use CrossEntropyLoss for num_classes=2
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'valid_loss': [], 'valid_acc': []
    }
    
    best_valid_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f"\t--> New Best Model Saved! (Val Loss: {valid_loss:.3f})")
            
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
        
    return history