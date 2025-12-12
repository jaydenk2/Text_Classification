import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(outputs, targets): # for f1 score, precision, recall
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', zero_division=0)
    acc = accuracy_score(targets, preds)
    return {
        'acc': acc,
        'prec': precision,
        'rec': recall,
        'f1': f1
    }
    
def train_one_epoch(model, iterator, optimizer, criterion, device): # training loop
    epoch_loss = 0
    model.train()
    all_preds = []
    all_labels = []
    for batch in tqdm(iterator, desc="Training", leave=False, mininterval=1.0, ncols=100):
        q1 = batch['q1'].to(device)
        q2 = batch['q2'].to(device)
        labels = batch['label'].to(device).long()
        optimizer.zero_grad()
        predictions = model(q1, q2)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        all_preds.append(predictions.detach())
        all_labels.append(labels.detach())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = epoch_loss / len(iterator)
    return metrics

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating", leave=False, mininterval=1.0, ncols=100):
            q1 = batch['q1'].to(device)
            q2 = batch['q2'].to(device)
            labels = batch['label'].to(device).long()
            predictions = model(q1, q2)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            all_preds.append(predictions)
            all_labels.append(labels)      
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = epoch_loss / len(iterator)
    return metrics


def train_model(model, train_iterator, valid_iterator, epochs=5, lr=0.001, save_path='best_model.pt', device=None): # main training function
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_valid_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch+1}/{epochs}")
        train_res = train_one_epoch(model, train_iterator, optimizer, criterion, device)
        valid_res = evaluate(model, valid_iterator, criterion, device)
        if valid_res['loss'] < best_valid_loss:
            best_valid_loss = valid_res['loss']
            torch.save(model.state_dict(), save_path)
            print(f"\t--> New Best Model Saved! (Val Loss: {valid_res['loss']:.4f})")
        # output metrics
        print(f"\tTrain Loss: {train_res['loss']:.3f} | Acc: {train_res['acc']:.3f} | F1: {train_res['f1']:.3f}")
        print(f"\t Val. Loss: {valid_res['loss']:.3f} | Acc: {valid_res['acc']:.3f} | F1: {valid_res['f1']:.3f}")
        print(f"\t Val. Prec: {valid_res['prec']:.3f} | Rec: {valid_res['rec']:.3f}")

    return "Training Complete"