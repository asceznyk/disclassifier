from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

import torch

from config import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, valid_loader=None):
    model.to(device)

    def run_epoch(split):
        is_train = split == 'train'
        loader = train_loader if is_train else valid_loader
        model.train(is_train)

        avg_loss = 0
        per = 0
        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, batch in pbar:  
            batch = [i.to(device) for i in batch]
            seq, mask, labels = batch
            
            with torch.set_grad_enabled(is_train):
                if is_train: model.zero_grad() 
                preds = model(seq, mask)
                loss = cost(preds, labels)
                avg_loss += loss.item() / len(loader)

            if is_train:
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step()

            pbar.set_description(f"epoch: {e+1}, loss: {loss.item():.3f}, avg: {avg_loss:.2f}") 
        
        return avg_loss

    best_loss = float('inf')
    cost = torch.nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for e in range(EPOCHS):
        train_loss = run_epoch('train')
        valid_loss = run_epoch('valid') if valid_loader is not None else train_loss

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'best.bert.classifier')

def calc_acc(model, loader):
    model.eval().to(device)
    labels, preds = [], []
    for (seq, mask, label) in loader:
        with torch.no_grad():
            pred = model(seq.to(device), mask.to(device)) 
            pred = pred.detach().cpu().numpy()

        preds.extend([i for i in np.argmax(pred, axis=1)])
        labels.extend([i for i in label.detach().cpu().numpy()])

    print(labels, preds)
    print(classification_report(np.array(labels), np.array(preds), zero_division=0))

