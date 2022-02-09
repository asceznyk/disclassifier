from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

import torch

from config import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fit(master, student, train_loader, valid_loader=None, ckpt_path=None):  
    def run_epoch(split, mode):
        is_train = split == 'train' 
        student.train(is_train)
        loader = train_loader if is_train else valid_loader

        avg_loss = 0
        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, batch in pbar: 
            batch = [i.to(device) for i in batch]
            seq, mask, labels = batch
            
            if mode == 'distil':
                labels = master(seq, mask)
                mask = None

            with torch.set_grad_enabled(is_train):  
                preds = student(seq, mask)
                loss = cost(preds, labels)
                avg_loss += loss.item() / len(loader)

            if is_train:
                student.zero_grad() 
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0) 
                optimizer.step()

            pbar.set_description(f"epoch: {e+1}, loss: {loss.item():.3f}, avg: {avg_loss:.2f}")     
        return avg_loss

    if student is None:
        student = master
        cost = torch.nn.CrossEntropyLoss()
        mode = 'finetune'
    else:  
        cost = torch.nn.KLDivLoss()
        mode = 'distil'

    best_loss = float('inf') 
    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE) 
    for e in range(EPOCHS):
        train_loss = run_epoch('train', mode)
        valid_loss = run_epoch('valid', mode) if valid_loader is not None else train_loss

        if ckpt_path is not None and valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(student.state_dict(), ckpt_path)

def calc_acc(model, loader):
    model.eval().to(device)
    labels, preds = [], []
    for (seq, mask, label) in loader:
        with torch.no_grad():
            pred = model(seq.to(device), mask.to(device)) 
            pred = pred.detach().cpu().numpy()

        preds.extend([i for i in np.argmax(pred, axis=1)])
        labels.extend([i for i in label.detach().cpu().numpy()])

    print(classification_report(np.array(labels), np.array(preds), zero_division=0))

