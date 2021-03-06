from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from model import *

def fit(model, train_loader, valid_loader=None, ckpt_path=None, cost_fn='entropy', epochs=EPOCHS):  
    def run_epoch(split):
        is_train = split == 'train' 
        model.train(is_train)
        loader = train_loader if is_train else valid_loader

        avg_loss = 0
        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, batch in pbar: 
            batch = [i.to(device) for i in batch]
            seq, mask, labels = batch
            
            with torch.set_grad_enabled(is_train):  
                preds = model(seq, mask)

                if cost_fn == 'kldiv':
                    preds =  F.log_softmax(preds, dim=-1)
                    labels = F.softmax(labels, dim=-1)

                loss = cost(preds, labels) 
                avg_loss += loss.item() / len(loader)

            if is_train:
                model.zero_grad() 
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step()

            pbar.set_description(f"epoch: {e+1}, loss: {loss.item():.3f}, avg: {avg_loss:.2f}")     
        return avg_loss

    model.to(device)

    if cost_fn == 'entropy':
        cost = nn.CrossEntropyLoss()
    elif cost_fn == 'mse': 
        cost = nn.MSELoss()
    else:
        cost = nn.KLDivLoss(reduction='batchmean')

    best_loss = float('inf') 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    for e in range(epochs):
        train_loss = run_epoch('train')
        valid_loss = run_epoch('valid') if valid_loader is not None else train_loss

        if ckpt_path is not None and valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)

