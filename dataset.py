from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizerFast

from config import *

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def create_loader(df, xcol, yid=None, randomize=1, bsize=BATCH_SIZE):
    tokens = tokenizer.batch_encode_plus(
        df[xcol].tolist(),
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True
    )

    sampler = RandomSampler if randomize else SequentialSampler

    input_ids = torch.tensor(tokens['input_ids'])
    attention_mask = torch.tensor(tokens['attention_mask']) 

    dataset = TensorDataset(input_ids, attention_mask, torch.tensor(df[yid].tolist())) if yid is not None else TensorDataset(input_ids, attention_mask) 

    return DataLoader(dataset, sampler=sampler(dataset), batch_size=bsize)

def split_preprocess_data(df, xcol, ycol, nan_txt='Other', split=0.1):
    df.columns = df.columns.str.lower()
    xcol, ycol = xcol.lower(), ycol.lower()

    df[xcol] = df[xcol].str.lower()
    df[ycol] = df[ycol].str.lstrip().str.rstrip() 
    
    df[ycol].fillna(nan_txt, inplace=True)
    
    y_uniq = np.unique(df[ycol]).tolist()
    yid = ycol+'_id'
    df[yid] = df[ycol].apply(lambda x: y_uniq.index(x))

    train_df, valid_df = train_test_split(df, test_size=split)
    train_df = train_df.reset_index(drop=1)
    valid_df = valid_df.reset_index(drop=1)
    train_loader, valid_loader = create_loader(train_df, xcol, yid), create_loader(valid_df, xcol, yid, randomize=0)
    
    return train_loader, valid_loader, train_df, valid_df, y_uniq

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

def predict_logits(model, df, xcol):
    model.eval().to(device)
    loader = create_loader(df, xcol, yid=None, randomize=0)
    bsize = BATCH_SIZE
    logits_df = {xcol:[], 'logits':[]}
    for i, (seq, mask) in tqdm(enumerate(loader), total=len(loader)):
        with torch.no_grad():
            pred = model(seq.to(device), mask.to(device))
            pred = pred.detach().cpu().numpy().tolist()
            print([(i*bsize+k) for k in range(bsize)])
            logits_df[xcol].extend([df.loc[(i*bsize)+k, xcol] for k in range(bsize)])
            logits_df['logits'].extend(pred)

    logits_df = pd.DataFrame(logits_df, columns=[xcol, 'logits'])
    return logits_df

def predict_labels(model, ckpt_path, test_df, labels, text_col, pred_col, label_col=None, n_samples=50):   
    model.load_state_dict(torch.load(ckpt_path))
    model.eval().cpu()

    sampled_df = test_df.sample(n_samples)
    enc = tokenizer.batch_encode_plus(sampled_df[text_col].tolist(), padding=True)
    seq, mask = torch.tensor(enc['input_ids'], enc['attention_mask'])
    preds = np.argmax(model(seq.cpu(), mask.cpu()).detach().cpu().numpy(), axis=1)    
    sampled_df[pred_col] = [labels[p] for p in preds.tolist()]

    full_cols = [text_col, pred_col] if label_col is None else [text_col, label_col, pred_col]

    return sampled_df[full_cols]

