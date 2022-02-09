import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizerFast

from config import *

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def split_preprocess_data(df, xcol, ycol, nan_txt='Other', split=0.1):
    df.columns = df.columns.str.lower()
    xcol, ycol = xcol.lower(), ycol.lower()

    df[xcol] = df[xcol].str.lower()
    df[ycol] = df[ycol].str.lstrip().str.rstrip() 
    
    df[ycol].fillna(nan_txt, inplace=True)
    
    y_uniq = np.unique(df[ycol]).tolist()
    df[ycol+'_id'] = df[ycol].apply(lambda x: y_uniq.index(x))

    train_df, valid_df = train_test_split(df, test_size=split)

    def create_loader(df, is_train=1):
        tokens = tokenizer.batch_encode_plus(
            df[xcol].tolist(),
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True
        )

        sampler = RandomSampler if is_train else SequentialSampler
        dataset = TensorDataset(torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']), torch.tensor(df[ycol+'_id'].tolist())) 

        return DataLoader(dataset, sampler=sampler(dataset), batch_size=BATCH_SIZE)
    
    return create_loader(train_df), create_loader(valid_df, is_train=0), train_df, valid_df, y_uniq

def predict(model, ckpt_path, test_df, labels, all_cols, n_samples=20):   
    model.load_state_dict(torch.load(ckpt_path))
    model = model.cpu()

    [text_col, label_col, pred_col] = all_cols

    sampled_df = test_df.sample(n_samples)
    enc = tokenizer.batch_encode_plus(sampled_df[text_col].tolist(), padding=True)
    seq, mask = torch.tensor(enc['input_ids']), torch.tensor(enc['attention_mask']) 
    preds = np.argmax(model(seq.cpu(), mask.cpu()).detach().cpu().numpy(), axis=1)    
    sampled_df[pred_col] = [labels[p] for p in preds.tolist()]

    return sampled_df[all_cols]

