import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizerFast

from config import *

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def lower_cols(df, xcol, ycol):
    df.columns = df.columns.str.lower()
    xcol, ycol = xcol.lower(), ycol.lower()
    return df, xcol, ycol

def preprocess_split_data(df, xcol, ycol, nan_txt='Other', split=0.1):
    df, xcol, ycol = lower_cols(df, xcol, ycol)
    df[xcol] = df[xcol].str.lower()
    df[ycol] = df[ycol].str.lstrip().str.rstrip()
    
    df[ycol].fillna(nan_txt, inplace=True)

    y_uniq = np.unique(df[ycol]).tolist()
    df[ycol+'_id'] = df[ycol].apply(lambda x: y_uniq.index(x))

    return train_test_split(df, test_size=split)

def create_loader(df, xcol, ycol, is_train=True):
    df, xcol, ycol = lower_cols(df, xcol, ycol)
    tokens = tokenizer.batch_encode_plus(
        df[xcol].tolist(),
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True
    )

    sampler = RandomSampler if is_train else SequentialSampler
    dataset = TensorDataset(torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']), torch.tensor(df[ycol+'_id'].tolist())) 

    return DataLoader(dataset, sampler=sampler(dataset), batch_size=BATCH_SIZE)


