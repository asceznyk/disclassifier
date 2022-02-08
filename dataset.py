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
    
    return create_loader(train_df), create_loader(valid_df, is_train=0)
