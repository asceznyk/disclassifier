from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import spacy

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizerFast

from config import *

nlp = spacy.load("en_core_web_sm")
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

def split_data(df, split=0.1):
    train_df, valid_df = train_test_split(df, test_size=split)
    train_df = train_df.reset_index(drop=1)
    valid_df = valid_df.reset_index(drop=1)
    return train_df, valid_df

def preprocess_data(df, xcol, ycol, nan_txt='Other'):
    df.columns = df.columns.str.lower()
    xcol, ycol = xcol.lower(), ycol.lower()
    df[xcol] = df[xcol].str.lower()
    df[ycol] = df[ycol].str.lstrip().str.rstrip()     
    df[ycol].fillna(nan_txt, inplace=True)  
    return df

def build_pos_dict(df, xcol):
    pos_dict = {}
    def org_pos(input_sentence):
        for token in nlp(input_sentence):
            pos = token.pos_
            text = token.text.lower()
            if pos not in pos_dict: pos_dict[pos] = [text]
            if text not in pos_dict[pos]: pos_dict[pos].append(text)
    
    df[xcol].apply(org_pos)
    return pos_dict

def augment_sentences(df, xcol, pos_dict):
    new_df = {xcol:[]}
    mask_token = '[MASK]'
    def make_sample(input_sentence, p_mask=0.1, p_pos=0.1, p_ng=0.25, max_ng=5):
        sentence = []
        for token in nlp(input_sentence):
            u = np.random.uniform()
            if u < p_mask:
                sentence.append(mask_token)
            elif u < (p_mask + p_pos): 
                sentence.append(np.random.choice(pos_dict[token.pos_]))
            else:
                sentence.append(token.text.lower())

        if len(sentence) > 2 and np.random.uniform() < p_ng:
            n = min(np.random.choice(range(1, 5+1)), len(sentence) - 1)
            start = np.random.choice(len(sentence) - n)
            for idx in range(start, start + n):
                sentence[idx] = mask_token

        sentence = ' '.join(sentence)
        if input_sentence != sentence: new_df[xcol].append(sentence)
        new_df[xcol].append(input_sentence)

    df[xcol].apply(make_sample) 
    return pd.DataFrame(new_df, columns=[xcol])

def calc_acc(model, loader):
    model.eval().to(device)
    labels, preds = [], []
    for (seq, mask, label) in loader:
        with torch.no_grad():
            pred = F.softmax(model(seq.to(device), mask.to(device)), dim=-1) 
            pred = pred.detach().cpu().numpy()

        preds.extend([i for i in np.argmax(pred, axis=1)])
        labels.extend([i for i in label.detach().cpu().numpy()])

    print(classification_report(np.array(labels), np.array(preds), zero_division=0))

def predict_logits(model, df, xcol, lcol):
    model.eval().to(device)
    loader = create_loader(df, xcol, yid=None, randomize=0, bsize=1)
    logits_df = {xcol:[], lcol:[]}
    for i, (seq, mask) in tqdm(enumerate(loader), total=len(loader)):
        with torch.no_grad():
            pred = model(seq.to(device), mask.to(device)) 
            logits_df[xcol].append(df.loc[i, xcol])
            logits_df['logits'].append(pred.detach().cpu().numpy().tolist()[0])

    logits_df = pd.DataFrame(logits_df, columns=[xcol, 'logits'])
    return logits_df

def predict_labels(model, ckpt_path, test_df, labels, text_col, pred_col, label_col=None, n_samples=100):   
    model.load_state_dict(torch.load(ckpt_path))
    model = model.eval().to(device)

    sampled_df = test_df.sample(n_samples)
    enc = tokenizer.batch_encode_plus(sampled_df[text_col].tolist(), padding=True)
    seq, mask = torch.tensor(enc['input_ids']).to(device), torch.tensor(enc['attention_mask']).to(device)
    preds = np.argmax(F.softmax(model(seq, mask), dim=-1).detach().cpu().numpy(), axis=1)    
    sampled_df[pred_col] = [labels[p] for p in preds.tolist()]
    full_cols = [text_col, pred_col] if label_col is None else [text_col, label_col, pred_col]

    return sampled_df[full_cols]

