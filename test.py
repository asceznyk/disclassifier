import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from config import *
from model import *
from dataset import *
from trainer import *

from transformers import AutoModel

def main(): 
    xcol, ycol = 'ptitle', 'category'  
    pcol = 'pred_'+ycol
    lcol = 'logits'

    full_df = pd.read_csv('/kaggle/input/catpreds/train_set.csv')
    test_df = pd.read_csv('/kaggle/input/catpreds/test_set.csv')

    train_loader, valid_loader, train_df, valid_df, labels = split_preprocess_data(full_df, xcol, ycol) 

    N_CLASS = len(labels)

    master = BertClassifier(AutoModel.from_pretrained('bert-base-uncased'), N_CLASS, HIDDEN_DIM)
    student = BiGRUClassifier(N_CLASS, VOCAB_SIZE, EMB_DIM, HIDDEN_DIM) 

    master_path = 'best.master.classifier'
    student_path = 'best.student.classifier'

    fit(master, train_loader, valid_loader, master_path)
    master.load_state_dict(torch.load(master_path))
    calc_acc(master, valid_loader)

    pos_dict = build_pos_dict(train_df, xcol)
    aug_df = augment_sentences(train_df, xcol, pos_dict)
    logits_df = predict_logits(master, aug_df, xcol, lcol)
    train_logits, valid_logits = train_test_split(logits_df, split=0.1)
    train_logits, valid_logits = create_loader(train_logits, xcol, lcol), create_loader(valid_logits, xcol, lcol)

    fit(student, train_logits, valid_logits, student_path)
    student.load_state_dict(torch.load(student_path))
    calc_acc(student, valid_logits)

    pred_df = predict(student, student_path, valid_df, labels, xcol, pcol, ycol)
    pred_df.to_csv('valid_preds.csv', index=False)
    pred_df = predict(student, student_path, test_df, labels, xcol, pcol)
    pred_df.to_csv('test_preds.csv', index=False)

    with open('labels.txt', 'w') as f:
        for l in labels:
            f.write('%s\n' % l)

if __name__ == '__main__':
    main()

