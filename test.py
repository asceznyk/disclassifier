import argparse
import numpy as np
import pandas as pd

from config import *
from model import *
from dataset import *
from trainer import *

from transformers import AutoModel

def main(): 
    xcol, ycol = 'ptitle', 'category'  
    pcol = 'pred_'+ycol

    test_df = pd.read_csv('/kaggle/input/catpreds/test_set.csv')

    train_loader, valid_loader, train_df, valid_df, labels = split_preprocess_data(pd.read_csv('/kaggle/input/catpreds/train_set.csv'), xcol, ycol) 

    N_CLASS = len(labels)

    master = BertClassifier(AutoModel.from_pretrained('bert-base-uncased'), N_CLASS, HIDDEN_DIM)
    #student = BiGRUClassifier(N_CLASS, VOCAB_SIZE, EMB_DIM, HIDDEN_DIM) 

    master_path = 'best.master.classifier'
    #student_path = 'best.student.classifier'

    #fit(master, train_loader, valid_loader, master_path)
    master.load_state_dict(torch.load(master_path))
    #calc_acc(master, valid_loader)

    pos_dict = build_pos_dict(train_df, xcol)
    print(pos_dict)
    aug_df = augment_sentences(train_df, xcol, pos_dict)
    aug_df.to_csv('aug_set.csv', index=False)

    logits_df = predict_logits(master, train_df, xcol)
    #loader = create_loader(logits_df, xcol, 'logits')
    #print(next(iter(loader)))

    #fit(student, train_loader, valid_loader, student_path)
    #student.load_state_dict(torch.load(student_path))
    #calc_acc(student, valid_loader)

    #pred_df = predict(student, student_path, valid_df, labels, xcol, pcol, ycol)
    #pred_df.to_csv('valid_preds.csv', index=False)

    with open('labels.txt', 'w') as f:
        for l in labels:
            f.write('%s\n' % l)

if __name__ == '__main__':
    main()

