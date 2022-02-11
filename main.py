import argparse
import numpy as np
import pandas as pd

from config import *
from model import *
from dataset import *
from trainer import *

from transformers import AutoModel

def main(args):
    xcol, ycol = args.input_col.lower(), args.label_col.lower() 
    pcol = 'pred_'+ycol
    yid = ycol+'_id'
    lcol = 'logits'

    labels = []
    with open('labels.txt', 'r') as f:
        for l in f: labels.append(l.replace('\n', ''))
    N_CLASS = len(labels)

    student_path = 'best.student.classifier'
    student = BiGRUClassifier(N_CLASS, VOCAB_SIZE, EMB_DIM, HIDDEN_DIM)

    full_df = pd.read_csv('full_set.csv')

    if args.is_train:
        train_df, valid_df = split_data(full_df)
        train_loader, valid_loader = create_loader(train_df, xcol, yid), create_loader(valid_df, xcol, yid, randomize=0)

        master = BertClassifier(AutoModel.from_pretrained('bert-base-uncased'), N_CLASS, HIDDEN_DIM) 
        master_path = 'best.master.classifier'

        fit(master, train_loader, valid_loader, master_path)
        master.load_state_dict(torch.load(master_path))
        calc_acc(master, valid_loader)

        pos_dict = build_pos_dict(train_df, xcol)
        aug_df = augment_sentences(train_df, xcol, pos_dict)
        logits_df = predict_logits(master, aug_df, xcol, lcol)
        train_logits, valid_logits = split_data(logits_df)
        train_logits, valid_logits = create_loader(train_logits, xcol, lcol), create_loader(valid_logits, xcol, lcol, randomize=0)

        fit(student, train_logits, valid_logits, student_path)
        student.load_state_dict(torch.load(student_path))
        calc_acc(student, valid_loader)

        pred_df = predict_labels(student, student_path, valid_df, labels, xcol, pcol, ycol)
        pred_df.to_csv('valid_preds.csv', index=False)
    else:
        student.load_state_dict(torch.load(student_path))
        pred_df = predict_labels(student, student_path, full_df, labels, xcol, pcol)
        pred_df.to_csv('test_preds.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type=int, default=1, help='train mode and test')
    parser.add_argument('--full_csv', type=str, help='full csv file with labels')
    parser.add_argument('--input_col', type=str, help='inputs to the seq classifier')
    parser.add_argument('--label_col', type=str, help='labels to the seq classifier')

    options = parser.parse_args()
    main(options)


