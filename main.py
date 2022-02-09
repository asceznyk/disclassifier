import argparse
import numpy as np
import pandas as pd

from config import *
from model import *
from dataset import *
from trainer import *

from transformers import AutoModel

def main(args):
    is_train, xcol, ycol = args.is_train, args.input_col, args.label_col  
    student = BiGRUClassifier(N_CLASS, VOCAB_SIZE, EMB_DIM, HIDDEN_DIM) 
    student_path = 'best.student.classifier'
    pcol = 'pred_'+ycol

    if is_train:
        train_loader, valid_loader, _, valid_df, labels = split_preprocess_data(pd.read_csv(args.full_csv), xcol, ycol) 

        master = BertClassifier(AutoModel.from_pretrained('bert-base-uncased'), N_CLASS, HIDDEN_DIM)

        master_path = 'best.master.classifier'

        fit(master, None, train_loader, valid_loader, master_path)
        master.load_state_dict(torch.load(master_path))
        calc_acc(master, valid_loader)

        fit(master, student, train_loader, valid_loader, student_path)
        student.load_state_dict(torch.load(student_path))
        calc_acc(student, valid_loader)

        pred_df = predict(student, student_path, valid_df, labels, xcol, pcol, ycol)
        pred_df.to_csv('valid_preds.csv', index=False)

        with open('labels.txt', 'w') as f:
            for l in labels:
                f.write('%s\n' % l)
    else: 
        test_df = pd.read_csv(args.full_csv)
        labels = []
        with open('labels.txt', 'r') as f:
            for l in f: labels.append(l.replace('\n', ''))

        student.load_state_dict(torch.load(student_path))
        pred_df = predict(student, student_path, test_df, labels, xcol, pcol)
        pred_df.to_csv('test_preds.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type=int, default=1, help='train mode and test')
    parser.add_argument('--full_csv', type=str, help='train csv file with labels')
    parser.add_argument('--input_col', type=str, help='inputs to the seq classifier')
    parser.add_argument('--label_col', type=str, help='labels to the seq classifier')

    options = parser.parse_args()
    main(options)


