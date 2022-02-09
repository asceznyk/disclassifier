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
    train_loader, valid_loader, _, valid_df, labels = split_preprocess_data(pd.read_csv(args.train_csv), xcol, ycol)
    N_CLASS = len(labels)

    master_path = 'best.master.classifier'
    student_path = 'best.student.classifier'
 
    pcol = 'pred_'+ycol
    student = BiGRUClassifier(N_CLASS, VOCAB_SIZE, EMB_DIM, HIDDEN_DIM) 
    if is_train:
        master = BertClassifier(AutoModel.from_pretrained('bert-base-uncased'), N_CLASS, HIDDEN_DIM)

        fit(master, None, train_loader, valid_loader, master_path)
        master.load_state_dict(torch.load(master_path))
        calc_acc(master, valid_loader)

        fit(master, student, train_loader, valid_loader, student_path)
        student.load_state_dict(torch.load(student_path))
        calc_acc(student, valid_loader)

        pred_df = predict(student, student_path, valid_df, labels, [xcol, ycol, pcol])
        pred_df.to_csv('valid_preds.csv', index=False)
    else:
        test_df = pd.read_csv(args.test_csv)
        student.load_state_dict(torch.load(student_path))
        pred_df = predict(student, student_path, test_df, labels, [xcol, pcol])
        pred_df.to_csv('test_preds.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type=int, default=1, help='train mode and test')
    parser.add_argument('--train_csv', type=str, help='train csv file with labels')
    parser.add_argument('--test_csv', type=str, help='test csv with inputs')
    parser.add_argument('--input_col', type=str, help='inputs to the seq classifier')
    parser.add_argument('--label_col', type=str, help='labels to the seq classifier')

    options = parser.parse_args()
    main(options)


