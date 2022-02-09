import numpy as np
import pandas as pd

from config import *
from model import *
from dataset import *
from trainer import *

from transformers import AutoModel, BertConfig

bertconf = BertConfig()

xcol, ycol = 'ptitle', 'category'
train_loader, valid_loader, train_df, valid_df, labels = split_preprocess_data(pd.read_csv('/kaggle/input/catpreds/train_set.csv'), xcol, ycol)

n_class = len(labels)

master = BertClassifier(AutoModel.from_pretrained('bert-base-uncased'), n_class, HIDDEN_DIM)
student = BiGRUClassifier(n_class, bertconf.vocab_size, master.emb_dim, HIDDEN_DIM)
#master_path = 'best.master.classifier'
student_path = 'best.student.classifier'

#fit(master, None, train_loader, valid_loader, master_path)
#master.load_state_dict(torch.load(master_path))
#calc_acc(master, valid_loader)

#fit(master, student, train_loader, valid_loader, student_path)
student.load_state_dict(torch.load(student_path))
calc_acc(student, valid_loader)

pred_df = predict(student, student_path, valid_df, labels, [xcol, ycol, 'pred_'+ycol])
pred_df.to_csv('preds_labels.csv')

