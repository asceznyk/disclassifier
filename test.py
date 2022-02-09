import numpy as np
import pandas as pd

from config import *
from model import *
from dataset import *
from trainer import *

from transformers import AutoModel, BertConfig

bertconf = BertConfig()
print(bertconf)

xcol, ycol = 'ptitle', 'Category'
train_loader, valid_loader, n_class = split_preprocess_data(pd.read_csv('/kaggle/input/catpreds/train_set.csv'), xcol, ycol)
 
master = BertClassifier(AutoModel.from_pretrained('bert-base-uncased'), n_class, HIDDEN_DIM)
student = BiGRUClassifier(n_class, BertConfig.vocab_size, master.emb_dim, HIDDEN_DIM)
master_path = 'best.master.classifier'
student_path = 'best.student.classifier'

fit(master, None, train_loader, valid_loader, master_path)
master.load_state_dict(torch.load(master_path))
calc_acc(master, valid_loader)

fit(master, student, train_loader, valid_loader, student_path)
student.load_state_dict(torch.load(student_path))
calc_acc(student, valid_loader)
