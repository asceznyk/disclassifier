import numpy as np
import pandas as pd

from config import *
from model import *
from dataset import *
from trainer import *

from transformers import AutoModel

xcol, ycol = 'ptitle', 'Category'
train_loader, valid_loader, n_class = split_preprocess_data(pd.read_csv('/kaggle/input/catpreds/train_set.csv'), xcol, ycol)
 
master = BertClassifier(AutoModel.from_pretrained('bert-base-uncased'), n_class, HIDDEN_DIM)

print(master)

train(master, train_loader, valid_loader)
calc_acc(master, valid_loader)

