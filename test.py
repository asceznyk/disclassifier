import numpy as np
import pandas as pd

from dataset import *

xcol, ycol = 'ptitle', 'Category'
train_loader, valid_loader = split_preprocess_data(pd.read_csv('/kaggle/input/catpreds/train_set.csv'), xcol, ycol)

print(next(iter(train_loader)))
print(next(iter(valid_loader)))


