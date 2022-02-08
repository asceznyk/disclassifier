import numpy as np
import pandas as pd

from dataset import *

xcol, ycol = 'ptitle', 'category'
train_df, valid_df = preprocess_split_data(pd.read_csv('/kaggle/input/catpreds/train_set.csv'), xcol, ycol)
train_loader = create_loader(train_df, xcol, ycol)
valid_loader = create_loader(valid_df, xcol, ycol, is_train=False)

print(train_loader, valid_loader)


