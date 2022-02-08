import numpy as np
import pandas as pd

xcol, ycol = 'ptitle', 'Category'
train_df, valid_df = preprocess_split_data(pd.read_excel('/content/drive/MyDrive/train_set_corvid.xls'))
train_loader = create_loader(train_df, xcol, ycol)
valid_loader = create_loader(valid_df, xcol, ycol, is_train=False)

print(train_loader, valid_loader)


