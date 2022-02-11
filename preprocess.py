import argparse
import numpy as np
import pandas as pd

from dataset import *

def main(args): 
    xcol, ycol = args.input_col.lower(), args.label_col.lower()
    full_df = preprocess_data(pd.read_csv(args.full_csv), xcol, ycol)
    labels = np.unique(full_df[ycol]).tolist()
    full_df[ycol+'_id'] = full_df[ycol].apply(lambda x: labels.index(x))
    full_df[[xcol,  ycol, ycol+'_id']].to_csv('full_set.csv', index=False)
    with open('labels.txt', 'w') as f: for l in labels: f.write('%s\n' % l)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_csv', type=str, help='full csv file with labels')
    parser.add_argument('--input_col', type=str, help='inputs to the seq classifier')
    parser.add_argument('--label_col', type=str, help='labels to the seq classifier')
    options = parser.parse_args()
    main(options)


