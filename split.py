#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# And pyspark.sql to get the spark session

# using this path_file when running the script
# !python split.py <path_file>

path_file=sys.argv[1]
df = pd.read_csv(path_file)
    
counts = df['user_id'].value_counts()
df = df[df['user_id'].isin(counts[counts >=10].index)]
    
# get unique user_ids
ids = df.user_id.unique()
np.random.seed(2020)
np.random.shuffle(ids)
train = int(0.6 * len(ids)) 
valid = int(0.2 * len(ids)) 
    
    
# split user_ids for three datasets
ids_train = np.array(ids[0:train])
ids_valid = np.array(ids[train:train + valid])
ids_test = np.array(ids[train + valid:])
    
train = df.loc[df['user_id'].isin(ids_train)]
valid = df.loc[df['user_id'].isin(ids_valid)]
test = df.loc[df['user_id'].isin(ids_test)]

# split interactions in validation set into two parts
valid_train = []
valid_valid = []
    
for i in ids_valid:
    act = valid.loc[valid['user_id'] == i]
    act = shuffle(act)
    length = int(act.shape[0]/2)
    to_train = act.iloc[:length]
    to_valid = act.iloc[length:]
    valid_train.append(to_train)
    valid_valid.append(to_valid)
        
        
# split interactions in test set into two parts           
test_train = []
test_test = []
    
for i in ids_test:
    act = test.loc[test['user_id'] == i]
    act = shuffle(act)
    length = int(act.shape[0]/2)
    to_train = act.iloc[:length]
    to_test = act.iloc[length:]
    test_train.append(to_train)
    test_test .append(to_test)
        
valid_train = pd.concat(valid_train)
valid_valid = pd.concat(valid_valid)
test_train = pd.concat(test_train)
test_test = pd.concat(test_test)

# final training set
train_df = [train, valid_train, test_train]
train = pd.concat(train_df)


# remove any items not observed in the training set
id_books = train.book_id.unique()
valid_valid = valid_valid.loc[valid_valid['book_id'].isin(id_books)]
test_test = test_test.loc[test_test['book_id'].isin(id_books)]
    
train.to_csv("train.csv",index= False)
print('train.csv has been saved')
valid_valid.to_csv("valid.csv",index= False)
print('valid.csv has been saved')
test_test.to_csv("test.csv",index= False)
print('test.csv has been saved')

