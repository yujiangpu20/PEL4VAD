import os
import glob

ori_path = './xd/train.list'
count = 0
with open('train_split.list', 'w+') as f:
    with open(ori_path, 'r') as b:
        for line in b:
            name = line.strip('\n').split('/')[-1]
            newline = 'train/'+name
            f.write(newline + '\n')

print('finish.')