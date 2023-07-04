import os
import shutil
import numpy as np

src_feat = '/data/pyj/vad-master/SHTech/rgb/test'
dst_feat = '/data/pyj/feat/SH_new/test'
# mislead = []
# count = 0
# for file in os.listdir(src_feat):
#     ori_feat = np.load(os.path.join(src_feat, file))
#     new_feat = np.load(os.path.join(dst_feat, file))
#     if ori_feat.shape[0] != new_feat.shape[0]:
#         mislead.append(file[:-6]+'.npy')
#         count += 1
#         # print(file, ori_feat.shape[0], new_feat.shape[0])

# print(count)
gt = np.load('./SH_gt.npy')
new_gt = np.zeros(0).astype(np.float32)
with open('./rgb/test.list', 'r') as f:
    f = f.readlines()
    count = 0
    for line in f:
        count += 1
        if count % 10 == 0:
            name = line.strip('\n').split('/')[1]
            ori_feat = np.load(os.path.join(src_feat, name))
            new_feat = np.load(os.path.join(dst_feat, name))
            if ori_feat.shape[0] == new_feat.shape[0]:
                labels = gt[: ori_feat.shape[0] * 16]
                new_gt = np.concatenate([new_gt, labels], axis=0)
            else:
                labels = gt[: new_feat.shape[0] * 16]
                new_gt = np.concatenate([new_gt, labels], axis=0)
            gt = gt[ori_feat.shape[0] * 16:]

np.save('./new_gt.npy', new_gt)