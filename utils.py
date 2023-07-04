# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def random_extract(feat, t_max):
   r = np.random.randint(len(feat)-t_max)
   return feat[r:r+t_max]


def uniform_extract(feat, t_max):
   r = np.linspace(0, len(feat)-1, t_max, dtype=np.uint16)
   return feat[r, :]


def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0, min_len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat


def process_feat(feat, length, is_random=True):
    if len(feat) > length:
        if is_random:
            return random_extract(feat, length)
        else:
            return uniform_extract(feat, length)
    else:
        return pad(feat, length)


def process_feat2(feat, length, is_random=False):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), length+1, dtype=np.int32)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat


def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num, num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i, k] = 1
    return gt


def create_logits(x1, x2, logit_scale):
    x2 = x2.squeeze(dim=1)
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    v2t_logits = torch.matmul(logit_scale * x1, x2.t())
    v2v_logits = torch.matmul(logit_scale * x1, x1.t())
    return v2t_logits, v2v_logits


def KLVDloss(preds, label, criterion):
    preds = F.softmax(preds, dim=1)
    preds = torch.log(preds + 1e-7)
    if torch.isnan(preds).any():
        loss = 0
    else:
        # preds = F.log_softmax(preds, dim=1)
        target = F.softmax(label * 10, dim=1)
        loss = criterion(preds, target)

    return loss


def get_cas(x_v, x_t, logits, labels, scale=10):
    x_v = x_v.permute(0, 2, 1)
    video_feat = torch.zeros(0).cuda()  # tensor([])
    token_feat = torch.zeros(0).cuda()  # tensor([])
    video_labels = torch.zeros(0).cuda()  # tensor([])
    bg_label = torch.tensor([0]).cuda()

    abn_logits = (scale * logits).exp() - 1
    abn_logits = F.normalize(abn_logits, p=1, dim=1)
    nor_logits = (scale * (1. - logits)).exp() - 1
    nor_logits = F.normalize(nor_logits, p=1, dim=1)

    abn_feat = torch.matmul(abn_logits.permute(0, 2, 1), x_v)
    nor_feat = torch.matmul(nor_logits.permute(0, 2, 1), x_v)

    for i in range(logits.shape[0]):
        if labels[i] == 0:
            fg = abn_feat[i, :, :]
            video_feat = torch.cat((video_feat, fg), dim=0)
            token_feat = torch.cat((token_feat, x_t[i, 0, :].view(1, -1)), dim=0)
            video_labels = torch.cat((video_labels, labels[i].view(1)))
        else:
            fg = abn_feat[i, :, :]
            bg = nor_feat[i, :, :]
            # foreground
            video_feat = torch.cat((video_feat, fg), dim=0)
            token_feat = torch.cat((token_feat, x_t[i, 1, :].view(1, -1)), dim=0)
            video_labels = torch.cat((video_labels, labels[i].view(1)))
            # background
            video_feat = torch.cat((video_feat, bg), dim=0)
            token_feat = torch.cat((token_feat, x_t[i, 0, :].view(1, -1)), dim=0)
            video_labels = torch.cat((video_labels, bg_label.view(1)))

    return video_feat, token_feat, video_labels


def pairwise_cosine_similarity(x, y):
    # x,y: (bs, n_head, seq_len, num_channels)
    y = y.permute(0, 1, 3, 2)
    dot = torch.matmul(x, y)
    x_dist = torch.norm(x, p=2, dim=3, keepdim=True)
    y_dist = torch.norm(y, p=2, dim=2, keepdim=True)
    dist = x_dist * y_dist
    cos = dot / (dist + 1e-8)
    # cos_dist = 1 - cos
    return cos


def pairwise_minus_l2_distance(x, y):
    # x,y: (bs, n_head, seq_len, num_channels)
    x = x.unsqueeze(3).detach()
    # ([128, 4, 200, 256]) -> ([128, 4, 1, 200, 256])
    y = y.unsqueeze(2)
    l2_dist = torch.sqrt(torch.sum((x-y)**2, dim=-1) + 1e-8)
    l2_dist = nn.InstanceNorm2d(l2_dist.size(1))(l2_dist)
    return  -l2_dist


def fixed_smooth(logits, t_size):
    ins_preds = torch.zeros(0).cuda()
    assert t_size > 1
    if len(logits) % t_size != 0:
        delta = t_size - len(logits) % t_size
        logits = F.pad(logits, (0,  delta), 'constant', 0)

    seq_len = len(logits) // t_size
    for i in range(seq_len):
        seq = logits[i * t_size: (i + 1) * t_size]
        avg = torch.mean(seq, dim=0)
        avg = avg.repeat(t_size)
        ins_preds = torch.cat((ins_preds, avg))

    return ins_preds


def slide_smooth(logits, t_size, mode='zero'):
    assert t_size > 1
    ins_preds = torch.zeros(0).cuda()
    padding = t_size - 1
    if mode == 'zero':
        logits = F.pad(logits, (0, padding), 'constant', 0)
    elif mode == 'constant':
        logits = F.pad(logits, (0, padding), 'constant', logits[-1])

    seq_len = int(len(logits) - t_size) + 1
    for i in range(seq_len):
        seq = logits[i: i + t_size]
        avg = torch.mean(seq, dim=0).unsqueeze(dim=0)
        ins_preds = torch.cat((ins_preds, avg))

    return ins_preds