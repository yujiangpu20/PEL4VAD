import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import gen_label


def CLAS2(logits, label, seq_len, criterion):
    logits = logits.squeeze()
    ins_logits = torch.zeros(0).cuda()  # tensor([])
    for i in range(logits.shape[0]):
        if label[i] == 0:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=1, largest=True)
        else:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
        tmp = torch.mean(tmp).view(1)
        ins_logits = torch.cat((ins_logits, tmp))

    clsloss = criterion(ins_logits, label)
    return clsloss


def KLV_loss(preds, label, criterion):
    preds = F.softmax(preds, dim=1)
    preds = torch.log(preds)
    if torch.isnan(preds).any():
        loss = 0
    else:
        # preds = F.log_softmax(preds, dim=1)
        target = F.softmax(label * 10, dim=1)
        loss = criterion(preds, target)

    return loss


def temporal_smooth(arr):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2-arr)**2)
    return loss


def temporal_sparsity(arr):
    loss = torch.sum(arr)
    # loss = torch.mean(torch.norm(arr, dim=0))
    return loss


def Smooth(logits, seq_len, lamda=8e-5):
    smooth_mse = []
    for i in range(logits.shape[0]):
        tmp_logits = logits[i][:seq_len[i]-1]
        sm_mse = temporal_smooth(tmp_logits)
        smooth_mse.append(sm_mse)
    smooth_mse = sum(smooth_mse) / len(smooth_mse)

    return smooth_mse * lamda


def Sparsity(logits, seq_len, lamda=8e-5):
    spar_mse = []
    for i in range(logits.shape[0]):
        tmp_logits = logits[i][:seq_len[i]]
        sp_mse = temporal_sparsity(tmp_logits)
        spar_mse.append(sp_mse)
    spar_mse = sum(spar_mse) / len(spar_mse)

    return spar_mse * lamda


def Smooth_Sparsity(logits, seq_len, lamda=8e-5):
    smooth_mse = []
    spar_mse = []
    for i in range(logits.shape[0]):
        tmp_logits = logits[i][:seq_len[i]]
        sm_mse = temporal_smooth(tmp_logits)
        sp_mse = temporal_sparsity(tmp_logits)
        smooth_mse.append(sm_mse)
        spar_mse.append(sp_mse)
    smooth_mse = sum(smooth_mse) / len(smooth_mse)
    spar_mse = sum(spar_mse) / len(spar_mse)

    return (smooth_mse + spar_mse) * lamda