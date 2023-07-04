from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve
import numpy as np
import torch


def cal_false_alarm(gt, preds, threshold=0.5):
    preds = list(preds.cpu().detach().numpy())
    gt = list(gt.cpu().detach().numpy())

    preds = np.repeat(preds, 16)
    preds[preds < threshold] = 0
    preds[preds >= threshold] = 1
    tn, fp, fn, tp = confusion_matrix(gt, preds, labels=[0, 1]).ravel()

    far = fp / (fp + tn)

    return far


def test_func(dataloader, model, gt, dataset):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        abnormal_preds = torch.zeros(0).cuda()
        abnormal_labels = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(gt.copy()).cuda()

        for i, (v_input, label) in enumerate(dataloader):
            v_input = v_input.float().cuda(non_blocking=True)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)

            logits, _ = model(v_input, seq_len)
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))
            labels = gt_tmp[: seq_len[0] * 16]
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
            else:
                abnormal_labels = torch.cat((abnormal_labels, labels))
                abnormal_preds = torch.cat((abnormal_preds, logits))
            gt_tmp = gt_tmp[seq_len[0] * 16:]

        pred = list(pred.cpu().detach().numpy())
        n_far = cal_false_alarm(normal_labels, normal_preds)
        fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
        roc_auc = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(rec, pre)

        if dataset == 'ucf-crime':
            return roc_auc, n_far
        elif dataset == 'xd-violence':
            return pr_auc, n_far
        elif dataset == 'shanghaiTech':
            return roc_auc, n_far
        else:
            raise RuntimeError('Invalid dataset.')
