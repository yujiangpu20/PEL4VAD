import torch.utils.data as data
from utils import process_feat
import numpy as np
import os


class UCFDataset(data.Dataset):
    def __init__(self, cfg, transform=None, test_mode=False):
        self.feat_prefix = cfg.feat_prefix
        if test_mode:
            self.list_file = cfg.test_list
        else:
            self.list_file = cfg.train_list
        self.max_seqlen = cfg.max_seqlen
        self.tranform = transform
        self.test_mode = test_mode
        self.normal_flag = 'Normal'
        self.abnormal_dict = {'Normal':0,'Abuse':1, 'Arrest':2, 'Arson':3, 'Assault':4,
                              'Burglary':5, 'Explosion':6, 'Fighting':7,'RoadAccidents':8,
                              'Robbery':9, 'Shooting':10, 'Shoplifting':11, 'Stealing':12, 'Vandalism':13}
        self.t_features = np.array(np.load(cfg.token_feat))
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.list_file))

    def __getitem__(self, index):
        # video_name = self.list[index].strip('\n').split('/')[-1][:-4]
        feat_path = os.path.join(self.feat_prefix, self.list[index].strip('\n'))
        video_idx = self.list[index].strip('\n').split('/')[-1].split('_')[0]
        if self.normal_flag in self.list[index]:
            video_ano = video_idx
            ano_idx = self.abnormal_dict[video_ano]
            label = 0.0
        else:
            video_ano = video_idx[:-3]
            ano_idx = self.abnormal_dict[video_ano]
            label = 1.0

        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        fg_feat = np.array(self.t_features[ano_idx, :], dtype=np.float16)
        bg_feat = np.array(self.t_features[0, :], dtype=np.float16)
        fg_feat = fg_feat.reshape(1, 512)
        bg_feat = bg_feat.reshape(1, 512)
        t_feat = np.concatenate((bg_feat, fg_feat), axis=0)
        if self.tranform is not None:
            v_feat = self.tranform(v_feat)
            t_feat = self.tranform(t_feat)

        if self.test_mode:
            return v_feat, label  # ano_idx , video_name
        else:
            v_feat = process_feat(v_feat, self.max_seqlen, is_random=False)
            return v_feat, t_feat, label, ano_idx

    def __len__(self):
        return len(self.list)


class XDataset(data.Dataset):
    def __init__(self, cfg, transform=None, test_mode=False):
        self.feat_prefix = cfg.feat_prefix
        if test_mode:
            self.list_file = cfg.test_list
        else:
            self.list_file = cfg.train_list

        self.max_seqlen = cfg.max_seqlen
        self.tranform = transform
        self.test_mode = test_mode
        self.t_features = np.load(cfg.token_feat)
        self.normal_flag = '_label_A'
        self.abnormal_dict = {'A': 0, 'B5': 1, 'B6': 2, 'G': 3, 'B1': 4, 'B4': 5, 'B2': 6}
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.list_file))

    def __getitem__(self, index):
        if self.normal_flag in self.list[index]:
            label = 0.0
        else:
            label = 1.0

        feat_path = os.path.join(self.feat_prefix, self.list[index].strip('\n'))
        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        tokens = self.list[index].strip('\n').split('_label_')[-1].split('__')[0].split('-')
        idx = self.abnormal_dict[tokens[0]]
        fg_feat = self.t_features[idx, :].reshape(1, 512)
        bg_feat = self.t_features[0, :].reshape(1, 512)
        t_feat = np.concatenate((bg_feat, fg_feat), axis=0)
        if self.tranform is not None:
            v_feat = self.tranform(v_feat)
            t_feat = self.tranform(t_feat)
        if self.test_mode:
            return v_feat, self.list[index]  #, idx
        else:
            v_feat = process_feat(v_feat, self.max_seqlen, is_random=False)
            return v_feat, t_feat, label, idx

    def __len__(self):
        return len(self.list)


class SHDataset(data.Dataset):
    def __init__(self, cfg, transform=None, test_mode=False):
        self.feat_prefix = cfg.feat_prefix
        if test_mode:
            self.list_file = cfg.test_list
        else:
            self.list_file = cfg.train_list

        self.max_seqlen = cfg.max_seqlen
        self.tranform = transform
        self.test_mode = test_mode
        self.abn_file = cfg.abn_label
        self.cls_dict = {'cycling': 1, 'chasing': 2, 'handcart': 3, 'fighting': 4,'skateboarding': 5,
                         'vehicle': 6, 'running': 7, 'jumping': 8, 'wandering': 9, 'lifting': 10,
                         'robbery': 11, 'climbing_over': 12, 'throwing': 13}
        self.tokens = np.array(np.load(cfg.token_feat))
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.list_file))
        self.abn_dict = {}
        self.abn_list = []

        with open(self.abn_file, 'r') as f:
            f = f.readlines()
            for line in f:
                name = line.strip('\n').split(' ')[0]
                label = line.strip('\n').split(' ')[1]
                action = label.split(',')
                self.abn_dict[name] = action
                self.abn_list.append(name)

    def __getitem__(self, index):
        video_name = self.list[index].strip('\n').split(' ')[0].split('/')[-1][:-6]
        video_path = os.path.join(self.feat_prefix, self.list[index].strip('\n').split(' ')[0])
        v_feat = np.array(np.load(video_path), dtype=np.float32)

        if self.tranform is not None:
            v_feat = self.tranform(v_feat)

        if not self.test_mode:
            if video_name in self.abn_list:
                cls = self.abn_dict[video_name]
                abn_idx = [self.cls_dict[i] for i in cls]
            else:
                abn_idx = [0]
            fg_feat = np.array(self.tokens[abn_idx, :]).reshape(-1, 512)
            fg_feat = np.mean(fg_feat, axis=0).reshape(1, 512)
            bg_feat = np.array(self.tokens[0, :]).reshape(1, 512)
            t_feat = np.concatenate((bg_feat, fg_feat), axis=0)

            label = float(self.list[index].strip('\n').split(' ')[1])
            v_feat = process_feat(v_feat, self.max_seqlen, is_random=False)
            return v_feat, t_feat, label, abn_idx[0]

        else:
            return v_feat, video_name

    def __len__(self):
        return len(self.list)
