import os.path

import clip
import torch
import numpy as np
from collections import OrderedDict
import json
import argparse


def prompt2vec(json_file, prompt_file, th=0.0, fixed_th=False, avg_th=False, clipbackbone='ViT-B/16'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load the CLIP model
    clipmodel, _ = clip.load(clipbackbone, device=device, jit=False)
    for param in clipmodel.parameters():
        param.requires_grad = False

    # clip_feat = torch.zeros(0).half().cuda()
    clip_feat = torch.zeros(0).cuda()

    # convert to token, will automatically padded to 77 with zeros
    with open(json_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        for label, prompt in json_data.items():
            # print(label)
            actionlist = []
            weights = []
            relation = list(prompt.values())
            if len(relation) == 1:
                dynamic_th = 1.0
            else:
                dynamic_th = sum(relation[1:]) / len(relation[1:])  # mean value of all related token

            # print(label, dynamic_th)
            if fixed_th:
                for token, score in prompt.items():
                    if score >= th:
                        actionlist.append(token)
                        weights.append(score)
            elif avg_th:
                for token, score in prompt.items():
                    if score >= dynamic_th:
                        actionlist.append(token)
                        weights.append(score)
            else:
                # all concept
                for token, score in prompt.items():
                    actionlist.append(token)
                    weights.append(score)

            actiontoken = clip.tokenize(actionlist).to(device)
            # query the vector from dictionary
            with torch.no_grad():
                actionembed = clipmodel.encode_text(actiontoken)
                actionembed = torch.mean(actionembed, dim=0)
                # actionembed2 = torch.tensor(actionembed.clone().detach(), dtype=torch.float32)
            clip_feat = torch.cat((clip_feat, actionembed.view(1, -1)), dim=0)

    np.save(prompt_file, np.array(clip_feat.detach().cpu().numpy()))


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser(description='PromptExtract')
    parser.add_argument('--dataset', default='ucf', help='anomaly video dataset')
    args = parser.parse_args()

    json_file = os.path.join('./json', args.dataset+'-concept.json')
    prompt_file = os.path.join('./prompt_feature', args.dataset+'-prompt.npy')
    prompt2vec(json_file, prompt_file, th=0.0, fixed_th=False, avg_th=True)
