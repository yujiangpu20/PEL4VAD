import os
import json
import numpy as np
import requests
import xlwt
from collections import OrderedDict, defaultdict
import argparse


def get_concept(dataset, labels, relations, file_name, filter=False):
    prompts = OrderedDict()
    for i, label in enumerate(labels):
        print(label)
        if ' ' in label:
            label = label.replace(' ', '_')
        else:
            pass
        if 'accident' in label:
            label = 'accident'
        else:
            pass

        token_score = {}
        for j, rel in enumerate(relations):
            print(rel, '-' * 20 + str(j))
            urls = 'http://api.conceptnet.io/query?node=/c/en/' + label + '&rel=/r/' + rel
            response = requests.get(urls)
            obj = response.json()

            if dataset == 'ucf' and label == 'accident':
                token_score['road accidents'] = 1.0
            elif dataset == 'xd' and label == 'accident':
                token_score['car accident'] = 1.0
            else:
                token_score[label] = 1.0

            for k in range(len(obj['edges'])):
                start = obj['edges'][k]['start']['label']
                end = obj['edges'][k]['end']['label']
                sim = 'http://api.conceptnet.io/relatedness?node1=/c/en/' + start + '&node2=/c/en/' + end
                sim_score = requests.get(sim).json()
                sss = sim_score['value']  # related score
                print(start + ', ' + end + ', ' + str(sss))

                # first step filtering in PEL
                if filter:
                    if sss > 0:
                        if label in start and label not in end:
                            token_score[end] = sss
                        elif label in end and label not in start:
                            token_score[start] = sss
                        else:
                            pass
                else:
                    if label in start and label not in end:
                        token_score[end] = sss
                    elif label in end and label not in start:
                        token_score[start] = sss
                    else:
                        pass

        prompts[label] = token_score
    json_str = json.dumps(prompts)
    with open(file_name, 'w') as ff:
        ff.write(json_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ConceptExtract')
    parser.add_argument('--dataset', default='ucf', help='anomaly video dataset')
    args = parser.parse_args()

    relist = []
    labels = []
    with open(os.path.join('./list', args.dataset + '_relation.list'), 'r') as f:
        for line in f.readlines():
            rel = line.strip('\n')
            relist.append(rel)

    with open(os.path.join('./list', args.dataset + '_label.list'), 'r') as f:
        for line in f.readlines():
            label = line.strip('\n')
            labels.append(label)

    file_name = os.path.join('./json', args.dataset+'-concept.json')
    get_concept(args.dataset, labels, relist, file_name, filter=True)
