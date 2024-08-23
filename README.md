# Learning Prompt-Enhanced Context features for Weakly-Supervised Video Anomaly Detection
**Authors**: Yujiang Pu, Xiaoyu Wu, Lulu Yang, Shengjin Wang

## Abstract
Video anomaly detection under weak supervision presents significant challenges, particularly due to the lack of frame-level annotations during training. While prior research has utilized graph convolution networks and self-attention mechanisms alongside multiple instance learning (MIL)-based classification loss to model temporal relations and learn discriminative features, these methods often employ multi-branch architectures to capture local and global dependencies separately, resulting in increased parameters and computational costs. Moreover, the coarse-grained interclass separability provided by the binary constraint of MIL-based loss neglects the fine-grained discriminability within anomalous classes. In response, this paper introduces a weakly supervised anomaly detection framework that focuses on efficient context modeling and enhanced semantic discriminability. We present a Temporal Context Aggregation (TCA) module that captures comprehensive contextual information by reusing the similarity matrix and implementing adaptive fusion. Additionally, we propose a Prompt-Enhanced Learning (PEL) module that integrates semantic priors using knowledge-based prompts to boost the discriminative capacity of context features while ensuring separability between anomaly sub-classes. Extensive experiments validate the effectiveness of our method's components, demonstrating competitive performance with reduced parameters and computational effort on three challenging benchmarks: UCF-Crime, XD-Violence, and ShanghaiTech datasets. Notably, our approach significantly improves the detection accuracy of certain anomaly sub-classes, underscoring its practical value and efficacy.

![image](https://github.com/Aaron-Pu/PEL4VAD/blob/master/list/framework.png)

**Contents**

[1. Introduction](#Introduction)  
[2. Requirements](#Requirements)  
[3. Datasets](#Datasets)  
[4. Quick Start](#Quick-Start)  
[5. Results and Models](#Results-and-Models)  
[6. Acknowledgement](#Acknowledgement)  
[7. Citation](#Citation)  


## Introduction
This repo is the official implementation of "Learning Prompt-Enhanced Context features for Weakly-Supervised Video Anomlay Detection" (Preprint on Jun.2023). The original paper can be found [here](https://arxiv.org/pdf/2306.14451.pdf). Please feel free to contact me if you have any questions.

## Requirements
The code requires ```python>=3.8``` and the following packages:
```
torch==1.8.0
torchvision==0.9.0
numpy==1.21.2
scikit-learn==1.0.1
scipy==1.7.2
pandas==1.3.4
tqdm==4.63.0
xlwt==2.5
```
The environment with required packages can be created directly by running the following command:
```
conda env create -f environment.yml
```

## Datasets
For the **UCF-Crime** and **XD-Violence** datasets, we use off-the-shelf features extracted by [Wu et al](https://github.com/Roc-Ng). For the **ShanghaiTech** dataset, we used this [repo](https://github.com/v-iashin/video_features) to extract I3D features (highly recommended:+1:).
| Dataset     | Origin Video   | I3D Features  |
| -------- | -------- | -------- |
| &nbsp;&nbsp;UCF-Crime | &nbsp;&nbsp;[homepage](https://www.crcv.ucf.edu/projects/real-world/) | [download link](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/EvYcZ5rQZClGs_no2g-B0jcB4ynsonVQIreHIojNnUmPyA?e=xNrGxc) |
| &nbsp;XD-Violence | &nbsp;&nbsp;[homepage](https://roc-ng.github.io/XD-Violence/) | [download link](https://roc-ng.github.io/XD-Violence/) |
| ShanghaiTech | &nbsp;&nbsp;[homepage](https://svip-lab.github.io/dataset/campus_dataset.html) | [download link](https://drive.google.com/file/d/1kIv502RxQnMer-8HB7zrU_GU7CNPNNDv/view?usp=drive_link) |

Before the Quick Start, please download above features and change **feat_prefix** in config.py to your local path.

## Quick Start
Please change the hyperparameters in **config.py** if necessary, where we keep default settings as mentioned in our paper. The example of configs for UCF-Crime is shown as follows:
```
dataset = 'ucf-crime'
model_name = 'ucf_'
metrics = 'AUC'  # the evaluation metric
feat_prefix = '/data/pyj/feat/ucf-i3d'  # the prefix path of the video features
train_list = './list/ucf/train.list'  # the split file of training set
test_list = './list/ucf/test.list'  #  the split file of test/infer set
token_feat = './list/ucf/ucf-prompt.npy'  # the prompt feature extracted by CLIP
gt = './list/ucf/ucf-gt.npy'  # the ground-truth of test videos

# TCA settings
win_size = 9  # the local window size
gamma = 0.6  # initialization for DPE
bias = 0.2  # initialization for DPE 
norm = True  # whether adaptive fusion uses normalization

# CC settings
t_step = 9  # the kernel size of causal convolution

# training settings
temp = 0.09  # the temperature for contrastive learning
lamda = 1  # the loss weight
seed = 9  # random seed

# test settings
test_bs = 10  # test batch size
smooth = 'slide'  # the type of score smoothing ['None', 'fixed': 10, slide': 7]
kappa = 7  # the smoothing window
ckpt_path = './ckpt/ucf__8636.pkl'
```

- Run the following command for **training**:
```
python main.py --dataset 'ucf' --mode 'train'  # dataset:['ucf', 'xd', 'sh']  mode:['train', 'infer']
```
- Run the following command for **test/inference**:
```
python main.py --dataset 'ucf' --mode 'infer'  # dataset:['ucf', 'xd', 'sh']  mode:['train', 'infer']
```

## Results and Models
Below are the results with score smoothing in the testing phase. Note that our experiments are conducted on a single Tesla A40 GPU, and different torch or cuda versions can lead to slightly different results.
| Dataset     | AUC (%)   | AP (%)  | FAR (%)  |  ckpt  |  log |
| --------     | -------- | -------- | -------- | -------- | -------- |
| &nbsp;&nbsp;UCF-Crime    |   &nbsp;&nbsp;**86.76**  |  &nbsp;33.99   |  &nbsp;&nbsp;&nbsp;0.47    |  &nbsp;[link](https://github.com/Aaron-Pu/PEL4VAD/blob/master/ckpt/ucf__8636.pkl)  |  [link](https://github.com/Aaron-Pu/PEL4VAD/blob/master/log_info.log)        |
| &nbsp;XD-Violence  |   &nbsp;&nbsp;94.94  |  &nbsp;**85.59**   |  &nbsp;&nbsp;&nbsp;0.57    |  &nbsp;[link](https://github.com/Aaron-Pu/PEL4VAD/blob/master/ckpt/xd__8526.pkl)        |       [link](https://github.com/Aaron-Pu/PEL4VAD/blob/master/log_info.log)   |
| ShanghaiTech |   &nbsp;&nbsp;**98.14**  |  &nbsp;72.56   |  &nbsp;&nbsp;&nbsp;0.00    |  &nbsp;[link](https://github.com/Aaron-Pu/PEL4VAD/blob/master/ckpt/SH__98.pkl)        |        [link](https://github.com/Aaron-Pu/PEL4VAD/blob/master/log_info.log)  |

## Acknowledgement
Our codebase mainly refers to [XDVioDet](https://github.com/Roc-Ng/XDVioDet) and [CLIP](https://github.com/openai/CLIP). We greatly appreciate their excellent contribution with nicely organized code!

## Citation
If this repo works positively for your research, please consider citing our paper. Thanks all!
```
@article{pu2023learning,
  title={Learning Prompt-Enhanced Context Features for Weakly-Supervised Video Anomaly Detection},
  author={Pu, Yujiang and Wu, Xiaoyu and Wang, Shengjin},
  journal={arXiv preprint arXiv:2306.14451},
  year={2023}
}
```
