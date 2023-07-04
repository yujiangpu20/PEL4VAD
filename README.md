# Learning Prompt-Enhanced Context features for Weakly-Supervised Video Anomaly Detection
**Authors**: Yujiang Pu, Xiaoyu Wu, Shengjin Wang

## Abstract
Video anomaly detection under weak supervision is challenging due to the absence of frame-level annotations during the training phase. Previous work has employed graph convolution networks or self-attention mechanisms to model temporal relations, along with multiple instance learning (MIL)-based classification loss to learn discriminative features. However, most of them utilize multi-branches to capture local and global dependencies separately, leading to increased parameters and computational cost. Furthermore, the binarized constraint of the MIL-based loss only ensures coarse-grained interclass separability, ignoring fine-grained discriminability within anomalous classes. In this paper, we propose a weakly supervised anomaly detection framework that emphasizes efficient context modeling and enhanced semantic discriminability. To this end, we first construct a temporal context aggregation (TCA) module that captures complete contextual information by reusing similarity matrix and adaptive fusion. Additionally, we propose a prompt-enhanced learning (PEL) module that incorporates semantic priors into the model by utilizing knowledge-based prompts, aiming at enhancing the discriminative capacity of context features while ensuring separability between anomaly sub-classes. Furthermore, we introduce a score smoothing (SS) module in the testing phase to suppress individual bias and reduce false alarms. Extensive experiments demonstrate the effectiveness of various components of our method, which achieves competitive performance with fewer parameters and computational effort on three challenging benchmarks: the UCF-crime, XD-violence, and ShanghaiTech datasets. The detection accuracy of some anomaly sub-classes is also improved with a great margin.

[[pdf](https://arxiv.org/pdf/2306.14451.pdf)] [[supp](https://drive.google.com/file/d/1CxvDFjiMg_RdEZA5_aOwwCEXlJuMMlxk/view?usp=drive_link)] [[video](https://drive.google.com/file/d/1A2E0_ylViA6LCQkb7XOQAum1VUoFMroL/view?usp=drive_link)]

![image](https://github.com/Aaron-Pu/PEL4VAD/blob/master/list/framework.png)

**Content**

1. Introduction
2. Requirements
3. Datasets
4. Quick Start
5. Acknowledgement
6. Citation


## Introduction
This repo is the official implementation of "Learning Prompt-Enhanced Context features for Weakly-Supervised Video Anomlay Detection" (under review). The original paper can be found [here](https://arxiv.org/pdf/2306.14451.pdf). We also submitted a [supplementary document](https://drive.google.com/file/d/1CxvDFjiMg_RdEZA5_aOwwCEXlJuMMlxk/view?usp=drive_link) with a [demo video](https://drive.google.com/file/d/1A2E0_ylViA6LCQkb7XOQAum1VUoFMroL/view?usp=drive_link) for peer review. Please feel free to contact me if you have any questions.

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
For the **UCF-Crime** and **XD-Violence** datasets, we use off-the-shelf features extracted by [Wu et al](https://github.com/Roc-Ng). For the **ShanghaiTech** dataset, we used this [repo](https://github.com/v-iashin/video_features) to extract features (highly recommended).
| Dataset     | Origin Video   | I3D Features  |
| -------- | -------- | -------- |
| UCF-Crime | [homepage](https://www.crcv.ucf.edu/projects/real-world/) | [download link](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/EvYcZ5rQZClGs_no2g-B0jcB4ynsonVQIreHIojNnUmPyA?e=xNrGxc) |
| XD-Violence | [homepage](https://roc-ng.github.io/XD-Violence/) | [download link](https://roc-ng.github.io/XD-Violence/) |
| ShanghaiTech | [homepage](https://svip-lab.github.io/dataset/campus_dataset.html) | [download link](https://drive.google.com/file/d/1kIv502RxQnMer-8HB7zrU_GU7CNPNNDv/view?usp=drive_link) |
Before the Quick Start, please download above features and change **feat_prefix** in config.py to your local path.

## Quick Start
Please modify the hyperparameters in **config.py** as necessary, where we keep default settings as mentioned in our paper. The example of configs for UCF-Crime is shown as follows:
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
smooth = 'slide'  # the type of score smoothing ['fixed': 10, slide': 7]
kappa = 7  # the smoothing window
ckpt_path = './ckpt/ucf__8636.pkl'
```

- Run the following command for training:
```
python main.py --dataset --train
```


