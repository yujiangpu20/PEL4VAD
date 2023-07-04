# Learning Prompt-Enhanced Context features for Weakly-Supervised Video Anomlay Detection
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
The required packages can be installed directly by running the following command:
