# Learning Prompt-Enhanced Context features for Weakly-Supervised Video Anomlay Detection
**Authors**: Yujiang Pu, Xiaoyu Wu, Shengjin Wang

## Abstract
Video anomaly detection under weak supervision is challenging due to the absence of frame-level annotations during the training phase. Previous work has employed graph convolution networks or self-attention mechanisms to model temporal relations, along with multiple instance learning (MIL)-based classification loss to learn discriminative features. However, most of them utilize multi-branches to capture local and global dependencies separately, leading to increased parameters and computational cost. Furthermore, the binarized constraint of the MIL-based loss only ensures coarse-grained interclass separability, ignoring fine-grained discriminability within anomalous classes. In this paper, we propose a weakly supervised anomaly detection framework that emphasizes efficient context modeling and enhanced semantic discriminability. To this end, we first construct a temporal context aggregation (TCA) module that captures complete contextual information by reusing similarity matrix and adaptive fusion. Additionally, we propose a prompt-enhanced learning (PEL) module that incorporates semantic priors into the model by utilizing knowledge-based prompts, aiming at enhancing the discriminative capacity of context features while ensuring separability between anomaly sub-classes. Furthermore, we introduce a score smoothing (SS) module in the testing phase to suppress individual bias and reduce false alarms. Extensive experiments demonstrate the effectiveness of various components of our method, which achieves competitive performance with fewer parameters and computational effort on three challenging benchmarks: the UCF-crime, XD-violence, and ShanghaiTech datasets. The detection accuracy of some anomaly sub-classes is also improved with a great margin.

[[pdf](https://arxiv.org/pdf/2306.14451.pdf)] [[supp](https://drive.google.com/file/d/1CxvDFjiMg_RdEZA5_aOwwCEXlJuMMlxk/view?usp=drive_link)] [[video](https://drive.google.com/file/d/1A2E0_ylViA6LCQkb7XOQAum1VUoFMroL/view?usp=drive_link)]

```
**Content**

1. Introduction
2. Requirements
3. Datasets
4. Quick Start
5. Acknowledgement
6. Citation
```

## Introduction
This repo is the official implementation of "Learning Prompt-Enhanced Context features for Weakly-Supervised Video Anomlay Detection" (under review). 
