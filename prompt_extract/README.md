# Prompt-Enhanced Learning with CLIP

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

Prompt-Enhanced Learning aims to enrich potential context information via knowledge-based prompts. Therefore, we utilize [ConceptNet](https://github.com/commonsense/conceptnet5) to extract concepts that are highly relevant to specific anomalies. 


## Approach

![PEL](PEL.png)



## Usage
Generate Anomaly Dictionary (.json file) by running the following command:

```
python concept_extract.py --dataset 'ucf'  # dataset:['ucf', 'xd', 'sh']
```

Extract prompt feature (.npy file) by running the following command:

```
python token_extract.py --dataset 'ucf'  # dataset:['ucf', 'xd', 'sh']
```
