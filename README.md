# From Text to Recommendations: LLM-Guided Interest Tokenization via T5 and Cross-Attention in Generative Models

This is project is based on the codebases [ETEGrec](https://github.com/BishopLiu/ETEGRec). We gratefully thank the authors for their wonderful works.

## Overview

We propose **ETEGRec**, a novel **E**nd-**T**o-**E**nd **G**enerative **Rec**ommender that unifies item tokenization and generative recommendation into a cohesive framework. Built on a dual encoder-decoder architecture, ETEGRec consists of an item tokenizer and a generative recommender. To enable synergistic interaction between these components, we propose a recommendation-oriented alignment strategy, which includes two key optimization objectives: sequence-item alignment and preference-semantic alignment. These objectives tightly couple the learning processes of the item tokenizer and the generative recommender, fostering mutual enhancement. Additionally, we develop an alternating optimization technique to ensure stable and efficient end-to-end training of the entire framework.

![model](./asset/model.png)

## Requirements

```
torch==2.4.0+cu121
numpy
accelerate
faiss
tqdm
scikit-learn
transformers
```

## Dataset

You can download the SASRec embeddings, pretrained RQVAE weights and interaction data used in our paper from [Google Drive](https://drive.google.com/drive/folders/1KiPpB7uq7eFc4qB74cFOxhtY3H8nWgAI?usp=sharing) 


## RQVAE Pretrain
```shell
cd RQVAE
bash run_pretrain.sh
```

## Train

```shell
bash run.sh
```

## Demo
```bash
cd demo
python app.py
```
