# From Text to Recommendations: LLM-Guided Interest Tokenization via T5 and Cross-Attention in Generative Models

ETEGRec is a SOTA end-to-end generative recommender system that integrates item tokenization and recommendation generation. However, its latent item tokenization process fails to take into account the high-level semantic information of user interests, which can lead to suboptimal recommendations. To address this, we introduce a brand new interest fusion mechanism that leverages user interest text embeddings to enhance the recommendation process. 

## Methodology

### LLM-Guided Interest Prediction

We utilize a large language model (LLM) to extract user interests from their purchase history and item metadata. The LLM generates structured interest categories, which are then used to enhance the recommendation process.

### Interest Fusion Mechanism

We implement a multi-layer cross-attention mechanism to fuse user interest embeddings with item sequence representations. This allows the model to better capture the semantic relationships between user interests and items, leading to more accurate recommendations.

### Integrating synchronous training and asynchronous inference procedure

During training, user interest are pre-processed. We use cached synchronous batch processing to ensure data consistency and quality. In the inference phase, we employ asynchronous processing to optimize user experience, allowing for real-time updates without blocking the user interface.

## Deployment

### Training

You should first install the dependencies below for ETEGRec. Also make sure some necessary libraries, e.g. `sentence-transformers`, are installed. I have provided the processed dataset (.processed.jsonl) along with the pretrained RQVAE weights and SASRec embeddings in the `data` folder. If you want to use your own dataset, modify the `data_process.py` script to suit your needs.

Then, run the shell script:
```bash
bash run.sh
```
to train the ETEGRec model. This will utilize the preprocessed data and pretrained weights to train the model.

### Demo
Run the following command to start the FastAPI server for inference:
```bash
cd demo
python app.py
```
this will start a web server watching port 5000. You can access the demo interface at `http://localhost:5000`.

# The following are the original README for ETEGRec:

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
