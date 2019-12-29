# Deep Short Text Classification with Knowledge Powered Attention
For the purpose of measuring the importance of knowledge, deep Short Text Classification with Knowledge powered Attention (STCKA) method introduces attention mechanisms, utilizing Concept towards Short Text (CST) attention and Concept towards Concept Set (C-CS) attention to acquire the weight of concepts from two aspects. And it can classify a short text with the help of conceptual information. Paper is available at [this site](https://arxiv.org/pdf/1902.08050.pdf).

#### For the purpose of reproducing this paper, we implemented this code.

## Requirements
* Python==3.7.4
* pytorch==1.3.1
* torchtext==0.3.1
* numpy
* tqdm

## Input data format
Snippets and TagMyNews Dataset can be available in dataset folder. The data format is as follows('\t' means TAB):

```
origin text \t concepts
...
```

## How to run
Train & Dev & Test:
Original dataset is randomly split into 80% for training and 20% for test. 20% of randomly selected training instances are used to form development set.

```
$ python main.py --epoch 100 --lr 2e-4 --train_data_path dataset/tagmynews.tsv --txt_embedding_path dataset/glove.6B.300d.txt  --cpt_embedding_path dataset/glove.6B.300d.txt  --embedding_dim 300 --train_batch_size 128 --hidden_size 64
```

More detailed configurations can be found in `config.py`, which is in utils folder.

## Cite
```
Chen J, Hu Y, Liu J, et al. Deep Short Text Classification with Knowledge Powered Attention[J]. 2019.
```

## Disclaimer

The code is for research purpose only and released under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).