# -*- coding: utf-8 -*-
import torch
from torchtext import data
from torchtext.vocab import Vectors
import numpy as np
from tqdm import tqdm

def load_dataset(train_data_path, dev_data_path, test_data_path, txt_wordVectors_path,\
                 cpt_wordVectors_path, train_batch_size, dev_batch_size, test_batch_size):

    tokenize = lambda x: x.split()
    txt_TEXT = data.Field(sequential=True, tokenize=tokenize, pad_token='<pad>',\
                                    lower=True, include_lengths=True, batch_first=True)
    cpt_TEXT = data.Field(sequential=True, tokenize=tokenize, pad_token='<pad>',\
                                    lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False, batch_first=True, unk_token=None)

    train_data = data.TabularDataset(path=train_data_path, format='tsv',
                                fields=[('text', txt_TEXT), ('concept', cpt_TEXT), ('label', LABEL)])
    if dev_data_path:
        dev_data = data.TabularDataset(path=dev_data_path, format='tsv',
                                     fields=[('text', txt_TEXT), ('concept', cpt_TEXT), ('label', LABEL)])
    if test_data_path:
        test_data = data.TabularDataset(path=test_data_path, format='tsv',
                                fields=[('text', txt_TEXT), ('concept', cpt_TEXT), ('label', LABEL)])
    # wordVectors_path = 'data/glove.6B/glove.6B.300d.txt'
    if txt_wordVectors_path:
        vectors = Vectors(name=txt_wordVectors_path)
        txt_TEXT.build_vocab(train_data, vectors=vectors)
        txt_word_embeddings = txt_TEXT.vocab.vectors
        print ("Vector size of Text Vocabulary: ", txt_TEXT.vocab.vectors.size())
    else:
        txt_TEXT.build_vocab(train_data)
        txt_word_embeddings = None
    
    if cpt_wordVectors_path:
        vectors = Vectors(name=cpt_wordVectors_path)
        cpt_TEXT.build_vocab(train_data, vectors=vectors)
        cpt_word_embeddings = cpt_TEXT.vocab.vectors
        print ("Vector size of Concept Vocabulary: ", cpt_TEXT.vocab.vectors.size())
    else:
        cpt_TEXT.build_vocab(train_data)
        cpt_word_embeddings = None
    LABEL.build_vocab(train_data)
    train_iter = data.Iterator(train_data, batch_size=train_batch_size, \
                                        train=True, sort=False, repeat=False, shuffle=True)
    dev_iter = None
    if dev_data_path:
        dev_iter = data.Iterator(dev_data, batch_size=dev_batch_size, \
                             train=False, sort=False, repeat=False, shuffle=True)
    test_iter = None
    if test_data_path:
        test_iter = data.Iterator(test_data, batch_size=test_batch_size, \
                                     train=False, sort=False, repeat=False, shuffle=False)

    txt_vocab_size = len(txt_TEXT.vocab)
    cpt_vocab_size = len(cpt_TEXT.vocab)

    label_size = len(LABEL.vocab)
    # print(src_TEXT.vocab.itos)
    # print(len(src_TEXT.vocab.itos))
    # print(src_vocab_size)
    # exit()

    # label_dict = dict(LABEL.vocab.stoi)
    # length = len(label_dict)
    # label_dict["<START>"] = length
    # label_dict["<STOP>"] = length+1

    return txt_TEXT, cpt_TEXT, txt_vocab_size, cpt_vocab_size, txt_word_embeddings, cpt_word_embeddings, \
           train_iter, dev_iter, test_iter, label_size

def train_test_split(all_iter, ratio):
    length = len(all_iter)
    train_data = []
    test_data = []
    train_start = 0
    train_end = int(length*ratio)
    ind = 0
    for batch in all_iter:
        if ind < train_end:
            train_data.append(batch)
        else:
            test_data.append(batch)
        ind += 1
    return train_data, test_data

def train_dev_split(train_iter, ratio):
    length = len(train_iter)
    train_data = []
    dev_data = []
    train_start = 0
    train_end = int(length*ratio)
    ind = 0
    for batch in train_iter:
        if ind < train_end:
            train_data.append(batch)
        else:
            dev_data.append(batch)
        ind += 1
    return train_data, dev_data


# if __name__ == '__main__':
#     # pass
