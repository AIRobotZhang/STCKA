# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math

class STCK_Atten(nn.Module):
    def __init__(self, text_vocab_size, concept_vocab_size, embedding_dim, txt_embedding_weights,\
                        cpt_embedding_weights, hidden_size, output_size, gama=0.5, num_layer=1, finetuning=True):
        super(STCK_Atten, self).__init__()

        self.gama = gama
        da = hidden_size
        db = int(da/2)
     
        self.txt_word_embed = nn.Embedding(text_vocab_size, embedding_dim)
        if isinstance(txt_embedding_weights, torch.Tensor):
            self.txt_word_embed.weight = nn.Parameter(txt_embedding_weights, requires_grad=finetuning)
        self.cpt_word_embed = nn.Embedding(concept_vocab_size, embedding_dim)
        if isinstance(cpt_embedding_weights, torch.Tensor):
            self.cpt_word_embed.weight = nn.Parameter(cpt_embedding_weights, requires_grad=finetuning)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layer, batch_first=True, bidirectional=True)
        self.W1 = nn.Linear(2*hidden_size+embedding_dim, da)
        self.w1 = nn.Linear(da, 1, bias=False)
        self.W2 = nn.Linear(embedding_dim, db)
        self.w2 = nn.Linear(db, 1, bias=False)
        self.output = nn.Linear(2*hidden_size+embedding_dim, output_size)

    def self_attention(self, H):
        # H: batch_size, seq_len, 2*hidden_size
        hidden_size = H.size()[-1]
        Q = H
        K = H
        V = H
        # batch_size, seq_len, seq_len
        atten_weight = F.softmax(torch.bmm(Q, K.permute(0, 2, 1))/math.sqrt(hidden_size), -1)
        A = torch.bmm(atten_weight, V) # batch_size, seq_len, 2*hidden_size
        A = A.permute(0, 2, 1)
        # q: short text representation
        q = F.max_pool1d(A, A.size()[2]).squeeze(-1) # batch_size, 2*hidden_size

        return q

    def cst_attention(self, c, q):
        # c: batch_size, concept_seq_len, embedding_dim
        # q: batch_size, 2*hidden_size
        # print(q.size())
        # print(c.size())
        q = q.unsqueeze(1)
        q = q.expand(q.size(0), c.size(1), q.size(2))
        c_q = torch.cat((c, q), -1) # batch_size, concept_seq_len, embedding_dim+2*hidden_size
        c_q = self.w1(F.tanh(self.W1(c_q))) # batch_size, concept_seq_len, 1
        alpha = F.softmax(c_q.squeeze(-1), -1) # batch_size, concept_seq_len

        return alpha

    def ccs_attention(self, c):
        # c: batch_size, concept_seq_len, embedding_dim
        c = self.w2(F.tanh(self.W2(c))) # batch_size, concept_seq_len, 1
        beta = F.softmax(c.squeeze(-1), -1) # batch_size, concept_seq_len

        return beta

    def forward(self, txt_wordid, cpt_wordid):
        # txt_wordid: batch_size, text_seq_len
        # cpt_wordid: batch_size, concept_seq_len
        input_txt = self.txt_word_embed(txt_wordid) # input_: batch_size, seq_len, emb_dim
        output, (hn, cn) = self.lstm(input_txt) # output: batch_size, seq_len, 2*hidden_size
        q = self.self_attention(output) # text representation
        input_cpt = self.cpt_word_embed(cpt_wordid) # input_: batch_size, concept_seq_len, emb_dim
        alpha = self.cst_attention(input_cpt, q) # batch_size, concept_seq_len
        beta = self.ccs_attention(input_cpt) # batch_size, concept_seq_len

        cpt_atten = F.softmax(self.gama*alpha+(1-self.gama)*beta, -1) # batch_size, concept_seq_len
        p = torch.bmm(cpt_atten.unsqueeze(1), input_cpt).squeeze(1) # batch_size, emb_dim

        hidden_rep = torch.cat((q, p), -1) # batch_size, 2*hidden_size+emb_dim

        logit = self.output(hidden_rep)

        return logit
