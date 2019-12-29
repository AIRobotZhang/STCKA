# -*- coding: utf-8 -*-
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def assess(logit, target):
    # logit: n, output_size
    # target: n
    # target_names: output_size
    pred = torch.argmax(logit, dim=1)
    if torch.cuda.is_available():
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
    acc = accuracy_score(target, pred)
    p = precision_score(target, pred, average='macro')
    r = recall_score(target, pred, average='macro')
    f1 = f1_score(target, pred, average='macro')

    return acc, p, r, f1
    