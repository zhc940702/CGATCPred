import numpy as np
import pickle
import torch
fw = open('data/drug_ATC_label.pckl', 'rb')
drug_label = pickle.load(fw)
fw.close()
def extra(drug_label, t):
    Co_occurrence = np.zeros((drug_label.shape[1], drug_label.shape[1]))
    x = 0
    for k in range(drug_label.shape[1]):
        for i in range(drug_label.shape[0]):
            if drug_label[i, k] == 1:
                for l in range(k + 1, drug_label.shape[1]):
                    if drug_label[i, l] == 1:
                        Co_occurrence[k, l] = Co_occurrence[k, l] + 1
    for i in range(drug_label.shape[1]):
        for j in range(drug_label.shape[1]):
            Co_occurrence[j, i] = Co_occurrence[i, j]
    number_occur = np.sum(drug_label, axis=0)
    adj = Co_occurrence/(number_occur + 1e-6)
    adj[adj < t] = 0
    _adj = adj * 0.25 / (adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(drug_label.shape[1], np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj