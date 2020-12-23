# -*- coding: utf-8 -*-

import pandas as pd
import scipy.io as sio
import numpy as np
from sklearn import metrics
from sklearn.metrics import hamming_loss

# hamming_loss（y_true，y_pred，*，sample_weight = None ）

def Hamming_loss(preLabels, test_targets, D):
    return hamming_loss(test_targets, preLabels)

def Intersect_set(a, b):
    countL = 0
    for i in range(len(a)):
        if a[i] == 1 and b[i] == 1:
            countL += 1
        else:
            continue
    return countL


def unionset(line1, line2):
    sum2 = 0
    for i in range(len(line1)):
        if (line1[i] == 0 and line2[i] == 1) or (line1[i] == 1 and line2[i] == 0) or (line1[i] == 1 and line2[i] == 1):
            sum2 += 1
    return sum2


def Aiming(preLabels, test_targets, D):
    # molecular
    sumsum1 = 0
    for i in range(D):
        line1, line2 = preLabels[i], test_targets[i]
        # denominator
        line1_count = 0
        for i in range(len(line1)):
            if line1[i] == 1:
                line1_count += 1
        sumsum1 += Intersect_set(line1, line2) / (line1_count + 1e-6)
    return sumsum1 / D


def Coverage(preLabels, test_targets, D):
    # molecular
    sumsum1 = 0
    for i in range(D):
        line1, line2 = preLabels[i], test_targets[i]
        # denominator
        line2_count = 0
        for i in range(len(line2)):
            if line2[i] == 1:
                line2_count += 1
        sumsum1 += Intersect_set(line1, line2) / (line2_count + 1e-6)
    return sumsum1 / D


def Abs_True_Rate(preLabels, test_targets, D):
    correct_pairs = 0
    for i in range(len(preLabels)):
        if np.all(preLabels[i] == test_targets[i]):
            correct_pairs += 1
    abs_true = correct_pairs / D
    return abs_true


def Abs_False_Rate(preLabels, test_targets, D):
    correct_pairs = 0.0
    for i in range(len(preLabels)):
        line1, line2 = preLabels[i], test_targets[i]
        correct_pairs += (unionset(line1, line2) - Intersect_set(line1, line2)) / 14
    abs_false = correct_pairs / D
    return abs_false


def Accuracy(preLabels, test_targets, D):
    acc_score = 0
    for i in range(len(preLabels)):
        item_inter = Intersect_set(preLabels[i], test_targets[i])
        item_union = unionset(preLabels[i], test_targets[i])
        acc_score += item_inter / (item_union + 1e-6)
    accuracy = acc_score / D
    return accuracy
