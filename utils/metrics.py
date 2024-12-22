'''Loading packages'''
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from sklearn import metrics

from sklearn.metrics import average_precision_score


'''
Defining the metrics for measuring the performance
Scenario 1: a binary classification problem
Scenario 2: multi-label classification problem
'''

'''Scenario 1: a binary classification problem'''
def print_metrics_binary(y_true, predictions, verbose=1):

    '''Preparing prediction results and groundtruth labels'''
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    '''Calculating confusion matrix, accuracy, precision and recall'''
    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])

    '''Calculating AUROC'''
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])
    
    '''Calculating AUPR (AUPRC)'''
    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)

    '''Calculating F1-score'''
    f1_score= 2*prec1*rec1/(prec1+rec1)
 
    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("f1_score = {}".format(f1_score))

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "f1_score":f1_score}
    
    
def print_metrics_multilabel(y_true, predictions, verbose=1):
    y_true = np.array(y_true)
    predictions = np.array(predictions)

    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)    
        
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")
    
    auprc_scores = average_precision_score(y_true, predictions, average=None)
    ave_auprc = average_precision_score(y_true, predictions, average='macro')
    
    predictions2 = np.zeros_like(predictions)
    for i in range(len(predictions2)):
        for j in range(len(predictions2[i])):
            if predictions[i][j]>=0.5:
                predictions2[i][j] = 1
 
    f1_0 = metrics.f1_score(y_true[:,0], predictions2[:,0])
    f1_1 = metrics.f1_score(y_true[:,1], predictions2[:,1])
    f1_2 = metrics.f1_score(y_true[:,2], predictions2[:,2])
    f1_3 = metrics.f1_score(y_true[:,3], predictions2[:,3])
    f1_4 = metrics.f1_score(y_true[:,4], predictions2[:,4])
    f1_5 = metrics.f1_score(y_true[:,5], predictions2[:,5])
    f1_6 = metrics.f1_score(y_true[:,6], predictions2[:,6])
    f1_7 = metrics.f1_score(y_true[:,7], predictions2[:,7])
    
    total_labels = np.array(list(y_true[:,0])+list(y_true[:,1])+list(y_true[:,2]))
    total_preds = np.array(list(predictions2[:,0])+list(predictions2[:,1])+list(predictions2[:,2]))
    
    ave_f1_micro = metrics.f1_score(total_labels, total_preds)
    ave_f1_macro = (f1_0+f1_1+f1_2+f1_3+f1_4+f1_5+f1_6+f1_7)/8
    f1_scores = [f1_0, f1_1, f1_2, f1_3, f1_4, f1_5, f1_6, f1_7]
    
    coverage_error = metrics.coverage_error(y_true, predictions)
    label_ranking_loss = metrics.label_ranking_loss(y_true, predictions)

    if verbose:
        print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))
        print("auprc_scores = {}".format(auprc_scores))
        print("ave_auprc = {}".format(ave_auprc))
        print("f1_scores = {}".format(f1_scores))
        print("ave_f1_micro = {}".format(ave_f1_micro))
        print("ave_f1_macro = {}".format(ave_f1_macro))
        print("coverage_error = {}".format(coverage_error))
        print("label_ranking_loss = {}".format(label_ranking_loss))

    return {"auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted,
            "auprc_scores": auprc_scores,
            "ave_auprc": ave_auprc,
            "f1_scores": f1_scores,
            "ave_f1_micro": ave_f1_micro,
            "ave_f1_macro": ave_f1_macro,
            "coverage_error": coverage_error,
            "label_ranking_loss": label_ranking_loss}