from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def evaluate(ytrue,ypred):

    metrics = pd.DataFrame([],columns=['accuracy','recall','precision','f1-score'])
    metrics['accuracy'] = [accuracy_score(ytrue,ypred)]
    metrics['recall'] = [recall_score(ytrue,ypred,average='macro')]
    metrics['precision'] = [precision_score(ytrue,ypred,average='macro',zero_division=0)]
    metrics['f1-score'] = [f1_score(ytrue,ypred,average='macro',zero_division=0)]

    return metrics

def get_binary_metrics(ytrue,ypred):

    labels =  np.unique(ytrue)
    results = pd.DataFrame([],columns = ['recall','precision','f1-score'],
                            index = labels)
    for l in labels:
        ytrue_l = (ytrue==l).astype(int)
        ypred_l = (ypred == l).astype(int)
        results.loc[l] = [recall_score(ytrue_l,ypred_l),
                          precision_score(ytrue_l,ypred_l,zero_division=0),
                          f1_score(ytrue_l,ypred_l,zero_division=0)  ]
    return results


def plot_confusion_matrix(ytrue,ypred,norm=False):
    #build confusion matrix
    plt.figure(figsize=(5,5))
    labels = np.unique(ytrue)
    cm = confusion_matrix(ytrue,ypred,labels = labels)
    if norm:
        cm = np.round(cm/cm.sum(axis=0),3)
        cm = np.nan_to_num(cm)
    #plot
    sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='.2f', cmap="Reds", vmin = 0.2)
    plt.title('Confusion matrix')
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.show()
