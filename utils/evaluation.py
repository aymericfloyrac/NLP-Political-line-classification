from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def evaluate(ytrue,ypred):

    metrics = pd.DataFrame([],columns=['accuracy','recall','precision','f1-score'])
    metrics['accuracy'] = [accuracy_score(ytrue,ypred)]
    metrics['recall'] = [recall_score(ytrue,ypred,average='macro')]
    metrics['precision'] = [precision_score(ytrue,ypred,average='macro',zero_division=0)]
    metrics['f1-score'] = [f1_score(ytrue,ypred,average='macro',zero_division=0)]

    return metrics

def get_predictions(model,loader):
    ypred = []
    ytrue = []
    gpu=True
    for it, (seq, attn_masks, labels) in enumerate(loader):
        #Clear gradients

        labels = labels.type(torch.LongTensor)
        if gpu:
            seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
        #Obtaining the logits from the model
        logits_val,attentions = model(seq, attn_masks)
        ypred.append(torch.argmax(logits_val,dim=1).cpu().numpy())
        ytrue.append(labels.cpu().numpy())

    ytrue = [item for sublist in ytrue for item in sublist]
    ypred = [item for sublist in ypred for item in sublist]
    
    return ytrue,ypred

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


def plot_confusion_matrix(ytrue,ypred,norm=False,label_map=None):
    #build confusion matrix
    plt.figure(figsize=(5,5))
    labels = np.unique(ytrue)
    cm = confusion_matrix(ytrue,ypred,labels = labels)
    if norm:
        cm = np.round(cm/cm.sum(axis=0),3)
        cm = np.nan_to_num(cm)
        fmt = '.2f'
    else:
        fmt = 'd'
    if not label_map is None:
        labels = [label_map[l] for l in labels]
    #plot
    sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt=fmt, cmap="Reds", vmin = 0.2)
    plt.title('Confusion matrix')
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.show()
