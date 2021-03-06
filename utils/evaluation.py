from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def evaluate(ytrue,ypred):
    """Function to evaluate a model 4 metrics
    Input : ytrue = array with true labels, ypred = array with predicts of model
    Output : dataframe with accuracy, recall, precision and f1-score"""
    metrics = pd.DataFrame([],columns=['accuracy','recall','precision','f1-score'])
    metrics['accuracy'] = [accuracy_score(ytrue,ypred)]
    metrics['recall'] = [recall_score(ytrue,ypred,average='macro')]
    metrics['precision'] = [precision_score(ytrue,ypred,average='macro',zero_division=0)]
    metrics['f1-score'] = [f1_score(ytrue,ypred,average='macro',zero_division=0)]

    return metrics

def get_predictions(model,loader,model_type,gpu = True):
    """Function to get predictions from model
    Input : model = model to use for predictions, loader = data associated to the model, model_type = sort of model used
    Output : ytrue = array of true labels, ypred = array of predicted labels"""
    ypred = []
    ytrue = []
    if model_type == 'rnn':
        hidden = model.init_hidden(loader.batch_size)

    for it, data in enumerate(loader):
        if model_type=='bert':
            seq,attn_masks,labels = data
        elif model_type in ['rnn','cnn']:
            seq,attn_masks,labels = data[0],torch.ones(1),data[1] #attn_mask is not important here
        else:
            raise ValueError(f'Model type "{model_type}" not supported.')

        labels = labels.type(torch.LongTensor)
        if gpu:
            seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
        #Obtaining the logits from the model
        if model_type == 'rnn':
            hidden = tuple([each.data for each in hidden])
            out, hidden = model(seq, hidden)
        elif model_type == 'cnn':
            out = model(seq)
        elif model_type=='bert':
            out, attentions_val = model(seq, attn_masks)
        else:
            raise ValueError(f'Model type "{model_type}" not supported.')
        ypred.append(torch.argmax(out,dim=1).cpu().numpy())
        ytrue.append(labels.cpu().numpy())

    ytrue = [item for sublist in ytrue for item in sublist]
    ypred = [item for sublist in ypred for item in sublist]

    return ytrue,ypred


def plot_confusion_matrix(ytrue,ypred,norm=False,label_map=None):
    """Function to get confusion matrix
    Input : ytrue = true labels, ypred = predicted labels, norm = option to display normalised values, label_map = display names of labels"""
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
