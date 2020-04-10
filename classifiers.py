from torch import nn
from transformers import CamembertModel
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(RNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        out = self.dropout(lstm_out[:,-1,:])
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out.view(batch_size, -1)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

class CNN(nn.Module):

    def __init__(self, max_features, embed_size):
        super(CNN, self).__init__()
        filter_sizes = [1,2,3,5]
        num_filters = 64
        self.embedding = nn.Embedding(max_features, embed_size)
        #self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        #self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(num_filters*len(filter_sizes), 5)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        l = self.fc1(x)
        return l

class CamembertClassifier(nn.Module):

    def __init__(self, pretrained_model_name='camembert-base'):
        super(CamembertClassifier, self).__init__()
        self.encoder = CamembertModel.from_pretrained(pretrained_model_name,output_attentions=True)
        self.cls_layer = nn.Linear(self.encoder.pooler.dense.out_features, 5)

    def forward(self, seq, attn_masks):
        cont_reps, _, attentions = self.encoder(seq, attention_mask = attn_masks)
        cls_rep = cont_reps[:, 0]
        logits = self.cls_layer(cls_rep)

        return logits,attentions


def train(model, model_type,criterion, optimizer, scheduler,train_loader, val_loader,
          n_epochs=1, gpu=False, print_every=1,print_validation_every=1):


    model = model.to(device)

    #for plotting
    hist = {'loss':[],'accuracy':[]}
    val_hist = {'loss':[],'accuracy':[]}
    for ep in range(n_epochs):
        running_loss = 0 #used by the scheduler
        running_accuracy = 0

        if model_type=='rnn':
            h = model.init_hidden(train_loader.batch_size)

        for it, data in enumerate(train_loader):
            #extract right info from data
            if model_type=='bert':
                seq,attn_masks,labels = data
            elif model_type in ['rnn','cnn']:
                seq,attn_masks,labels = data[0],torch.ones(1),data[1] #attn_mask is not important here
            else:
                raise ValueError(f'Model type "{model_type}" not supported.')

            labels = labels.type(torch.LongTensor)
            #Clear gradients
            optimizer.zero_grad()
            #Converting these to cuda tensors
            if gpu:
              seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)
            #Obtaining the logits from the model
            if model_type == 'rnn':
                h = tuple([e.data for e in h])
                output,h = model(seq,h)
            elif model_type == 'cnn':
                output = model(seq)
            elif model_type =='bert':
                output,attentions = model(seq, attn_masks)
            else:
                raise ValueError(f'Model type "{model_type}" not supported.')

            #Computing loss
            loss = criterion(output.squeeze(-1), labels)
            running_loss += loss
            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            optimizer.step()

            #accuracy update
            accuracy = torch.sum(torch.argmax(output,dim=1)==labels)/float(labels.size(0))
            running_accuracy += accuracy

            if (it + 1) % print_every == 0:
                print("Iteration {} of epoch {} complete. Loss : {}, Accuracy {} ".format(it+1, ep+1, loss.item(),accuracy))

        #scheduler step
        if not scheduler is None:
            scheduler.step(running_loss)

        #update training history
        hist['loss'].append(running_loss/it) #mean
        hist['accuracy'].append(running_accuracy/it) #mean

        #VALIDATION
        model.eval()
        n_batch_validation = 0
        loss_validation = 0
        accuracy_validation = 0
        #init hidden if rnn
        if model_type == 'rnn':
            val_h = model.init_hidden(val_loader.batch_size)

        for it, data in enumerate(val_loader):

            #extract right info from data
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
                val_h = tuple([each.data for each in val_h])
                out, val_h = model(seq, val_h)
            elif model_type == 'cnn':
                out = model(seq)
            elif model_type=='bert':
                out, attentions_val = model(seq, attn_masks)
            else:
                raise ValueError(f'Model type "{model_type}" not supported.')

            n_batch_validation+=1
            #Computing loss
            _loss = float(criterion(out.squeeze(-1), labels))
            #computing scores
            ypred = torch.argmax(out,dim=1).cpu().numpy()
            ytrue = labels.cpu().numpy()
            _accu = torch.sum(torch.argmax(output,dim=1)==labels)/float(labels.size(0))
            loss_validation += _loss
            accuracy_validation += _accu
        #validation printing
        if ep % print_validation_every==0:
            print("EVALUATION Validation set : mean loss {} || mean accuracy {}".format(loss_validation/n_batch_validation, accuracy_validation/n_batch_validation))

        val_hist['loss'].append(loss_validation/n_batch_validation)
        val_hist['accuracy'].append(accuracy_validation/n_batch_validation)

        model.train()

    #plot history
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
    ax1.plot(hist['loss'])
    ax1.plot(val_hist['loss'])
    ax1.set_title('Evolution of training loss')

    ax2.plot(hist['accuracy'])
    ax2.plot(val_hist['accuracy'])
    ax2.set_title('Evolution of training accuracy')

    plt.tight_layout()
    plt.show()
