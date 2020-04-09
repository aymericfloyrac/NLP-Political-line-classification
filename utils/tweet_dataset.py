import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoModel, AutoTokenizer
import re

class TweetDatasetBERT(Dataset):

    def __init__(self, df, maxlen, model_name='camembert-base'):

        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.iloc[index, 'tweet']
        label = self.df.iloc[index, 'label']

        #Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence) #Tokenize the sentence
        if self.tokenizer.cls_token is None:
          bos_token = self.tokenizer.bos_token
        else:
          bos_token = self.tokenizer.cls_token

        if self.tokenizer.sep_token is None:
          eos_token = self.tokenizer.eos_token
        else:
          eos_token = self.tokenizer.sep_token

        tokens = [bos_token] + tokens + [eos_token] #Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.maxlen:
            tokens = tokens + [self.tokenizer.pad_token for _ in range(self.maxlen - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.maxlen-1] + [eos_token] #Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor
        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label


def create_word_ix(df):
    word_to_ix = {}
    for sent in df['tweet']:
        for word in re.findall(r"[\w']+|[.,!?;]", sent):
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else len(to_ix) for w in seq]
    idxs = torch.tensor(idxs, dtype=torch.long).unsqueeze(0)
    return idxs


class TweetDataset(Dataset):

    def __init__(self, df,word_ix):
        self.df = df
        self.word_ix = word_ix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.iloc[index, 'tweet']
        label = self.df.iloc[index, 'label']
        #split and isolate punctuation
        sent = re.findall(r"[\w']+|[.,!?;]", sentence)
        idxs = prepare_sequence(sent,self.word_idx)


        return idxs,label

		
		
class Prepa_CNN(Dataset):

    def __init__(self,df, maxlen, n_most_common_words, batch_size):

        self.df = df
        self.n_most_common_words = n_most_common_words
        self.tokenizer = Tokenizer(self.n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        self.maxlen = maxlen
        self.batch_size = batch_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        sentence = self.df.loc[index, 'tweet']
        label = self.df.loc[index, 'label']
        self.tokenizer.fit_on_texts(sentence)
        sequences = self.tokenizer.texts_to_sequences(sentence)
        word_index = self.tokenizer.word_index
        X = pad_sequences(sequences, maxlen= self.maxlen)
        data = TensorDataset(torch.from_numpy(X), torch.from_numpy(label))
        loader = DataLoader(data, shuffle=True, batch_size = self.batch_size, drop_last=True)

        return loader