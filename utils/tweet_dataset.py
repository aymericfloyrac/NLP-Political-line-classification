import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoModel, AutoTokenizer
import re
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset

class TweetDatasetBERT(Dataset):

    def __init__(self, df, maxlen, model_name='camembert-base'):

        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df['tweet'].iloc[index]
        label = self.df['label'].iloc[index]

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



def build_tweet_dataset(df,tokenizer = None):
    if tokenizer is None:
        n_most_common_words = 8000
        tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(df['tweet'].values)

    max_len = 250
    sequences = tokenizer.texts_to_sequences(df['tweet'].values)
    word_index = tokenizer.word_index
    X = pad_sequences(sequences, maxlen=max_len)
    y = df['label'].values
    data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    return data,tokenizer


class TweetDataset(Dataset):

    def __init__(self, df,word_ix):
        self.df = df
        self.word_ix = word_ix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df['tweet'].iloc[index]
        label = self.df['label'].iloc[index]
        #split and isolate punctuation
        sent = re.findall(r"[\w']+|[.,!?;]", sentence)
        idxs = prepare_sequence(sent,self.word_ix)
        idxs = pad_sequence(idxs,)

        return idxs,label
