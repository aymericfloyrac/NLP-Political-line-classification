import json
import pandas as pd
import numpy as np
import re
import nltk
from gensim.models.phrases import Phrases, Phraser
from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import emoji
from tqdm import tqdm
import warnings
import os
from multiprocessing import Pool
import multiprocessing
import glob


def remove_hashtags(tokens):
    tokens = map(lambda x: x.replace('#', ''), tokens)
    return list(tokens)

def remove_url(tokens):
    tokens = filter(lambda x: "http" not in x, tokens)
    return list(tokens)

def remove_html(tokens):
    tokens = filter(lambda x: x[0]+x[-1] != '<>', tokens)
    return list(tokens)

def remove_emoji(tokens):
    tokens = [emoji.get_emoji_regexp().sub(u'', ''.join(tokens))]
    return tokens

def clean_tweets(tweet, keep_emoji):
    tokenizer = TweetTokenizer()
    tokenized_sentences = []
    tokens = tokenizer.tokenize(tweet)
    tokens = remove_url(tokens)
    tokens = remove_html(tokens)
    tokens = remove_hashtags(tokens)
    tokens = list(map(lambda x: x.replace('@', ''), tokens))
    tokens = list(map(lambda x: x.replace('"', ''), tokens))
    tokens = list(map(lambda x: x.replace('[', ''), tokens))
    tokens = list(map(lambda x: x.replace(']', ''), tokens))
    tokens = list(map(lambda x: x.lower(), tokens))
    u = TreebankWordDetokenizer().detokenize(tokens)
    tokenized_sentences.append(u)
    if keep_emoji == 'no':
        tokens = remove_emoji(tokenized_sentences)
        return tokens
    else:
        return tokenized_sentences

def clean_tweet_ML(df):
    for i in range(len(df)):
        try:
            df['tweet'][i] = df['tweet'][i].split(':')[1]
        except:
            pass
    var_list = ['tweet']
    for var in var_list:
        if var != 'tweet':
            #lower
            df[var] = df[var].str.lower()
            #on enlève les accents
            df[var] = df[var].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
            #on enlève les ponctuations
            df[var] = df[var].str.replace('[^\w\s]','')
    return df
