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
    """Function to remove hashtags in text"""
    tokens = map(lambda x: x.replace('#', ''), tokens)
    return list(tokens)

def remove_url(tokens):
    """Function to remove URL in text"""
    tokens = filter(lambda x: "http" not in x, tokens)
    return list(tokens)

def remove_html(tokens):
    """Function to remove html in text"""
    tokens = filter(lambda x: x[0]+x[-1] != '<>', tokens)
    return list(tokens)

def remove_emoji(tokens):
    """Function to remove emoji in text"""
    tokens = [emoji.get_emoji_regexp().sub(u'', ''.join(tokens))]
    return tokens

def clean_tweets(tweet, keep_emoji):
    """Function to clean text of tweet"""
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
    """Function to finish to clean tweet"""
    for i in range(len(df)):
        try:
            df['tweet'][i] = df['tweet'][i].split(':')[1]
        except:
            pass
    var_list = ['tweet']
    for var in var_list:
        if var != 'tweet':
            df[var] = df[var].str.lower()
            df[var] = df[var].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
            df[var] = df[var].str.replace('[^\w\s]','')
    return df

def normalise_text(text):
    """Function to clear text of tweet"""
    text = text.str.lower() 
    text = text.str.replace(r"\#","")
    text = text.str.replace(r"http\S+","URL") 
    text = text.str.replace(r"@","")
    #text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    #text = text.str.replace('[^\w\s#@/:%.,_-]', ' ', flags=re.UNICODE)
    return text
