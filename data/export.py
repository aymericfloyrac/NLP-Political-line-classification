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


def get_list(line):
    """Function to get a list from row data
    Input : long string
    Output : list"""
    return list(line.split(", \'"))

def get_tweet_and_name_2(line):
    """Function to get a list from row data
    Input : list
    Output : tweet, name of account,name of usermention_name, text tweet, date of tweet"""
    liste = get_list(line)
    t = 0
    n = 0
    c = 0
    m = 0
    rt = 0
    if "{'created_at':" in liste[0]:
            date = liste[0][len("{'created_at': "):]
    if 'RT @' not in liste[3][:20]:
        for item in liste:
            if "user':" in item:
                c += 1
            if 'text\'' in item and t == 0:
                t += 1
                tweet = item[len("text': "):]
                text = item[len("text': "):]
            if 'name\'' in item and 'screen_name' not in item and n == 0 and c == 1:
                n += 1
                personne =  item[len("name': "):]
            if 'name\'' in item and 'screen_name' not in item and m == 0:
                m += 1
                usermention_name =  item[len("name': "):]
    if 'RT @' in liste[3][:20]:
        for item in liste:
            if "user':" in item:
                c += 1
            if 'text\'' in item and t == 0:
                t += 1
                text = item[len("text': "):]
            if "retweeted_status':" in item:
                rt += 1
            if 'text' in item and t == 1 and rt == 1:
                t += 1
                tweet = item[len("text': "):]
            if 'name\'' in item and 'screen_name' not in item and n == 0 and c == 1:
                n += 1
                personne =  item[len("name': "):]
            if 'name\'' in item and 'screen_name' not in item and m == 0:
                m += 1
                usermention_name =  item[len("name': "):]
    return tweet, personne, usermention_name, text, date
	
def clean_variables(liste):
    """Function to clean a list
    Input : list
    Output : cleaned list"""
    for i in range(len(liste)):
        liste[i] = liste[i][1:-1]
    return liste

def dummy_mentionne_candidat(candidat,var,data):
    """Function to create dummy variable for each candidate
    Input : candidat = name of candidate, var = variable, data = dataframe"""
    cont = [int(candidat in x) for x in list(data[var].str.split(" "))]
    return cont

def remove_hashtags(tokens):
    """Function to remove hashtags"""
    tokens = map(lambda x: x.replace('#', ''), tokens)
    return list(tokens)

def remove_url(tokens):
    """Function to remove URL"""
    tokens = filter(lambda x: "http" not in x, tokens)
    return list(tokens)

def remove_html(tokens):
    """Function to remove HTML"""
    tokens = filter(lambda x: x[0]+x[-1] != '<>', tokens)
    return list(tokens)

def remove_emoji(tokens):
    """Function to remove EMOJI"""
    tokens = [emoji.get_emoji_regexp().sub(u'', ''.join(tokens))]
    return tokens

def clean_tweets(tweet):
    """Function to clean raw tweet"""
    tokenizer = TweetTokenizer()
    tokenized_sentences = []
    tokens = tokenizer.tokenize(tweet)
    tokens = remove_url(tokens)
    tokens = remove_html(tokens)
    tokens = remove_hashtags(tokens)
    tokens = list(map(lambda x: x.replace('@', ''), tokens))
    tokens = list(map(lambda x: x.lower(), tokens))
    u = TreebankWordDetokenizer().detokenize(tokens)
    tokenized_sentences.append(u)
    #tokens = remove_emoji(tokenized_sentences)
    return tokenized_sentences


def get_csv(data):
    """Function to get a clean csv from row data
    Input : file.txt
    Output : df_new.csv"""
    tweets = []
    names = []
    user_mention_name = []
    text = []
    date = []
    for i in range(len(data)):
        try:
            tweets.append(get_tweet_and_name_2(data[i])[0])
            names.append(get_tweet_and_name_2(data[i])[1])
            user_mention_name.append(get_tweet_and_name_2(data[i])[2])
            text.append(get_tweet_and_name_2(data[i])[3])
            date.append(get_tweet_and_name_2(data[i])[4])
        except:
            pass

    df = pd.DataFrame({'nom': names,'tweet': tweets, 'user_nom': user_mention_name, 'texte': text, 'date': date})
    var_list = ['nom','user_nom','tweet','texte']
    for var in var_list:
        if var != 'tweet':
            if var != 'texte':
                df[var] = df[var].str.lower()
                df[var] = df[var].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
                df[var] = df[var].str.replace('[^\w\s]','')
        if var == 'tweet':
            df[var] = df[var].str.lower()
        if var == 'texte':
            df[var] = df[var].str.lower()

    liste_candidats = ['emmanuel macron','marine le pen','francois fillon','benoit hamon','jeanluc melenchon','n dupontaignan','nathalie arthaud','jacques cheminade','philippe poutou','francois asselineau','jean lassalle']
    liste_retweet_candidats = ['emmanuelmacron','mlp_officiel','francoisfillon','benoithamon','jlmelenchon','dupontaignan','n_arthaud','jcheminade','philippepoutou','upr_asselineau','jeanlassalle']
    to_keep = []
    for i in range(len(df)):
        if df['nom'][i] in liste_candidats:
            to_keep.append(i)
        for item in liste_candidats:
            if df['user_nom'][i] == item:
                if 'rt @' in df['texte'][i]:
                    for item in liste_retweet_candidats:
                        if 'rt @'+item in df['texte'][i][0:19]:
                            to_keep.append(i)
    
    df = df[df.index.isin(set(to_keep))].reset_index(drop=True)
    
    liste_candidats = ['macron','pen','fillon','hamon','melenchon','dupontaignan','arthaud','cheminade','poutou','asselineau','lassalle']
    mention_col = ['cont_' + x for x in liste_candidats]
    for i in range(len(liste_candidats)):
        df[mention_col[i]] = dummy_mentionne_candidat(liste_candidats[i],'user_nom',df)
  
    df["partie_politique_associe"] = "RAS"
    df["partie_politique_associe"][df.cont_macron == 1] = "la republique en marche"
    df["partie_politique_associe"][df.cont_pen== 1] = "front national"
    df["partie_politique_associe"][df.cont_hamon == 1] = "parti socialiste"
    df["partie_politique_associe"][df.cont_lassalle == 1] = "resistons"
    df["partie_politique_associe"][df.cont_cheminade == 1] = "solidarite et progres"
    df["partie_politique_associe"][df.cont_dupontaignan == 1] = "debout la france"
    df["partie_politique_associe"][df.cont_fillon == 1] = "les republicains"
    df["partie_politique_associe"][df.cont_arthaud == 1] = "lutte ouvriere"
    df["partie_politique_associe"][df.cont_poutou == 1] = "nouveau parti anticapitaliste"
    df["partie_politique_associe"][df.cont_melenchon == 1] = "france insoumise"
    df["partie_politique_associe"][df.cont_asselineau == 1] = "union populaire republicaine"
    df["couleur_politique"] = "RAS"
    df["couleur_politique"][(df.partie_politique_associe == "france insoumise") | (df.partie_politique_associe == "lutte ouvriere") | (df.partie_politique_associe == "nouveau parti anticapitaliste")] = "EG"
    df["couleur_politique"][(df.partie_politique_associe == "parti socialiste") | (df.partie_politique_associe == "solidarite et progres")] = "G"
    df["couleur_politique"][(df.partie_politique_associe == "la republique en marche")] = "C"
    df["couleur_politique"][(df.partie_politique_associe == "les republicains") | (df.partie_politique_associe == "resistons")] = "D"
    df["couleur_politique"][(df.partie_politique_associe == "union populaire republicaine") | (df.partie_politique_associe == "front national") | (df.partie_politique_associe == "debout la france")] = "ED"

    df_new = df.loc[df.astype(str).drop_duplicates(['tweet','user_nom']).index]
    df_new = df_new.reset_index(drop=True)
    return df_new

def f(file):
    brut = open(path_input + folder + file,'r').read().split('\n')
    prepare = get_csv(brut)
    prepare.to_csv(path_output+ folder + file +".csv", sep=',', index=False)

if __name__=='__main__':

    liste_folder = ["24032020/","25032020/","26032020/","27032020/","01042020/","02042020/"]
    #version parallélisée
    for folder in liste_folder:
        print(folder)
        editFiles = []
        for item in os.listdir(path_input + folder):
            if "x" in item:
                editFiles.append(item)
        with Pool(processes = 2) as p:
            p.map(f, [file for file in editFiles])


    for mycsvdir in liste_folder:
        csvfiles = glob.glob(os.path.join(path_output + mycsvdir, '*.csv'))
        dataframes = []
        for csvfile in csvfiles:
            df = pd.read_csv(csvfile, sep = ",")
            dataframes.append(df)
        result = pd.concat(dataframes, ignore_index=True)
        result.to_csv(path_output+ mycsvdir +"base_tot_jour.csv", sep=',', index=False)

    dataframes = []
    csvfiles = []
    for mycsvdir in liste_folder:
        csvfile = glob.glob(os.path.join(path_output + mycsvdir, 'base_tot_jour.csv'))
        csvfiles.append(csvfile)
    for csv in csvfiles:
        df = pd.read_csv(csv[0], sep = ",")
        dataframes.append(df)
    term = pd.concat(dataframes, ignore_index=True)
    term = term[term["couleur_politique"] != "RAS"]
    term.to_csv(path_output + "base_tweets_finale_emoji.csv", sep=',', index=False)
