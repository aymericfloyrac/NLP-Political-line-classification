from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

def random_split_dataset(df,validation=True,deep=False):

    if validation:
        test_prop = 0.4
    else:
        test_prop = 0.3

    #convert labels
    LE = LabelEncoder()
    y = LE.fit_transform(df['couleur_politique'])
    label_map = {i:label for i,label in enumerate(LE.classes_)}
    df['label'] = y
    df = df[['tweet','label']]
    #split
    sss = StratifiedShuffleSplit(n_splits=1,test_size=test_prop,random_state=0)
    for train_index,test_index in sss.split(df,y):
        dftrain,dftest = df.iloc[train_index],df.iloc[test_index]
        ytrain,ytest = y[train_index],y[test_index]

    #convert tweets
    vectorizer = CountVectorizer()
    Xtrain = vectorizer.fit_transform(dftrain['tweet'])
    Xtest = vectorizer.transform(dftest['tweet'])

    if validation:
        val_sss = StratifiedShuffleSplit(n_splits=1,test_size=.5,random_state=0)
        for val_index,test_index in val_sss.split(Xtest,ytest):
            Xval,Xtest = Xtest[val_index],Xtest[test_index]
            yval,ytest = ytest[val_index],ytest[test_index]

        if deep:
            dfval = dftest.iloc[val_index]
            dftest = dftest.iloc[test_index]
            dftrain.reset_index(inplace=True,drop=True)
            dfval.reset_index(inplace=True,drop=True)
            dftest.reset_index(inplace=True,drop=True)
            return dftrain,ytrain,dfval,yval,dftest,ytest,label_map

        return Xtrain,ytrain,Xval,yval,Xtest,ytest,label_map

    if deep:
        dftrain.reset_index(inplace=True,drop=True)
        dftest.reset_index(inplace=True,drop=True)
        return dftrain,ytrain,dftest,ytest,label_map

    return Xtrain,ytrain,Xtest,ytest,label_map


def time_split_dataset(df,validation=True,deep=False):

#convert labels
    LE = LabelEncoder()
    y = LE.fit_transform(df['couleur_politique'])
    label_map = {i:label for i,label in enumerate(LE.classes_)}
    df['label'] = y

    dico_mois = {'Mar' : 3, 'Apr' : 4, 'May': 5}
    df = df.replace({"mois": dico_mois})
    df = df.sort_values(by=['mois', 'jour']).reset_index(drop = True)

    #time split train test
    df = df.reset_index(drop=True)
    frontiere_tv = round(len(df[df['mois'] == 3]) + int(len(df[df['mois'] == 4])/2))
    df = df[['tweet','label']]

    dftrain, dftest = df.iloc[:frontiere_tv].reset_index(drop=True), df.iloc[frontiere_tv:].reset_index(drop=True)
    ytrain,ytest = y[:frontiere_tv],y[frontiere_tv:]

    #convert tweets
    vectorizer = CountVectorizer()
    Xtrain = vectorizer.fit_transform(dftrain['tweet'])
    Xtest = vectorizer.transform(dftest['tweet'])

    if validation:
        frontiere_vt = round(Xtest.shape[0]/2)
        Xval,Xtest = Xtest[:frontiere_vt],Xtest[frontiere_vt:]
        yval,ytest = ytest[:frontiere_vt],ytest[frontiere_vt:]

        if deep:
            dfval = dftest.iloc[:frontiere_vt]
            dftest = dftest.iloc[frontiere_vt:]
            dftrain.reset_index(inplace=True,drop=True)
            dfval.reset_index(inplace=True,drop=True)
            dftest.reset_index(inplace=True,drop=True)
            return dftrain,ytrain,dfval,yval,dftest,ytest,label_map

        return Xtrain,ytrain,Xval,yval,Xtest,ytest,label_map

    if deep:
        dftrain.reset_index(inplace=True,drop=True)
        dftest.reset_index(inplace=True,drop=True)
        return dftrain,ytrain,dftest,ytest,label_map

    return Xtrain,ytrain,Xtest,ytest,label_map