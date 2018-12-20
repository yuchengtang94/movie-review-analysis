import numpy as np 
import pandas as pd 
import nltk
import os
import gc
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import matplotlib.pyplot as plt
import warnings
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()
from string import punctuation
import re
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', -1)

from keras.models import load_model

def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus

def predict_sentences(sentences, model_path):
    test_X = pre_process_test_set(sentences)
    model = load_model(model_path)
    res = model.predict_classes(test_X, verbose=1)
    return res

def pre_process_test_set(test):
    test = clean_review(test)    
    max_features = 13732
    max_words = 48
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(test))
    X_test = tokenizer.texts_to_sequences(test)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)
    return X_test