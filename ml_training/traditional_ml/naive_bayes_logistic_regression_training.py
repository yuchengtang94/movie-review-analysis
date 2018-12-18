
# coding: utf-8

# ## Reference: 
#     
# https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
# 
# https://www.kaggle.com/snehithatiger/movie-review-sentiment-analysis

# In[ ]:


import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split   
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from joblib import dump, load


# In[2]:


train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv',  sep='\t')
sampleSub = pd.read_csv('../input/sampleSubmission.csv')


# In[3]:


train.head()


# In[4]:


test.head()


# ## Feature: Tokenization & Stemming 

# In[5]:


from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer,WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import re

stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()

# https://www.kaggle.com/nikitpatel/blending-with-lr-xgb-mnb-adaboost-kne-lsvc
def clean(review_raw):
    review_clean=[]
    for i in range(0,len(review_raw)):
        review=str(review_raw[i])
        review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_clean.append(review)
    return review_clean

# train.Phrase=clean(train.Phrase.values)
# test.Phrase=clean(test.Phrase.values)


# ## Feature: CountVectorizer

# In[6]:


cv = CountVectorizer(max_features = None, tokenizer=TweetTokenizer().tokenize)
cv.fit(train.Phrase)
data = cv.transform(train.Phrase)
x_final_test_cv = cv.transform(test.Phrase)

x_train_cv, x_test_cv, y_train_cv, y_test_cv = train_test_split(data, train.Sentiment, test_size = 0.1)


# ## Feature: TF-IDF

# In[7]:


tf_idf = TfidfVectorizer(tokenizer=TweetTokenizer().tokenize)


# In[8]:


tf_idf.fit(train.Phrase)


# In[9]:


data = tf_idf.transform(train.Phrase)
x_final_test_cv = tf_idf.transform(test.Phrase)

x_train_tf, x_test_tf, y_train_tf, y_test_tf = train_test_split(data, train.Sentiment, test_size = 0.1)


# ## Training

# In[10]:


def fit_classifier(classifier, x_train_feature, y_train, x_test_feature, y_test):

    classifier.fit(x_train_feature, y_train)
    y_pred = classifier.predict(x_test_feature)
    print(classifier)
    print('Accuracy: ', metrics.accuracy_score( y_pred , y_test))
    return classifier

def predict_final_result(classifier, x_final_test_feature):

    y_final_test_pred = nb_classifier.predict(x_final_test_feature)
    return y_final_test_pred

def save(model, path):
    dump(model, path)
    
# sampleSub.to_csv("naive_bayes_cv.csv", index=False)


# In[11]:


# save(cv, 'model/cv.joblib')
# save(cv, 'model/tf_idf.joblib')


# ## Naive Bayes classifier

# In[12]:


nb_classifier = MultinomialNB()
fit_classifier(nb_classifier, x_train_cv, y_train_cv, x_test_cv, y_test_cv)
save(nb_classifier, 'model/nb_cv.joblib')
fit_classifier(nb_classifier, x_train_tf, y_train_tf, x_test_tf, y_test_tf)
save(nb_classifier, 'model/nb_tf.joblib')
# predict_final_result(nb_classifier, x_final_test_cv)


# ## Logistic Regression classifier

# In[13]:


lr_classifier = LogisticRegression() 
fit_classifier(lr_classifier, x_train_cv, y_train_cv, x_test_cv, y_test_cv)
save(lr_classifier, 'model/lr_cv.joblib')
fit_classifier(lr_classifier, x_train_tf, y_train_tf, x_test_tf, y_test_tf)
save(lr_classifier, 'model/lr_tf.joblib')
# predict_final_result(lr_classifier, x_final_test_cv)


# ## Simple Prediction
# 
# 0 - negative
# 
# 1 - somewhat negative
# 
# 2 - neutral
# 
# 3 - somewhat positive
# 
# 4 - positive

# In[14]:


sentences = ["this movie is bad", "this movie is fucking bad", "this movie is fucking good", "this movie is funny"]


# In[15]:


def predict_sentences(sentences, classifier, transformer):
    
    sentence_cv = transformer.transform(sentences)
    return classifier.predict(sentence_cv)


# In[16]:


sentences_label = predict_sentences(sentences, nb_classifier, cv)
for sentence, label in zip(sentences, sentences_label):
        print("Sentence:  " + sentence + "\nLabel:  " + str(label))

