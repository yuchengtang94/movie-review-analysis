
# coding: utf-8

# In[1]:


import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split   
from sklearn.svm import SVC


# In[2]:


test_mode = 1

train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv',  sep='\t')
sampleSub = pd.read_csv('../input/sampleSubmission.csv')


# In[3]:


if test_mode:
    train_data = train.Phrase[:10000]
    sentiment = train.Sentiment[:10000]
else:
    train_data = train.Phrase
    sentiment = train.Sentiment
x_train, x_test, y_train, y_test = train_test_split(train_data, sentiment, test_size = 0.2)


# In[4]:


cv = CountVectorizer(max_features = None)


# In[5]:


cv.fit(x_train)


# In[6]:


x_train_cv = cv.transform(x_train)
x_test_cv = cv.transform(x_test)


# In[7]:


SVM_classifier = SVC()


# In[8]:


SVM_classifier.fit(x_train_cv, y_train)


# In[9]:


y_pred = SVM_classifier.predict(x_test_cv)
print('Accuracy of test: ', metrics.accuracy_score( y_pred , y_test))


# In[10]:


from joblib import dump, load
if not debug_mode:
    nb_path = 'model/svm_cv.joblib'
    dump(SVM_classifier, nb_path)

