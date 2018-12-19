
seed = 0

import random
import numpy as np
from tensorflow import set_random_seed

random.seed(seed)
np.random.seed(seed)
set_random_seed(seed)


import pandas as pd

train = pd.read_csv('../input/train.tsv',  sep="\t")
test = pd.read_csv('../input/test.tsv',  sep="\t")

test_mode = 1

if test_mode:
    train = train[:1000]

# In[ ]:


train.head()


# The mean and max lengths of phrases are the following:


train['Phrase'].str.len().mean()

train['Phrase'].str.len().max()

train['Sentiment'].value_counts()


# The values correspond to sentiments as follows:
# 
# ```
# 0 - negative
# 1 - somewhat negative
# 2 - neutral
# 3 - somewhat positive
# 4 - positive
# ```
# 
# We can see that most phrases are neutral, followed by the 'somewhats' (somewhat positive - somewhat negative). Then, far behind are positive and negative phrases. This (seemingly) makes classification harder, since most phrases are congregated towards the middle/neutral.
# 
# To train our model, we need to format our data.
# 
# Since we are dealing with text, we will first convert everything to lowercase. Then, we will tokenize our text. Currently, we build the tokenizer only on the training data. We could add to the ingredients the testing data, but results may go up or down. Testing is needed to determine whether or not adding the testing data will help. After the tokenization, we also need to pad the rows with zeros.
# 
# Apart from that, we need to convert the numerical output to categorical. Specifically, we need to one-hot encode the labels.
# 
# Finally, we need to shuffle our data as well.


def format_data(train, test, max_features, maxlen):
    """
    Convert data to proper format.
    1) Shuffle
    2) Lowercase
    3) Sentiments to Categorical
    4) Tokenize and Fit
    5) Convert to sequence (format accepted by the network)
    6) Pad
    7) Voila!
    """
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    
    train = train.sample(frac=1).reset_index(drop=True)
    train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())
    test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())

    X = train['Phrase']
    test_X = test['Phrase']
    Y = to_categorical(train['Sentiment'].values)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X))

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=maxlen)
    test_X = tokenizer.texts_to_sequences(test_X)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    return X, Y, test_X


maxlen = 125
max_features = 10000

X, Y, test_X = format_data(train, test, max_features, maxlen)



from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=seed)


# We will now start building our model.
# 
# Since we are dealing with sequences (the tokens) the input should be an embedding layer. To avoid overfitting, we will use `SpatialDropout` to drop some of the neurons in the embedding.
# 
# Afterwards, we build a CNN as normal. Since data is one-dimensional, we only need one-dimensional convolutions. After each convolution, we use `MaxPooling` to merge neighboring activations together. This will reduce the size of each layer and will keep the network from overfitting.

from keras.layers import Input, Dense, Embedding, Flatten
from keras.layers import SpatialDropout1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential


model = Sequential()

# Input / Embdedding
model.add(Embedding(max_features, 150, input_length=maxlen))

# CNN
model.add(SpatialDropout1D(0.2))

model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

# Output layer
model.add(Dense(5, activation='sigmoid'))

epochs = 5
batch_size = 16

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, verbose=1)


sub = pd.read_csv('../input/sampleSubmission.csv')

sub['Sentiment'] = model.predict_classes(test_X, batch_size=batch_size, verbose=1)
# sub.to_csv('sub_cnn.csv', index=False)

