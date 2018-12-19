import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import load_model
epochs = 5
batch_size = 16
max_features = 10000
maxlen = 125


sentences = ["this movie is bad", "this movie is fucking bad", "this movie is fucking good", "this movie is funny"]

def cnn_preprocessing(test_X):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(test_X))
    test_X = tokenizer.texts_to_sequences(test_X)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    return test_X


def predict_sentences(sentences, model_path, preprocess_func):
    test_X = preprocess_func(sentences)
    model = load_model(model_path)
    res = model.predict_classes(test_X, batch_size=batch_size, verbose=1)
    return res
