from joblib import dump, load
nb_path = 'model/nb_cv.joblib'
cv_path = 'model/cv.joblib'

classifier = load(nb_path)
transformer = load(cv_path)

def predict_single_sentence(sentence):
    """

    predict single sentences
    :param sentence: sentence str
    :return: integer label 0 ~ 4
    """
    return predict_sentences([sentence])[0]

def predict_sentences(sentences):
    """
    
    predict sentences list
    :param sentences: list of sentences str
    :return: list of label integer
    """
    sentence_cv = transformer.transform(sentences)

    sentences_label = classifier.predict(sentence_cv)
    for sentence, label in zip(sentences, sentences_label):
        print("Sentence:  " + sentence + "\nLabel:  " + str(label))
    return sentences_label