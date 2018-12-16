from joblib import dump, load

class NBPredictor:

    def __init__(self,path_prefix=''):
        nb_path = path_prefix+'model/nb_cv.joblib'
        cv_path = path_prefix+'model/cv.joblib'
        self.classifier = load(nb_path)
        self.transformer = load(cv_path)

    def predict_single_sentence(self, sentence):
        """

        predict single sentences
        :param sentence: sentence str
        :return: integer label 0 ~ 4
        """
        return self.predict_sentences([sentence])[0]

    def predict_sentences(self, sentences):
        """

        predict sentences list
        :param sentences: list of sentences str
        :return: list of label integer
        """
        sentence_cv = self.transformer.transform(sentences)

        sentences_label = self.classifier.predict(sentence_cv)
        for sentence, label in zip(sentences, sentences_label):
            print("Sentence:  " + sentence + "\nLabel:  " + str(label))
        return sentences_label
