from joblib import dump, load

class MLPredictor:

    def __init__(self,path_prefix=''):
        nb_path = path_prefix+'model/nb_cv.joblib'
        lr_path = path_prefix+'model/lr_cv.joblib'
        cv_path = path_prefix+'model/cv.joblib'
        self.nb_classifier = load(nb_path)
        self.lr_classifier = load(lr_path)
        self.transformer = load(cv_path)

    def predict_single_sentence(self, sentence):
        """

        predict single sentences
        :param sentence: sentence str
        :return: integer label 0 ~ 4
        """
        return self.predict_sentences([sentence])[0]

    def predict_sentences(self, sentences, classifier_type='nb'):
        """

        :param sentences: sentence list
        :param classifier_type: classifier type
        :return: sentences label list
        """
        sentence_cv = self.transformer.transform(sentences)

        if classifier_type == 'lr':
            sentences_label = self.lr_classifier.predict(sentence_cv)
        else:
            sentences_label = self.nb_classifier.predict(sentence_cv)
        # for sentence, label in zip(sentences, sentences_label):
        #     print("Sentence:  " + sentence + "\nLabel:  " + str(label))
        return sentences_label
