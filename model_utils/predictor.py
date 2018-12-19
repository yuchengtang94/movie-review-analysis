import numpy as np
from model_utils.ml_predictor import MLPredictor
from model_utils import cnn_predictor


def predict(sentences, path='../ml_training/'):
    """

    :param sentences: sentence list
    :param path: path to ml directory
    :return: score list predicted by multiple classifier
    """
    results = []
    ml_predictor = MLPredictor(path_prefix= path + 'traditional_ml/')
    lr_result = ml_predictor.predict_sentences(sentences, classifier_type='lr')
    nb_result = ml_predictor.predict_sentences(sentences, classifier_type='nb')
    cnn_result = cnn_predictor.predict_sentences(sentences, path + '/cnn/cnn.h5')
    results.append(lr_result)
    results.append(nb_result)
    results.append(cnn_result)

    return ensemble_average(results)


def ensemble_average(results):
    """
    Calculating average score for results predicted by different classifier
    :param results:
    :return: average score list
    """
    return np.round(np.average(np.array(results), axis=0)).astype(np.uint)