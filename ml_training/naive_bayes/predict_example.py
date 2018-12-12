from naive_bayes_predictor import NBPredictor

sentences = ["this movie is bad", "this movie is fucking bad", "this movie is fucking good", "this movie is funny"]


nb_predictor = NBPredictor()

print("predicting one sentence.....")

nb_predictor.predict_single_sentence(sentences[0])

print("predicting sentences.....")

nb_predictor.predict_sentences(sentences)