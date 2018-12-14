from naive_bayes_predictor import NBPredictor

sentences = ["this movie is bad", "this movie is fucking bad", "this movie is fucking good", "this movie is funny", "The whole thing just ... works.", "What ensues is enjoyably rooted in sentiment, genre stereotype and Rocky mythology, but does drag a little between the brutal but brilliantly staged fight scenes.", "It's just another Rocky sequel, and that sucks."]


nb_predictor = NBPredictor()

print("predicting one sentence.....")

nb_predictor.predict_single_sentence(sentences[0])

print("predicting sentences.....")

nb_predictor.predict_sentences(sentences)
