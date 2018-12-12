from naive_bayes_predictor import predict_sentences, predict_single_sentence

sentences = ["this movie is bad", "this movie is fucking bad", "this movie is fucking good", "this movie is funny"]

print("predicting sentences.....")
predict_sentences(sentences)

print("predicting single sentence.....")
predict_single_sentence(sentences[0])