from model_utils.predictor import predict

sentences = ["this movie is bad", "this movie is fucking bad", "this movie is fucking good", "this movie is funny"]

results = predict(sentences, 'ml_training/')

print(results)