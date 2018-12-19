## Movie Review Sentiment Analysis (Kernels Only)

#### Folder Structure:

```
ml_training (train ml & dl classifier)
	--cnn
	    cnn.h5(cnn model)
	    cnn.py(train cnn classifier)
	--input(data folder)
	--lstm
	    LSTM and Bidirectional LSTM.py(train LSTM classifier)
	--traditional_ml
		models(save traditional ml models)
		naive_bayes_logistic_regression_training.py(train naive bayes and logistic regression)
		SVM_training.py(train naive bayes and logistic regression)
model_utils
    cnn_predictor.py (cnn)
    ml_predictor.py (ml)
    predictor.py (predictor)
web_app
    --crawler (crawler folder)
    --static
    app.py
    templates (web app html templates)
```
### Instruction Classifier Model Running

The *.ipynb files are for our development, the runnable scripts are .py files.

#### Traditional ML 

Author: Yucheng Tang(yuchengtang@brandeis.edu)

Run with:

```
python naive_bayes_logistic_regression_training.py
python SVM_training.py
```

- The naive bayes and logistic regression will run for all data.
- The SVM will run with 10000 sentences because it is very slow.

#### CNN
Author: Chuangxiong Yi



#### LSTM
Author: Yuan Zhou

Run with the following for training LSTM and Bidirectional LSTM:
```
python LSTM and Bidirectional LSTM.py
```




### Web App

Author: Jinli Yu

Run with:

```
FLASK_APP=app.py flask run
```
