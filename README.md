## Movie Review Sentiment Analysis (Kernels Only)

#### Build Dependency

```
build build_env.sh
```

#### Folder Structure:

```
ml_training (train ml & dl classifier)
	--cnn
	    cnn.h5(cnn model)
	    cnn.py(train cnn classifier)
	--input(data folder)
	--lstm
	    lstm_bilstm.py(train LSTM classifier)
	--traditional_ml
		    models(save traditional ml models)
		    naive_bayes_logistic_regression_training.py(train naive bayes and logistic regression)
		    SVM_training.py(train naive bayes and logistic regression)
model_utils
    cnn_predictor.py (cnn)
    ml_predictor.py (ml)
    lstm_predictor.py (lstm)
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

You can test with
```
python cnn.py
```

It will run with 1000 sentences.

#### LSTM
Author: Yuan Zhou

You can test with
```
python lstm_bilstm.py
```

It will run with 1000 sentences.

#### Note

To run with full data, you can set

```
test_mode = 0
```
In SVM cnn LSTM training script.

### Web App

Author: Jinli Yu

dependency:

install by pip:
flask
flask_wtf
bootstrap_flask
selenium
BeautifulSoup4

install by brew(in Mac) or apt-get(in linux)
chromedriver
reference: https://tecadmin.net/setup-selenium-chromedriver-on-ubuntu/

Run with:

```
cd web_app
FLASK_APP=app.py flask run
```
