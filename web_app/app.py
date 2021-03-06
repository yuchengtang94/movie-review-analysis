# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

from flask_bootstrap import Bootstrap
from crawler import crawler
import sys
path = sys.path
sys.path.append("..")
from model_utils.predictor import predict
sys.path = path

app = Flask(__name__)
app.secret_key = 'dev'

bootstrap = Bootstrap(app)

app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

def map_label(label):
    m = ['negative','somewhat negative','neutral','somewhat positive','positive']
    return m[label] if label >= 0 and label < len(m) else 'not available'

class SearchForm(FlaskForm):
    moviename = StringField("moviename",validators=[DataRequired(),Length(1, 100)])
    submit = SubmitField()




@app.route('/', methods=['GET', 'POST'])
def index():
    form = SearchForm()
    return render_template('index.html', form=form)

@app.route("/search",methods=['GET', "POST"])
def search():
    # print("in search controler")
    form = SearchForm(request.form)
    # if request.method == 'POST' and form.validate():
    # print("go to rotten tomatoes")
    reviews = crawler.crawl_reviews_by_movie(form.moviename.data)
    if(reviews == None):
        # print("failed to get reviews")
        return render_template('reviews.html',reviews=reviews,movie_name=form.moviename.data)
    # print("success to get reviews")
    # nb_predictor = NBPredictor(path_prefix='../ml_training/traditional_ml/') #switch predictor here
    sentences = []
    for review in reviews:
        sentences.append(review['comment'])
    results = predict(sentences)
    for review, result in zip(reviews, results):
        review['predict'] = map_label(result)
        review['score'] = str(int(review['score'])/10) if review['score'].isdigit() else "N/A"
    return render_template('reviews.html',reviews=reviews,movie_name=form.moviename.data)
    # else:
    #     print("form not validate")
    #     pdb.set_trace()
    #     return render_template('index.html')

app.run(debug=True, host='0.0.0.0')
