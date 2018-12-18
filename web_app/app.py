# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, PasswordField
from wtforms.validators import DataRequired, Length

from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'dev'

bootstrap = Bootstrap(app)
db = SQLAlchemy(app)

app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True


class HelloForm(FlaskForm):
    moviename = StringField('movie', validators=[DataRequired(), Length(1, 100)])
    submit = SubmitField()


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/form', methods=['GET', 'POST'])
def test_form():
    form = HelloForm()
    return render_template('form.html', form=form)


@app.route('/nav', methods=['GET', 'POST'])
def test_nav():
    return render_template('nav.html')


@app.route('/pagination', methods=['GET', 'POST'])
def test_pagination():
    db.drop_all()
    db.create_all()
    for i in range(100):
        m = Message()
        db.session.add(m)
    db.session.commit()
    page = request.args.get('page', 1, type=int)
    pagination = Message.query.paginate(page, per_page=10)
    messages = pagination.items
    return render_template('pagination.html', pagination=pagination, messages=messages)


@app.route('/utils', methods=['GET', 'POST'])
def test_utils():
    return render_template('utils.html')

@app.route("/search",methods=['GET', "POST"])
def search():
    return render_template('index.html')

app.run(debug=True, host='0.0.0.0')
