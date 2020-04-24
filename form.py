from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

class chatForm(FlaskForm):
    box = StringField('Input')
    submit = SubmitField('Enter')