# Here we will create the views
# Views are same as routes in Django

import os
from predict import Predict

from flask import Flask, render_template
from form import chatForm
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)
topSecret = os.urandom(50)
app.config['SECRET_KEY'] = topSecret

a = Predict()

@app.route('/', methods=['GET', 'POST'])
def home():
    userInput = False
    form = chatForm()
    if form.validate_on_submit():
        userInput = form.box.data
        response = a.getResult(userInput)
        return render_template('home.html', form=form, userInput=userInput, response=response)
    return render_template('home.html', form=form, userInput=userInput)


if __name__ == '__main__':
    app.run(debug=True)
