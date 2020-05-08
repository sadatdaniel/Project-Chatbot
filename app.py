# Here we will create the views
# Views are same as routes in Django

import os
from predict import Predict

from flask import Flask, render_template
from form import chatForm
from flask_bootstrap import Bootstrap
from flask_restful import Resource, Api
from check import authenticate, identity
from flask_jwt import JWT, jwt_required


app = Flask(__name__)
Bootstrap(app)
topSecret = os.urandom(50)
app.config['SECRET_KEY'] = topSecret
api = Api(app)
jwt = JWT(app,authenticate,identity)

a = Predict()


class ChatAPI(Resource):

  #  @jwt_required()
  # commented out to avoid having access token. Implemented to make the application
  # future-proof

    def post(self,userInput):
        
        response = a.getResult(userInput,3)

        return response
        


api.add_resource(ChatAPI, '/api/<string:userInput>')

if __name__ == '__main__':
    app.run(debug=True)
