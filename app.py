import numpy as np

from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load('main.pkl')
@app.route('/')
def home():
   return render_template('index.html')

# creating the Decorators so that we can link our pages
@app.route('/index',methods=['POST','GET'])
def index():
    # collecting the data from html
    if request.method == "POST":
        hours =int(request.form['content'])
        final_hours =[np.array(hours)]
        prediction=model.predict([final_hours])
        output = np.round(prediction[0], 2)
        return render_template('index.html', pedicted="Marks Scored should be {}".format((output)))
if __name__ == "__main__":
    app.run(debug=True)

