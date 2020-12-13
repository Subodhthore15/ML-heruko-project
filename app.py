import numpy as np
from flask import Flask, request, render_template, url_for
import pickle
import math

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=="POST":
        int_features  = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)] # 2d array required
        prediction = model.predict(final_features)

        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text='Number of Weekly Riders Should be {}'.format(math.floor(output)))


if __name__ == "__main__":
    app.run()
