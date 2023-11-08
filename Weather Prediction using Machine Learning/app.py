from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('weather.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    #val1 = request.form['Country']
    #val2 = request.form['State']
    val1 = request.form['State']
    val2 = request.form['Month']
    val3 = request.form['Day']
    val4 = request.form['Year']
    arr = np.array([ val1, val2, val3, val4])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', data=np.round(pred, 3))


if __name__ == '__main__':
    app.run(debug=True)
