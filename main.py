from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

model_pk = pickle.load(open("flower-v1.pkl", "rb"))


@app.route('/api_predict', methods=["GET", "POST"])
def api_predict():
    if request.method == "GET":
        return "PLEASE Send POST request"
    elif request.method == "POST":

        print("Hello" + str(request.get_json()))

        data = request.get_json()

        sepal_length = data['sepal_length']
        sepal_width = data['sepal_width']
        petal_length = data['petal_length']
        pepal_width = data['petal_width']

        in1 = np.array([[sepal_length, sepal_width, petal_length, pepal_width]])

        prediction = model_pk.predict(in1)

        return str(prediction)

if __name__ == '__main__':
    app.run()
