from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

model_pk = pickle.load(open("flower-v1.pkl", "rb"))


@app.route('/api_predict', methods=["GET", "POST"])
def api_predict():
    if request.method == "GET":
        return "Send POST request PLEASE"
    elif request.method == "POST":

        #print("Hello" + str(request.get_json()))
        #data = request.get_json()

        #sepal_length = data['sepal_length']
        #sepal_width = data['sepal_width']
        #petal_length = data['petal_length']
        #petal_width = data['petal_width']

        sepal_length = float(request.form.get('sepal_length'))
        sepal_width = float(request.form.get('sepal_width'))
        petal_length = float(request.form.get('petal_length'))
        petal_width = float(request.form.get('petal_width'))

        
        in1 = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        prediction = model_pk.predict(in1)

        return str(prediction)

if __name__ == '__main__':
    app.run()
