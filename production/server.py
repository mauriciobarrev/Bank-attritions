'''
I'll apply the Machine Learning model into an API to use it on web (in this case in localhost).
It generates the results of the model on JSON to work on different ways in production!
'''

import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # first entry of data_balanced.csv for testing
    X_test = np.array([-0.9702856200111142, -0.47680555992043105, 0.7638417007469629, -1.4631379688590884])
    prediction = model.predict(X_test.reshape(1, -1))
    return jsonify({"prediction" : int(prediction)})

if __name__ == '__main__':
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8080)