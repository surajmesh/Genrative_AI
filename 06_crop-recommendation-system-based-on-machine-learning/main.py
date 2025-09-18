from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("NBClassifier.joblib")

@app.route('/')
def home():
    # Render the input form
    return render_template('index.html')

@app.route('/crop', methods=['GET'])
def crop():
    # Render the input form
    return render_template('crop.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        # Extract input values from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare input for the model
        input_features = np.array([[N, P, K, temperature, humidity , ph , rainfall]])

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Render the results page with the prediction
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
