import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
model_filename = 'best_knn_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the Salary Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Assuming the input will be a dictionary like {'YearsExperience': 5.0}
        # Convert input to DataFrame as the model was trained with DataFrame
        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)

        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # For deployment, you might use a production-ready WSGI server like Gunicorn
    # For local testing, you can run:
    # app.run(debug=True, host='0.0.0.0', port=5000)
    print("To run the Flask app locally, execute: python app.py")
    print("Or for deployment, use a WSGI server.")
