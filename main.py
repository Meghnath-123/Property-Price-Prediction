
# flask, scikit-learn, pandas, pickle-mixin
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

# Inspect the pipeline to understand its structure
print(pipe)

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')  # Fixed typo
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    # Validate form inputs
    if not location or not bhk or not bath or not sqft:
        return "Error: All fields are required."

    print(location, bhk, bath, sqft)
    input_df = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input_df)[0] * 1e5
    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)

