from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np


car_price_model_path = 'model02.pkl'  # Path to your trained car price model

# Load the car price prediction model
with open(car_price_model_path, 'rb') as file:
    car_price_model = pickle.load(file)

app = Flask(__name__)


@app.route('/car_price_predict', methods=['POST'])
def car_price_predict():
    # Extract data from the form for car price prediction
    mileage = request.form.get('Mileage', type=float)
    age = request.form.get('Age', type=int)

    # Validate inputs
    if mileage is None or age is None:
        return render_template('car_price.html', prediction_text='Invalid input. Please provide both mileage and age.')

    # Prepare the input features for the model
    final_features = np.array([[mileage, age]])

    # Make car price prediction
    prediction = car_price_model.predict(final_features)
    output = round(prediction[0], 2)  # Round the prediction to 2 decimal places

    return render_template('car_price.html', prediction_text=f'Predicted Car Price: ${output}')

@app.route('/')
def car_price_section():
    return render_template('car_price.html')

if __name__ == "__main__":
    app.run(debug=True)
