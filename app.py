from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load('models/best_model_10_features.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values from the form
    input_data = [
        float(request.form['u-g']),        # Feature 1: u-g color
        float(request.form['g-r']),        # Feature 2: g-r color
        float(request.form['r-i']),        # Feature 3: r-i color
        float(request.form['i-z']),        # Feature 4: i-z color
        float(request.form['petroR50_r']), # Feature 5: Petrosian radius (r-band)
        float(request.form['petroFlux_r']),# Feature 6: Petrosian flux (r-band)
        float(request.form['psfMag_r']),   # Feature 7: PSF magnitude (r-band)
        float(request.form['expAB_r']),    # Feature 8: Exponential magnitude (r-band)
        float(request.form['modelFlux_r']),# Feature 9: Model flux (r-band)
        float(request.form['redshift'])    # Feature 10: Redshift
    ]

    # Convert input data to a numpy array and reshape for prediction
    input_array = np.array(input_data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_array)

    # Map prediction to class label
    class_label = 'STARFORMING' if prediction[0] == 0 else 'STARBURST'

    # Render the result page
    return render_template('inner-page.html', prediction=class_label)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)