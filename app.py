import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Model loading logic using built-in pickle
MODEL_PATH = "model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        # Fallback if named model.pkl or linear_model.pkl
        alt_path = "linear_model.pkl"
        if os.path.exists(alt_path):
            with open(alt_path, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError("Model file (model.pkl or linear_model.pkl) not found!")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Feature list matching the unpickled object structure exactly
FEATURE_NAMES = [
    'number of bedrooms', 'number of bathrooms', 'living area', 'lot area',
    'number of floors', 'waterfront present', 'number of views',
    'condition of the house', 'grade of the house',
    'Area of the house(excluding basement)', 'Area of the basement',
    'Built Year', 'Renovation Year', 'Postal Code', 'Lattitude',
    'Longitude', 'living_area_renov', 'lot_area_renov',
    'Number of schools nearby', 'Distance from the airport'
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.12);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background: var(--bg-gradient);
            background-attachment: fixed;
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        /* Animated Entrance Card */
        .container {
            width: 100%;
            max-width: 1100px;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeIn 0.8s ease-out forwards;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(30px) scale(0.98);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        h1 {
            text-align: center;
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 2rem;
        }

        /* Result Animation Box */
        .result-box {
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid #818cf8;
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 2rem;
            animation: pulseGlow 2s infinite alternate, bounce 0.6s ease-out;
        }

        @keyframes pulseGlow {
            0% { box-shadow: 0 0 10px rgba(129, 140, 248, 0.2); }
            100% { box-shadow: 0 0 25px rgba(129, 140, 248, 0.6); }
        }

        @keyframes bounce {
            0% { transform: translateY(-10px); }
            50% { transform: translateY(5px); }
            100% { transform: translateY(0); }
        }

        .result-title {
            font-size: 1.1rem;
            color: #cbd5e1;
        }

        .result-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #38bdf8;
            margin-top: 0.3rem;
        }

        /* Form Layout */
        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.2rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
            color: #cbd5e1;
            text-transform: capitalize;
        }

        input {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            color: #fff;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            font-size: 0.95rem;
            transition: all 0.3s ease;
        }

        input:focus {
            outline: none;
            border-color: #818cf8;
            box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.25);
            background: rgba(15, 23, 42, 0.9);
        }

        /* Animated Submit Button */
        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1.5rem;
            padding: 1rem;
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
            background: linear-gradient(90deg, #4f46e5, #7c3aed);
        }

        .btn-submit:active {
            transform: translateY(0);
        }
    </style>
</head>
<body>

<div class="container">
    <h1>🏠 House Price Prediction</h1>
    <p class="subtitle">Enter property details below to estimate value</p>

    {% if prediction %}
    <div class="result-box">
        <div class="result-title">Estimated Property Value</div>
        <div class="result-value">${{ prediction }}</div>
    </div>
    {% endif %}

    <form action="/predict" method="POST" class="grid-form">
        {% for feature in features %}
        <div class="form-group">
            <label for="{{ feature }}">{{ feature }}</label>
            <input 
                type="number" 
                step="any" 
                name="{{ feature }}" 
                id="{{ feature }}" 
                value="{{ inputs.get(feature, '') }}" 
                required
            >
        </div>
        {% endfor %}

        <button type="submit" class="btn-submit">✨ Calculate Prediction</button>
    </form>
</div>

</body>
</html>
"""

# Default input values for pre-filling the form on initial load
DEFAULT_INPUTS = {
    'number of bedrooms': 3,
    'number of bathrooms': 2,
    'living area': 2000,
    'lot area': 5000,
    'number of floors': 1,
    'waterfront present': 0,
    'number of views': 0,
    'condition of the house': 3,
    'grade of the house': 7,
    'Area of the house(excluding basement)': 1500,
    'Area of the basement': 500,
    'Built Year': 2000,
    'Renovation Year': 0,
    'Postal Code': 98001,
    'Lattitude': 47.5,
    'Longitude': -122.2,
    'living_area_renov': 2000,
    'lot_area_renov': 5000,
    'Number of schools nearby': 2,
    'Distance from the airport': 15
}

@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML_TEMPLATE, 
        features=FEATURE_NAMES, 
        inputs=DEFAULT_INPUTS, 
        prediction=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not loaded properly on server.", 500

    form_inputs = {}
    feature_values = []

    for name in FEATURE_NAMES:
        val = float(request.form.get(name, 0))
        form_inputs[name] = val
        feature_values.append(val)

    # Convert array into DataFrame with feature names to retain scikit-learn feature compatibility
    input_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
    
    raw_prediction = model.predict(input_df)[0]
    formatted_prediction = f"{raw_prediction:,.2f}"

    return render_template_string(
        HTML_TEMPLATE, 
        features=FEATURE_NAMES, 
        inputs=form_inputs, 
        prediction=formatted_prediction
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
