import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained linear regression model using standard pickle
try:
    with open("linear_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model file: {e}")

# HTML, CSS with Animations, and Layout embedded in app.py
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent-primary: #6366f1;
            --accent-glow: #818cf8;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: rgba(255, 255, 255, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.15) 0%, transparent 40%);
        }

        .container {
            width: 100%;
            max-width: 1000px;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        header h1 {
            font-size: 2.2rem;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .section-title {
            color: var(--accent-glow);
            font-size: 1.1rem;
            margin: 1.5rem 0 1rem 0;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.4rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.2rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }

        .form-group input, .form-group select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.65rem 0.9rem;
            color: #fff;
            font-size: 0.95rem;
            transition: all 0.3s ease;
        }

        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
        }

        .btn-submit {
            margin-top: 2rem;
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: #fff;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        .btn-submit:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
            background: linear-gradient(135deg, #4f46e5, #4338ca);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 12px;
            text-align: center;
            animation: pulseGlow 1.5s ease-out;
        }

        @keyframes pulseGlow {
            0% { transform: scale(0.95); opacity: 0; }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); opacity: 1; }
        }

        .result-box h2 {
            color: #10b981;
            font-size: 1.8rem;
        }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>🏡 House Price Predictor</h1>
        <p>Fill in the specifications below to estimate property market value</p>
    </header>

    {% if prediction %}
    <div class="result-box">
        <h2>Estimated Price: ${{ prediction }}</h2>
    </div>
    {% endif %}

    <form method="POST" action="/predict">
        
        <div class="section-title">1. Structure & Size</div>
        <div class="grid">
            <div class="form-group">
                <label>Bedrooms</label>
                <input type="number" name="number of bedrooms" value="{{ form_data.get('number of bedrooms', 3) }}" required>
            </div>
            <div class="form-group">
                <label>Bathrooms</label>
                <input type="number" step="0.25" name="number of bathrooms" value="{{ form_data.get('number of bathrooms', 2) }}" required>
            </div>
            <div class="form-group">
                <label>Living Area (sq ft)</label>
                <input type="number" step="any" name="living area" value="{{ form_data.get('living area', 2000) }}" required>
            </div>
            <div class="form-group">
                <label>Lot Area (sq ft)</label>
                <input type="number" step="any" name="lot area" value="{{ form_data.get('lot area', 5000) }}" required>
            </div>
            <div class="form-group">
                <label>Floors</label>
                <input type="number" step="0.5" name="number of floors" value="{{ form_data.get('number of floors', 1) }}" required>
            </div>
            <div class="form-group">
                <label>Area (Excl. Basement)</label>
                <input type="number" step="any" name="Area of the house(excluding basement)" value="{{ form_data.get('Area of the house(excluding basement)', 1500) }}" required>
            </div>
            <div class="form-group">
                <label>Basement Area (sq ft)</label>
                <input type="number" step="any" name="Area of the basement" value="{{ form_data.get('Area of the basement', 500) }}" required>
            </div>
        </div>

        <div class="section-title">2. Quality & Ratings</div>
        <div class="grid">
            <div class="form-group">
                <label>Waterfront Present</label>
                <select name="waterfront present">
                    <option value="0" {% if form_data.get('waterfront present') == '0' %}selected{% endif %}>No</option>
                    <option value="1" {% if form_data.get('waterfront present') == '1' %}selected{% endif %}>Yes</option>
                </select>
            </div>
            <div class="form-group">
                <label>Views (0-4)</label>
                <input type="number" min="0" max="4" name="number of views" value="{{ form_data.get('number of views', 0) }}" required>
            </div>
            <div class="form-group">
                <label>House Condition (1-5)</label>
                <input type="number" min="1" max="5" name="condition of the house" value="{{ form_data.get('condition of the house', 3) }}" required>
            </div>
            <div class="form-group">
                <label>House Grade (1-13)</label>
                <input type="number" min="1" max="13" name="grade of the house" value="{{ form_data.get('grade of the house', 7) }}" required>
            </div>
        </div>

        <div class="section-title">3. History & Location</div>
        <div class="grid">
            <div class="form-group">
                <label>Built Year</label>
                <input type="number" name="Built Year" value="{{ form_data.get('Built Year', 1995) }}" required>
            </div>
            <div class="form-group">
                <label>Renovation Year (0 if none)</label>
                <input type="number" name="Renovation Year" value="{{ form_data.get('Renovation Year', 0) }}" required>
            </div>
            <div class="form-group">
                <label>Postal Code</label>
                <input type="number" name="Postal Code" value="{{ form_data.get('Postal Code', 98001) }}" required>
            </div>
            <div class="form-group">
                <label>Latitude</label>
                <input type="number" step="any" name="Lattitude" value="{{ form_data.get('Lattitude', 47.51) }}" required>
            </div>
            <div class="form-group">
                <label>Longitude</label>
                <input type="number" step="any" name="Longitude" value="{{ form_data.get('Longitude', -122.21) }}" required>
            </div>
            <div class="form-group">
                <label>Living Area (Renovated)</label>
                <input type="number" step="any" name="living_area_renov" value="{{ form_data.get('living_area_renov', 2000) }}" required>
            </div>
            <div class="form-group">
                <label>Lot Area (Renovated)</label>
                <input type="number" step="any" name="lot_area_renov" value="{{ form_data.get('lot_area_renov', 5000) }}" required>
            </div>
            <div class="form-group">
                <label>Schools Nearby</label>
                <input type="number" name="Number of schools nearby" value="{{ form_data.get('Number of schools nearby', 2) }}" required>
            </div>
            <div class="form-group">
                <label>Distance from Airport</label>
                <input type="number" step="any" name="Distance from the airport" value="{{ form_data.get('Distance from the airport', 15) }}" required>
            </div>
        </div>

        <button type="submit" class="btn-submit">Calculate Property Price</button>
    </form>
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, form_data={}, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return "Model not loaded properly.", 500

    form_data = request.form.to_dict()

    # Features in exact order expected by linear_model.pkl
    feature_keys = [
        'number of bedrooms', 'number of bathrooms', 'living area', 'lot area',
        'number of floors', 'waterfront present', 'number of views',
        'condition of the house', 'grade of the house',
        'Area of the house(excluding basement)', 'Area of the basement',
        'Built Year', 'Renovation Year', 'Postal Code', 'Lattitude', 'Longitude',
        'living_area_renov', 'lot_area_renov', 'Number of schools nearby',
        'Distance from the airport'
    ]

    try:
        # Build input vector matching required types
        input_values = [float(form_data[key]) for key in feature_keys]
        df_input = pd.DataFrame([input_values], columns=feature_keys)
        
        # Predict
        pred_value = model.predict(df_input)[0]
        formatted_pred = f"{pred_value:,.2f}"
    except Exception as e:
        formatted_pred = f"Error in calculation: {e}"

    return render_template_string(HTML_TEMPLATE, form_data=form_data, prediction=formatted_pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
