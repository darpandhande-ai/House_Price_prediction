import os
import pickle
import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# Load the pickle model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "linear_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 31, 49, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --input-bg: rgba(11, 15, 25, 0.6);
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --accent: #8b5cf6;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.15) 0%, transparent 40%);
            background-attachment: fixed;
        }

        .container {
            width: 100%;
            max-width: 950px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 35px;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #818cf8;
            margin: 25px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            transition: color 0.3s ease;
        }

        .input-group:focus-within label {
            color: var(--primary);
        }

        .input-group input, .input-group select {
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 12px 16px;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
            background: rgba(15, 23, 42, 0.8);
        }

        .btn-submit {
            width: 100%;
            margin-top: 35px;
            padding: 16px;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: white;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px -5px rgba(99, 102, 241, 0.6);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Result Modal / Box */
        .result-box {
            margin-top: 30px;
            padding: 24px;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            text-align: center;
            display: none;
            animation: slideUp 0.5s ease-out;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-box h3 {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .result-box .price {
            font-size: 2.2rem;
            font-weight: 700;
            color: #34d399;
            text-shadow: 0 0 20px rgba(52, 211, 153, 0.3);
        }

        .spinner {
            display: none;
            width: 22px;
            height: 22px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>🏠 House Price Predictor</h1>
        <p>Fill in the specifications below to estimate property market value</p>
    </div>

    <form id="predictionForm">
        <!-- 1. Structure & Size -->
        <div class="section-title">1. Structure & Size</div>
        <div class="grid">
            <div class="input-group">
                <label>Bedrooms</label>
                <input type="number" name="bedrooms" value="3" required step="1" min="0">
            </div>
            <div class="input-group">
                <label>Bathrooms</label>
                <input type="number" name="bathrooms" value="2" required step="0.25" min="0">
            </div>
            <div class="input-group">
                <label>Living Area (sq ft)</label>
                <input type="number" name="sqft_living" value="2000" required step="1">
            </div>
            <div class="input-group">
                <label>Lot Area (sq ft)</label>
                <input type="number" name="sqft_lot" value="5000" required step="1">
            </div>
            <div class="input-group">
                <label>Floors</label>
                <input type="number" name="floors" value="1" required step="0.5" min="1">
            </div>
            <div class="input-group">
                <label>Area (Excl. Basement)</label>
                <input type="number" name="sqft_above" value="1500" required step="1">
            </div>
            <div class="input-group">
                <label>Basement Area (sq ft)</label>
                <input type="number" name="sqft_basement" value="500" required step="1">
            </div>
        </div>

        <!-- 2. Quality & Ratings -->
        <div class="section-title">2. Quality & Ratings</div>
        <div class="grid">
            <div class="input-group">
                <label>Waterfront Present</label>
                <select name="waterfront">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div class="input-group">
                <label>Views (0-4)</label>
                <input type="number" name="view" value="0" min="0" max="4" required>
            </div>
            <div class="input-group">
                <label>House Condition (1-5)</label>
                <input type="number" name="condition" value="3" min="1" max="5" required>
            </div>
            <div class="input-group">
                <label>House Grade (1-13)</label>
                <input type="number" name="grade" value="7" min="1" max="13" required>
            </div>
        </div>

        <!-- 3. History & Location -->
        <div class="section-title">3. History & Location</div>
        <div class="grid">
            <div class="input-group">
                <label>Built Year</label>
                <input type="number" name="yr_built" value="1995" required>
            </div>
            <div class="input-group">
                <label>Renovation Year (0 if none)</label>
                <input type="number" name="yr_renovated" value="0" required>
            </div>
            <div class="input-group">
                <label>Postal Code</label>
                <input type="number" name="zipcode" value="98001" required>
            </div>
            <div class="input-group">
                <label>Latitude</label>
                <input type="number" name="lat" value="47.5100" step="0.0001" required>
            </div>
            <div class="input-group">
                <label>Longitude</label>
                <input type="number" name="long" value="-122.2100" step="0.0001" required>
            </div>
            <div class="input-group">
                <label>Living Area (Renovated)</label>
                <input type="number" name="sqft_living15" value="2000" required>
            </div>
            <div class="input-group">
                <label>Lot Area (Renovated)</label>
                <input type="number" name="sqft_lot15" value="5000" required>
            </div>
            <div class="input-group">
                <label>Schools Nearby</label>
                <input type="number" name="schools_nearby" value="2" required>
            </div>
            <div class="input-group">
                <label>Distance from Airport</label>
                <input type="number" name="distance_airport" value="15" required>
            </div>
        </div>

        <button type="submit" class="btn-submit" id="submitBtn">
            <span>Calculate Property Price</span>
            <div class="spinner" id="spinner"></div>
        </button>
    </form>

    <div class="result-box" id="resultBox">
        <h3>Estimated Market Value</h3>
        <div class="price" id="predictedPrice">$0</div>
    </div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const btn = document.getElementById('submitBtn');
        const spinner = document.getElementById('spinner');
        const resultBox = document.getElementById('resultBox');
        
        // Show spinner state
        spinner.style.display = 'block';
        btn.style.opacity = '0.8';
        btn.disabled = true;

        const formData = new FormData(this);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.success) {
                document.getElementById('predictedPrice').innerText = '$' + data.prediction;
                resultBox.style.display = 'block';
                resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            alert('Something went wrong. Please try again.');
        } finally {
            spinner.style.display = 'none';
            btn.style.opacity = '1';
            btn.disabled = false;
        }
    });
</script>

</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model file not found."})

    try:
        # Features ordered exactly as defined in linear_model.pkl
        feature_values = [
            float(request.form.get("bedrooms")),
            float(request.form.get("bathrooms")),
            float(request.form.get("sqft_living")),
            float(request.form.get("sqft_lot")),
            float(request.form.get("floors")),
            float(request.form.get("waterfront")),
            float(request.form.get("view")),
            float(request.form.get("condition")),
            float(request.form.get("grade")),
            float(request.form.get("sqft_above")),
            float(request.form.get("sqft_basement")),
            float(request.form.get("yr_built")),
            float(request.form.get("yr_renovated")),
            float(request.form.get("zipcode")),
            float(request.form.get("lat")),
            float(request.form.get("long")),
            float(request.form.get("sqft_living15")),
            float(request.form.get("sqft_lot15")),
            float(request.form.get("schools_nearby")),
            float(request.form.get("distance_airport")),
        ]

        features_array = np.array([feature_values])
        prediction = model.predict(features_array)[0]

        formatted_price = f"{max(0, prediction):,.2f}"

        return jsonify({"success": True, "prediction": formatted_price})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
