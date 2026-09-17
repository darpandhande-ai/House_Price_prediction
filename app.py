import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained scikit-learn model from the pickle file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Feature list based on model metadata
FEATURE_KEYS = [
    'number of bedrooms',
    'number of bathrooms',
    'living area',
    'lot area',
    'number of floors',
    'waterfront present',
    'number of views',
    'condition of the house',
    'grade of the house',
    'Area of the house(excluding basement)',
    'Area of the basement',
    'Built Year',
    'Renovation Year',
    'lot_area_renov',
    'Number of schools nearby',
    'Distance from the airport'
]

# HTML/CSS/JS UI Design Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="emerald">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real Estate AI Price Predictor & Analytics Dashboard</title>

    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">

    <!-- FontAwesome & Chart.js -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root[data-theme="emerald"] {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #064e3b 50%, #022c22 100%);
            --card-bg: rgba(15, 23, 42, 0.75);
            --card-border: rgba(52, 211, 153, 0.2);
            --accent-primary: #10b981;
            --accent-secondary: #06b6d4;
            --accent-glow: rgba(16, 185, 129, 0.4);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(30, 41, 59, 0.7);
            --chart-color-1: #10b981;
            --chart-color-2: #3b82f6;
        }

        :root[data-theme="cyber"] {
            --bg-gradient: linear-gradient(135deg, #180828 0%, #2e0854 50%, #0c021a 100%);
            --card-bg: rgba(24, 8, 40, 0.8);
            --card-border: rgba(236, 72, 153, 0.3);
            --accent-primary: #ec4899;
            --accent-secondary: #8b5cf6;
            --accent-glow: rgba(236, 72, 153, 0.5);
            --text-main: #ffffff;
            --text-muted: #c084fc;
            --input-bg: rgba(45, 16, 75, 0.6);
            --chart-color-1: #ec4899;
            --chart-color-2: #8b5cf6;
        }

        :root[data-theme="royal"] {
            --bg-gradient: linear-gradient(135deg, #0a192f 0%, #1e3a8a 50%, #0f172a 100%);
            --card-bg: rgba(15, 23, 42, 0.82);
            --card-border: rgba(96, 165, 250, 0.3);
            --accent-primary: #3b82f6;
            --accent-secondary: #f59e0b;
            --accent-glow: rgba(59, 130, 246, 0.5);
            --text-main: #f1f5f9;
            --text-muted: #93c5fd;
            --input-bg: rgba(30, 58, 138, 0.4);
            --chart-color-1: #3b82f6;
            --chart-color-2: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background 0.4s ease, border-color 0.4s ease, color 0.3s ease;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1rem;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .dashboard-container {
            width: 100%;
            max-width: 1400px;
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }

        /* Header Layout */
        .header-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            backdrop-filter: blur(16px);
            border-radius: 24px;
            padding: 1.5rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            animation: fadeInDown 0.8s ease-out;
        }

        .header-title h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, #fff, var(--text-muted));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .currency-selector, .theme-selector {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(0, 0, 0, 0.25);
            padding: 6px 14px;
            border-radius: 30px;
            border: 1px solid var(--card-border);
        }

        .currency-selector select {
            background: transparent;
            border: none;
            color: var(--text-main);
            font-weight: 600;
            font-size: 0.85rem;
            outline: none;
            cursor: pointer;
        }

        .currency-selector select option {
            background: #0f172a;
            color: var(--text-main);
        }

        .theme-btn {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border: 2px solid transparent;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .theme-btn:hover { transform: scale(1.2); }
        .theme-btn.emerald { background: #10b981; }
        .theme-btn.cyber { background: #ec4899; }
        .theme-btn.royal { background: #3b82f6; }

        /* Main Content Layout */
        .main-grid {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .main-grid { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            backdrop-filter: blur(16px);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            animation: fadeInUp 0.8s ease-out;
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-main);
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 0.75rem;
        }

        /* Form Controls */
        .inputs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 1.2rem;
            max-height: 520px;
            overflow-y: auto;
            padding-right: 8px;
        }

        .inputs-grid::-webkit-scrollbar {
            width: 6px;
        }
        .inputs-grid::-webkit-scrollbar-thumb {
            background: var(--card-border);
            border-radius: 10px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .input-group label {
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: capitalize;
        }

        .input-group input, .input-group select {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 0.75rem 1rem;
            border-radius: 12px;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 15px var(--accent-glow);
        }

        .submit-btn {
            width: 100%;
            margin-top: 1.5rem;
            padding: 1rem;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: white;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 10px 25px var(--accent-glow);
            transition: all 0.3s ease;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px var(--accent-glow);
        }

        /* Analytics Panel */
        .analytics-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .prediction-badge {
            background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .prediction-badge::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
            opacity: 0.3;
            pointer-events: none;
        }

        .prediction-label {
            font-size: 0.9rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }

        .prediction-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.8rem;
            font-weight: 700;
            color: var(--accent-primary);
            text-shadow: 0 0 20px var(--accent-glow);
        }

        .chart-container {
            position: relative;
            height: 300px;
            width: 100%;
        }

        /* Animations */
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
    </style>
</head>
<body>

<div class="dashboard-container">
    <!-- Top Header -->
    <header class="header-card">
        <div class="header-title">
            <h1><i class="fa-solid fa-chart-line" style="color: var(--accent-primary);"></i> RealEstate AI Predictor</h1>
        </div>
        <div class="header-controls">
            <!-- Currency Selection -->
            <div class="currency-selector">
                <i class="fa-solid fa-coins" style="color: var(--accent-primary); font-size: 0.85rem;"></i>
                <select id="currencySelect" onchange="updateCurrency()">
                    <option value="USD" data-symbol="$" data-rate="1">USD ($)</option>
                    <option value="INR" data-symbol="₹" data-rate="83.0">INR (₹)</option>
                    <option value="EUR" data-symbol="€" data-rate="0.92">EUR (€)</option>
                    <option value="GBP" data-symbol="£" data-rate="0.79">GBP (£)</option>
                </select>
            </div>

            <!-- Theme Selection -->
            <div class="theme-selector">
                <span style="font-size: 0.8rem; color: var(--text-muted); font-weight: 600;">Theme:</span>
                <button class="theme-btn emerald" onclick="setTheme('emerald')" title="Emerald Green"></button>
                <button class="theme-btn cyber" onclick="setTheme('cyber')" title="Cyberpunk Pink"></button>
                <button class="theme-btn royal" onclick="setTheme('royal')" title="Royal Blue"></button>
            </div>
        </div>
    </header>

    <!-- Main Section -->
    <div class="main-grid">
        <!-- Input Form Section -->
        <div class="card">
            <div class="card-title">
                <i class="fa-solid fa-sliders" style="color: var(--accent-primary);"></i>
                Property Parameters
            </div>
            <form id="predictionForm">
                <div class="inputs-grid">
                    <div class="input-group">
                        <label>Bedrooms</label>
                        <input type="number" name="number of bedrooms" value="3" step="1" required>
                    </div>
                    <div class="input-group">
                        <label>Bathrooms</label>
                        <input type="number" name="number of bathrooms" value="2.5" step="0.25" required>
                    </div>
                    <div class="input-group">
                        <label>Living Area (sqft)</label>
                        <input type="number" name="living area" value="2000" step="10" required>
                    </div>
                    <div class="input-group">
                        <label>Lot Area (sqft)</label>
                        <input type="number" name="lot area" value="5000" step="10" required>
                    </div>
                    <div class="input-group">
                        <label>Floors</label>
                        <input type="number" name="number of floors" value="2" step="0.5" required>
                    </div>
                    <div class="input-group">
                        <label>Waterfront Present</label>
                        <select name="waterfront present">
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>
                    <div class="input-group">
                        <label>Number of Views</label>
                        <input type="number" name="number of views" value="0" min="0" max="4" required>
                    </div>
                    <div class="input-group">
                        <label>Condition (1-5)</label>
                        <input type="number" name="condition of the house" value="3" min="1" max="5" required>
                    </div>
                    <div class="input-group">
                        <label>Grade (1-13)</label>
                        <input type="number" name="grade of the house" value="7" min="1" max="13" required>
                    </div>
                    <div class="input-group">
                        <label>Area Above Basement</label>
                        <input type="number" name="Area of the house(excluding basement)" value="1500" required>
                    </div>
                    <div class="input-group">
                        <label>Basement Area</label>
                        <input type="number" name="Area of the basement" value="500" required>
                    </div>
                    <div class="input-group">
                        <label>Built Year</label>
                        <input type="number" name="Built Year" value="1995" required>
                    </div>
                    <div class="input-group">
                        <label>Renovation Year</label>
                        <input type="number" name="Renovation Year" value="0" required>
                    </div>
                    <div class="input-group">
                        <label>Renovated Lot Area</label>
                        <input type="number" name="lot_area_renov" value="5000" required>
                    </div>
                    <div class="input-group">
                        <label>Nearby Schools</label>
                        <input type="number" name="Number of schools nearby" value="3" required>
                    </div>
                    <div class="input-group">
                        <label>Airport Distance (mi)</label>
                        <input type="number" name="Distance from the airport" value="15" required>
                    </div>
                </div>

                <button type="submit" class="submit-btn">
                    <i class="fa-solid fa-wand-magic-sparkles"></i> Predict Property Value
                </button>
            </form>
        </div>

        <!-- Analytics Output Dashboard -->
        <div class="analytics-panel">
            <div class="prediction-badge pulse">
                <div class="prediction-label">Estimated Market Value</div>
                <div class="prediction-value" id="predictionOutput">$0.00</div>
            </div>

            <div class="card" style="flex: 1;">
                <div class="card-title">
                    <i class="fa-solid fa-chart-pie" style="color: var(--accent-secondary);"></i>
                    Feature Impact Analysis
                </div>
                <div class="chart-container">
                    <canvas id="analyticsChart"></canvas>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    let rawBasePrediction = 0; // Store standard USD prediction base amount

    // Theme Switcher Logic
    function setTheme(themeName) {
        document.documentElement.setAttribute('data-theme', themeName);
        updateChartColors();
    }

    // Currency Switcher Logic
    function updateCurrency() {
        if (rawBasePrediction > 0) {
            renderCurrencyValue(rawBasePrediction);
        }
    }

    function renderCurrencyValue(usdAmount) {
        const select = document.getElementById('currencySelect');
        const selectedOption = select.options[select.selectedIndex];
        const rate = parseFloat(selectedOption.getAttribute('data-rate'));
        const code = select.value;

        const convertedAmount = usdAmount * rate;
        
        animateValue("predictionOutput", 0, convertedAmount, 800, code);
    }

    // Chart Setup using Chart.js
    let analyticsChart;
    function initChart() {
        const ctx = document.getElementById('analyticsChart').getContext('2d');
        analyticsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Living Area', 'Basement', 'Grade', 'Schools', 'Distance'],
                datasets: [{
                    label: 'Relative Weight Impact',
                    data: [65, 35, 80, 20, 15],
                    backgroundColor: [
                        getComputedStyle(document.documentElement).getPropertyValue('--accent-primary').trim(),
                        getComputedStyle(document.documentElement).getPropertyValue('--accent-secondary').trim(),
                        '#3b82f6',
                        '#f59e0b',
                        '#8b5cf6'
                    ],
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans' } },
                        grid: { display: false }
                    },
                    y: {
                        ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans' } },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    }
                }
            }
        });
    }

    function updateChartColors() {
        if (!analyticsChart) return;
        const color1 = getComputedStyle(document.documentElement).getPropertyValue('--accent-primary').trim();
        const color2 = getComputedStyle(document.documentElement).getPropertyValue('--accent-secondary').trim();
        
        analyticsChart.data.datasets[0].backgroundColor[0] = color1;
        analyticsChart.data.datasets[0].backgroundColor[1] = color2;
        analyticsChart.update();
    }

    // Form Submission Handler
    document.getElementById('predictionForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const data = {};
        formData.forEach((value, key) => data[key] = parseFloat(value));

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                rawBasePrediction = result.prediction;
                renderCurrencyValue(rawBasePrediction);
            } else {
                alert("Error making prediction: " + result.error);
            }
        } catch (err) {
            console.error(err);
            alert("Failed to communicate with prediction server.");
        }
    });

    function animateValue(id, start, end, duration, currencyCode) {
        const obj = document.getElementById(id);
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const current = progress * (end - start) + start;
            obj.innerHTML = new Intl.NumberFormat('en-US', { 
                style: 'currency', 
                currency: currencyCode,
                maximumFractionDigits: 2
            }).format(current);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // Initialize Chart on load
    window.onload = initChart;
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'success': False, 'error': 'Model not loaded correctly.'})

    try:
        data = request.json
        # Map parameters in the exact feature sequence expected by scikit-learn
        feature_vector = [float(data.get(k, 0)) for k in FEATURE_KEYS]
        
        # Convert to 2D numpy array for prediction
        input_array = np.array([feature_vector])
        
        # Calculate prediction
        prediction = model.predict(input_array)[0]

        return jsonify({
            'success': True,
            'prediction': float(prediction)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
