from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained linear regression model
MODEL_PATH = 'linear_pkl.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing required model file: '{MODEL_PATH}' in this directory.")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Unified UI Design with modern gradients, typography, and crisp input layouts
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Insurance Cost Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 50px 0;
        }
        .predictor-card {
            background: #ffffff;
            border: none;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        .card-header-custom {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .card-header-custom h2 {
            font-weight: 700;
            margin-bottom: 5px;
        }
        .btn-predict {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            border: none;
            color: white;
            font-weight: 600;
            padding: 12px;
            transition: all 0.3s ease;
        }
        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
            color: white;
        }
        .result-box {
            background-color: #e8f5e9;
            border-left: 5px solid #2e7d32;
            color: #1b5e20;
            border-radius: 8px;
        }
        .form-label {
            font-weight: 600;
            color: #495057;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="row justify-content-center">
        <div class="col-md-8 col-lg-6">
            <div class="card predictor-card">
                <div class="card-header-custom">
                    <h2>Insurance Predictor</h2>
                    <p class="mb-0 text-white-50">Estimate Medical Charges via Machine Learning</p>
                </div>
                <div class="card-body p-4">
                    
                    {% if prediction_text %}
                    <div class="result-box p-3 mb-4 text-center">
                        <h4 class="mb-1 text-secondary fs-6 text-uppercase">Estimated Annual Cost</h4>
                        <h2 class="fw-bold m-0 text-success">{{ prediction_text }}</h2>
                    </div>
                    {% endif %}

                    <form action="/" method="POST">
                        <div class="row">
                            <div class="col-sm-6 mb-3">
                                <label for="age" class="form-label">Age (Years)</label>
                                <input type="number" class="form-control" id="age" name="age" min="1" max="120" required value="{{ inputs.get('age', '') }}" placeholder="e.g. 28">
                            </div>

                            <div class="col-sm-6 mb-3">
                                <label for="bmi" class="form-label">BMI (Body Mass Index)</label>
                                <input type="number" step="0.01" class="form-control" id="bmi" name="bmi" min="10" max="60" required value="{{ inputs.get('bmi', '') }}" placeholder="e.g. 24.5">
                            </div>
                        </div>

                        <div class="row">
                            <div class="col-sm-6 mb-3">
                                <label for="sex" class="form-label">Sex</label>
                                <select class="form-select" id="sex" name="sex" required>
                                    <option value="" disabled {% if 'sex' not in inputs %}selected{% endif %}>Choose Option</option>
                                    <option value="0" {% if inputs.get('sex') == 0 %}selected{% endif %}>Female</option>
                                    <option value="1" {% if inputs.get('sex') == 1 %}selected{% endif %}>Male</option>
                                </select>
                            </div>

                            <div class="col-sm-6 mb-3">
                                <label for="children" class="form-label">Number of Children</label>
                                <input type="number" class="form-control" id="children" name="children" min="0" max="15" required value="{{ inputs.get('children', '') }}" placeholder="e.g. 0">
                            </div>
                        </div>

                        <div class="row">
                            <div class="col-sm-6 mb-3">
                                <label for="smoker" class="form-label">Smoker Status</label>
                                <select class="form-select" id="smoker" name="smoker" required>
                                    <option value="" disabled {% if 'smoker' not in inputs %}selected{% endif %}>Choose Option</option>
                                    <option value="0" {% if inputs.get('smoker') == 0 %}selected{% endif %}>No</option>
                                    <option value="1" {% if inputs.get('smoker') == 1 %}selected{% endif %}>Yes</option>
                                </select>
                            </div>

                            <div class="col-sm-6 mb-3">
                                <label for="region" class="form-label">Geographic Region</label>
                                <select class="form-select" id="region" name="region" required>
                                    <option value="" disabled {% if 'region' not in inputs %}selected{% endif %}>Choose Region</option>
                                    <option value="0" {% if inputs.get('region') == 0 %}selected{% endif %}>Northeast</option>
                                    <option value="1" {% if inputs.get('region') == 1 %}selected{% endif %}>Northwest</option>
                                    <option value="2" {% if inputs.get('region') == 2 %}selected{% endif %}>Southeast</option>
                                    <option value="3" {% if inputs.get('region') == 3 %}selected{% endif %}>Southwest</option>
                                </select>
                            </div>
                        </div>

                        <div class="d-grid mt-4">
                            <button type="submit" class="btn btn-predict btn-lg text-uppercase">Calculate Cost</button>
                        </div>
                    </form>

                </div>
            </div>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    inputs = {}

    if request.method == 'POST':
        try:
            # Extract attributes from request
            age = float(request.form.get('age'))
            sex = int(request.form.get('sex'))
            bmi = float(request.form.get('bmi'))
            children = int(request.form.get('children'))
            smoker = int(request.form.get('smoker'))
            region = int(request.form.get('region'))

            # Track form state values 
            inputs = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}

            # Map array layout expected by scikit-learn
            features = np.array([[age, sex, bmi, children, smoker, region]])
            
            # Predict
            raw_prediction = model.predict(features)[0][0]
            prediction_text = f"${max(0.0, raw_prediction):,.2f}"
            
        except Exception as e:
            prediction_text = "Prediction Error"

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text, inputs=inputs)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
