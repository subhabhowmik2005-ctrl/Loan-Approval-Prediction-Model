# from flask import Flask, render_template, request, jsonify
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load model
# try:
#     with open('loan_model.pkl', 'rb') as f:
#         data = pickle.load(f)
#         model = data['model']
#         model_accuracy = data['accuracy']
# except FileNotFoundError:
#     model = None
#     model_accuracy = 0.0

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None:
#         return jsonify({'error': 'Model not loaded'}), 500
    
#     try:
#         data = request.get_json()
        
#         features = np.array([[
#             int(data['education']),
#             int(data['self_employed']),
#             float(data['income_annum']),
#             float(data['loan_amount']),
#             float(data['loan_term']),
#             float(data['cibil_score']),
#             float(data['residential_assets_value']),
#             float(data['commercial_assets_value']),
#             float(data['luxury_assets_value']),
#             float(data['bank_asset_value'])
#         ]])
        
#         prediction = model.predict(features)[0]
#         # prediction is 0 or 1
        
#         result_text = "Approved" if prediction == 1 else "Rejected"
        
#         return jsonify({
#             'status': result_text,
#             'accuracy': f"{model_accuracy*100:.2f}%"
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True, port=5007)
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import traceback

app = Flask(__name__)

# ---------------- LOAD MODEL SAFELY ---------------- #
model = None
model_accuracy = 0.0

try:
    model_path = os.path.join(os.path.dirname(__file__), 'Loan_model.pkl')
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

        # Case 1: You saved {'model': model, 'accuracy': acc}
        if isinstance(data, dict):
            model = data.get('model', None)
            model_accuracy = data.get('accuracy', 0.0)
        else:
            # Case 2: You saved only the model
            model = data
            model_accuracy = 0.0

    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ Error loading model:", e)
    model = None


# ---------------- ROUTES ---------------- #
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        # Safe conversion with defaults
        features = np.array([[ 
            int(float(data.get('education', 0))),
            int(float(data.get('self_employed', 0))),
            float(data.get('income_annum', 0)),
            float(data.get('loan_amount', 0)),
            float(data.get('loan_term', 0)),
            float(data.get('cibil_score', 0)),
            float(data.get('residential_assets_value', 0)),
            float(data.get('commercial_assets_value', 0)),
            float(data.get('luxury_assets_value', 0)),
            float(data.get('bank_asset_value', 0))
        ]])

        prediction = model.predict(features)[0]
        result_text = "Approved" if prediction == 1 else "Rejected"

        return jsonify({
            'status': result_text,
            'accuracy': f"{model_accuracy*100:.2f}%"
        })

    except Exception as e:
        traceback.print_exc()  # Shows full error in terminal
        return jsonify({'error': str(e)}), 400


# ---------------- RUN APP ---------------- #
if __name__ == '__main__':
    app.run(debug=True, port=5007)
