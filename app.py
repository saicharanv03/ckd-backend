from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

def sigmoid(z):
    return 1 / (1 + pow(2.718281828459045, -z))

def encode_yes_no(value):
    return 1 if value and str(value).lower() == 'yes' else 0

def encode_normal_abnormal(value):
    return 1 if value and str(value).lower() == 'abnormal' else 0

def compute_risk_score(data):
    # Hypothetical logistic regression weights
    weights = {
        'age': 0.4,
        'bp': 0.3,
        'sg': -1.2,
        'al': 1.5,
        'su': 1.0,
        'rbc': 1.2,
        'pc': 1.3,
        'pcc': 1.1,
        'ba': 1.0,
        'bgr': 0.6,
        'bu': 0.5,
        'sc': 1.4,
        'sod': -0.5,
        'pot': 0.7,
        'hemo': -1.1,
        'pcv': -0.8,
        'wc': 0.4,
        'rc': 0.3,
        'htn': 1.5,
        'dm': 1.5,
        'cad': 1.2,
        'appet': -0.5,
    }
    bias = -3.5

    # Normalization helper
    def normalize(value, min_val, max_val):
        try:
            v = float(value)
            return (v - min_val) / (max_val - min_val)
        except:
            return 0

    z = 0
    z += weights['age'] * normalize(data.get('age', 0), 1, 120)
    z += weights['bp'] * normalize(data.get('bp', 0), 40, 200)
    z += weights['sg'] * normalize(data.get('sg', 1.005), 1.005, 1.025)
    z += weights['al'] * normalize(data.get('al', 0), 0, 5)
    z += weights['su'] * normalize(data.get('su', 0), 0, 5)
    z += weights['rbc'] * encode_normal_abnormal(data.get('rbc', 'normal'))
    z += weights['pc'] * encode_normal_abnormal(data.get('pc', 'normal'))
    z += weights['pcc'] * encode_normal_abnormal(data.get('pcc', 'normal'))
    z += weights['ba'] * encode_normal_abnormal(data.get('ba', 'normal'))
    z += weights['bgr'] * normalize(data.get('bgr', 0), 50, 400)
    z += weights['bu'] * normalize(data.get('bu', 0), 5, 200)
    z += weights['sc'] * normalize(data.get('sc', 0), 0.1, 10)
    z += weights['sod'] * normalize(data.get('sod', 0), 100, 200)
    z += weights['pot'] * normalize(data.get('pot', 0), 2, 10)
    z += weights['hemo'] * normalize(data.get('hemo', 0), 3, 20)
    z += weights['pcv'] * normalize(data.get('pcv', 0), 10, 60)
    z += weights['wc'] * normalize(data.get('wc', 0), 1, 20)
    z += weights['rc'] * normalize(data.get('rc', 0), 1, 10)
    z += weights['htn'] * encode_yes_no(data.get('htn', 'no'))
    z += weights['dm'] * encode_yes_no(data.get('dm', 'no'))
    z += weights['cad'] * encode_yes_no(data.get('cad', 'no'))
    appet_val = str(data.get('appet', 'good')).lower()
    z += weights['appet'] * (-0.5 if appet_val == 'good' else 0.5)
    z += bias
    return sigmoid(z)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'Missing JSON data'}), 400

    try:
        risk = compute_risk_score(data)
        risk_percent = round(risk * 100, 1)
        status = 'Low risk' if risk < 0.5 else 'High risk'
        message = f"{status} of Chronic Kidney Disease (Risk: {risk_percent}%)"
        return jsonify({
            'risk_score': risk,
            'risk_percent': risk_percent,
            'status': status,
            'message': message
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
