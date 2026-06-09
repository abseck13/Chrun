from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("data/churn_model_clean.pkl")

@app.route('/')
def home():
    return render_template("predict.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['Age'])
        account_manager = int(request.form['Account_Manager'])
        years = float(request.form['Years'])
        num_sites = int(request.form['Num_Sites'])

        input_data = np.array([[age, account_manager, years, num_sites]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)